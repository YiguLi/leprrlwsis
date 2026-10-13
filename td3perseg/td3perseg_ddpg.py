import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from PER_buffer import PrioritizedReplayBuffer
from random_process import OrnsteinUhlenbeckProcess
from util import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()


class SP_TD3(object):
    """TD3 + PER + Lagrangian cost constraint.

    Adapted from PER_ddpg.SP_DDPG. TD3 upgrades over DDPG:
      1. Twin critics per channel (critic_p/critic_p2 for profit,
         critic_c/critic_c2 for cost). Profit targets and policy loss
         take the MIN of each pair (clipped double Q-learning); the cost
         aggregate defaults to MAX (safe: cost underestimation would make
         the Lagrangian treat constraints as cheaper than they are) and
         cost bootstrap targets are clamped to >= 0 (costs are
         non-negative by construction).
      2. Delayed policy update: actor + target nets are soft-updated
         only every `policy_delay` critic updates.
      3. Target policy smoothing: clipped Gaussian noise added to
         target actions, scaled per dimension by action amplitude.

    Everything else (PER buffer, importance sampling, EMA dual ascent
    for lam, exploration noise, hierarchical observe interface) is
    identical to SP_DDPG.
    """

    def __init__(self, nb_states, nb_actions, args, lam_max=None, lam_min=0.1,
                 ema_lam=False, ema_alpha=0.02, ema_lam_lr=0.01,
                 record_update_stats=False,
                 policy_delay=2, policy_noise=0.1, noise_clip=0.3,
                 cost_critic_mode='max',
                 wt_noise=0.05, sig_noise_max=0.2, sig_noise_min=0.01,
                 wt_noise_floor=0.01,
                 nstep=1, target_agg='min'):

        # Determine device first
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        print(f"Using device: {self.device}")

        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions
        # L 层观测标记: myfunc 测试函数用它区分 5 维规则原配料观测
        # 与 PER 基线 5 维观测 (两者 nb_states 同为 5, 无法按维度区分).
        self.distilled_obs = True

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        # ---- TD3: twin critics per channel ----
        # profit channel: critic_p / critic_p2
        self.critic_p = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target_p = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim_p = Adam(self.critic_p.parameters(), lr=args.rate)

        self.critic_p2 = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target_p2 = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim_p2 = Adam(self.critic_p2.parameters(), lr=args.rate)

        # cost channel: critic_c / critic_c2
        self.critic_c = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target_c = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim_c = Adam(self.critic_c.parameters(), lr=args.rate)

        self.critic_c2 = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target_c2 = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim_c2 = Adam(self.critic_c2.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target_p, self.critic_p)
        hard_update(self.critic_target_p2, self.critic_p2)
        hard_update(self.critic_target_c, self.critic_c)
        hard_update(self.critic_target_c2, self.critic_c2)

        # Create replay buffer (PER reused unchanged)
        self.memory = PrioritizedReplayBuffer(size=args.rmsize,
                                              alpha=args.PER_alpha)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions,
                                                       theta=args.ou_theta,
                                                       mu=args.ou_mu,
                                                       sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # ---- H 层 n 步回报 (默认 nstep=1 = 历史 1-step 口径, 逐位不变) ----
        #   动机: γ=0.98 时真值 ≈ 块奖励/(1−γ) = 50× 块奖励, 1-step 目标必须把
        #   奖励信息沿 50 步自助传播; 冻结批实测 (agentutil 离线验证) 40k 次更新
        #   只到真值的 34 %, 而 n=10 只需 20k 次就到 90 %。n 步把需要自助传播的
        #   深度从 50 降到 50/n。
        #   target_agg='mean': 去掉 twin-min 的悲观偏差 (冻结批 Qbar +18 %);
        #   'min' = TD3 原口径 (默认, 不动历史 run)。
        self.nstep = max(1, int(nstep))
        self.target_agg = str(target_agg)
        assert self.target_agg in ('min', 'mean'), self.target_agg
        self._pending = []        # nstep>1 时的滑窗队列: 最近 n-1 个块 (等待后继块结算)
        # 每条转移自带的 bootstrap 系数 (γ^n; episode 截断处为 0), 按 memory 下标存。
        # nstep>1 的 update_policy 用它替代全局 discount; nstep==1 时完全不读。
        self._boot = (np.full(int(args.rmsize), float(self.discount), dtype=np.float32)
                      if self.nstep > 1 else None)

        #
        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True

        # Ensure all models and tensors are on the same device
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic_p.to(self.device)
        self.critic_target_p.to(self.device)
        self.critic_p2.to(self.device)
        self.critic_target_p2.to(self.device)
        self.critic_c.to(self.device)
        self.critic_target_c.to(self.device)
        self.critic_c2.to(self.device)
        self.critic_target_c2.to(self.device)

        # TODO dual variable
        self.lam = args.lam
        self.lr_lam = args.lr_lam
        self.lam_max = lam_max  # None means no upper bound
        self.lam_min = lam_min  # lower bound on dual variable
        # self.cost_limit = args.cost_single_limit * (1 - args.discount**args.max_episode_length)/(1-args.discount) # long tern cost <= d
        # self.cost_limit = 0
        self.cost_limit = args.cost_single_limit
        self.lam_loss = 0.0  # critic_c - cost_limit, updated in update_policy()

        # B2: 优先级归一化的 TD running-mean (|TD_p|、|TD_c| 各自 EMA)
        self.td_mean_p = None
        self.td_mean_c = None

        # ---- EMA dual-ascent mode (lam updated from ACTUAL collected cost) ----
        # When ema_lam=True, update_policy() no longer touches lam; instead the
        # training loop feeds per-block actual cost via update_ema_cost() and
        # advances the dual variable via update_lam_ema(d_block).
        self.ema_lam = ema_lam
        self.ema_alpha = ema_alpha
        self.ema_lam_lr = ema_lam_lr
        self.ema_cost = None  # EMA of per-block actual cost (fed from loop)

        self.PER_beta = args.PER_beta
        self.PER_epsilon = args.PER_epsilon
        self.beta_increment_per_sampling = args.beta_increment_per_sampling
        self.nb_turbine = args.nb_turbine

        # ---- H 层更新数据记录（每批统计；主循环块末汇总写入 TB/csv） ----
        self.record_update_stats = record_update_stats
        self.update_stats = []
        # 值函数回归空间缩放 s (见 hsrl_cost 文件末尾小节):
        #   1.0 = 不归一化 (默认, 未接线时行为不变); main 只对 H 层设为 hc.value_scale(),
        #   L 层保持 1.0 (r2/c2 本就 O(1))。
        self.value_scale = 1.0
        # cost 通道可单独缩放 s_c (hsrl_cost: profit/cost 双尺度 + λ 自动重标);
        #   None ⇒ 与 s_p 相同 = 旧行为; main 对 H 层设 hc.cost_scale()
        self.cost_scale = None

        # ---- TD3 specific ----
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        assert cost_critic_mode in ('min', 'max'), \
            "cost_critic_mode must be 'min' or 'max', got {}".format(cost_critic_mode)
        self.cost_critic_mode = cost_critic_mode
        self.update_count = 0  # counts critic updates for delayed policy update

        # Per-dimension action amplitude (for target smoothing noise scaling).
        # Matches CheckAction() legal ranges: H = 5 turbines [0,0.5] + 1 signal
        # [-0.5,0.5] (amp 0.5 each); L = 1 BESS action [-charge, charge].
        charge=0.50
        if self.nb_actions == self.nb_turbine:
            amp = np.full(self.nb_actions, 0.5)
        elif self.nb_actions == 1:
            amp = np.full(self.nb_actions, charge)
        elif self.nb_actions == self.nb_turbine + 1:
            amp = np.array([0.5] * self.nb_turbine + [0.5])
        else:
            amp = np.ones(self.nb_actions)
        self.action_amp = np.asarray(amp, dtype=np.float32)

        # ---- 探索噪声幅度 (逐维; 与 td3perrb 上层同口径) ----
        # 旧实现: action += eps * ((np.random.random(n) - 0.5)/10)[0]
        #   → (a) 幅度仅 ±0.05 (对 SOC 目标信号 = ±0.25 MWh, 不足一个块的跟踪能力)
        #     (b) [0] 索引使同一个标量加到全部维度 (维度耦合, 非逐维噪声)
        # 新实现: 逐维均匀噪声 ±amp_dim * eps, 信号维另有地板 sig_noise_min:
        #   WT 5 维: ±wt_noise * eps        (0.05, 与原幅度一致, 仅改为逐维独立抽样)
        #   信号维 : ±max(sig_noise_max*eps, sig_noise_min)  (0.2 → 地板 0.05)
        # eps 由 1 线性衰减到 0, 衰减完成的【决策次数】= args.epsilon
        # (H 每 5 个环境步决策一次 ⇒ args.epsilon=4000 对应 20k 环境步)。
        # 下层 (nb_actions=1) 走 else 分支: 幅度 = wt_noise·eps、无地板,
        # 与改造前 td3per 下层的 ±0.05·eps 完全等价 (epsilon 由 --l_epsilon 控制)。
        self.wt_noise = float(wt_noise)
        self.sig_noise_max = float(sig_noise_max)
        self.sig_noise_min = float(sig_noise_min)
        self.wt_noise_floor = float(wt_noise_floor)
        # ---- 风机探索噪声: **恒定 `wt_noise`（默认 0.05），不随 ε 衰减** ----
        #   用户口径（2026-10）: 全部风机、全部 TD3 族方法, 风机行为噪声恒为 0.05。
        #   实现 = 让 WT 维的 amp 与 floor 相等 ⇒ 凸组合
        #   `explore_amp*eps + explore_floor*(1-eps)` 恒等于 `wt_noise`
        #   （公式本身不动 ⇒ `test_noise_format_uniform.py` 的一致性判据仍成立）。
        #   第 6 维 SOC 信号仍按 `sig_noise_max -> sig_noise_min` 衰减;
        #   L 层（nb_actions==1）保持旧口径（main 随后会覆写 explore_amp/explore_floor）。
        if self.nb_actions == self.nb_turbine + 1:
            amp_e = np.array([self.wt_noise] * self.nb_turbine + [self.sig_noise_max],
                             dtype=np.float32)
            floor_e = np.array([self.wt_noise] * self.nb_turbine + [self.sig_noise_min],
                               dtype=np.float32)
        elif self.nb_actions == self.nb_turbine:
            # 纯风机维（aifflor / yawflor / nolow 的 td3per·td3perseg）: 全维恒定
            amp_e = np.full(self.nb_actions, self.wt_noise, dtype=np.float32)
            floor_e = np.full(self.nb_actions, self.wt_noise, dtype=np.float32)
        else:
            # L 层（nb_actions==1）: 旧口径不变
            amp_e = np.full(self.nb_actions, self.wt_noise, dtype=np.float32)
            floor_e = np.full(self.nb_actions, self.wt_noise_floor, dtype=np.float32)
        self.explore_amp = amp_e
        self.explore_floor = floor_e

    # ---- TD3 helper: aggregate a critic pair (min or max) ----
    def _pair_min(self, q1, q2):
        return torch.min(q1, q2)

    def _aggregate_pair(self, q1, q2):
        """Aggregate a twin-critic pair. Default TD3 uses min."""
        if self.cost_critic_mode == 'max':
            return torch.max(q1, q2)
        return torch.min(q1, q2)

    def update_policy(self):
        # A2: β 退火 —— 每次更新推进 self.PER_beta (0.4→1)，写回实例属性
        # (原 PER_buffer.sample 内递增是局部变量，退火永不生效；此处推进后传入)
        self.PER_beta = min(1.0, self.PER_beta + self.beta_increment_per_sampling)

        # Sample batch
        state_batch, action_batch, profit_batch, cost_batch,\
        next_state_batch, terminal_batch, weights, batch_idxes \
        = self.memory.sample(self.batch_size, self.PER_beta, self.beta_increment_per_sampling)
        # reshape data
        profit_batch = profit_batch.reshape(-1, 1)
        cost_batch = cost_batch.reshape(-1, 1)
        # ---- 值函数回归空间归一化: r1/c1 进 critic 的唯一一次 /s ----
        # buffer 里存的仍是原始口径 ⇒ relabel / warmup 填入 / 对偶 / 日志全部不受影响。
        _SP = float(getattr(self, 'value_scale', 1.0)) or 1.0
        _SC = getattr(self, 'cost_scale', None)
        _SC = _SP if _SC is None else (float(_SC) or _SP)
        if _SP != 1.0:
            profit_batch = profit_batch / _SP
        if _SC != 1.0:
            cost_batch = cost_batch / _SC
        terminal_batch = terminal_batch.reshape(-1, 1)

        # Ensure all tensors are on the correct device
        state_batch = to_tensor(state_batch).to(self.device)
        action_batch = to_tensor(action_batch).to(self.device)
        profit_batch = to_tensor(profit_batch).to(self.device)
        cost_batch = to_tensor(cost_batch).to(self.device)
        next_state_batch = to_tensor(next_state_batch).to(self.device)
        terminal_batch = to_tensor(terminal_batch).to(self.device)
        weights = torch.Tensor(weights).to(self.device)

        # ---- TD3: target policy smoothing (clipped Gaussian noise,
        #      scaled per dimension by action amplitude) ----
        amp_t = torch.tensor(self.action_amp, dtype=torch.float32, device=self.device)
        a_next = self.actor_target(next_state_batch)
        noise = torch.randn_like(a_next) * (self.policy_noise * amp_t)
        noise = noise.clamp(-self.noise_clip * amp_t, self.noise_clip * amp_t)
        a_smoothed = a_next + noise

        # ---- TD3: twin targets, take min of the pair (clipped double Q) ----
        q_p1_next = self.critic_target_p([next_state_batch, a_smoothed])
        q_p2_next = self.critic_target_p2([next_state_batch, a_smoothed])
        q_c1_next = self.critic_target_c([next_state_batch, a_smoothed])
        q_c2_next = self.critic_target_c2([next_state_batch, a_smoothed])

        if self.target_agg == 'mean':
            # 去掉 twin-min 的悲观偏差: 冻结批实测 Qbar 6.997 -> 8.235 (真值 9.159)
            min_p_next = 0.5 * (q_p1_next + q_p2_next)
        else:
            min_p_next = self._pair_min(q_p1_next, q_p2_next)
        # cost aggregate: max (safe, avoid underestimation) or min (TD3 paper).
        # clamp(min=0): costs are non-negative by construction (c1 = c+0.1*P_F,
        # c2 = 1000*slack >= 0), so a negative cost bootstrap target is
        # impossible and only corrupts the Lagrangian — clamp enforces it.
        min_c_next = self._aggregate_pair(q_c1_next, q_c2_next).clamp(min=0.0)

        # A1: terminal 语义修复 —— bootstrap 当且仅当非终止转移
        # nstep>1: 每条转移自带 bootstrap 系数 γ^n (episode 截断处为 0),
        #          替代全局 discount; nstep==1 时 _disc 就是 float(discount),
        #          表达式与历史版本逐位相同。
        if self.nstep > 1:
            _disc = torch.tensor(self._boot[batch_idxes], dtype=torch.float32,
                                 device=self.device).reshape(-1, 1)
        else:
            _disc = self.discount
        target_q_batch_p = profit_batch + \
            _disc * (1 - terminal_batch) * min_p_next
        target_q_batch_c = cost_batch + \
            _disc * (1 - terminal_batch) * min_c_next

        # ---- Critic updates (all four, every call) ----
        self.critic_p.zero_grad()
        self.critic_p2.zero_grad()
        self.critic_c.zero_grad()
        self.critic_c2.zero_grad()

        q_batch_p = self.critic_p([state_batch, action_batch])
        q_batch_p2 = self.critic_p2([state_batch, action_batch])
        q_batch_c = self.critic_c([state_batch, action_batch])
        q_batch_c2 = self.critic_c2([state_batch, action_batch])

        # Compute the value losses (twin profit + twin cost).
        # Twin critics share the same min-target graph, so a single combined
        # backward is required (standard TD3); then clip & step each network.
        value_loss_p = torch.mean(torch.mean(weights * (q_batch_p - target_q_batch_p) ** 2, dim=1))
        value_loss_p2 = torch.mean(torch.mean(weights * (q_batch_p2 - target_q_batch_p) ** 2, dim=1))
        value_loss_c = torch.mean(torch.mean(weights * (q_batch_c - target_q_batch_c) ** 2, dim=1))
        value_loss_c2 = torch.mean(torch.mean(weights * (q_batch_c2 - target_q_batch_c) ** 2, dim=1))

        critic_loss = value_loss_p + value_loss_p2 + value_loss_c + value_loss_c2
        critic_loss.backward()

        nn.utils.clip_grad_norm_(self.critic_p.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optim_p.step()

        nn.utils.clip_grad_norm_(self.critic_p2.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optim_p2.step()

        nn.utils.clip_grad_norm_(self.critic_c.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optim_c.step()

        nn.utils.clip_grad_norm_(self.critic_c2.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optim_c2.step()

        # ---- TD3: delayed policy update ----
        self.update_count += 1
        if self.update_count % self.policy_delay == 0:
            self.actor.zero_grad()
            # Fresh forward through the (just updated) critics: min of twins.
            min_p_cur = self._pair_min(
                self.critic_p([state_batch, self.actor(state_batch)]),
                self.critic_p2([state_batch, self.actor(state_batch)]))
            # clamp(min=0) on cost: prevents the actor from exploiting a
            # (pathological) negative cost Q inside the Lagrangian objective.
            min_c_cur = self._aggregate_pair(
                self.critic_c([state_batch, self.actor(state_batch)]),
                self.critic_c2([state_batch, self.actor(state_batch)])).clamp(min=0.0)

            # cost 为 +
            policy_loss = -(min_p_cur - self.lam * min_c_cur)  # TODO
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            # Gradient Clipping
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
            self.actor_optim.step()

            # TODO update dual variable
            if not self.ema_lam:
                lam_loss = (min_c_cur * _SC - self.cost_limit)   # 还原原始口径 (仅 non-EMA 分支)
                lam_loss = lam_loss.mean().item()
                self.lam_loss = lam_loss  # for curriculum convergence check
                self.lam += lam_loss * self.lr_lam
                self.lam = max(self.lam, self.lam_min)
                if self.lam_max is not None:
                    self.lam = min(self.lam, self.lam_max)
            # else: EMA mode — lam is advanced externally via update_lam_ema()

            # Target update (all targets, only on delayed steps)
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target_p, self.critic_p, self.tau)
            soft_update(self.critic_target_p2, self.critic_p2, self.tau)
            soft_update(self.critic_target_c, self.critic_c, self.tau)
            soft_update(self.critic_target_c2, self.critic_c2, self.tau)

        # ---- PER priorities: twin-min TD errors (updated every call) ----
        # B2: 优先级 = |TD_p|/mean_p + |TD_c|/mean_c (去 lam，量纲归一化)
        min_p_cur_b = self._pair_min(q_batch_p, q_batch_p2)
        min_c_cur_b = self._aggregate_pair(q_batch_c, q_batch_c2)
        TD_errors_p = abs(target_q_batch_p - min_p_cur_b) * _SP   # ×s_p   # ×s: PER/日志保持原始口径
        TD_errors_c = abs(target_q_batch_c - min_c_cur_b) * _SC   # ×s_c

        td_errors_p_np = TD_errors_p.detach().cpu().numpy()
        td_errors_c_np = TD_errors_c.detach().cpu().numpy()
        mean_p = float(np.mean(td_errors_p_np))
        mean_c = float(np.mean(td_errors_c_np))
        if self.td_mean_p is None:
            self.td_mean_p, self.td_mean_c = mean_p, mean_c
        else:
            self.td_mean_p = 0.99 * self.td_mean_p + 0.01 * mean_p
            self.td_mean_c = 0.99 * self.td_mean_c + 0.01 * mean_c
        td_errors = (td_errors_p_np / (self.td_mean_p + 1e-6)
                     + td_errors_c_np / (self.td_mean_c + 1e-6))
        new_priorities = td_errors + self.PER_epsilon
        self.memory.update_priorities(batch_idxes, new_priorities)

        # ---- H 层更新数据收集：本批（batch）统计，供主循环块末汇总 ----
        if self.record_update_stats:
            self.update_stats.append({
                'td_p_mean': float(np.mean(td_errors_p_np)),
                'td_c_mean': float(np.mean(td_errors_c_np)),
                'q_p_mean': float(q_batch_p.mean().item()),
                'q_p2_mean': float(q_batch_p2.mean().item()),
                'q_c_mean': float(q_batch_c.mean().item()),
                'q_c2_mean': float(q_batch_c2.mean().item()),
                'min_p_next_mean': float(min_p_next.mean().item()),
                'min_c_next_mean': float(min_c_next.mean().item()),
                'target_p_mean': float(target_q_batch_p.mean().item()),
                'target_c_mean': float(target_q_batch_c.mean().item()),
                'reward_mean': float(profit_batch.mean().item()),
                'cost_mean': float(cost_batch.mean().item()),
                'weight_mean': float(weights.mean().item()),
            })
            # 值函数口径字段还原成原始口径 (s=1 时恒等) —— update_stats_H.csv /
            # analyse_q_scale.py / 历史 run 因此可直接对比。
            if _SP != 1.0 or _SC != 1.0:
                _d = self.update_stats[-1]
                for _k in ('reward_mean', 'q_p_mean', 'q_p2_mean',
                           'min_p_next_mean', 'target_p_mean'):
                    if _k in _d:
                        _d[_k] = float(_d[_k]) * _SP      # profit 通道用 s_p
                for _k in ('cost_mean', 'q_c_mean', 'q_c2_mean',
                           'min_c_next_mean', 'target_c_mean'):
                    if _k in _d:
                        _d[_k] = float(_d[_k]) * _SC      # cost 通道用 s_c

    # ========================================================================
    # EMA dual-ascent (actual-cost based lam update)
    #   EMA_t   = (1-α)·EMA_{t-1} + α·x_t          (x_t = per-block actual cost)
    #   λ      ← clip(λ + η·(EMA − d_block), λ_min, λ_max)
    # Called from the training loop (once per 5-step block, after warmup).
    # ========================================================================
    def update_ema_cost(self, x):
        """EMA update with the actual per-block cost (EMA dual-ascent mode)."""
        if self.ema_cost is None:
            self.ema_cost = float(x)
        else:
            self.ema_cost = (1.0 - self.ema_alpha) * self.ema_cost \
                + self.ema_alpha * float(x)

    def update_lam_ema(self, d_block):
        """Dual ascent on the actual-cost EMA: lam += η·(EMA − d_block), clipped."""
        if self.ema_cost is None:
            return
        self.lam += self.ema_lam_lr * (self.ema_cost - float(d_block))
        self.lam = max(self.lam, self.lam_min)
        if self.lam_max is not None:
            self.lam = min(self.lam, self.lam_max)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()

        self.critic_p.eval()
        self.critic_target_p.eval()

        self.critic_p2.eval()
        self.critic_target_p2.eval()

        self.critic_c.eval()
        self.critic_target_c.eval()

        self.critic_c2.eval()
        self.critic_target_c2.eval()

    def observe(self, r_t, c_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, c_t, s_t1, done)
            self.s_t = s_t1

    # ---- n 步回报: 写入侧 --------------------------------------------------
    def _push_transition(self, s_t, a_t, r_t, c_t, s_t1, done, boot):
        """写一条转移进 memory, 并记录它的 bootstrap 系数 boot。

        boot 只在 nstep>1 时被 update_policy 读取 (nstep==1 走原分支, 不读)。
        下标 = append 之前的 memory._next_idx —— PrioritizedReplayBuffer.append
        也用同一个值作为 segment-tree 下标, 两者一致。
        """
        idx = self.memory._next_idx
        self.memory.append(s_t, a_t, r_t, c_t, s_t1, done)
        if self._boot is not None and 0 <= idx < len(self._boot):
            self._boot[idx] = boot

    def _emit_nstep(self, start, n_used):
        """把 _pending[start : start+n_used] 折叠成一条 n 步转移写入 memory。

        reward = Σ_{k<n} γ^k·r_k (成本通道同式); next_state = 第 n 个后继状态;
        boot = γ^n, 但窗口内出现 done 就截断且 boot=0 (由 done 表达终止)。

        注意: 正常路径 (滑窗) 每次只有 start=0, n_used=nstep —— 每来一个块结算一条;
        start>0 只出现在 episode 末的尾部结算 (逐条截断)。
        """
        seg = self._pending[start:start + n_used]
        gam = float(self.discount)
        r_sum = 0.0
        c_sum = 0.0
        trunc = False
        for k, (_s, _a, _r, _c, _s2, _d) in enumerate(seg):
            r_sum += (gam ** k) * float(_r)
            c_sum += (gam ** k) * float(_c)
            if _d:
                trunc = True
                break
        boot = 0.0 if trunc else gam ** n_used
        self._push_transition(seg[0][0], seg[0][1], r_sum, c_sum, seg[-1][4],
                              1.0 if trunc else float(seg[-1][5]), boot)

    def observe_jump(self, s_t, a_t, r_t, c_t, s_t1, done):
        if not self.is_training:
            return
        if self.nstep <= 1:
            self.memory.append(s_t, a_t, r_t, c_t, s_t1, done)
            return
        self._pending.append((s_t, a_t, r_t, c_t, s_t1, done))
        # ---- 滑窗 (stride-1): 每来一个新块就结算**队首**那一条, 队列长度稳定在 n-1 ----
        #   为什么必须滑窗而不是"攒满 n 个清空": 分块版每条转移的起点是 b1,b(1+n),...
        #   ⇒ 只有 1/n 的块状态成为**训练输入**, 其余只作为 bootstrap 的后继 (target 侧)
        #   出现, 网络从未以它们为输入拟合过。滑窗版每个块都是某条转移的起点 ⇒
        #   **状态覆盖 = 块数**, 同时每条仍带 n 步目标 (标准做法: Rainbow/Ape-X 的
        #   n_step wrapper 即 stride-1)。
        #   代价: 条目数 ×n (rmsize 口径变回"≈块数"), 且相邻两条共享 n-1 个奖励
        #   ⇒ 批内相关、PER 优先级在同一奖励段上扩散 (看 weight_mean)。
        while len(self._pending) >= self.nstep:
            self._emit_nstep(0, self.nstep)
            self._pending.pop(0)
        if done:
            # episode 末: 尾部 (不足 n 步) 逐条按截断结算, 防止跨 episode 混算
            k = len(self._pending)
            for i in range(k):
                self._emit_nstep(i, k - i)
            self._pending = []

    def random_action(self):
        charge=0.5
        if self.nb_actions == self.nb_turbine:
            action = np.random.uniform(0., 0.5, self.nb_actions)
        if self.nb_actions == 1:
            action = np.random.uniform(-charge, charge, self.nb_actions)
        if self.nb_actions == self.nb_turbine + 1:
            action_H = np.random.uniform(0., 0.5, self.nb_turbine)
            # 第 6 维 = SOC 目标信号 (语义已由 shadow price 改为 SOC 目标):
            #   必须落在 CheckAction 的合法区间 [-0.5, 0.5] (映射 soc_target = 3+5*a6 ∈[0.5,5.5]);
            #   旧写法用 U(-10,10) 会在 warmup 产出 target ±50 的病态数据,
            #   把 c2 打到 2e4 量级 → lam2 顶格/PER 优先级被污染 (见 run157 诊断)
            action_sig = np.random.uniform(-0.5, 0.5, 1)
            action = np.concatenate((action_H, action_sig))
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)

        # 逐维探索噪声: 均匀 U(-amp, +amp), amp = max(amp_dim*eps, floor_dim)
        # (信号维幅值 0.2*eps 且地板 0.05; WT 维 0.05*eps; 见 __init__)
        if self.is_training:
            eps = max(self.epsilon, 0.0)
            # ε 收敛到**地板**而非 0: amp = amp_full·ε + floor·(1−ε)
            #   ε=1 ⇒ amp = explore_amp（起始幅度不变）
            #   ε=0 ⇒ amp = explore_floor（**永不完全停止探索**）
            #   ε 线性降到 0 的用时 = args.epsilon 决策（H=4000⇒20k 环境步）
            amp = self.explore_amp * eps + self.explore_floor * (1.0 - eps)
            action = action + (np.random.random(self.nb_actions) - 0.5) * 2.0 * amp

        if decay_epsilon and self.is_training:   # 评测(is_training=False) 不消耗 ε 预算
            self.epsilon -= self.depsilon

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(torch.load('{}/actor_{}.pkl'.format(output, self.nb_actions)))
        self.critic_p.load_state_dict(torch.load('{}/critic_p_{}.pkl'.format(output, self.nb_actions)))
        self.critic_c.load_state_dict(torch.load('{}/critic_c_{}.pkl'.format(output, self.nb_actions)))

        # TD3: twin-2 weights — fall back to warm-start copy from twin-1
        # (e.g. loading a trained PER model that has no critic_p2/critic_c2).
        try:
            self.critic_p2.load_state_dict(torch.load('{}/critic_p2_{}.pkl'.format(output, self.nb_actions)))
            self.critic_c2.load_state_dict(torch.load('{}/critic_c2_{}.pkl'.format(output, self.nb_actions)))
        except Exception:
            print('critic_p2/critic_c2 not found - copying from critic_p/critic_c (warm-start)')
            hard_update(self.critic_p2, self.critic_p)
            hard_update(self.critic_c2, self.critic_c)

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target_p, self.critic_p)
        hard_update(self.critic_target_p2, self.critic_p2)
        hard_update(self.critic_target_c, self.critic_c)
        hard_update(self.critic_target_c2, self.critic_c2)

    def save_model(self, output):
        torch.save(self.actor.state_dict(), '{}/actor_{}.pkl'.format(output, self.nb_actions))
        torch.save(self.critic_p.state_dict(), '{}/critic_p_{}.pkl'.format(output, self.nb_actions))
        torch.save(self.critic_p2.state_dict(), '{}/critic_p2_{}.pkl'.format(output, self.nb_actions))
        torch.save(self.critic_c.state_dict(), '{}/critic_c_{}.pkl'.format(output, self.nb_actions))
        torch.save(self.critic_c2.state_dict(), '{}/critic_c2_{}.pkl'.format(output, self.nb_actions))

    def save_checkpoint(self, output, step):
        """Save checkpoint in a dedicated subfolder per checkpoint step."""
        import os
        ckpt_dir = os.path.join(output, 'checkpoint_{}'.format(step))
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.actor.state_dict(),
                   '{}/actor_{}.pkl'.format(ckpt_dir, self.nb_actions))
        torch.save(self.critic_p.state_dict(),
                   '{}/critic_p_{}.pkl'.format(ckpt_dir, self.nb_actions))
        torch.save(self.critic_p2.state_dict(),
                   '{}/critic_p2_{}.pkl'.format(ckpt_dir, self.nb_actions))
        torch.save(self.critic_c.state_dict(),
                   '{}/critic_c_{}.pkl'.format(ckpt_dir, self.nb_actions))
        torch.save(self.critic_c2.state_dict(),
                   '{}/critic_c2_{}.pkl'.format(ckpt_dir, self.nb_actions))

    def seed(self, s):
        torch.manual_seed(s)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(s)
