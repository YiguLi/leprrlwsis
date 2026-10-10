"""
PRLE Extended SP-DDPG Agent
============================
Extends SP_DDPG from PER_ddpg.py to use PRLEBuffer instead of
PrioritizedReplayBuffer. The observe_jump method is extended to
accept an optional context parameter for reward relabeling.

All other methods (update_policy, select_action, random_action, reset,
save_model, save_checkpoint, load_weights, seed) are inherited unchanged.
"""

import numpy as np
import torch

from model import (Actor, Critic)
from prle.prle_buffer import PRLEBuffer
from random_process import OrnsteinUhlenbeckProcess
from util import *

criterion = torch.nn.MSELoss()


class PRLE_DDPG(object):
    """
    SP-DDPG agent variant that uses PRLEBuffer for reward relabeling support.

    The only difference from PER_ddpg.SP_DDPG:
      1. Uses PRLEBuffer instead of PrioritizedReplayBuffer
      2. observe_jump() accepts an optional 'context' parameter
      3. observe() accepts an optional 'context' parameter
    """

    def __init__(self, nb_states, nb_actions, args, lam_max=None, lam_min=0.1,
                 ema_lam=False, ema_alpha=0.02, ema_lam_lr=0.01,
                 record_update_stats=False):

        # Determine device first
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        print(f"Using device: {self.device}")

        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.prate)

        self.critic_p = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target_p = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim_p = torch.optim.Adam(self.critic_p.parameters(), lr=args.rate)

        self.critic_c = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target_c = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim_c = torch.optim.Adam(self.critic_c.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target_p, self.critic_p)
        hard_update(self.critic_target_c, self.critic_c)

        # Create PRLE replay buffer (block_size=5 for H, =1 for L)
        block_size = 5 if nb_actions >= 6 else 1
        self.memory = PRLEBuffer(size=args.rmsize,
                                  alpha=args.PER_alpha,
                                  block_size=block_size)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions,
                                                       theta=args.ou_theta,
                                                       mu=args.ou_mu,
                                                       sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True

        # Ensure all models and tensors are on the same device
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic_p.to(self.device)
        self.critic_target_p.to(self.device)
        self.critic_c.to(self.device)
        self.critic_target_c.to(self.device)

        # Dual variable
        self.lam = args.lam
        self.lr_lam = args.lr_lam
        self.lam_max = lam_max  # None means no upper bound
        self.lam_min = lam_min  # lower bound on dual variable
        self.cost_limit = args.cost_single_limit
        self.lam_loss = 0.0  # critic_c - cost_limit, updated in update_policy()

        # EMA dual-ascent mode (lam updated from ACTUAL collected cost, 与 SP_DDPG 一致)
        self.ema_lam = ema_lam
        self.ema_alpha = ema_alpha
        self.ema_lam_lr = ema_lam_lr
        self.ema_cost = None

        # B2: 优先级归一化的 TD running-mean (|TD_p|、|TD_c| 各自 EMA)
        self.td_mean_p = None
        self.td_mean_c = None

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

    def update_policy(self):
        """Sample batch, compute TD errors, update actor/critic/dual. Identical to SP_DDPG."""
        # A2: β 退火 —— 每次更新推进 self.PER_beta (0.4→1)，写回实例属性
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

        # Prepare for the target q batch -- profit
        next_q_values_p = self.critic_target_p([
            next_state_batch,
            self.actor_target(next_state_batch),
        ])

        # Prepare for the target q batch -- cost
        next_q_values_c = self.critic_target_c([
            next_state_batch,
            self.actor_target(next_state_batch),
        ])

        target_q_batch_p = profit_batch + \
            self.discount * (1 - terminal_batch) * next_q_values_p
        target_q_batch_c = cost_batch + \
            self.discount * (1 - terminal_batch) * next_q_values_c

        # Critic update
        self.critic_p.zero_grad()
        self.critic_c.zero_grad()

        q_batch_p = self.critic_p([
            state_batch,
            action_batch
        ])
        q_batch_c = self.critic_c([
            state_batch,
            action_batch
        ])

        TD_errors_p = abs(target_q_batch_p - q_batch_p) * _SP   # ×s_p   # ×s: PER/日志保持原始口径
        TD_errors_c = abs(target_q_batch_c - q_batch_c) * _SC   # ×s_c

        # Compute the value loss
        value_loss_p = torch.mean(torch.mean(weights * (q_batch_p - target_q_batch_p) ** 2, dim=1))
        value_loss_p.backward()
        nn = torch.nn
        nn.utils.clip_grad_norm_(self.critic_p.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optim_p.step()

        # Compute the value loss for cost
        value_loss_c = torch.mean(torch.mean(weights * (q_batch_c - target_q_batch_c) ** 2, dim=1))
        value_loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic_c.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optim_c.step()

        # Actor update
        self.actor.zero_grad()
        policy_loss = -(self.critic_p([state_batch, self.actor(state_batch)]) -
            self.lam * self.critic_c([state_batch, self.actor(state_batch)]))

        policy_loss = policy_loss.mean()
        policy_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)

        self.actor_optim.step()

        # Update dual variable (critic-based; 仅在非 EMA 模式)
        if not self.ema_lam:
            lam_loss = (self.critic_c([state_batch, self.actor(state_batch)]) * _SC - self.cost_limit)   # 还原原始口径 (仅 non-EMA 分支)
            lam_loss = lam_loss.mean().item()
            self.lam_loss = lam_loss
            self.lam += lam_loss * self.lr_lam
            self.lam = max(self.lam, self.lam_min)
            if self.lam_max is not None:
                self.lam = min(self.lam, self.lam_max)
        # else: EMA mode — lam is advanced externally via update_lam_ema()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target_p, self.critic_p, self.tau)
        soft_update(self.critic_target_c, self.critic_c, self.tau)

        # For prioritized experience replay
        # B2: 优先级 = |TD_p|/mean_p + |TD_c|/mean_c (去 lam，量纲归一化)
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
        # 记录的是"这次更新实际吃进去的数据"：TD 误差（PER 优先级来源）、
        # critic 预测 Q、bootstrap 目标、batch 奖励/成本水平、ISW 权重。
        if self.record_update_stats:
            self.update_stats.append({
                'td_p_mean': float(np.mean(td_errors_p_np)),
                'td_c_mean': float(np.mean(td_errors_c_np)),
                'q_p_mean': float(q_batch_p.mean().item()),
                'q_c_mean': float(q_batch_c.mean().item()),
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
    # EMA dual-ascent (actual-cost based lam update, 与 SP_DDPG 一致)
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
        self.critic_c.eval()
        self.critic_target_c.eval()

    def observe(self, r_t, c_t, s_t1, done, context=None):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, c_t, s_t1, done, context=context)
            self.s_t = s_t1

    def observe_jump(self, s_t, a_t, r_t, c_t, s_t1, done, context=None):
        """Store transition with optional context for reward relabeling."""
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, c_t, s_t1, done, context=context)

    def random_action(self):
        charge = 10.
        if self.nb_actions == self.nb_turbine:
            action = np.random.uniform(0., 0.5, self.nb_actions)
        if self.nb_actions == 1:
            action = np.random.uniform(-charge, charge, self.nb_actions)
        if self.nb_actions == self.nb_turbine + 1:
            action_H = np.random.uniform(0., 0.5, self.nb_turbine)
            action_L = np.random.uniform(-charge, charge, 1)
            action = np.concatenate((action_H, action_L))
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)
        action += self.is_training * max(self.epsilon, 0) * ((np.random.random(self.nb_actions) - 0.5) / 10)[0]

        if decay_epsilon and self.is_training:   # 评测(is_training=False) 不消耗 ε 预算
            self.epsilon -= self.depsilon

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None:
            return
        import torch
        self.actor.load_state_dict(torch.load('{}/actor_{}.pkl'.format(output, self.nb_actions)))
        self.critic_p.load_state_dict(torch.load('{}/critic_p_{}.pkl'.format(output, self.nb_actions)))
        self.critic_c.load_state_dict(torch.load('{}/critic_c_{}.pkl'.format(output, self.nb_actions)))
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target_p, self.critic_p)
        hard_update(self.critic_target_c, self.critic_c)

    def save_model(self, output):
        import torch
        torch.save(self.actor.state_dict(), '{}/actor_{}.pkl'.format(output, self.nb_actions))
        torch.save(self.critic_p.state_dict(), '{}/critic_p_{}.pkl'.format(output, self.nb_actions))
        torch.save(self.critic_c.state_dict(), '{}/critic_c_{}.pkl'.format(output, self.nb_actions))

    def save_checkpoint(self, output, step):
        import os
        ckpt_dir = os.path.join(output, 'checkpoint_{}'.format(step))
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.actor.state_dict(),
                   '{}/actor_{}.pkl'.format(ckpt_dir, self.nb_actions))
        torch.save(self.critic_p.state_dict(),
                   '{}/critic_p_{}.pkl'.format(ckpt_dir, self.nb_actions))
        torch.save(self.critic_c.state_dict(),
                   '{}/critic_c_{}.pkl'.format(ckpt_dir, self.nb_actions))

    def seed(self, s):
        torch.manual_seed(s)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(s)
