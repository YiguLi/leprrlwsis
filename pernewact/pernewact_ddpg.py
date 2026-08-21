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


class PERNEWACT_DDPG(object):
    """
    PER + 有界行为输出（clamp + straight-through）。

    与 PER_ddpg.SP_DDPG 的差异（行为生成逻辑）:
      1. 风机轴向因子输出经 clamp [a_lo, a_hi] + straight-through 梯度
         （前向=clamp 值保证行为有界；反传梯度=1 直通，无 sigmoid/tanh 饱和）。
      2. raw 卫生界 [-raw_C, raw_C]：防 actor 原始输出数值爆炸（行为已被 clamp 保护）。
      3. update_policy 中所有 self.actor()/self.actor_target() 输出（target ×2、
         policy_loss、lam_loss）先 clamp 再进 critic，保证 critic 只在界内动作上被求值。
      4. 探索噪声加在 raw 上（clamp 前），行为恒界内。
      5. record_q=True 时暂存本批 q_p_mean / q_c_mean，供训练循环记录（仅上层 H 启用）。

    clamp 只作用于前 nb_turbine 维（风机）；信号维（action[nb_turbine]）与
    L 层电池动作不 clamp，保留原 CheckAction 裁剪/惩罚逻辑。
    """

    def __init__(self, nb_states, nb_actions, args, lam_max=None, lam_min=0.1,
                 ema_lam=False, ema_alpha=0.02, ema_lam_lr=0.01,
                 a_lo=0.0, a_hi=0.5, raw_C=3.0, record_q=False):

        # Determine device first
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        print(f"Using device: {self.device}")

        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # ---- pernewact: 有界行为参数 ----
        self.a_lo = a_lo
        self.a_hi = a_hi
        self.raw_C = raw_C
        # H 层（nb_actions > 1）对前 nb_turbine 维 clamp；L 层（nb_actions == 1）不 clamp
        self.clamp_turbine = nb_actions > 1
        self.record_q = record_q
        self.q_p_mean = None   # 本批 profit critic 均值（record_q=True 时由 update_policy 写入）
        self.q_c_mean = None   # 本批 cost critic 均值

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic_p = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target_p = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim_p = Adam(self.critic_p.parameters(), lr=args.rate)

        self.critic_c = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target_c = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim_c = Adam(self.critic_c.parameters(), lr=args.rate)

        hard_update(self.actor_target,self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target_p, self.critic_p)
        hard_update(self.critic_target_c, self.critic_c)

        #Create replay buffer
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
        self.critic_c.to(self.device)
        self.critic_target_c.to(self.device)

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

    # ========================================================================
    # pernewact: 有界行为（clamp + straight-through）
    # ========================================================================

    def _bound_raw(self, a):
        """raw 卫生界：防 actor 原始输出数值爆炸（仅 H 层风机动作启用）。"""
        if not self.clamp_turbine:
            return a
        return torch.clamp(a, -self.raw_C, self.raw_C)

    def _bound_tensor(self, a_raw):
        """tensor 版 clamp + straight-through：
        前向返回 clamp 后值（行为有界）；反传梯度=1 直通（无饱和）。
        只 clamp 前 nb_turbine 维（风机），其余维（信号）原样。
        """
        if not self.clamp_turbine:
            return a_raw
        n = min(self.nb_turbine, a_raw.shape[-1])
        a_clamped = torch.clamp(a_raw[..., :n], self.a_lo, self.a_hi)
        a_out = torch.cat([a_clamped, a_raw[..., n:]], dim=-1)
        return a_raw + (a_out - a_raw).detach()

    def _bound_np(self, a):
        """numpy 版 clamp（行为层，无梯度需求）：只 clamp 前 nb_turbine 维。"""
        if not self.clamp_turbine:
            return a
        n = min(self.nb_turbine, len(a))
        a = a.copy()
        a[:n] = np.clip(a[:n], self.a_lo, self.a_hi)
        return a

    # ========================================================================

    def update_policy(self):
        # A2: β 退火 —— 每次更新推进 self.PER_beta (0.4→1)，写回实例属性
        # (原 PER_buffer.sample 内递增是局部变量，退火永不生效；此处推进后传入)
        self.PER_beta = min(1.0, self.PER_beta + self.beta_increment_per_sampling)

        # Sample batch
        state_batch, action_batch, profit_batch, cost_batch,\
        next_state_batch, terminal_batch, weights, batch_idxes \
        = self.memory.sample(self.batch_size, self.PER_beta, self.beta_increment_per_sampling)
        #reshape data
        profit_batch = profit_batch.reshape(-1, 1)
        cost_batch = cost_batch.reshape(-1, 1)
        terminal_batch = terminal_batch.reshape(-1, 1)

        # weights = np.ones_like(profit_batch)
        # prPurple('weights = {}'.format(weights))

        # Ensure all tensors are on the correct device
        state_batch = to_tensor(state_batch).to(self.device)
        action_batch = to_tensor(action_batch).to(self.device)
        profit_batch = to_tensor(profit_batch).to(self.device)
        cost_batch = to_tensor(cost_batch).to(self.device)
        next_state_batch = to_tensor(next_state_batch).to(self.device)
        terminal_batch = to_tensor(terminal_batch).to(self.device)
        weights = torch.Tensor(weights).to(self.device)

        # Prepare for the target q batch -- profit
        # pernewact: target 动作先 clamp（+ raw 卫生界）再进 target critic，
        # 保证 Q 只在界内动作上被求值（外推区不进入 bootstrap 链）。
        a_next = self._bound_tensor(self._bound_raw(self.actor_target(next_state_batch)))
        next_q_values_p = self.critic_target_p([
            next_state_batch,
            a_next,
        ])

        # Prepare for the target q batch -- cost
        next_q_values_c = self.critic_target_c([
            next_state_batch,
            a_next,
        ])

        # A1: terminal 语义修复 —— bootstrap 当且仅当非终止转移
        # (terminal_batch = 存储的 done: 1=终止/块结束 → (1-done)=0 切断 bootstrap;
        #  0=正常转移 → 正常 bootstrap。修复原公式把 done 直接当乘子的反转)
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

        TD_errors_p = abs(target_q_batch_p - q_batch_p)
        TD_errors_c = abs(target_q_batch_c - q_batch_c)

        # Compute the value loss
        value_loss_p = torch.mean(torch.mean(weights * (q_batch_p - target_q_batch_p) ** 2, dim=1))
        value_loss_p.backward()
        # Gradient Clipping
        nn.utils.clip_grad_norm_(self.critic_p.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optim_p.step()

        # Compute the value loss for cost
        value_loss_c = torch.mean(torch.mean(weights * (q_batch_c - target_q_batch_c) ** 2, dim=1))
        value_loss_c.backward()
        # Gradient Clipping
        nn.utils.clip_grad_norm_(self.critic_c.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optim_c.step()

        # Actor update
        self.actor.zero_grad()
        # cost 为 +
        # pernewact: actor 输出先 clamp（+ raw 卫生界）再进 critic，梯度沿
        # straight-through 直达 actor —— actor 永远只收到"界内动作的 Q 梯度"。
        a_cur = self._bound_tensor(self._bound_raw(self.actor(state_batch)))
        policy_loss = -(self.critic_p([state_batch, a_cur]) - \
            self.lam * self.critic_c([state_batch, a_cur])) #TODO

        policy_loss = policy_loss.mean()
        policy_loss.backward(retain_graph=True)
        #Gradient Clipping
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        
        # check gradient
        # for name, parms in self.actor.named_parameters():	
        #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)

        self.actor_optim.step()

        # TODO update dual variable
        if not self.ema_lam:
            lam_loss = (self.critic_c([state_batch, a_cur]) - self.cost_limit)
            lam_loss = lam_loss.mean().item()
            self.lam_loss = lam_loss  # for curriculum convergence check: critic_c - cost_limit
            # prGreen(lam_loss)
            self.lam += lam_loss * self.lr_lam
            # prLightPurple('New lambda = {}'.format(self.lam))
            self.lam = max(self.lam, self.lam_min)
            if self.lam_max is not None:
                self.lam = min(self.lam, self.lam_max)
        # else: EMA mode — lam is advanced externally via update_lam_ema()
        #       using the EMA of actual collected costs (see training loop).

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target_p, self.critic_p, self.tau)
        soft_update(self.critic_target_c, self.critic_c, self.tau)

        #For prioritized exprience replay
        #Update priorities of experiences with TD errors
        # Ensure tensors are on CPU before converting to numpy
        # B2: 优先级 = |TD_p|/mean_p + |TD_c|/mean_c (去 lam，量纲归一化)
        # 各自除以 running-mean 后相加：两路 TD 同量级，避免利润/成本任一侧主导采样，
        # 且与 lam 解耦（lam 已改 EMA 实际成本，不再进入回放分布）。
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

        # ---- pernewact: 上层 Q 值记录（仅 record_q=True，即 H 层） ----
        if self.record_q:
            self.q_p_mean = float(q_batch_p.mean().item())
            self.q_c_mean = float(q_batch_c.mean().item())

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

        self.critic_c.eval()
        self.critic_target_c.eval()

    def observe(self, r_t, c_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, c_t, s_t1, done)
            self.s_t = s_t1

    def observe_jump(self, s_t, a_t, r_t, c_t, s_t1, done):
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, c_t, s_t1, done)


    def random_action(self):
        charge = 10.
        if self.nb_actions == self.nb_turbine:
            action = np.random.uniform(self.a_lo, self.a_hi, self.nb_actions)
        if self.nb_actions == 1:
            action = np.random.uniform(-charge, charge, self.nb_actions)
        if self.nb_actions == self.nb_turbine + 1:
            action_H = np.random.uniform(self.a_lo, self.a_hi, self.nb_turbine)
            action_L = np.random.uniform(-charge, charge, 1)
            action = np.concatenate((action_H, action_L))
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        # pernewact: actor 输出(raw) + 噪声 → raw 卫生界 → clamp 风机维到 [a_lo, a_hi]
        raw = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)
        raw += self.is_training * max(self.epsilon,0) * ((np.random.random(self.nb_actions)-0.5)/10)[0] # noise [0.0, 1.0)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        if self.clamp_turbine:
            raw = np.clip(raw, -self.raw_C, self.raw_C)   # raw 卫生界（H 层）
        action = self._bound_np(raw)                      # clamp 风机维（行为有界）

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
       
        hard_update(self.actor_target,self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target_p, self.critic_p)
        hard_update(self.critic_target_c, self.critic_c)

    def save_model(self, output):
        torch.save(self.actor.state_dict(), '{}/actor_{}.pkl'.format(output, self.nb_actions))
        torch.save(self.critic_p.state_dict(), '{}/critic_p_{}.pkl'.format(output, self.nb_actions))
        torch.save(self.critic_c.state_dict(), '{}/critic_c_{}.pkl'.format(output, self.nb_actions))

    def save_checkpoint(self, output, step):
        """Save checkpoint in a dedicated subfolder per checkpoint step."""
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
