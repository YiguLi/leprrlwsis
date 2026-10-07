# -*- coding: utf-8 -*-
"""`SP_TD3_PAVE` —— `aifflor.aifflor_ddpg.SP_TD3` 的**子类**，只加 PAVE 式 critic 几何正则。

为什么用子类而不是复制：基线路径由 `super().update_policy()` 原样走完，**逐位不变**
（`--critic_act relu --pave_w_* 0` 时本 arm ≡ aifflor 基线）；新增的只是
*(1) profit 双 critic 可切换激活*、*(2) 一个附加的几何优化步*。

与父类的差异清单（**只有这些**）：
  1. `__init__` 末尾：若 `critic_act != 'relu'` 或任一 `pave_w_* > 0`，把
     `critic_p / critic_p2 / critic_target_p / critic_target_p2` 换成
     `CriticPAVE`（**先 `load_state_dict` 继承父类同 seed 的初始化 ⇒ 参数逐位相同**），
     然后重建这两个 Adam（参数对象变了；lr 从旧优化器原样取回）并 `hard_update`
     目标网络。cost 双 critic 一律不动。
  2. `update_policy`：`super()` 之后再走一次 `_pave_step()`（**只更新 profit 双 critic**，
     不碰 actor、不碰 cost、不更新 PER 优先级、不改 `update_count`）。
  3. `_pave_step()`：从同一个 PER 缓冲取一个 batch，算 MPR/VFC/Curv（见
     `aifflorpave_pave.pave_terms`），加权求和后 backward + clip_grad_norm(2.0) + step。
     权重全 0 或 `pave_report_only` 时**只测量不施加**（用于标定权重与看 M/M/μ 量级）。
  4. 诊断：`<args.output>/pave_stats.csv` 追加 `n,mpr,vfc,curv,gscale,loss`（每
     `pave_log_every` 步一行，`--pave_log_every 0` 关闭）。

⚠ L 层（BESS 残差动作）由 main 用**基类** `SP_TD3` 构造 ⇒ 残差/零初始化设计不受影响。
"""
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from aifflor.aifflor_ddpg import SP_TD3
from aifflorpave.aifflorpave_pave import CriticPAVE, pave_terms, describe_pave
from util import to_tensor, hard_update


class SP_TD3_PAVE(SP_TD3):
    """TD3 + PER + 拉格朗日成本约束 + PAVE 式 critic 几何正则。"""

    def __init__(self, nb_states, nb_actions, args,
                 critic_act='relu',
                 pave_w_mpr=0.0, pave_w_vfc=0.0, pave_w_curv=0.0,
                 pave_sigma=0.05, pave_delta=0.0, pave_nv=1,
                 pave_every=1, pave_report_only=False, pave_log_every=50,
                 **kwargs):
        super(SP_TD3_PAVE, self).__init__(nb_states, nb_actions, args, **kwargs)
        self.args = args          # 基类不留 args 引用；这里留一份供 pave_stats.csv 定位 output/

        self.critic_act = str(critic_act).lower()
        self.pave_w_mpr = float(pave_w_mpr)
        self.pave_w_vfc = float(pave_w_vfc)
        self.pave_w_curv = float(pave_w_curv)
        self.pave_sigma = float(pave_sigma)
        self.pave_delta = float(pave_delta)
        self.pave_nv = int(pave_nv)
        self.pave_every = max(1, int(pave_every))
        self.pave_report_only = bool(pave_report_only)
        self.pave_log_every = int(pave_log_every)

        # Curv 只在 C² 激活下有意义（ReLU 分段线性 ⇒ ∇²_aaQ ≡ 0）；
        # report_only（标定模式）下也测，便于一次性拿到三项量级。
        self.pave_curv_on = bool((self.pave_w_curv != 0.0 or self.pave_report_only)
                                 and self.critic_act != 'relu')
        self.pave_need_critic = (self.critic_act != 'relu'
                                 or self.pave_w_mpr != 0.0
                                 or self.pave_w_vfc != 0.0
                                 or self.pave_w_curv != 0.0
                                 or self.pave_report_only)
        self.pave_enabled = bool(self.pave_need_critic)
        self._pave_calls = 0
        self._pave_ema = None
        self._pave_csv = None
        self._pave_csv_path = None
        if self.pave_enabled:
            self._swap_profit_critics()
        if getattr(self, 'pave_w_curv', 0.0) and self.critic_act == 'relu':
            print('[PAVE] ⚠ --pave_w_curv>0 但 --critic_act relu：ReLU 下 ∇²_aaQ≡0，'
                  'Curv 项被自动置 0（要它生效请用 --critic_act silu）')

    # ------------------------------------------------------------------ 内部
    def _swap_profit_critics(self):
        """把 profit 双 critic（含目标）换成 CriticPAVE，继承同参数后重建 Adam。"""
        lr_p = self.critic_optim_p.param_groups[0]['lr']
        lr_p2 = self.critic_optim_p2.param_groups[0]['lr']
        for online, target in (('critic_p', 'critic_target_p'),
                               ('critic_p2', 'critic_target_p2')):
            old = getattr(self, online)
            na = old.fc2.weight.shape[1] - old.fc4.weight.shape[0]
            new = CriticPAVE(old.fc1.weight.shape[1], na,
                             hidden1=old.fc1.weight.shape[0],
                             hidden2=old.fc4.weight.shape[0],
                             act=self.critic_act).to(self.device)
            new.load_state_dict(old.state_dict())      # 参数逐位继承（初始化与 RL 无关）
            setattr(self, online, new)
            tgt = CriticPAVE(old.fc1.weight.shape[1], na,
                             hidden1=old.fc1.weight.shape[0],
                             hidden2=old.fc4.weight.shape[0],
                             act=self.critic_act).to(self.device)
            setattr(self, target, tgt)
            hard_update(tgt, new)
        self.critic_optim_p = Adam(self.critic_p.parameters(), lr=lr_p)
        self.critic_optim_p2 = Adam(self.critic_p2.parameters(), lr=lr_p2)

    def _pave_step(self):
        """附加的几何优化步：只更新 profit 双 critic。"""
        (state_batch, action_batch, _profit, _cost, next_state_batch,
         _terminal, _weights, _idxes) = self.memory.sample(
            self.batch_size, self.PER_beta, self.beta_increment_per_sampling)
        s = to_tensor(state_batch).to(self.device)
        a = to_tensor(action_batch).to(self.device)
        s2 = to_tensor(next_state_batch).to(self.device)

        self.critic_p.zero_grad()
        self.critic_p2.zero_grad()
        mpr, vfc, curv, gs = pave_terms([self.critic_p, self.critic_p2], s, a, s2,
                                        sigma=self.pave_sigma, delta=self.pave_delta,
                                        n_v=self.pave_nv, do_curv=self.pave_curv_on)
        loss = (self.pave_w_mpr * mpr + self.pave_w_vfc * vfc
                + self.pave_w_curv * curv)
        if not self.pave_report_only and float(loss.detach()) != 0.0:
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic_p.parameters(), max_norm=2.0, norm_type=2)
            self.critic_optim_p.step()
            nn.utils.clip_grad_norm_(self.critic_p2.parameters(), max_norm=2.0, norm_type=2)
            self.critic_optim_p2.step()
        vals = (float(mpr.detach()), float(vfc.detach()), float(curv.detach()),
                float(gs), float(loss.detach()))
        self._pave_ema = vals if self._pave_ema is None else tuple(
            0.99 * o + 0.01 * n for o, n in zip(self._pave_ema, vals))
        self._pave_log(vals)

    def _pave_log(self, vals):
        """把几何正则的量级写进 <output>/pave_stats.csv（标定权重用）。"""
        out = getattr(self.args, 'output', None)
        if not out or self.pave_log_every <= 0:
            return
        if self._pave_csv is None:
            self._pave_csv_path = os.path.join(out, 'pave_stats.csv')
            new = not os.path.exists(self._pave_csv_path)
            self._pave_csv = open(self._pave_csv_path, 'a', newline='')
            if new:
                self._pave_csv.write('n,mpr,vfc,curv,gscale,loss\n')
        if self._pave_calls % self.pave_log_every == 0:
            self._pave_csv.write('%d,%.6g,%.6g,%.6g,%.6g,%.6g\n'
                                 % ((self._pave_calls,) + vals))
            self._pave_csv.flush()

    # ------------------------------------------------------------------ 对外
    def update_policy(self):
        """父类更新 + 每 `pave_every` 次追加一个几何步（关掉时逐位等于父类）。"""
        super(SP_TD3_PAVE, self).update_policy()
        if self.pave_enabled and (self._pave_calls % self.pave_every == 0):
            self._pave_step()
        self._pave_calls += 1

    def pave_summary(self):
        if not self.pave_enabled:
            return '[PAVE] off（基线口径）'
        e = self._pave_ema or (float('nan'),) * 5
        return (describe_pave(self.pave_w_mpr, self.pave_w_vfc, self.pave_w_curv,
                              self.critic_act, self.pave_sigma, self.pave_delta,
                              self.pave_nv, self.pave_report_only)
                + ' | EMA(mpr,vfc,curv,gscale,loss)=(%.4g, %.4g, %.4g, %.4g, %.4g)'
                % e)
