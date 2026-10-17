#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run87 诊断 4：物理侧的"降出力 → 降波动"到底有多强 + critic 的分风速段估值（只读）。

两件事：
  (1) **物理（FLORIS 模型）**：对 u 网格 × 均匀轴向因子 a 网格，算 ``P(u, a)`` 与
      ``|P(u+Δu, a) − P(u, a)|``（Δu = 2.5 m/s，= 违约步的中位 |Δu|）
      ⇒ 升/降 a 对单步功率跳变的**真实**控制权（不看网络、不看数据）。
  (2) **critic 的分段估值**：checkpoint 上按 u 分桶，算"a 整体 +0.1（少降出力）"的
      ΔQ_p / ΔQ_c / net（λ_phys = 50），与物理侧对照。

用法: python agentutil/probe_run87_physics_vs_critic.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import model as M            # noqa: E402
import aifflor.aifflor_turbine as AT  # noqa: E402

LAM_ACT, LAM_SCALE = 5.0, 0.1
LAM_PHYS = LAM_ACT / LAM_SCALE          # 50 元/(MW·step)

# ---------------------------------------------------------------- (1) 物理
print('=' * 100)
print('(1) 物理：FLORIS 风场功率对均匀轴向因子 a 的敏感度（Δu = 2.5 m/s）')
print('=' * 100)
m, _info = AT.build_model(verbose=False)
us = np.arange(8.0, 18.01, 1.0)
A_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 1.0 / 3.0]
print('  %6s' % 'u' + ''.join('%12s' % ('a=%.2f' % a) for a in A_GRID))
P = {}
for a in A_GRID:
    P[a] = np.array([AT.farm_power(m, u, np.full(AT.NT, a)) for u in us])
for i, u in enumerate(us):
    print('  %6.2f' % u + ''.join('%12.3f' % P[a][i] for a in A_GRID))
print('\n  单步功率跳变 |P(u+2.5) − P(u)| [MW]（违约阈值 3 MW）')
print('  %6s' % 'u' + ''.join('%12s' % ('a=%.2f' % a) for a in A_GRID))
for i, u in enumerate(us[:-1]):
    d = np.array([abs(P[a][i + 1] - P[a][i]) for a in A_GRID])
    print('  %6.2f' % u + ''.join('%12.3f' % x for x in d))
print('\n  ⇒ 若 a 越小跳变越小，则"降出力"在物理上确实有效；比较 a=0.20 与 a=0.05 的比值即为控制权')

# ---------------------------------------------------------------- (2) critic
print('\n' + '=' * 100)
print('(2) critic 的分风速段估值：a 整体 +0.1（少降出力）的边际效应')
print('=' * 100)
RUN = os.path.join('output', 'AIFFLOR-run87')
CKPT = os.path.join(RUN, 'checkpoint_272000')
sd_a = torch.load(os.path.join(CKPT, 'actor_6.pkl'), map_location='cpu')
sd_p = torch.load(os.path.join(CKPT, 'critic_p_6.pkl'), map_location='cpu')
sd_c = torch.load(os.path.join(CKPT, 'critic_c_6.pkl'), map_location='cpu')
actor = M.Actor(4, 6, hidden1=sd_a['fc1.weight'].shape[0], hidden2=sd_a['fc2.weight'].shape[0])
actor.load_state_dict(sd_a); actor.eval()
cp = M.Critic(4, 6, hidden1=sd_p['fc1.weight'].shape[0], hidden2=sd_p['fc4.weight'].shape[0])
cp.load_state_dict(sd_p); cp.eval()
cc = M.Critic(4, 6, hidden1=sd_c['fc1.weight'].shape[0], hidden2=sd_c['fc4.weight'].shape[0])
cc.load_state_dict(sd_c); cc.eval()


def arr(s):
    return np.fromstring(str(s).strip().lstrip('[').rstrip(']'), sep=' ')


it = pd.read_csv(os.path.join(RUN, 'df_hsrl.csv'),
                 usecols=['step', 'observation', 'u', 'P_F'], chunksize=100000)
O, U, PF = [], [], []
for ch in it:
    keep = ch['step'].to_numpy() > 200000          # 只看训练后段的状态分布
    if not keep.any():
        continue
    c = ch[keep]
    O.append(np.vstack([arr(x) for x in c['observation']]))
    U.append(c['u'].to_numpy(float))
    PF.append(c['P_F'].to_numpy(float))
O = np.vstack(O); U = np.concatenate(U); PF = np.concatenate(PF)
S = np.stack([O[:, 0] / 1000, O[:, 1] / 10, O[:, 2] / 0.5, O[:, 3] / 5], 1)
S = S[:60000]; U = U[:60000]; PF = PF[:60000]
st = torch.tensor(S, dtype=torch.float32)
with torch.no_grad():
    a0 = actor(st)
    a_up = a0.clone(); a_up[:, :5] = (a_up[:, :5] + 0.1).clamp(0, 0.5)
    qp0, qc0 = cp([st, a0]), cc([st, a0])
    qp1, qc1 = cp([st, a_up]), cc([st, a_up])
dQp = 1000 * (qp1 - qp0).squeeze(1).numpy()
dQc = 100 * (qc1 - qc0).squeeze(1).numpy()
net = dQp - LAM_PHYS * dQc
a_np = a0[:, :5].mean(1).detach().numpy()

print('  %13s %6s %8s %9s %9s %11s %11s %9s' %
      ('u 区间', 'n', 'a_det', 'ΔQ_p元', 'ΔQ_c', 'λ·ΔQ_c元', 'net元', 'net<0%'))
for lo, hi in [(0, 9), (9, 11), (11, 12.5), (12.5, 14), (14, 40)]:
    msk = (U >= lo) & (U < hi)
    if msk.sum() < 50:
        continue
    print('  [%4.1f,%4.1f) %6d %8.3f %9.1f %9.3f %11.1f %11.1f %9.1f'
          % (lo, hi, msk.sum(), a_np[msk].mean(), dQp[msk].mean(), dQc[msk].mean(),
             LAM_PHYS * dQc[msk].mean(), net[msk].mean(), 100 * (net[msk] < 0).mean()))
print('\n  解读：net = ΔQ_p − 50·ΔQ_c 是"a +0.1（少降出力）"的净值。')
print('  net>0 ⇒ critic 认为**少降更好**（actor 不该再降）；net<0 ⇒ 应该**多降**。')
print('  再由 1) 的物理表判断：即使 critic 说该降，物理上的降幅是否足以把 |ΔP| 压到 3 MW 以下。')
