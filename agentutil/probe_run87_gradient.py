#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run87 诊断 3：checkpoint 上的 critic 梯度 —— λ1 钉顶时 actor 到底收到什么信号（只读）。

在 checkpoint_272000 上取状态 → actor 给动作 → 用 **critic_p / critic_c** 算
  f(a) = q_p(a) − λ·q_c(a)            （actor 的 loss 就是 −f，λ = lam1 的 actor 值）
再看 ∂f/∂桨距 的方向与量级，以及"把 5 维桨距整体 +0.1"这个反事实的收益/成本差：

  ΔQ_p = 1000·Δq_p  [元]        ΔQ_c = 100·Δq_c  [MW·step]
  net = ΔQ_p − λ_phys·ΔQ_c     λ_phys = λ / lam_scale = 10·λ

用法: python agentutil/probe_run87_gradient.py [run_dir] [ckpt]
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

import model as M  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join('output', 'AIFFLOR-run87')
CKPT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RUN, 'checkpoint_272000')
LAM = 5.0          # run87 的 lam1 峰值（actor 空间）
S_P, S_C = 1000.0, 100.0

sd_a = torch.load(os.path.join(CKPT, 'actor_6.pkl'), map_location='cpu')
sd_p = torch.load(os.path.join(CKPT, 'critic_p_6.pkl'), map_location='cpu')
sd_c = torch.load(os.path.join(CKPT, 'critic_c_6.pkl'), map_location='cpu')
nb_s, h1 = sd_a['fc1.weight'].shape[1], sd_a['fc1.weight'].shape[0]
h2 = sd_a['fc2.weight'].shape[0]
actor = M.Actor(nb_s, 6, hidden1=h1, hidden2=h2)
actor.load_state_dict(sd_a)
actor.eval()
cp = M.Critic(nb_s, 6, hidden1=sd_p['fc1.weight'].shape[0], hidden2=sd_p['fc4.weight'].shape[0])
cp.load_state_dict(sd_p); cp.eval()
cc = M.Critic(nb_s, 6, hidden1=sd_c['fc1.weight'].shape[0], hidden2=sd_c['fc4.weight'].shape[0])
cc.load_state_dict(sd_c); cc.eval()
print('actor: nb_states=%d hidden=(%d,%d)' % (nb_s, h1, h2))

# ---- 取状态：observation = [pr, u, E, P_G_last] → Normalize_H ----
obs = pd.read_csv(os.path.join(RUN, 'df_hsrl.csv'),
                  usecols=['step', 'observation', 'u', 'pr', 'SOC', 'P_F'], chunksize=100000)


def _arr(s):
    return np.fromstring(str(s).strip().lstrip('[').rstrip(']'), sep=' ')


S, U, PF = [], [], []
for ch in obs:
    O = np.vstack([_arr(x) for x in ch['observation'].to_numpy()])
    S.append(np.stack([O[:, 0] / 1000.0, O[:, 1] / 10.0, O[:, 2] / 0.5, O[:, 3] / 5.0], 1))
    U.append(ch['u'].to_numpy(float))
    PF.append(ch['P_F'].to_numpy(float))
S = np.vstack(S); U = np.concatenate(U); PF = np.concatenate(PF)
idx = np.random.RandomState(0).choice(len(S), size=min(40000, len(S)), replace=False)
S, U, PF = S[idx], U[idx], PF[idx]
st = torch.tensor(S, dtype=torch.float32)
print('状态样本 %d（u: p10 %.1f p50 %.1f p90 %.1f）' % (len(S), np.percentile(U, 10), np.median(U), np.percentile(U, 90)))

with torch.no_grad():
    a0 = actor(st)


def crit_grad(net, a, lam_sign=1.0):
    a = a.clone().requires_grad_(True)
    q = net([st, a])
    g, = torch.autograd.grad(q.sum(), a)
    return q.detach(), g


with torch.no_grad():
    qp0 = cp([st, a0]); qc0 = cc([st, a0])
qp0, gp = crit_grad(cp, a0)
qc0, gc = crit_grad(cc, a0)

print('\n[1] critic 对 5 维桨距的梯度（归一化空间；∂q/∂pitch）')
print('  %8s %12s %12s %14s %14s' % ('dim', '|∂q_p/∂a|', '|∂q_c/∂a|', 'λ·|∂q_c/∂a|', '占比 λc/p'))
for j in range(5):
    ap, ac = float(gp[:, j].abs().mean()), float(gc[:, j].abs().mean())
    print('  a%d      %12.4e %12.4e %14.4e %13.2f%%' % (j, ap, ac, LAM * ac, 100 * LAM * ac / max(ap, 1e-12)))
print('  （占比 < 100%% ⇒ 利润梯度压过成本梯度 ⇒ actor 学的是"少降出力"）')

# ---- 反事实：5 维桨距整体 +0.1（更多降出力）----
sh = 0.1
a_up = a0.clone()
a_up[:, :5] = (a_up[:, :5] + sh).clamp(0.0, 0.5)
with torch.no_grad():
    qp_up, qc_up = cp([st, a_up]), cc([st, a_up])
dqp = (qp_up - qp0).squeeze(1).numpy()
dqc = (qc_up - qc0).squeeze(1).numpy()
dQp, dQc = S_P * dqp, S_C * dqc
net = dQp - (LAM / 0.1) * dQc          # λ_phys = λ/lam_scale = 10λ

print('\n[2] 反事实：5 维桨距 +%.1f（更多降出力），critic 的预测' % sh)
print('  ΔQ_p（利润, 元）      : 均值 %+9.2f | p10 %+9.2f | p90 %+9.2f'
      % (dQp.mean(), np.percentile(dQp, 10), np.percentile(dQp, 90)))
print('  ΔQ_c（成本, MW·step） : 均值 %+9.4f | p10 %+9.4f | p90 %+9.4f'
      % (dQc.mean(), np.percentile(dQc, 10), np.percentile(dQc, 90)))
print('  net = ΔQ_p − %.0f·ΔQ_c : 均值 %+9.2f | 为负的占比 %.1f%%'
      % (LAM / 0.1, net.mean(), 100 * (net < 0).mean()))
print('  （net < 0 ⇒ critic 自己认为"降出力不划算"，actor 没有理由学它）')

# ---- 高风速子集（违约几乎都发生在 u>12.5）----
for lo in (11.0, 12.5, 13.5):
    m = U > lo
    if m.sum() < 100:
        continue
    print('\n  u > %.1f 的子集 (%d 步):' % (lo, int(m.sum())))
    print('    ΔQ_p %+8.2f 元 | ΔQ_c %+8.4f MW·step | net %+9.2f 元 | net<0 占比 %.1f%%'
          % (dQp[m].mean(), dQc[m].mean(), net[m].mean(), 100 * (net[m] < 0).mean()))
    print('    实际 P_F 均值 %.2f | 该子集桨距均值 %.4f' % (PF[m].mean(), float(a0[m, :5].mean())))

# ---- 需要多大的 λ 才能让"降出力"划算：λ* = ΔQ_p/ΔQ_c ----
mask = np.abs(dQc) > 1e-9
lam_star = dQp[mask] / dQc[mask]
print('\n[3] 反事实的临界乘子 λ* = ΔQ_p/ΔQ_c（物理口径；> 当前 λ_phys=%.0f 才值得降出力）' % (LAM / 0.1))
print('  λ*: p10 %+9.1f | p50 %+9.1f | p90 %+9.1f（负值说明 ΔQ_c<0 而 ΔQ_p<0：降出力同时减成本与利润）'
      % tuple(np.percentile(lam_star, [10, 50, 90])) if mask.any() else '  (ΔQ_c 全 ~0)')
print('  为使 net>0 需要 λ_phys > p90(λ*) = %.1f' % np.percentile(lam_star, 90))
