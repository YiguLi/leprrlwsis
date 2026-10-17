#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run87 诊断 8：**奖励的梯度是明确的——但它的一阶项在期望上抵消了**（只读）

用户口径：现有奖励/成本的单步梯度是明确的（`∂r1/∂a = pr·dt·∂P/∂a`、
`∂c1/∂a = sign(ΔP)·∂P/∂a`）。本脚本用 FLORIS 把这两项逐条算出来，回答：
  * 单步梯度有多大（带号）；
  * 它在 (u, Δu) 上取期望后还剩多少 —— 若 `E[∂c1/∂a] ≈ 0` 而 `E|∂c1/∂a|` 很大，
    就说明"梯度明确但方向随状态翻转"，actor 学到的是**零均值噪声**而不是平坦地形。

做法（逐条，不训练）：
  ΔP(a)  = |P(u_t, a) − P(u_{t−1}, a)|,   u_{t−1} = u_t − Δu（日志里的 du）
  ∂c1/∂a = (1 + K_V·1{P_F>3}) · sign(P1 − P0) · (∂P/∂a|_u_t − ∂P/∂a|_u_{t−1})
  ∂r1/∂a = pr·dt · ∂P/∂a|_u_t
  net    = ∂r1/∂a − λ_phys · ∂c1/∂a
（a 取日志里 5 台机的均值；物理表用均匀 a，与 aifflor 的逐台最优只差常数）
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import aifflor.aifflor_turbine as AT  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join('output', 'AIFFLOR-run87')
K_V = 20.0          # run87 的 c1 口径
LAM_PHYS = 50.0     # run87 的 λ1（actor 5.0 ÷ lam_scale 0.1）

# ---------------------------------------------------------------- 日志
rows = []
prev_u = None
for ch in pd.read_csv(os.path.join(RUN, 'df_hsrl.csv'),
                      usecols=['u', 'pr', 'P_F', 'applied_action_H'], chunksize=100000):
    act = np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                     for x in ch['applied_action_H']])
    u = ch['u'].to_numpy(float)
    du = np.empty_like(u)
    du[0] = u[0] - prev_u if prev_u is not None else np.nan
    du[1:] = u[1:] - u[:-1]
    prev_u = u[-1]
    rows.append(pd.DataFrame({'u': u, 'du': du, 'pr': ch['pr'].to_numpy(float),
                              'PF': ch['P_F'].to_numpy(float),
                              'a': act[:, :5].mean(1)}))
D = pd.concat(rows, ignore_index=True).dropna()
D = D[(D.a > -0.05) & (D.a < 0.55) & (np.abs(D.du) < 8)]
D = D.sample(n=min(40000, len(D)), random_state=0)
print('样本 %d 步（u p50 %.1f | |Δu| p50 %.2f）' % (len(D), np.median(D.u), np.median(D.du.abs())))

# ---------------------------------------------------------------- P(u,a) 表
UG = np.arange(4.0, 24.01, 0.25)
A = np.arange(0.0, 0.351, 0.01)
model, _ = AT.build_model(verbose=False)
farm, _ = AT.eval_batch(model, np.tile(UG, A.size),
                        np.stack([np.repeat(A, UG.size)] * AT.NT, 1))
P = farm.reshape(A.size, UG.size)          # (na, nu)


def p_of(u, a):
    i = np.clip(np.searchsorted(A, a) - 0, 0, A.size - 1)
    a = np.clip(a, A[0], A[-1])
    lo = np.clip(np.floor((a - A[0]) / 0.01).astype(int), 0, A.size - 2)
    frac = (a - A[lo]) / 0.01
    pu = np.array([np.interp(u[k], UG, P[lo[k]]) for k in range(len(u))])
    pu1 = np.array([np.interp(u[k], UG, P[lo[k] + 1]) for k in range(len(u))])
    return pu * (1 - frac) + pu1 * frac


def dpda(u, a, da=0.01):
    return (p_of(u, a + da) - p_of(u, a - da)) / (2 * da)


u_t = D.u.to_numpy()
u_p = u_t - D.du.to_numpy()
a = D.a.to_numpy()
pr = D.pr.to_numpy()
PF = D.PF.to_numpy()

P1 = p_of(u_t, a)
P0 = p_of(u_p, a)
sign = np.sign(P1 - P0)
d1 = dpda(u_t, a)
d0 = dpda(u_p, a)
slope = 1.0 + K_V * (PF > 3.0).astype(float)      # c1 的分段斜率
dc1 = slope * sign * (d1 - d0)                    # ∂c1/∂a（一步，a_prev 固定）
dr1 = pr * (1.0 / 60.0) * d1                      # ∂r1/∂a
net = dr1 - LAM_PHYS * dc1

print('\n[1] 单步梯度（a = 5 台机均值；λ_phys = %.0f, K_V = %.0f）' % (LAM_PHYS, K_V))
print('  %-22s %12s %12s %12s' % ('', '带号均值', '均值|·|', 'E[·]/E|·|'))
for nm, v in (('∂r1/∂a  [元/单位a]', dr1),
              ('∂c1/∂a  [MW·step/单位a]', dc1),
              ('λ·∂c1/∂a [元/单位a]', LAM_PHYS * dc1),
              ('net      [元/单位a]', net)):
    print('  %-22s %12.3f %12.3f %11.1f%%'
          % (nm, v.mean(), np.abs(v).mean(), 100 * v.mean() / (np.abs(v).mean() + 1e-12)))

print('\n[2] 符号一致性（∂c1/∂a 与 sign(ΔP)）: %.1f%% 同号；E[sign(ΔP)] = %+.3f'
      % (100 * np.mean(np.sign(dc1) == sign), sign.mean()))

print('\n[3] 高风 + 大风变子集（u>13.5 且 |Δu|>2.5，违约的主要来源）')
m = (u_t > 13.5) & (np.abs(D.du.to_numpy()) > 2.5)
for nm, v in (('∂c1/∂a', dc1[m]), ('λ·∂c1/∂a', LAM_PHYS * dc1[m]), ('net', net[m])):
    print('  %-12s 带号均值 %+9.3f | 均值|·| %9.3f | 正占比 %.1f%%'
          % (nm, v.mean(), np.abs(v).mean(), 100 * (v > 0).mean()))
print('  n = %d 步' % m.sum())

print('\n[4] 把 λ 抬到 200（= actor 20）也不会改变\"符号翻转\"这件事，只会放大 |·|：')
for lam in (50.0, 200.0):
    v = dr1 - lam * dc1
    print('  λ_phys=%5.0f: net 带号均值 %+9.2f | 均值|·| %9.2f | 正占比 %.1f%%'
          % (lam, v.mean(), np.abs(v).mean(), 100 * (v > 0).mean()))
print('\n结论口径：若"带号均值 ≪ 均值|·|"，则梯度**明确但方向随状态翻转** ⇒')
print('          平滑 actor 只能拟合其平均（≈0），这不是奖励缺失，也不是 TD3/DDPG 的局部性。')
