#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run87 诊断 5：**降出力 → 降违约**的物理权衡曲线（FLORIS 直接算，不看网络）（只读）。

对日志里真实的 (u, Δu) 组合，问："若把 5 台机的轴向因子锁在 a，违约率会是多少？"
再配上同一点的出力，就得到"违约率 ↔ 出力"的整条 Pareto 曲线：

  a=0.20 → 全 14.9% / u>14 43.3%   但 P(14)=10.17 MW
  a=0.02 → 全  1.5% / u>14  8.4%   但 P(14)= 3.84 MW

⇒ 判断"降出力是否有效"以及"要降到多低才有效"（曲线是**凸**的：0.20→0.12 只降 2pp，
   0.05→0.02 降 19pp ⇒ 有效区在 a<0.05 的悬崖里）。

用法: python agentutil/probe_run87_control_authority.py [run_dir]
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

# ---- (u, Δu) 样本 ----
U, DU, prev = [], [], None
for ch in pd.read_csv(os.path.join(RUN, 'df_hsrl.csv'), usecols=['u', 'P_F'], chunksize=150000):
    u = ch['u'].to_numpy(float)
    du = np.empty_like(u)
    du[0] = u[0] - prev if prev is not None else np.nan
    du[1:] = u[1:] - u[:-1]
    prev = u[-1]
    U.append(u); DU.append(du)
U = np.concatenate(U); DU = np.concatenate(DU)
m = np.isfinite(DU) & (U > 0)
U, DU = U[m], DU[m]
sel = np.random.RandomState(0).choice(len(U), size=min(40000, len(U)), replace=False)
U, DU = U[sel], DU[sel]
print('样本 %d（u p50 %.1f | |Δu| p50 %.2f p90 %.2f）' % (len(U), np.median(U),
      np.median(np.abs(DU)), np.percentile(np.abs(DU), 90)))

# ---- P(u, a) 表 ----
A = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
UG = np.arange(6.0, 24.01, 0.25)
model, _ = AT.build_model(verbose=False)
farm, _ = AT.eval_batch(model, np.tile(UG, len(A)),
                        np.stack([np.repeat(A, UG.size)] * AT.NT, 1))
P = farm.reshape(len(A), UG.size)


def pinch(u):
    return np.array([np.interp(np.clip(u, UG[0], UG[-1]), UG, r) for r in P])


jump = np.abs(pinch(U + DU) - pinch(U))

print('\n[a] 固定 a 时的违约率（用真实 (u,Δu) 分布）')
print('  %6s %10s %12s %10s %10s %10s' % ('a', '全样本', 'u>12.5', 'u>14', 'P(u=14)', 'P(u=16)'))
for i, a in enumerate(A):
    print('  %6.2f %9.2f%% %11.2f%% %9.2f%% %9.2f MW %9.2f MW'
          % (a, 100 * (jump[i] > 3).mean(), 100 * (jump[i][U > 12.5] > 3).mean(),
             100 * (jump[i][U > 14] > 3).mean(),
             np.interp(14.0, UG, P[i]), np.interp(16.0, UG, P[i])))

print('\n[b] 边际效果（相邻 a 之间的违约率变化，u>14 段）—— 曲线是否凸')
prev = None
for i, a in enumerate(A):
    v = 100 * (jump[i][U > 14] > 3).mean()
    if prev is not None:
        print('  a %.2f → %.2f: 违约率 %5.2f%% → %5.2f%%（Δ%+.2f pp，Δa=%.2f ⇒ %.0f pp/0.1a）'
              % (A[i - 1], a, prev, v, v - prev, a - A[i - 1], (v - prev) / (a - A[i - 1]) * 0.1))
    prev = v

print('\n[c] 盈亏平衡的 λ（只考虑"多降 0.12→0.02"，u>14 段）')
dv = (100 * (jump[A.index(0.12)][U > 14] > 3).mean() - 100 * (jump[A.index(0.02)][U > 14] > 3).mean()) / 100
dP = np.interp(14.0, UG, P[A.index(0.12)]) - np.interp(14.0, UG, P[A.index(0.02)])
for pr in (300.0, 580.0, 980.0):
    d_rev = dP * pr / 60.0                        # 元/步
    lam_be = d_rev / (dv * 25.0) if dv > 0 else np.inf   # c1(viol) ≈ 25 MW·step
    print('  电价 %.0f 元/MWh: 少卖 %.2f 元/步 | 违约降 %.1f pp ⇒ 盈亏平衡 λ ≈ %.2f 元/(MW·step)'
          % (pr, d_rev, 100 * dv, lam_be))
print('  （run87 的 λ_phys = 50 元/(MW·step) —— 远高于上面任何一个盈亏平衡点 ⇒ 价格**不是**瓶颈）')
