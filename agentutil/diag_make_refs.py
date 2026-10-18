#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断用参考最优（只读、离线算，跑一次即可）。

产出 `output/diag/refs_*.csv`：
  * `refs_u.csv` : 每个 u 的 `P_max(u)` 与 5 维最优动作 `a*(u)`（生产物理，坐标上升）
  * `refs_s1.csv`: u=10 的单点最优（实验 1-S1）
  * `refs_e2.csv`: 实验 2 的一维 `s=0.33·s` 参数族 —— 用**动态规划**在 day122(clean)
                   的真实 u(t)/pr(t) 上求当天最优 `J*` 与 `s*(t)`，并给"最优常数 s"
用法: python agentutil/diag_make_refs.py [--lam_phys 20] [--day 122]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import aifflor.aifflor_turbine as AT                     # noqa: E402
from aifflor.aifflor_env import AifflorPhysics           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--lam_phys', type=float, default=20.0, help='实验 2 的 λ（物理价）')
ap.add_argument('--day', type=int, default=122)
ap.add_argument('--csv', default='FFC_speed_1.csv')
a = ap.parse_args()
OUT = os.path.join('output', 'diag')
os.makedirs(OUT, exist_ok=True)

PH = AifflorPhysics(nt=5, calibrated=False, power_scale=AT.LEVEL_COEF)
K_V, P_F_LIMIT, DT = 20.0, 3.0, 1.0 / 60.0


def P_of(us, acts):
    """acts: (n,5) 或 (5,)；us: 标量或 (n,)。返回风场出力 [MW]。"""
    us = np.atleast_1d(np.asarray(us, float))
    acts = np.atleast_2d(np.asarray(acts, float))
    if acts.shape[0] == 1 and us.size > 1:
        acts = np.repeat(acts, us.size, axis=0)
    farm, _ = AT.eval_batch(PH.model, np.clip(us, max(AT.U_MIN_ENV, 0.1), None), acts)
    return np.asarray(farm, float) * PH.power_scale


def coord_ascent(u, starts=((1 / 3.0,) * 5, (0.2,) * 4 + (1 / 3.0,)), step=0.02, passes=4):
    """5 维坐标上升最大化 P(u, a)。每个维度把 25 个候选**一次批量**评估（快 25 倍）。"""
    cands = np.arange(0.02, 0.5001, step)
    best = None
    for s0 in starts:
        a = np.clip(np.asarray(s0, float), AT.A_PHYS_LO, AT.A_HI)
        val = float(P_of([u], a)[0])
        for _ in range(passes):
            for j in range(5):
                A = np.repeat(a[None, :], cands.size, 0)
                A[:, j] = cands
                vals = P_of(np.full(cands.size, u), A)
                k = int(np.argmax(vals))
                if float(vals[k]) > val + 1e-9:
                    val, a = float(vals[k]), A[k].copy()
        if best is None or val > best[0]:
            best = (val, a.copy())
    return best[0], best[1]


print('=' * 82)
print('1) 每个 u 的最优出力与动作（生产物理；坐标上升）')
rows = []
for u in np.arange(4.0, 18.01, 0.5):
    pmax, amax = coord_ascent(u)
    rows.append(dict(u=u, P_max=pmax, **{'a%d' % j: amax[j] for j in range(5)}))
U = pd.DataFrame(rows)
U.to_csv(os.path.join(OUT, 'refs_u.csv'), index=False)
print(U.to_string(index=False, float_format=lambda v: '%.4f' % v))

S1 = U[np.isclose(U.u, 10.0)].iloc[0]
S1.to_frame().T.to_csv(os.path.join(OUT, 'refs_s1.csv'), index=False)
print('\n2) S1 (u=10): P_max=%.4f  a*=%s' % (S1.P_max, ['%.3f' % S1['a%d' % j] for j in range(5)]))

# ---------------------------------------------------------------------------
# 3) 实验 2：一维 s 参数族的当天最优（DP）
# ---------------------------------------------------------------------------
raw = pd.read_csv(a.csv)
u_day = raw.iloc[:1440, a.day].to_numpy(float)
cp = np.arange(1, 1441)                       # env 的 CalPrice: 每步先 +1 再判档
pr_day = np.where((cp <= 7 * 60) | (cp > 23 * 60), 300.0,
                  np.where(((cp > 11 * 60) & (cp <= 17 * 60)) | ((cp > 21 * 60) & (cp <= 23 * 60)),
                           580.0, 980.0))

S = np.arange(0.0, 1.0001, 0.02)              # s 网格（51 点）
P = np.zeros((len(S), 1440))
for i, s in enumerate(S):
    P[i] = P_of(u_day, np.full(5, 0.33 * s))  # 逐分钟出力 [MW]
P0 = float(P_of([u_day[0]], np.full(5, AT.A_NOMINAL))[0])   # env.reset 的 P_G_prev（标称动作）
lam = a.lam_phys

# DP：状态 = 上一分钟的 s 下标；成本含 |P_t - P_{t-1}| 的 c1
nS = len(S)
Jprev = np.array([-lam * (abs(P[k, 0] - P0) + K_V * max(0.0, abs(P[k, 0] - P0) - P_F_LIMIT))
                  for k in range(nS)])
back = np.zeros((1440, nS), dtype=np.int16)
for t in range(1, 1440):
    pf = np.abs(P[:, t][:, None] - P[None, :, t - 1])          # (cur, prev)
    c1 = pf + K_V * np.maximum(0.0, pf - P_F_LIMIT)
    rev = pr_day[t] * P[:, t] * DT
    tot = rev[:, None] - lam * c1 + Jprev[None, :]
    back[t] = np.argmax(tot, axis=1)
    Jprev = tot.max(axis=1)
k_end = int(np.argmax(Jprev))
J_star = float(Jprev[k_end])
k = np.zeros(1440, dtype=int); k[-1] = k_end
for t in range(1439, 0, -1):
    k[t - 1] = back[t, k[t]]
s_star = S[k]

# 最优常数 s 作为对照
best_c = (-1e18, None)
for i, s in enumerate(S):
    pf = np.abs(np.diff(P[i]))
    c1 = pf + K_V * np.maximum(0.0, pf - P_F_LIMIT)
    J = float((pr_day[1:] * P[i, 1:] * DT).sum() - lam * c1.sum())
    if J > best_c[0]:
        best_c = (J, s)
E2 = pd.DataFrame(dict(minute=np.arange(1440), u=u_day, pr=pr_day, s_star=s_star,
                       P_star=P[k, np.arange(1440)]))
E2.to_csv(os.path.join(OUT, 'refs_e2.csv'), index=False)
pd.DataFrame([dict(J_star=J_star, s_const=best_c[1], J_const=best_c[0], lam_phys=lam,
                   n_s=len(S), s_step=0.02)]).to_csv(os.path.join(OUT, 'refs_e2_head.csv'),
                                                     index=False)
print('\n3) 实验 2（λ_phys=%.1f, day=%d, clean）: J*=%.1f | 最优常数 s=%.2f (J=%.1f)'
      % (lam, a.day, J_star, best_c[1], best_c[0]))
print('   s* 分布: p5 %.2f p50 %.2f p95 %.2f | 违约率 %.2f%%'
      % (*np.percentile(s_star, [5, 50, 95]), 100 * np.mean(np.abs(np.diff(P[k])) > P_F_LIMIT)))
print('\n已写: %s/{refs_u.csv, refs_s1.csv, refs_e2.csv, refs_e2_head.csv}' % OUT)
