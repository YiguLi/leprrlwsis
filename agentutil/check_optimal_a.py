#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""应该在 高风速 一直降载到违约率很低" —— 先算**奖励自己的最优点**在哪（只读）。

对 noise 池 day122 的真实 1440 步风速序列，枚举"常数 a"策略，用**生产物理**算
  revenue(a) = Σ pr·P(u_t,a)/60
  c1(a)      = Σ [P_F + K_V·(P_F−3)⁺],  P_F = |P(u_t,a) − P(u_{t−1},a)|
  J(a)       = revenue(a) − λ·c1(a)
然后看 argmax 的 a 与它对应的违约率。

若 argmax 的违约率仍很高 ⇒ "降载到违约率很低"与"收益项"**在目标里是冲突的**，
网络没做到不是学不会，而是奖励不允许；若 argmax 在 a≈0.03（近停机）⇒ 那是学习失败。

用法: python agentutil/check_optimal_a.py [--day 122]
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

import aifflor.aifflor_turbine as AT                      # noqa: E402
from aifflor.aifflor_env import AifflorPhysics            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--day', type=int, default=122)
ap.add_argument('--csv', default='FFC_speed_noise_1.csv')
a = ap.parse_args()

# ---- 当天真实 (u_t, pr_t) 序列（noise 池，1440 步；pr 用与 env 相同的日程表）----
raw = pd.read_csv(a.csv)
u = raw.iloc[0:1440, a.day].to_numpy(float)
# 电价：300/580/980 的日程表（env.CalPrice 的三档，按小时段）；这里直接取 rollout 的 pr 更稳
roll = pd.read_csv('output/fixed_action/noise_day122_a0.10.csv')
pr = roll.pr.to_numpy(float) if len(roll) == 1440 else np.full(1440, 580.0)
assert roll.u.shape[0] == 1440
u = roll.u.to_numpy(float)          # 用 rollout 里记录的实际 u（与本池逐位一致）

ph = AifflorPhysics(nt=5, calibrated=False, power_scale=AT.LEVEL_COEF)
U_MIN = float(AT.U_MIN_ENV)


def power_all(us, aa):
    us = np.clip(np.asarray(us, float), max(U_MIN, 0.1), None)
    acts = np.full((us.size, AT.NT), float(np.clip(aa, AT.A_PHYS_LO, AT.A_HI)))
    farm, _ = AT.eval_batch(ph.model, us, acts)
    return np.asarray(farm, float) * ph.power_scale


print('day=%d | pool=%s | %d 步 | u 均值 %.2f' % (a.day, a.csv, len(u), u.mean()))
A_GRID = np.array([0.025, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 1 / 3.0])
print('\n  %7s %9s %9s %9s %9s %10s' % ('a', 'P_G均', '违约率', 'c1(KV20)', '收入/天', 'J(KV20,λ=50)'))
best = (-1e18, None)
rows = []
for aa in A_GRID:
    P = power_all(u, aa)
    pf = np.abs(np.diff(P))
    c20 = pf + 20 * np.maximum(0, pf - 3)
    c100 = pf + 100 * np.maximum(0, pf - 3)
    rev = float((pr[1:] * P[1:] / 60).sum())
    J = rev - 50.0 * c20.sum()
    rows.append(dict(a=aa, PG=P.mean(), viol=100 * (pf > 3).mean(), c20=c20.sum(),
                     c100=c100.sum(), rev=rev, J50_20=J))
    if J > best[0]:
        best = (J, aa)
    print('  %7.3f %9.2f %8.2f%% %9.1f %9.0f %12.0f'
          % (aa, P.mean(), 100 * (pf > 3).mean(), c20.sum(), rev, J))

R = pd.DataFrame(rows)
print('\n  目标函数的 argmax: a = %.3f（该点违约率 %.1f%%、出力 %.2f MW、收入 %.0f 元/天）'
      % (best[1], float(R.loc[R.a == best[1], 'viol'].iloc[0]),
         float(R.loc[R.a == best[1], 'PG'].iloc[0]),
         float(R.loc[R.a == best[1], 'rev'].iloc[0])))
for kv, key, lab in ((20, 'c20', 'K_V=20'), (100, 'c100', 'K_V=100')):
    for lam in (5.0, 50.0, 200.0):
        J = R.rev - lam * R[key]
        i = int(np.argmax(J.to_numpy()))
        print('  %-8s λ=%5.0f ⇒ argmax a=%.3f | 该点违约率 %5.1f%% | 收入 %7.0f 元/天'
              % (lab, lam, R.a[i], R.loc[i, 'viol'], R.loc[i, 'rev']))
