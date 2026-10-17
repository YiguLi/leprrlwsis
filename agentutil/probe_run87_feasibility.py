#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run87 诊断 7：违约是"不可观测/不可行"还是"策略没做"？（只读，修正版）

用户论点（本脚本要检验）：每步已知 `u_t` 与上一步出力 `P_G(t−1)`、出手由自己掌握 ⇒
本步的 `|ΔP_G|` 完全可控（FLORIS 准稳态：任意目标水平 ≤ 可达上限都能用 a 精确做到），
"没有 Δu"不构成信息缺口。

三个可判定量：
1. **强制违约**：`P_G(t−1) − (可达到的最大出力 + 0.5) > 3` ⇒ 本步无论怎么出都违约。
2. **斜坡规则**：`P_G(t) = min(P_G(t−1) + 3, 可达上限)`（只做"每步最多升 3 MW"，
   该降就降）⇒ 违约率 = 该风序列下"逐点可行"策略的下界参照。
3. 对照实际策略（run87）的违约率与平均出力。

⚠ 可达上限用**日志里同风速区间出现过的最大出力**（按构造可达，避免"均匀 a"低估
逐台最优的事实；均匀 a 在 u≈16 时低估约 9%，会把 9% 的正常记录误判为不自洽）。
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

RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join('output', 'AIFFLOR-run87')
THR, PB = 3.0, 0.5

# ---------------- 读日志：P_G(t−1)、P_G(t)、u、P_F、step ----------------
prev_g, cur_g, U, PF, ST = [], [], [], [], []
for ch in pd.read_csv(os.path.join(RUN, 'df_hsrl.csv'),
                      usecols=['step', 'observation', 'observation2', 'u', 'P_F'],
                      chunksize=100000):
    a1 = np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                    for x in ch['observation']])
    a2 = np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                    for x in ch['observation2']])
    prev_g.append(a1[:, 3]); cur_g.append(a2[:, 3])
    U.append(ch['u'].to_numpy(float)); PF.append(ch['P_F'].to_numpy(float))
    ST.append(ch['step'].to_numpy(np.int64))
PGp = np.concatenate(prev_g); PGc = np.concatenate(cur_g)
U = np.concatenate(U); PF = np.concatenate(PF); ST = np.concatenate(ST)

# ---------------- 可达上限：按 u 分箱取"观测到的最大出力"（构造可达） ----------------
binw = 0.25
ub = np.floor(U / binw).astype(int)
cap = {}
for b in np.unique(ub):
    m = ub == b
    cap[b] = np.percentile(PGc[m], 99.5) + 0.25          # 留一点余量给 5 台机的最优分配
# 平滑相邻箱，减少样本噪声
keys = np.array(sorted(cap)); vals = np.array([cap[k] for k in keys])
vals_s = np.convolve(np.pad(vals, 2, mode='edge'), np.ones(5) / 5, mode='valid')
cap_s = dict(zip(keys, vals_s))
pm = np.array([cap_s[b] for b in ub])

ok = (PGp <= pm + PB + 0.6) & (PGp >= -0.6) & (PGc >= -0.6)
viol = PF > THR
print('总步 %d | 自洽 %.2f%% | 实际违约率 %.2f%%'
      % (len(ST), 100 * ok.mean(), 100 * viol[ok].mean()))
PGp, PGc, pm, U, PF, viol = PGp[ok], PGc[ok], pm[ok], U[ok], PF[ok], viol[ok]

# ---------------- 1) 强制违约 ----------------
forced = (PGp - (pm + PB)) > THR
print('\n[1] 强制违约（满出力+满放电仍拉不回 3 MW 内）: 全步 %.3f%% | 违约步 %.1f%%'
      % (100 * forced.mean(), 100 * forced.sum() / max(viol.sum(), 1)))
print('    ⇒ %.1f%% 的违约步在**本步物理上可避免**（只要把出力放到目标水平）'
      % (100 * (1 - forced.sum() / max(viol.sum(), 1))))

# ---------------- 2) 斜坡规则（逐点可行、不管收益） ----------------
g = np.empty_like(PGp)
g[0] = PGp[0]
for t in range(1, len(g)):
    g[t] = min(g[t - 1] + THR, pm[t] + PB)
    g[t] = max(g[t], 0.0)
PF_r = np.abs(np.diff(np.concatenate([[PGp[0]], g])))
print('\n[2] "每步最多升 3 MW、到顶就跟着风走"的规则: 违约率 %.2f%% | 平均出力 %.2f MW（上限 %.2f）'
      % (100 * (PF_r > THR).mean(), g.mean(), pm.mean()))
print('    实际策略: 违约率 %.2f%% | 平均出力 %.2f MW' % (100 * viol.mean(), PGp.mean()))
print('    ⇒ 规则用 %.1f%% 的出力拿到 %.2f%% 的违约率；实际策略出力更低(%.0f%%)却违约更多 ⇒ 被规则**双杀**'
      % (100 * g.mean() / pm.mean(), 100 * (PF_r > THR).mean(),
         100 * PGp.mean() / pm.mean()))

# ---------------- 3) 强制违约与"上一拍是否预留" ----------------
lag = (PGp - pm - PB)                   # >0 ⇒ 上一拍高出本拍上限
print('\n[3] 上一拍出力高于"本拍上限+3"的时刻: %.3f%%（这些时刻风一垮就必违约）'
      % (100 * (lag > THR).mean()))
print('    实际策略里"违约但非强制"的步占 %.2f%% ⇒ 这些是**策略选择**（没把出力压到目标）'
      % (100 * ((viol) & (~forced)).mean()))
