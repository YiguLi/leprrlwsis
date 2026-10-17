#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run87 诊断 9：**"违约率 < 1%" 能不能做到、要付多少出力**（只读，纯开环重放）

问题：给定真实的 u(t) 序列与 3 MW/步 的硬限，只用"预留（水平低于可达上限）+ 每步最多升 3 MW"
这一族策略，违约率与平均出力的 Pareto 前沿长什么样？1% 在第几个点上？

策略族（r = 预留比例）：
    P(t) = min( P(t−1) + 3 , (1−r)·cap(u_t) )        cap = 该风速下可达的最大风场出力
对照：
    * 实际策略（run87 日志）                  违约 5.25% / 出力 4.13 MW
    * 3 行规则（r=0，顶在上限）               违约 3.98% / 出力 6.34 MW
    * **神谕**：知道 u_{t+1} ⇒ P(t) = min(P(t−1)+3, cap(u_t), cap(u_{t+1})+3)
    * **|Δu| 条件预留**：r 随 |Δu(t)| 增大（用可观测的 Δu 近似"预判"）

用法: python agentutil/probe_run87_violation_frontier.py [run_dir]
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

# ---------------- 读 (u, P_G(t−1), P_G(t), P_F, pr) ----------------
pg_prev, pg_cur, U, PF, PR = [], [], [], [], []
for ch in pd.read_csv(os.path.join(RUN, 'df_hsrl.csv'),
                      usecols=['observation', 'observation2', 'u', 'P_F', 'pr'],
                      chunksize=100000):
    a1 = np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                    for x in ch['observation']])
    a2 = np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                    for x in ch['observation2']])
    pg_prev.append(a1[:, 3]); pg_cur.append(a2[:, 3])
    U.append(ch['u'].to_numpy(float)); PF.append(ch['P_F'].to_numpy(float))
    PR.append(ch['pr'].to_numpy(float))
PGp = np.concatenate(pg_prev); PGc = np.concatenate(pg_cur)
U = np.concatenate(U); PF = np.concatenate(PF); PR = np.concatenate(PR)

# ---------------- cap(u)：数据里同风速区间出现过的最大出力（构造可达） ----------------
binw = 0.25
ub = np.floor(U / binw).astype(int)
cap = {b: np.percentile(PGc[ub == b], 99.5) + 0.25 for b in np.unique(ub)}
keys = np.array(sorted(cap)); vals = np.array([cap[k] for k in keys])
vals = np.convolve(np.pad(vals, 2, mode='edge'), np.ones(5) / 5, mode='valid')
CAP = dict(zip(keys, vals))
cap_t = np.array([CAP[b] for b in ub])

ok = (PGp <= cap_t + 1.2) & (PGp >= -0.6) & (PGc >= -0.6)
PGp, PGc, U, PF, PR, cap_t = PGp[ok], PGc[ok], U[ok], PF[ok], PR[ok], cap_t[ok]
print('样本 %d 步（%.1f 天）| 实际: 违约 %.2f%% | 出力 %.2f MW | 上限均值 %.2f MW'
      % (len(U), len(U) / 1440, 100 * (PF > THR).mean(), PGp.mean(), cap_t.mean()))


def sim(rfun):
    P = np.empty_like(cap_t)
    P[0] = (1 - rfun(0)) * cap_t[0]
    for t in range(1, len(P)):
        P[t] = min(P[t - 1] + THR, (1 - rfun(t)) * cap_t[t])
        P[t] = max(P[t], 0.0)
    d = np.abs(np.diff(P))
    return (d > THR), P


print('\n[A] 预留比例 r 的 Pareto 前沿（策略：每步最多升 3 MW，顶到 (1−r)·cap）')
print('  %6s %10s %11s %11s %10s' % ('r', '违约率', '平均出力', '占上限', '违约次数/天'))
dU = np.concatenate([[0.0], np.diff(U)])
for r in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60):
    v, P = sim(lambda t, r=r: r)
    print('  %6.2f %9.2f%% %10.2f MW %10.1f%% %10.1f'
          % (r, 100 * v.mean(), P.mean(), 100 * P.mean() / cap_t.mean(), 1440 * v.mean()))

print('\n[B] 神谕（知道 u_{t+1}）与 |Δu| 条件预留')
v_omni, _ = sim(lambda t: 0.0)          # 占位
P = np.empty_like(cap_t)
cap_next = np.concatenate([cap_t[1:], cap_t[-1:]])
P[0] = cap_t[0]
for t in range(1, len(P)):
    P[t] = max(0.0, min(P[t - 1] + THR, cap_t[t], cap_next[t] + THR))
d = np.abs(np.diff(P)); print('  %-22s 违约 %6.2f%% | 出力 %5.2f MW（占上限 %5.1f%%）'
                              % ('神谕(知 u_{t+1})', 100 * (d > THR).mean(), P.mean(),
                                 100 * P.mean() / cap_t.mean()))
# |Δu| 条件预留：|Δu| 大 ⇒ 多留
for r0, rmax, thr in ((0.05, 0.5, 2.0), (0.02, 0.6, 2.5)):
    rf = (lambda t, r0=r0, rmax=rmax, thr=thr: min(rmax, r0 + rmax * min(1.0, abs(dU[t]) / thr)))
    v, P = sim(rf)
    print('  %-22s 违约 %6.2f%% | 出力 %5.2f MW（占上限 %5.1f%%）'
          % ('|Δu| 条件预留 r0=%.2f' % r0, 100 * v.mean(), P.mean(), 100 * P.mean() / cap_t.mean()))

print('\n[C] 收入口径：三种点的"每步收入"（按日志电价）')
for nm, r in (('实际策略', None), ('r=0（顶上限）', 0.0), ('r=0.30', 0.30)):
    if r is None:
        rev = (PR * PGp / 60).mean()
        print('  %-14s 出力 %5.2f MW | 收入 %7.3f 元/步 | 违约 %.2f%%'
              % (nm, PGp.mean(), rev, 100 * (PF > THR).mean()))
    else:
        v, P = sim(lambda t, r=r: r)
        rev = (PR * P / 60).mean()
        print('  %-14s 出力 %5.2f MW | 收入 %7.3f 元/步 | 违约 %.2f%%'
              % (nm, P.mean(), rev, 100 * v.mean()))
