#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把"同一天、同一动作、仅风速池不同（noise vs clean）"的两条 rollout 叠在一起比（只读）。

读取 `agentutil/run_fixed_action_day.py` 落盘的 CSV：
    output/fixed_action/noise_day<J>_act<...>.csv
    output/fixed_action/clean_day<J>_act<...>.csv
画 3 联：u(t)、P_G(t)、P_F(t)（叠加 + 3 MW 线），并把两组的 P_F 分位与违约率打在标题上。

用法:
    python agentutil/plot_pool_compare.py                       # 默认 day=122, act=0.2_0.2_0.2_0.2_0.3
    python agentutil/plot_pool_compare.py --day 122 --act 0.2_0.2_0.2_0.2_0.2
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
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.environ.get('TEMP', '/tmp'), 'mplcache'))

import matplotlib                     # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--day', type=int, default=122)
ap.add_argument('--act', default='0.2_0.2_0.2_0.2_0.3')
ap.add_argument('--dir', default=os.path.join('output', 'fixed_action'))
a = ap.parse_args()

paths = {}
for pool in ('noise', 'clean'):
    p = os.path.join(a.dir, '%s_day%d_act%s.csv' % (pool, a.day, a.act))
    if not os.path.exists(p):
        raise SystemExit('缺文件: %s（先跑 run_fixed_action_day.py）' % p)
    paths[pool] = p

D = {k: pd.read_csv(v) for k, v in paths.items()}
cols = {'noise': '#d62728', 'clean': '#1f77b4'}

fig, ax = plt.subplots(3, 1, figsize=(11, 9), dpi=140, sharex=True)
for k, d in D.items():
    ax[0].plot(d.t, d.u, lw=0.9, color=cols[k], label='%s (σ(Δu)=%.3f)' % (k, np.diff(d.u).std()))
    ax[1].plot(d.t, d.P_G, lw=0.9, color=cols[k], label='%s (mean %.2f MW)' % (k, d.P_G.mean()))
    ax[2].plot(d.t, d.P_F, lw=0.8, color=cols[k],
               label='%s (mean %.3f, p95 %.2f, viol %.2f%%)'
                     % (k, d.P_F.mean(), np.percentile(d.P_F, 95), 100 * (d.P_F > 3).mean()))
ax[0].set_ylabel('$u$ [m/s]')
ax[0].set_title('day j=%d | action %s | FLORIS | BESS idle (g=0) | noise vs clean wind pool'
                % (a.day, a.act), fontsize=10)
ax[1].set_ylabel('$P_G$ [MW]')
ax[2].axhline(3.0, color='k', ls='--', lw=1.0)
ax[2].set_ylabel('$P_F$ [MW]')
ax[2].set_xlabel('minute of day')
for _a in ax:
    _a.legend(fontsize=8, loc='upper left')
    _a.grid(alpha=0.3, lw=0.5)
fig.tight_layout()
out = os.path.join(a.dir, 'compare_pool_day%d_act%s.png' % (a.day, a.act))
fig.savefig(out)
plt.close(fig)
print('图: %s' % out)
for k, d in D.items():
    pf = d.P_F.to_numpy()
    print('%-6s P_F 均值 %.3f | p95 %.3f | max %.3f | 违约 %.2f%% | c1(KV20) %.1f | 收入 %.0f 元/天'
          % (k, pf.mean(), np.percentile(pf, 95), pf.max(), 100 * (pf > 3).mean(),
             (pf + 20 * np.maximum(0, pf - 3)).sum(), float((d.pr * d.P_G / 60).sum())))
