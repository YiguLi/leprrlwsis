#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 run 前 N 个 episode 实际用到的 **FFC 风速日**（`FFC_speed_noise_1.csv` 的对应列）画出来。

天号从 TensorBoard 的 `wind_day` 标量读（run 里每 episode 记一次 `env.j`），
所以画的一定是**这次 run 真正用过的那几天**，不是猜的。
同时校验：CSV 那一列是否与 TB 里记的 `wind_speed` 逐位相同（同一天应当相同）。

用法:
    python agentutil/plot_fcc_wind_days.py [run_dir] [--days 5] [--csv FFC_speed_noise_1.csv]
输出:
    output/wind_days/fcc_day<j>.png
"""
import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.environ.get('TEMP', '/tmp'), 'mplcache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader  # noqa: E402

DAY = 1440


def _scalar(v):
    """tf2 SummaryWriter 把标量写在 tensor.float_val 里。"""
    if v.HasField('simple_value'):
        return float(v.simple_value)
    t = v.tensor
    if t.float_val:
        return float(t.float_val[0])
    if t.double_val:
        return float(t.double_val[0])
    if t.int64_val:
        return float(t.int64_val[0])
    return float('nan')


ap = argparse.ArgumentParser()
ap.add_argument('run_dir', nargs='?', default=os.path.join('output', 'AIFFLOR-run87'))
ap.add_argument('--days', type=int, default=5)
ap.add_argument('--csv', default='FFC_speed_noise_1.csv')
ap.add_argument('--out', default=os.path.join('output', 'wind_days'))
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.run_dir, 'events.out.tfevents*')))
if not files:
    raise SystemExit('未找到事件文件: %s' % a.run_dir)

# ---- 从 TB 取 wind_day（每 episode 一个）与前 N 天的 wind_speed 序列 ----
days_used, wind = [], []
need = a.days * DAY
for f in files:
    for e in EventFileLoader(f).Load():
        if not e.HasField('summary'):
            continue
        for v in e.summary.value:
            if v.tag == 'wind_day':
                days_used.append(int(_scalar(v)))
            elif v.tag == 'wind_speed':
                wind.append((int(e.step), _scalar(v)))
        if len(wind) > need + 2 and len(days_used) >= a.days:
            break
    else:
        continue
    break

wind = np.array(sorted(wind), dtype=float)
print('TB: wind_day 前 %d 个 = %s' % (min(len(days_used), a.days), days_used[:a.days]))

# ---- 读 FFC 池，画每一天 ----
pool = pd.read_csv(a.csv)
print('FFC 池: %s, shape=%s（第 0 列=行号，列 1.. 才是"天"）' % (a.csv, pool.shape))

os.makedirs(a.out, exist_ok=True)
run_name = os.path.basename(os.path.normpath(a.run_dir))
made = []
for k in range(min(a.days, len(days_used))):
    j = days_used[k]
    series = pool.iloc[:, j].to_numpy(dtype=float)[:DAY]
    lo, hi = k * DAY, (k + 1) * DAY
    env_u = wind[(wind[:, 0] >= lo) & (wind[:, 0] < hi)][:, 1]
    # TB 里 step=t 的 wind_speed 其实是 u(t+1)（env.step 先更新 u 再记录）⇒ 校验用 env[t] vs series[t+1]
    diff = (np.abs(env_u[:len(series) - 1] - series[1:]).max()
            if len(env_u) > 1 else float('nan'))

    x = np.arange(len(series))
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=140)
    ax.plot(x, series, lw=1.1, color='#2ca02c', label='FFC column %d' % j)
    ax.axhline(series.mean(), color='#d62728', ls='--', lw=1.0,
               label='mean %.2f m/s' % series.mean())
    ax.axhline(np.percentile(series, 95), color='#ff7f0e', ls=':', lw=1.0,
               label='p95 %.2f m/s' % np.percentile(series, 95))
    ax.set_xlim(0, DAY - 1)
    ax.set_xlabel('minute of day')
    ax.set_ylabel('$u$ [m/s]')
    ax.set_title('%s  ep%d  |  FFC_speed_noise_1.csv col %d  |  match vs env log '
                 '(shift 1): max|d| = %.1e' % (run_name, k + 1, j, diff), fontsize=10)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    p = os.path.join(a.out, 'fcc_day%d.png' % j)
    fig.savefig(p)
    plt.close(fig)
    made.append(p)
    print('  episode %d -> FFC col %3d | mean %5.2f | min %5.2f | max %5.2f | p95 %5.2f '
          '| max|d| vs env(shift1) %.1e -> %s'
          % (k + 1, j, series.mean(), series.min(), series.max(),
             np.percentile(series, 95), diff, p))

print('\n输出目录: %s' % os.path.abspath(a.out))
