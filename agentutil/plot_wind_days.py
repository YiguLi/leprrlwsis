#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 TensorBoard 事件文件里取 `wind_speed`，把**前 5 个风速日**各画一张图（每天分开）。

每天 = 1440 步（1 episode = 24 h × 60 min，见 main 的 `args.max_episode_length = 1440`）。

用法:
    python agentutil/plot_wind_days.py [run_dir] [--days 5]
输出:
    <run_dir>/wind_day1.png ... wind_day5.png（默认与事件文件同目录，即 output/…）
若传 `--out <dir>` 则写到指定目录。
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

# matplotlib 的字体缓存目录可能不可写 ⇒ 指到临时目录（只影响缓存，不影响图）
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.environ.get('TEMP', '/tmp'), 'mplcache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader  # noqa: E402


def _scalar(v):
    """tf2 的 SummaryWriter 把标量写成 `tensor`（DT_FLOAT/float_val），不是 simple_value。"""
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

DAY = 1440

ap = argparse.ArgumentParser()
ap.add_argument('run_dir', nargs='?', default=os.path.join('output', 'AIFFLOR-run87'))
ap.add_argument('--days', type=int, default=5)
ap.add_argument('--tag', default='wind_speed')
ap.add_argument('--out', default=None)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.run_dir, 'events.out.tfevents*')))
if not files:
    raise SystemExit('未找到事件文件: %s' % a.run_dir)
out_dir = a.out or os.path.join('output', 'wind_days')
os.makedirs(out_dir, exist_ok=True)

# ---- 流式读取，只留目标 tag，凑够 days*1440 步就停 ----
steps, vals = [], []
limit = a.days * DAY + 1
for f in files:
    for e in EventFileLoader(f).Load():
        if not e.HasField('summary'):
            continue
        for v in e.summary.value:
            if v.tag == a.tag:
                steps.append(int(e.step))
                vals.append(_scalar(v))
        if steps and steps[-1] > limit:
            break
    if steps and steps[-1] > limit:
        break

steps = np.asarray(steps)
vals = np.asarray(vals)
order = np.argsort(steps, kind='stable')
steps, vals = steps[order], vals[order]
print('共取到 %d 个 `%s` 点，step ∈ [%d, %d]' % (len(steps), a.tag, steps[0], steps[-1]))

run_name = os.path.basename(os.path.normpath(a.run_dir))
made = []
for d in range(a.days):
    lo, hi = d * DAY, (d + 1) * DAY
    m = (steps >= lo) & (steps < hi)
    if not m.any():
        print('  day%d: 无数据，跳过' % (d + 1))
        continue
    x = steps[m] - lo
    y = vals[m]
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=140)
    ax.plot(x, y, lw=1.1, color='#1f77b4', label='wind speed $u$')
    ax.axhline(y.mean(), color='#d62728', ls='--', lw=1.0,
               label='mean %.2f m/s' % y.mean())
    ax.axhline(np.percentile(y, 95), color='#ff7f0e', ls=':', lw=1.0,
               label='p95 %.2f m/s' % np.percentile(y, 95))
    ax.set_xlim(0, min(DAY - 1, x.max()))
    ax.set_xlabel('minute of day (step within episode)')
    ax.set_ylabel('$u$ [m/s]')
    ax.set_title('%s — day %d (steps %d–%d, n=%d)' % (run_name, d + 1, lo, hi - 1, int(m.sum())),
                 fontsize=10)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    p = os.path.join(out_dir, 'wind_day%d.png' % (d + 1))
    fig.savefig(p)
    plt.close(fig)
    made.append(p)
    print('  day%d: n=%4d | 均值 %5.2f | min %5.2f | max %5.2f | p95 %5.2f  →  %s'
          % (d + 1, int(m.sum()), y.mean(), y.min(), y.max(), np.percentile(y, 95), p))

print('\n输出目录: %s' % os.path.abspath(out_dir))
