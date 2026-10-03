# -*- coding: utf-8 -*-
"""跨 run 健康度汇总：判断某个 run 的问题是独有还是系统性。

对每个 `output/*-runN/df_*.csv` 抽关键健康指标：
  · 对偶残差 `ema_c1_H/d_block_H − 1`（H）、`ema_c2_L/d_block_L − 1`（L）
    残差 > 0 = 违反预算；对偶长期贴着**上界**且残差 > 0 ⇒ 压不住
  · gap_pen（块末 SOC 可行性罚）分位
  · 动作贴界比例
  · profit 趋势（后半/前半）

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\summarise_runs_health.py
    ... --pattern "TD3PER-run4*"
"""
import argparse
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def num(df, c):
    return pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float) \
        if c in df.columns else None


def half_trend(x):
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float('nan')
    h = x.size // 2
    a = x[:h].mean()
    return (x[h:].mean() - a) / abs(a) if abs(a) > 1e-12 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='output')
    ap.add_argument('--pattern', default='*-run*')
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.base, args.pattern))):
        if not os.path.isdir(d):
            continue
        f = glob.glob(os.path.join(d, 'df_*.csv'))
        f = [p for p in f if 'episode' not in os.path.basename(p)
             and 'test9' not in os.path.basename(p)]
        if not f:
            continue
        df = pd.read_csv(f[0], low_memory=False)
        r = {'run': os.path.basename(d), 'n': len(df)}
        e1, d1 = num(df, 'ema_c1_H'), num(df, 'd_block_H')
        e2, d2 = num(df, 'ema_c2_L'), num(df, 'd_block_L')
        l1, l2 = num(df, 'lam1'), num(df, 'lam2')
        g = num(df, 'gap_pen')
        if e1 is not None and d1 is not None:
            v = (e1 / d1 - 1)
            r['H_resid'] = float(np.nanmean(v[-len(v) // 5:]))
        if e2 is not None and d2 is not None:
            v = (e2 / d2 - 1)
            r['L_resid'] = float(np.nanmean(v[-len(v) // 5:]))
        for k, v in (('lam1', l1), ('lam2', l2)):
            if v is not None and np.isfinite(v).any():
                r[k + '_end'] = float(v[np.isfinite(v)][-1])
        if g is not None:
            r['gap_p99'] = float(np.nanpercentile(g, 99)) if np.isfinite(g).any() else np.nan
            r['gap_mean'] = float(np.nanmean(g)) if np.isfinite(g).any() else np.nan
        if 'r1' in df.columns:
            r['r1_trend'] = half_trend(num(df, 'r1'))
        # 动作贴界
        ac = [c for c in df.columns if c.startswith('action/a_H_clipped_')]
        if ac:
            A = np.vstack([pd.to_numeric(df[c], errors='coerce').to_numpy(float)
                           for c in sorted(ac)]).T
            A = A[np.isfinite(A).all(axis=1)]
            if A.size:
                r['sat'] = float(((A <= 1e-6) | (A >= 0.5 - 1e-6)).mean())
        rows.append(r)

    if not rows:
        print('未找到匹配的 run')
        return 0
    keys = ['H_resid', 'L_resid', 'lam1_end', 'lam2_end', 'gap_p99', 'gap_mean',
            'r1_trend', 'sat']
    print('=' * 118)
    print('跨 run 健康度汇总（%-d 个 run）' % len(rows))
    print('=' * 118)
    hdr = ['%-16s' % 'run', '%7s' % 'n'] + ['%10s' % k for k in keys]
    print('  ' + ' '.join(hdr))
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, float('nan'))
            cells.append('%10.4f' % v if isinstance(v, float) else '%10s' % v)
        print('  %-16s %7d %s' % (r['run'], r['n'], ' '.join(cells)))

    print('\n' + '=' * 118)
    print('列含义')
    print('=' * 118)
    print('  H_resid  = ema_c1_H/d_block_H − 1 末段均值  (>0 违反预算; 对偶贴上界且>0 ⇒ 压不住)')
    print('  L_resid  = ema_c2_L/d_block_L − 1 末段均值  (同上)')
    print('  lam1_end / lam2_end = 末值 (对照 LAM_MIN=0.0967 / 上限)')
    print('  gap_p99 / gap_mean  = 块末 SOC 可行性罚的 p99 / 均值')
    print('  r1_trend = 步奖励后半/前半 − 1  (应 > 0)')
    print('  sat      = H 层动作贴 0 或 0.5 边界的比例 (应小)')
    print()
    # 统计
    for k in ('L_resid', 'lam2_end', 'H_resid', 'gap_p99', 'r1_trend', 'sat'):
        v = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            print('  %-9s 跨 run: 中位 %9.4f   范围 %.4f ~ %.4f   (>0 的占比 %.0f%%)'
                  % (k, np.median(v), v.min(), v.max(), 100 * (v > 0).mean()))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
