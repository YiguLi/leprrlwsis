#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""diag_q_climb_cost.py —— 「critic 数字爬升」到底花了多少预算 & 在爬什么？（只读）

拆两大类爬升（`update_stats_H.csv` 单文件即可，不需 MATLAB）：

  A. **尺度建立（虚假爬升）**：critic 从接近 0 的输出爬到收益量级。
     证据 = 训练早期 reward_mean 已经稳定，但 |q_p| 还在涨。
  B. **跟随收益改善（真实爬升）**：策略变好 ⇒ 真值 Q 变大 ⇒ |q_p| 跟着涨。
     证据 = |q_p| 与 reward_mean 同向（分段相关高）。

输出三组量：

  1. 尺度瞬态：|q_p| 达终值 50% / 90% 用掉多少预算（沿用 analyse_q_scale 口径）；
  2. **爬升预算占比**：|q_p| 仍低于终值 90% 的那些块，其 |td_p| 之和 占 全部 |td_p| 之和
     的比例 —— 直接回答"critic 把多少误差预算花在爬升期"；
  3. 相关性：mean(|q_p|) 与 mean(reward_mean) 的分段 Pearson r（==1 说明纯跟随收益）。

用法：
    python agentutil/diag_q_climb_cost.py                     # 扫 output/*/update_stats_H.csv
    python agentutil/diag_q_climb_cost.py output/<run> [run2 ...]
    python agentutil/diag_q_climb_cost.py --bin 10            # 分段数
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass


def _read(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine='python', on_bad_lines='skip')


def stats(path, nbin=10):
    df = _read(path)
    if len(df) < 50 or not {'q_p_mean', 'td_p_mean', 'reward_mean'} <= set(df.columns):
        return None
    q = df['q_p_mean'].abs().to_numpy(dtype=float)
    td = df['td_p_mean'].to_numpy(dtype=float)
    rw = df['reward_mean'].to_numpy(dtype=float)
    n = len(q)

    # ---- 1. 尺度瞬态（与 analyse_q_scale.transient_stats 同口径） ----
    tail = max(50, int(n * 0.05))
    q_final = float(np.nanmean(q[-tail:]))

    def frac(thr):
        hit = q >= thr * q_final
        return float(np.argmax(hit)) / n if hit.any() else float('nan')

    f50, f90 = frac(0.5), frac(0.9)

    # ---- 2. 爬升期吃掉多少 TD 误差预算 ----
    clim = q < 0.9 * q_final                      # 爬升期掩码
    td_sum = float(np.nansum(td))
    share = float(np.nansum(td[clim]) / td_sum) if td_sum > 0 else float('nan')
    td_climb = float(np.nanmean(td[clim])) if clim.any() else float('nan')
    td_plateau = float(np.nanmean(td[~clim])) if (~clim).any() else float('nan')
    # 爬升期强度比 = 爬升期平均 |td_p| / 平台期平均 |td_p|。>1 ⇒ 爬升期更"贵"（在追误差）；
    # <1 ⇒ 爬升期反而是量级更小的阶段（爬升由目标变大驱动，不是拟合变差）。
    ts_ratio = td_climb / td_plateau if td_plateau > 0 else float('nan')

    # ---- 3. 分段相关：|q_p| 是"跟随收益"还是"自己爬" ----
    edges = np.linspace(0, n, nbin + 1).astype(int)
    qs = np.array([np.nanmean(q[edges[i]:edges[i + 1]]) for i in range(nbin)])
    rs = np.array([np.nanmean(rw[edges[i]:edges[i + 1]]) for i in range(nbin)])
    ok = np.isfinite(qs) & np.isfinite(rs)
    r_corr = float(np.corrcoef(qs[ok], rs[ok])[0, 1]) if ok.sum() > 2 else float('nan')

    # 相对 TD 误差三段
    ratio = td / np.maximum(q, 1e-9)
    rel_first = float(np.nanmean(ratio[:n // 10]))
    rel_last = float(np.nanmean(ratio[-n // 10:]))

    return dict(path=path, rows=n, q_final=q_final, f50=f50, f90=f90,
                climb_share=share, td_climb=td_climb, td_plateau=td_plateau,
                ts_ratio=ts_ratio,
                r_corr=r_corr, rel_first=rel_first, rel_last=rel_last,
                rw_first=float(np.nanmean(rw[:n // 10])),
                rw_last=float(np.nanmean(rw[-n // 10:])),
                q_curve=[round(float(v), 1) for v in qs])


def fmt(s):
    name = os.path.dirname(s['path']).replace('output' + os.sep, '')
    return ('{:<18s} 行数={:<6d} |Q_p|终值={:9.1f}\n'
            '   f50={:<5.3f} f90={:<5.3f} 预算: 爬升期td占比={:<5.3f} '
            '(爬升期均值={:8.1f} vs 平台期={:8.1f}, 强度比={:.2f})\n'
            '   reward: 前10%={:8.1f} 末10%={:8.1f} (×{:.2f})  '
            '分段corr(|Q_p|,reward)={:+.3f}\n'
            '   relTD: 前10%={:.3f} 末10%={:.3f}   |Q_p|分段曲线={}'
            .format(name, s['rows'], s['q_final'], s['f50'], s['f90'],
                    s['climb_share'], s['td_climb'], s['td_plateau'], s['ts_ratio'],
                    s['rw_first'], s['rw_last'],
                    s['rw_last'] / max(1e-9, s['rw_first']), s['r_corr'],
                    s['rel_first'], s['rel_last'], s['q_curve']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='*', help='run 目录或 update_stats_H.csv；缺省扫 output/*/')
    ap.add_argument('--bin', type=int, default=10, help='分段点数（默认 10）')
    a = ap.parse_args()

    paths = []
    for p in a.paths:
        if os.path.isdir(p):
            paths.append(os.path.join(p, 'update_stats_H.csv'))
        else:
            paths.append(p)
    if not paths:
        paths = sorted(glob.glob(os.path.join('output', '*', 'update_stats_H.csv')))
    paths = [p for p in paths if os.path.isfile(p)]

    print('=' * 100)
    print('「critic 数字爬升」成本诊断（只读）  列含义见本文件 docstring')
    print('=' * 100)
    out = []
    for p in paths:
        try:
            s = stats(p, nbin=a.bin)
        except Exception as e:                        # 容忍训练中截断的文件
            print('  SKIP {} : {}'.format(p, e))
            continue
        if s is None:
            print('  SKIP {} : 数据不足或缺列'.format(p))
            continue
        out.append(s)
        print(fmt(s))
        print('-' * 100)

    if out:
        print('汇总（中位数，n={}）: f50={:.3f} f90={:.3f} 爬升期td占比={:.3f} '
              '分段corr中位={:+.3f} relTD 前→末={:.3f}→{:.3f}'.format(
                  len(out),
                  float(np.nanmedian([s['f50'] for s in out])),
                  float(np.nanmedian([s['f90'] for s in out])),
                  float(np.nanmedian([s['climb_share'] for s in out])),
                  float(np.nanmedian([s['r_corr'] for s in out])),
                  float(np.nanmedian([s['rel_first'] for s in out])),
                  float(np.nanmedian([s['rel_last'] for s in out]))))
        print('解读: corr→+1 ⇒ |Q_p| 主要在跟随收益改善（真实爬升，不是浪费）；')
        print('      corr≈0 且爬升期td占比大 ⇒ 主要在建立尺度（虚假爬升，可用初始化/监督拟合省掉）。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
