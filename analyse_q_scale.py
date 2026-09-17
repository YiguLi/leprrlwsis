#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""critic「Q 尺度瞬态」诊断（可复用函数 + CLI）
==============================================
读 `update_stats_H.csv`（主循环每块写入的 H 层更新统计），量化两件事：

  * **尺度瞬态**：|Q_p| 从初值爬到终值 50% / 90% 用掉了多少训练预算占比
    （Adam 下每参数每步只走 ~lr，与梯度大小无关 ⇒ "把输出搬到正确量级"纯是距离问题）
  * **相对 TD 误差** `td_p/|Q_p|` 的走势（前 10% / 中段 / 末 10%）——判断拟合是否到平台

用法
----
    python analyse_q_scale.py                       # 扫 output/*/update_stats_H.csv
    python analyse_q_scale.py <csv|通配> [<csv> ...]
    from analyse_q_scale import transient_stats     # 供 A/B 脚本复用
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass


def transient_stats(path, tail_frac=0.05, n_curve=10):
    """返回 None（数据太少）或统计字典。"""
    df = pd.read_csv(path)
    if len(df) < 50 or 'q_p_mean' not in df:
        return None
    q = df['q_p_mean'].abs().to_numpy(dtype=float)
    td = df['td_p_mean'].to_numpy(dtype=float)
    st = df['step'].to_numpy(dtype=float)
    n = len(q)
    q_final = float(q[-max(50, int(n * tail_frac)):].mean())

    def frac(thr):
        hit = q >= thr * q_final
        return float(np.argmax(hit)) / n if hit.any() else float('nan')

    ratio = td / np.maximum(q, 1e-9)
    return dict(path=path, rows=n, step0=float(st[0]), step1=float(st[-1]),
                q_final=q_final, f50=frac(0.5), f90=frac(0.9),
                r_first=float(ratio[:n // 10].mean()),
                r_mid=float(ratio[n // 2 - n // 20:n // 2 + n // 20].mean()),
                r_last=float(ratio[-n // 10:].mean()),
                curve=[int(v) for v in q[::max(1, n // n_curve)][:n_curve]])


def fmt(s):
    if s is None:
        return '  (数据不足)'
    return ('{:<44s} 行数={:<6d} 步 {:.0f}→{:.0f}\n'
            '   |Q_p| 终值(末5%)={:9.1f} | 达50%={:<6.3f}预算 达90%={:<6.3f}预算\n'
            '   |Q_p| 曲线(每10%一点)={}\n'
            '   相对TD误差 td_p/|Q_p|: 前10%={:.3f} 中段={:.3f} 末10%={:.3f}'
            .format(s['path'], s['rows'], s['step0'], s['step1'], s['q_final'],
                    s['f50'], s['f90'], s['curve'], s['r_first'], s['r_mid'], s['r_last']))


def main():
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob(os.path.join('output', '*', 'update_stats_H.csv')))
    if not args:
        print('未找到 update_stats_H.csv')
        return
    print('=' * 104)
    for a in args:
        for path in (sorted(glob.glob(a)) if ('*' in a or '?' in a) else [a]):
            print(fmt(transient_stats(path)))
            print('-' * 104)


if __name__ == '__main__':
    main()
