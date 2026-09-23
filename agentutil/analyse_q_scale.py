#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""critic「Q 尺度瞬态」诊断（可复用函数 + CLI）
==============================================
读 `update_stats_H.csv`（主循环每块写入的 H 层更新统计），量化两件事：

  * **尺度瞬态**：|Q_p| 从初值爬到终值 50% / 90% 用掉了多少训练进度占比（f50 / f90）。
    ⚠ 这两个数只说明"|Q| 是不是从低处起步、花了多久到量级"：**不度量学到的内容，
    也不度量预算的产出**；把它读成"浪费了这么多训练预算"是超出该指标的推断。
    ⚠ 早先被引用的"42~46% / 63~69%"测于 `--value_scale` 之前的 s=1 配置
    （2026-09-17 `df0561a`），不是当前配置的实测值。机制与实测更正见
    `hsrl_agent_init.py` 模块 docstring 的"机制更正"一节。
    注：`q_p_mean` 与 `td_p_mean` 都已还原成**原始元**口径，两者相除是有意义的相对误差。
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

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块 (hsrl_*, myfunc, util, td3*/...)
sys.path.insert(0, _HERE)                              # agentutil 内互相 import

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
