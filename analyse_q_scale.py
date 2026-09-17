#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""临时分析（用后删除）：量化 critic 的"尺度瞬态"与相对 TD 误差走势。

问: Q 网络对更小的数字是否更好？
答的关键量:
  * |Q_p| 从现在(~0)爬到终值的 50%/90% 用了多少训练步 → 尺度瞬态占总预算的比例
  * td_p/|q_p| 的走势 → 相对拟合误差是否在收敛（还是卡在 ~20%）
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = r'C:\Users\yigul\Desktop\llm enhanced wsis\code\HSRL'
os.chdir(ROOT)

files = sorted(glob.glob(os.path.join('output', '*', 'update_stats_H.csv')))[:4]
print('=' * 104)
for f in files:
    df = pd.read_csv(f)
    if len(df) < 50 or 'q_p_mean' not in df:
        continue
    q = df['q_p_mean'].abs().to_numpy()
    td = df['td_p_mean'].to_numpy()
    st = df['step'].to_numpy()
    qe = q[-max(50, len(q) // 20):].mean()          # 末段均值 = "终值"
    i50 = np.argmax(q >= 0.5 * qe) if (q >= 0.5 * qe).any() else -1
    i90 = np.argmax(q >= 0.9 * qe) if (q >= 0.9 * qe).any() else -1
    n = len(df)
    ratio = td / np.maximum(q, 1e-9)
    r10 = ratio[:n // 10].mean()
    rmid = ratio[n // 2 - n // 20:n // 2 + n // 20].mean()
    rlast = ratio[-n // 10:].mean()
    print('{:<42s} 行数={:<6d} 步 {:.0f}→{:.0f}'.format(f, n, st[0], st[-1]))
    print('   |Q_p| 终值(末5%)={:8.1f} | 达 50% 用的块数={:<6d}({:.0f}% 预算) 达 90%={:<6d}({:.0f}%)'
          .format(qe, i50, 100.0 * i50 / n, i90, 100.0 * i90 / n))
    print('   |Q_p| 曲线(每 10% 一个点): {}'.format(
        [int(v) for v in q[::max(1, n // 10)][:10]]))
    print('   相对 TD 误差 td_p/|q_p|: 前10%={:.3f}  中段={:.3f}  末10%={:.3f}'.format(r10, rmid, rlast))
    print('-' * 104)
