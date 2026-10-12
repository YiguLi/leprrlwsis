# -*- coding: utf-8 -*-
"""9 用例面板的**测量功效**：run57 的 test9 逐轮趋势能分辨多大的改进？"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

df = pd.read_csv('output/TD3PERNOLOW-run57/df_test9.csv', encoding='utf-8-sig')
rows = []
for i in range(len(df) // 45):
    s = df.iloc[i * 45:(i + 1) * 45]
    rows.append((s[s.wind_level == 'HIGH'].profit.mean(),
                 s[s.wind_level == 'MEDIUM'].profit.mean(),
                 s[s.wind_level == 'LOW'].profit.mean(),
                 s.turbine_profit.sum(), s.profit.sum()))
R = pd.DataFrame(rows, columns=['HIGH', 'MEDIUM', 'LOW', 'turb', 'total'])
print('逐轮 sd:')
for c in R.columns:
    print('  %-7s mean %9.2f  sd %8.2f  (=%.1f%%)'
          % (c, R[c].mean(), R[c].std(), 100 * R[c].std() / abs(R[c].mean())))
print()
for c in ('HIGH', 'turb', 'total'):
    x = np.arange(len(R))
    b, a = np.polyfit(x, R[c], 1)
    r = R[c] - (a + b * x)
    se = np.sqrt((r ** 2).sum() / (len(R) - 2) / ((x - x.mean()) ** 2).sum())
    print('  %-7s 趋势 %+8.2f/轮  t=%+5.2f  ⇒ 趋势的 95%%半宽 = %+.1f (%.2f%%/轮)'
          % (c, b, b / se, 1.96 * se, 100 * 1.96 * se / abs(R[c].mean())))
print()
sd = R['turb'].std()
for eff in (0.5, 2.0, 5.0):
    delta = eff / 100 * R['turb'].mean()
    n = (1.96 * sd / delta) ** 2
    print('  想检出"每轮 +%.1f%%（%+.0f 元）"⇒ 需要 n ≈ %.0f 轮 = %.0f 环境步'
          % (eff, delta, n, n * 480))
print('  run57 实际可用: 31 轮 = 14880 步 ⇒ 只能检出 ≥ %.2f%%/轮'
      % (100 * 1.96 * sd / np.sqrt(31) / R['turb'].mean()))
print()
print('  ⇒ 面板本身每一步的风序列是固定的（seed=42），逐轮波动来自**策略抖动**，')
print('    不是风况 ⇒ 用同一批 45 步做逐检查点配对比较可消掉它；')
print('    但作为"改进幅度"的度量，±%.1f%% 的分辨力不足以看清 1~3%% 的杠杆。'
      % (100 * 1.96 * sd / R['turb'].mean()))
