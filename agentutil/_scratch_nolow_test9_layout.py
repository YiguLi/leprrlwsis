# -*- coding: utf-8 -*-
r"""run57 的 test9 正确的"评测噪声 vs 策略改进"统计（行布局已核实）。

行布局（实测，勿再猜）
----------------------
`nine_case_test` 每步追加 **10 行**（9 个 (wind,price) 用例 + 1 行 OVERALL），
场景 45 步 ⇒ **一次评测 = 450 行**，但 csv 里同一 `step` 值只出现 10 行 ⇒
同一 `step` 值的 10 行是**一个场景步**（不是一次评测）。
⇒ 一次评测块 = 连续 450 行 = 45 个不同的 step 值 × 10 行。
⚠ 早先把 45 行当一次评测是错的（那 45 行只是 4.5 个场景步）。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CSV = sys.argv[1] if len(sys.argv) > 1 else 'output/TD3PERNOLOW-run57/df_test9.csv'
df = pd.read_csv(CSV, encoding='utf-8-sig')
print('总行数 %d' % len(df))
# 每个 step 值 = 一个场景步（10 行）；连续 45 个 step 值 = 一次评测
steps_uniq = df['step'].drop_duplicates().to_numpy()
print('step 唯一值 %d 个 ⇒ 场景步数 %d ⇒ 评测次数 %d'
      % (len(steps_uniq), len(steps_uniq), len(steps_uniq) // 45))
df['eval'] = np.searchsorted(steps_uniq, df['step'].to_numpy()) // 45
ne = int(df['eval'].max()) + 1

cases = df[df.wind_level != 'OVERALL']
print('\n===== [A] 确定性检查：逐用例 c1 的跨评测 sd（c1 只由 P_F 决定）=====')
piv = cases.pivot_table(index='eval', columns=['wind_level', 'price_level'], values='c1')
for k in piv.columns:
    print('    %-18s 均值 %9.4f  跨评测 sd %.3e' % (str(k), piv[k].mean(), piv[k].std()))
print('  ⇒ sd ≈ 0 ⇒ **每次评测的风/价序列逐位相同**（build_9case_scenario(seed=42) 生效）')

print('\n===== [B] 逐评测汇总（%d 次）=====' % ne)
rows = []
for e, g in df.groupby('eval'):
    c = g[g.wind_level != 'OVERALL']
    o = g[g.wind_level == 'OVERALL']
    rows.append(dict(eval=e, step=int(o['step'].iloc[0]),
                     turb=c['turbine_profit'].sum(), profit=c['profit'].sum(),
                     econ=c['profit_econ'].sum(), pen=c['penalty_NN_H'].sum(),
                     penalty_per_45=o['penalty_NN_H'].iloc[0]))
R = pd.DataFrame(rows)
print(R.to_string(index=False, float_format=lambda v: '%11.2f' % v))

print('\n===== [C] 噪声 / 分辨力 =====')
for c in ('turb', 'profit', 'pen'):
    v = R[c]
    print('  %-7s 均值 %11.2f  sd %8.2f (%.2f%%)  min %.0f max %.0f'
          % (c, v.mean(), v.std(), 100 * v.std() / abs(v.mean()), v.min(), v.max()))
sd = R['turb'].std()
for eff in (1.0, 2.0, 5.0):
    n = (1.96 * sd / (eff / 100 * R['turb'].mean())) ** 2
    print('    检出"每次评测 %+.0f%%"需 n ≈ %.0f 次评测 = %.0f 环境步'
          % (eff, n, n * 480))
print('  run57 有 %d 次评测 ⇒ 本面板分辨力 ≈ ±%.2f%%（1.96·sd/√n / 均值）'
      % (ne, 100 * 1.96 * sd / np.sqrt(ne) / R['turb'].mean()))

print('\n===== [D] 趋势：前半 → 后半（真实改进的唯一可信读数）=====')
n = len(R)
h = n // 2
for c in ('turb', 'profit', 'pen'):
    x = np.arange(n)
    beta, a0 = np.polyfit(x, R[c], 1)
    res = R[c] - (a0 + beta * x)
    se = np.sqrt((res ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum())
    print('  %-7s 前半 %11.2f → 后半 %11.2f (%+.2f%%)  斜率 %+8.2f/评测  t=%+5.2f'
          % (c, R[c].iloc[:h].mean(), R[c].iloc[h:].mean(),
             100 * (R[c].iloc[h:].mean() - R[c].iloc[:h].mean()) / abs(R[c].iloc[:h].mean()),
             beta, beta / se))
print('\n  逐 15 次评测的均值：')
R['grp'] = R['eval'] // 15
print(R.groupby('grp')[['turb', 'profit', 'pen']].mean().to_string(
    float_format=lambda v: '%11.2f' % v))
