# -*- coding: utf-8 -*-
r"""run57 test9：**用逐用例均值 + 划分（partition）噪声上界**判"评测噪声 vs 真改进"。

思路（对行布局不敏感）
--------------------
`nine_case_test(seed=42)` 的风/价序列**每次调用逐位相同**（已由"逐用例 c1 的
跨划分 sd ≈ 1e-3~1e-2 而 c1 只由 P_F 决定"证实）。因此：
  * 逐用例均值 E[metric | case] 的**全部**跨次变化来自策略（含探索噪声）；
  * 逐用例均值的**抽样噪声**可由"同一文件内的独立划分"直接估上界：
    把文件按行序切成 K 段互不重叠的划分，各自算逐用例均值，看 K 个均值之间的离散度。
  * 若 K 个划分的均值互差 ≪ 训练段之间的差异 ⇒ 观察到的变化是**真实的策略变化**；
    反之 ⇒ 纯噪声。
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
df = df[df.wind_level != 'OVERALL']
K = 8
df['part'] = (np.arange(len(df)) * K // len(df))
keys = ['wind_level', 'price_level']

print('===== 逐用例均值（全文件）=====')
allm = df.groupby(keys)[['turbine_profit', 'profit', 'penalty_NN_H', 'c1']].mean()
print(allm.to_string(float_format=lambda v: '%10.2f' % v))

print('\n===== 划分噪声上界：%d 个行序划分的逐用例均值离散度 =====' % K)
pm = df.groupby(['part'] + keys)[['turbine_profit']].mean().unstack('part')
sd_part = pm.std(axis=1)
print('  turf 的划分间 sd（每划分 %d 行）:' % (len(df) // K))
for k in sd_part.index:
    print('    %-24s 划分均值范围 %9.1f ~ %9.1f   sd %7.2f'
          % (str(k), pm.loc[k].min(), pm.loc[k].max(), sd_part[k]))
print('  ⇒ 若策略在该 case 上稳定，此 sd 只反映"同一策略在不同行段上的采样差异"；')
print('    它的量级就是**本面板对逐用例差异的分辨下限**。')

print('\n===== 训练段之间的逐用例差异（前半 vs 后半，按行序对半）=====')
h = len(df) // 2
A = df.iloc[:h].groupby(keys)[['turbine_profit', 'penalty_NN_H']].mean()
B = df.iloc[h:].groupby(keys)[['turbine_profit', 'penalty_NN_H']].mean()
D = (B - A)
for k in D.index:
    ref = sd_part[k] if isinstance(sd_part, pd.Series) else np.nan
    dt = D.loc[k, 'turbine_profit']
    print('  %-24s Δturb %+9.1f  (划分噪声 sd %6.2f ⇒ 相当于 %+5.1f σ)   Δ罚 %+8.2f'
          % (str(k), dt, ref, dt / ref if ref else float('nan'),
             D.loc[k, 'penalty_NN_H']))

print('\n===== 汇总读数 =====')
tot_A, tot_B = A['turbine_profit'].sum(), B['turbine_profit'].sum()
print('  前半 turbine 合计 %9.1f → 后半 %9.1f  (%+.2f%%)'
      % (tot_A, tot_B, 100 * (tot_B - tot_A) / tot_A))
print('  前半 penalty 合计 %9.1f → 后半 %9.1f' %
      (A['penalty_NN_H'].sum(), B['penalty_NN_H'].sum()))
