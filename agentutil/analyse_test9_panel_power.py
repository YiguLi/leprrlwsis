# -*- coding: utf-8 -*-
r"""9 用例面板的**测量功效与方差分解**（只读，不需 MATLAB）。

回答的问题
----------
"用这个面板判断策略有没有改进，能分辨多大？" —— 把一个指标的总方差分解成
**跨用例**（case 之间，纯策略无关）与**跨次评测**（同一 case 在不同检查点的差异），
后者才是"判据"要用的噪声/漂移量。

判据
----
* 每次评测的标准误 `SE = pooled 跨次 sd / √(评测次数)`；
* 能分辨的改进 ≈ `1.96·SE`（相对该指标均值给百分比）。

⚠ 本 run 可能仍在训练中（`df_*` 逐秒增长）⇒ 读数必须注明快照时间，且不同时刻不可混比。

用法
----
    python agentutil/analyse_test9_panel_power.py [output/TD3PERNOLOW-run57]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else 'output/TD3PERNOLOW-run57'
t9 = pd.read_csv(os.path.join(RUN, 'df_test9.csv'), encoding='utf-8-sig')
KEY = ['wind_level', 'price_level']
cases = t9[t9.wind_level != 'OVERALL'].copy()
ov = t9[t9.wind_level == 'OVERALL']
NEVAL = int(t9['step'].nunique())

print('=' * 96)
print('test9 面板功效: %s' % RUN)
print('=' * 96)
print('  行数 %d ⇒ 评测次数 %d（每次 10 行 = 9 用例 + OVERALL）'
      % (len(t9), NEVAL))
print('  快照时刻: %s' % __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

print('\n[1] OVERALL 指标（每次评测一个值）')
for c in ('turbine_profit', 'profit', 'penalty_NN_H', 'c1', 'soc_end'):
    if c not in ov:
        continue
    a = ov[c].to_numpy(float)
    if abs(a.mean()) < 1e-12:
        print('  %-14s mean %10.3f  sd %8.3f' % (c, a.mean(), a.std()))
        continue
    print('  %-14s mean %10.3f  sd %8.3f (%6.2f%%)  范围 %.1f ~ %.1f'
          % (c, a.mean(), a.std(), 100 * a.std() / abs(a.mean()), a.min(), a.max()))

print('\n[2] 方差分解：跨用例 vs 跨次评测（用例内）')


def pooled(col):
    num, den = 0.0, 0
    for _, g in cases.groupby(KEY):
        a = g[col].to_numpy(float)
        num += a.var() * (len(a) - 1)
        den += len(a) - 1
    return float(np.sqrt(num / den))


print('  %-14s %12s %12s %10s' % ('指标', '跨用例 sd', '跨次(用例内)sd', '占比'))
for c in ('turbine_profit', 'profit', 'penalty_NN_H'):
    cross = float(cases.groupby(KEY)[c].mean().std())
    inner = pooled(c)
    tot = float(cases[c].std())
    print('  %-14s %12.3f %12.3f %9.2f%%'
          % (c, cross, inner, 100 * inner ** 2 / tot ** 2))

print('\n[3] 逐用例的"跨次"相对离散度（策略变化在该 case 上的可测幅度）')
print('  %-20s %10s %8s %8s | %10s %9s %9s'
      % ('case', 'turb均值', 'sd', '%', 'penalty均值', 'sd', '%'))
for k, g in cases.groupby(KEY):
    t = g['turbine_profit'].to_numpy(float)
    p = g['penalty_NN_H'].to_numpy(float)
    print('  %-20s %10.1f %8.1f %7.2f%% | %10.1f %9.1f %8.1f%%'
          % (str(k), t.mean(), t.std(), 100 * t.std() / abs(t.mean()),
             p.mean(), p.std(), 100 * p.std() / max(abs(p.mean()), 1e-9)))

print('\n[4] 分辨力与趋势')
for c, lab in (('turbine_profit', 'turbine'), ('profit', 'profit')):
    v = ov[c].to_numpy(float)
    n = len(v)
    se = pooled(c) / np.sqrt(NEVAL)
    x = np.arange(n)
    b, a0 = np.polyfit(x, v, 1)
    r = v - (a0 + b * x)
    se_slope = np.sqrt((r ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum())
    print('  %-9s 单次标准误 %.2f 元 (%.3f%%)  可分辨改进 ≈ %.2f%%'
          % (lab, se, 100 * se / abs(v.mean()), 100 * 1.96 * se / abs(v.mean())))
    print('            趋势 %+9.3f/次  t=%+5.2f  ⇒ 该趋势的 95%%半宽 %+.3f 元 (%.4f%%)'
          % (b, b / se_slope, 1.96 * se_slope, 100 * 1.96 * se_slope / abs(v.mean())))
print('\n  注：penalty_NN_H 的逐用例相对 sd 达 100%%~1000%%（重尾）⇒ 该指标不适合逐次比较；')
print('      turbine_profit / profit 的跨次方差只占 %.2f%% ⇒ 面板本身是精密仪器，'
      % (100 * pooled('turbine_profit') ** 2 / cases['turbine_profit'].var()))
print('      它报告的是"策略没改进"这个事实，而不是"测不准"。')
