# -*- coding: utf-8 -*-
"""run57：H 层动作的逐维越界归因 + 动作随风速/时间的结构（分块读，不整表载入）。"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CSV = sys.argv[1] if len(sys.argv) > 1 else 'output/TD3PERNOLOW-run57/df_td3pernolow.csv'
COEF = 400.0

acc = []
for ch in pd.read_csv(CSV, usecols=['step', 'original_action_H', 'applied_action_H',
                                    'u', 'pr', 'turbine_profit', 'penalty_NN_H',
                                    'wind_level', 'price_level', 'P_F'],
                      chunksize=200000, encoding='utf-8-sig', low_memory=False):
    acc.append(ch)
df = pd.concat(acc, ignore_index=True)
print('行数 %d  step %d~%d' % (len(df), df['step'].min(), df['step'].max()))


def parse(s):
    return np.array([[float(x) for x in t.strip().strip('[]').split()]
                     for t in s.astype(str)])


raw = parse(df['original_action_H'])
app = parse(df['applied_action_H'])
print('raw shape %s  applied shape %s' % (raw.shape, app.shape))

viol_lo = np.clip(-raw[:, :5], 0, None)      # 越下界量 (raw<0)
viol_hi = np.clip(raw[:, :5] - 0.5, 0, None)  # 越上界量 (raw>0.5)
tot = viol_lo + viol_hi
pred = COEF * tot.sum(axis=1)
print('\n---- 逐维越界归因 (原始动作, 未 clip) ----')
print('  %-4s %10s %10s %10s %12s %12s' % ('dim', 'mean(a)', 'sd(a)',
                                           '越界率%', '平均越下', '平均越上'))
for j in range(5):
    print('  a%-3d %10.4f %10.4f %10.2f %12.5f %12.5f'
          % (j + 1, raw[:, j].mean(), raw[:, j].std(),
             100 * ((viol_lo[:, j] > 0) | (viol_hi[:, j] > 0)).mean(),
             viol_lo[:, j].mean(), viol_hi[:, j].mean()))
print('\n  罚列核对: 日志 penalty_NN_H 均值 %.4f vs 由动作复算 %.4f（差 %.3g）'
      % (df['penalty_NN_H'].mean(), pred.mean(),
         abs(df['penalty_NN_H'].mean() - pred.mean())))
contrib = COEF * tot.sum(axis=0)
print('  逐维罚贡献(元/块): %s   合计 %.3f'
      % (np.round(contrib, 3), contrib.sum()))
print('  ⇒ 罚的主力是 %s'
      % ', '.join('a%d(%.1f%%)' % (j + 1, 100 * contrib[j] / contrib.sum())
                  for j in np.argsort(-contrib)[:3]))

# clip 前后差（= 被裁掉的量）
print('\n  clip 位移 |raw−applied| 均值 %.5f  最大 %.4f'
      % (np.abs(raw - app).mean(), np.abs(raw - app).max()))

print('\n---- 动作随风速档的结构 (applied, 后 75% 训练段) ----')
lat = df[df['step'] > df['step'].max() * 0.25].copy()
for j in range(5):
    lat['a%d' % (j + 1)] = app[lat.index, j]
print(lat.groupby('wind_level', observed=True)[['a1', 'a2', 'a3', 'a4', 'a5',
                                                'u', 'turbine_profit']]
      .mean().to_string(float_format=lambda v: '%9.4f' % v))
print('\n  （对照: warmup 先验带 [0.05,0.45]，单机 Cp 最优 1/3；'
      '风场均匀族峰 0.20，生产 readme 记 a=0.20 平台 [0.15,0.30]）')
print('  全体后 75%% 的 5 维均值: %s'
      % np.round([lat['a%d' % (j + 1)].mean() for j in range(5)], 4))

print('\n---- 动作随时间的漂移 (每 12.5% 训练段) ----')
df['_seg'] = pd.qcut(df['step'], 8, duplicates='drop')
for j in range(5):
    df['a%d' % (j + 1)] = app[:, j]
print(df.groupby('_seg', observed=True)[['a1', 'a2', 'a3', 'a4', 'a5',
                                         'turbine_profit', 'penalty_NN_H']]
      .mean().to_string(float_format=lambda v: '%10.4f' % v))
