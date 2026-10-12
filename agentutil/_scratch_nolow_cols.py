# -*- coding: utf-8 -*-
"""run57 逐步日志的关键列分块统计（不整表载入内存）。"""
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
WANT = sys.argv[2].split(',') if len(sys.argv) > 2 else [
    'step', 'lam1', 'lam2', 'd_block_H', 'ema_c1_H', 'c1', 'c1_step', 'P_F',
    'pr', 'SOC', 'soc_target', 'pb_rule', 'pb_cmd', 'pbrule', 'clip_amount',
    'is_infeasible', 'inevitable_violation', 'unfeasible_gap', 'wind_level',
    'price_level', 'turbine_profit', 'bess_profit', 'revenue', 'degradation',
    'soc_deviation', 'soc_delta', 'c1_fluct', 'penalty_NN_H', 'P_G', 'u',
    'pbphysicsmin', 'pbphysicsmax', 'pbrulemin', 'pbrulemax', 'w_wind', 'w_bess',
    'phase_rest_frac', 'phase_rest_h', 'phase_minute', 'applied_action_L',
]
CH = 200000

head = pd.read_csv(CSV, nrows=0, encoding='utf-8-sig')
cols = [c for c in WANT if c in head.columns]
missing = [c for c in WANT if c not in head.columns]
print('可用列: %s' % cols)
if missing:
    print('缺列(跳过): %s' % missing)

acc = []
for chunk in pd.read_csv(CSV, usecols=cols, chunksize=CH, encoding='utf-8-sig',
                         low_memory=False):
    for c in cols:
        if chunk[c].dtype == object:
            chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
    acc.append(chunk)
df = pd.concat(acc, ignore_index=True)
print('行数 %d  step %d ~ %d' % (len(df), df['step'].min(), df['step'].max()))

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 50)
print('\n===== 描述统计 =====')
print(df.describe(percentiles=[.01, .05, .5, .95, .99]).T
      .to_string(float_format=lambda v: '%12.4f' % v))

print('\n===== 分 12 段均值（每段 %d 步）=====' % (len(df) // 12))
df['_seg'] = pd.qcut(df['step'], 12, duplicates='drop')
print(df.groupby('_seg', observed=True)[cols].mean().to_string(
    float_format=lambda v: '%11.3f' % v))

print('\n===== 关键时间序列（每 2%% 采样）=====')
sub = df.iloc[::max(1, len(df) // 50)]
keep = [c for c in ('step', 'lam1', 'd_block_H', 'ema_c1_H', 'c1', 'P_F',
                    'pb_rule', 'pb_cmd', 'SOC', 'clip_amount', 'penalty_NN_H',
                    'soc_deviation', 'P_G', 'u', 'pr') if c in df.columns]
print(sub[keep].to_string(index=False, float_format=lambda v: '%10.3f' % v))
