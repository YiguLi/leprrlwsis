#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run87 诊断 2：升桨距（降出力）**到底能不能**降违约率 —— 两维分层（只读）。

C-2 已看到"同一 |Δu| 桶内高桨距 P_F 更低"，但高桨距与**高风速**天然相关（u 越大 |Δu| 越大）。
本脚本同时按住 (u 桶 × |Δu| 桶)，再看 (a) 平均 P_F、(b) **违约率 P(P_F>3)** 在桨距三分位上的差异。

用法: python agentutil/analyse_run87_causality.py [run_dir]
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join('output', 'AIFFLOR-run87')
LOG = os.path.join(RUN, 'df_hsrl.csv')
USECOLS = ['step', 'P_F', 'c1', 'u', 'pr', 'SOC', 'original_action_H', 'pbrule',
           'pb_original', 'g', 'lam1', 'd_block_H']


def _arr(s):
    return np.fromstring(str(s).strip().lstrip('[').rstrip(']'), sep=' ')


rows, prev_u = [], None
for ch in pd.read_csv(LOG, usecols=USECOLS, chunksize=60000):
    ch = ch.reset_index(drop=True)
    act = np.vstack([_arr(x) for x in ch['original_action_H'].to_numpy()])
    u = ch['u'].to_numpy(float)
    du = np.empty_like(u)
    du[0] = u[0] - prev_u if prev_u is not None else np.nan
    du[1:] = u[1:] - u[:-1]
    prev_u = u[-1]
    rows.append(pd.DataFrame({
        'step': ch['step'].to_numpy(np.int64), 'P_F': ch['P_F'].to_numpy(float),
        'c1': ch['c1'].to_numpy(float), 'u': u, 'du': du, 'pr': ch['pr'].to_numpy(float),
        'SOC': ch['SOC'].to_numpy(float), 'pbrule': ch['pbrule'].to_numpy(float),
        'pb': ch['pb_original'].to_numpy(float), 'g': ch['g'].to_numpy(float),
        'lam1': ch['lam1'].to_numpy(float), 'd': ch['d_block_H'].to_numpy(float),
        'pitch': act[:, :5].mean(axis=1), 'pitch_max': act[:, :5].max(axis=1),
        'pitch_min': act[:, :5].min(axis=1),
    }))

D = pd.concat(rows, ignore_index=True)
D['viol'] = (D.P_F > 3.0 + 1e-4).astype(float)
D['addu'] = D.du.abs()
D = D.dropna(subset=['addu'])
print('样本 %d 步' % len(D))

# 桨距三分位（全局）
q1, q2 = D.pitch.quantile([1 / 3, 2 / 3])
D['pk'] = pd.cut(D.pitch, [-1, q1, q2, 1], labels=['低桨距', '中', '高桨距'])
print('桨距三分位阈值: q33 %.3f | q67 %.3f（全期均值 %.3f）' % (q1, q2, D.pitch.mean()))

D['ub'] = pd.cut(D.u, [0, 9, 11, 12.5, 14, 40])
D['wb'] = pd.cut(D.addu, [0, 0.1, 0.3, 0.7, 30])

print('\n[1] 违约率 P(P_F>3) —— u 桶 × |Δu| 桶 × 桨距三分位')
t = D.groupby(['ub', 'wb', 'pk'], observed=True).agg(v=('viol', 'mean'), n=('viol', 'size'))
piv = t['v'].unstack('pk').round(4)
piv['低-高'] = (piv['低桨距'] - piv['高桨距']).round(4)
print(piv.to_string())
print('\n  （"低-高" = 低桨距组违约率 − 高桨距组违约率，>0 表示升桨距确实降违约）')

print('\n[2] 平均 P_F —— 同上分层')
t2 = D.groupby(['ub', 'wb', 'pk'], observed=True).agg(v=('P_F', 'mean'))
p2 = t2['v'].unstack('pk').round(3)
p2['高/低'] = (p2['高桨距'] / p2['低桨距']).round(3)
print(p2.to_string())

print('\n[3] 违约质量分布：违约步里有百分之几落在哪些 u / |Δu| 区间')
V = D[D.viol > 0]
print('  违约步 %d（占 %.2f%%）' % (len(V), 100 * len(V) / len(D)))
print('  u 分位: p10 %.2f | p50 %.2f | p90 %.2f ; |Δu| p50 %.2f p90 %.2f'
      % (V.u.quantile(.1), V.u.median(), V.u.quantile(.9),
         V.addu.median(), V.addu.quantile(.9)))
print('  违约步中 u>12.5 占 %.1f%%；|Δu|>0.7 占 %.1f%%' % (100 * (V.u > 12.5).mean(), 100 * (V.addu > 0.7).mean()))

print('\n[4] 策略的桨距 vs 风速（是否在"最该降出力"的高风速段反而贴低桨距）')
print(D.groupby('ub', observed=True).agg(u=('u', 'mean'), pitch=('pitch', 'mean'),
                                         pmin=('pitch_min', 'mean'), pmax=('pitch_max', 'mean'),
                                         viol=('viol', 'mean')).round(4).to_string())

print('\n[5] BESS 侧贡献：违约步的 pbrule 变化幅度 vs 正常步')
D['d_pbrule'] = D.pbrule.diff().abs()
D['d_pb'] = D.pb.diff().abs()
print(D.groupby('viol').agg(P_F=('P_F', 'mean'), d_pbrule=('d_pbrule', 'mean'),
                            d_pb=('d_pb', 'mean'), pitch=('pitch', 'mean')).round(4).to_string())
print('  （若违约步的 |Δpbrule| 远大于正常步，说明 P_F 的主因是**下层 BESS 指令跳变**，H 的桨距只能间接管）')
