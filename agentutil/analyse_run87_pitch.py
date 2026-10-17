#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run87 诊断：λ1 长期钉顶时，策略为何没学会"降出力（升桨距）压波动"。（只读）

用 `output/AIFFLOR-run87/df_hsrl.csv`（逐步日志，568k 行）回答四个问题：

A. λ1 / d_block_H 的真实轨迹 —— 是否真的"几乎一直钉在 max"；
B. 违约（P_F > 3 MW）随训练怎么变；c1 的水平；
C. 升桨距（= 降出力）到底有没有**控制权**：
   - 违约步 vs 非违约步的桨距对比；
   - 在同一 Δu 桶内，高桨距组 vs 低桨距组的 P_F 对比（把"风本身变快"这个混杂因素按住）；
D. 训练前后桨距动作有没有移动（学习证据）。

用法:
    python agentutil/analyse_run87_pitch.py [run_dir]
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join('output', 'AIFFLOR-run87')
LOG = os.path.join(RUN, 'df_hsrl.csv')

USECOLS = ['step', 'P_F', 'c1', 'lam1', 'd_block_H', 'ema_c1_H', 'pr', 'u', 'SOC',
           'original_action_H', 'g', 'pb_original', 'pbrule', 'r1', 'clip_amount']
CHUNK = 60000


def _arr(s):
    """'[ 0.2  0.2 ... ]' → ndarray。"""
    return np.fromstring(str(s).strip().lstrip('[').rstrip(']'), sep=' ')


rows = []
prev = None            # 上一行 (u, P_F) 用于 Δu
cols_pitch = None
for ch in pd.read_csv(LOG, usecols=USECOLS, chunksize=CHUNK, engine='c'):
    ch = ch.reset_index(drop=True)
    act = np.vstack([_arr(x) for x in ch['original_action_H'].to_numpy()])
    if cols_pitch is None:
        cols_pitch = ['a%d' % i for i in range(act.shape[1])]
    u = ch['u'].to_numpy(dtype=float)
    du = np.empty_like(u)
    du[0] = np.nan
    du[1:] = u[1:] - u[:-1]
    if prev is not None:
        du[0] = u[0] - prev
    prev = u[-1]
    df = pd.DataFrame({
        'step': ch['step'].to_numpy(dtype=np.int64),
        'P_F': ch['P_F'].to_numpy(dtype=float),
        'c1': ch['c1'].to_numpy(dtype=float),
        'lam1': ch['lam1'].to_numpy(dtype=float),
        'd': ch['d_block_H'].to_numpy(dtype=float),
        'ema_c1': ch['ema_c1_H'].to_numpy(dtype=float),
        'pr': ch['pr'].to_numpy(dtype=float),
        'u': u, 'du': du,
        'SOC': ch['SOC'].to_numpy(dtype=float),
        'g': ch['g'].to_numpy(dtype=float),
        'pb': ch['pb_original'].to_numpy(dtype=float),
        'pbrule': ch['pbrule'].to_numpy(dtype=float),
        'r1': ch['r1'].to_numpy(dtype=float),
        'clip': ch['clip_amount'].to_numpy(dtype=float),
    })
    for i, c in enumerate(cols_pitch):
        df[c] = act[:, i]
    df['pitch'] = act[:, :5].mean(axis=1)          # 5 台机平均桨距
    df['viol'] = (df['P_F'] > 3.0 + 1e-4).astype(float)
    rows.append(df)

D = pd.concat(rows, ignore_index=True)
print('=' * 100)
print('run87：%d 步（≈%d 块），step %d → %d' % (len(D), len(D) // 5, D.step.min(), D.step.max()))
print('=' * 100)

# ---------------------------------------------------------------- A. λ1 轨迹
print('\n[A] λ1 / d_block_H / EMA(sum_c1)')
lam = D['lam1'].dropna()
print('  lam1: min %.4f | median %.4f | mean %.4f | max %.4f | 唯一值 %d'
      % (lam.min(), lam.median(), lam.mean(), lam.max(), lam.nunique()))
for thr in (0.99, 0.95):
    print('  占比 lam1 ≥ %.0f%%·max: %.1f%%' % (thr * 100, 100 * (lam >= thr * lam.max()).mean()))
print('  d_block_H: min %.4g | median %.4g | 末值 %.4g' % (D['d'].min(), D['d'].median(), D['d'].iloc[-1]))
ok = D.dropna(subset=['ema_c1'])
print('  EMA(sum_c1) 中位 %.3f | 末值 %.3f  (与 d_block_H 末值 %.4g 比: %s)'
      % (ok['ema_c1'].median(), ok['ema_c1'].iloc[-1] if len(ok) else np.nan,
         D['d'].iloc[-1], 'EMA ≫ d' if len(ok) and ok['ema_c1'].iloc[-1] > D['d'].iloc[-1] else 'EMA ≲ d'))

# ---------------------------------------------------------------- B. 违约随训练
print('\n[B] 违约率 / c1 随训练（每 100k 步一档）')
print('  %8s %8s %9s %9s %8s %8s %8s' % ('step', 'P_F均', 'P_F_p95', '违约率%', 'c1均', 'lam1均', 'pitch均'))
for lo in range(0, int(D.step.max()) + 1, 100000):
    m = (D.step >= lo) & (D.step < lo + 100000)
    if not m.any():
        continue
    s = D[m]
    print('  %8d %8.3f %9.3f %9.3f %8.3f %8.3f %8.4f'
          % (lo, s.P_F.mean(), s.P_F.quantile(.95), 100 * s.viol.mean(), s.c1.mean(),
             s.lam1.mean(), s.pitch.mean()))
print('  全期违约率 %.3f%%（共 %d 步），P_F>3 的步占 %.3f%%'
      % (100 * D.viol.mean(), int(D.viol.sum()), 100 * D.viol.mean()))

# ---------------------------------------------------------------- C. 控制权
print('\n[C-1] 违约步 vs 正常步（同期对比）')
v, n = D[D.viol > 0], D[D.viol == 0]
print('  %10s %9s %9s %9s %9s %9s' % ('组', 'P_F均', 'pitch均', 'u均', '|Δu|均', 'SOC均'))
print('  %10s %9.3f %9.4f %9.3f %9.4f %9.3f' % ('违约', v.P_F.mean(), v.pitch.mean(), v.u.mean(), v.du.abs().mean(), v.SOC.mean()))
print('  %10s %9.3f %9.4f %9.3f %9.4f %9.3f' % ('正常', n.P_F.mean(), n.pitch.mean(), n.u.mean(), n.du.abs().mean(), n.SOC.mean()))
print('  桨距在违约步的分位: p10 %.3f | p50 %.3f | p90 %.3f（域 0~0.5）'
      % (v.pitch.quantile(.1), v.pitch.median(), v.pitch.quantile(.9)))
print('  违约步中 pitch ≥ 0.49（贴满升桨）占 %.1f%%；pitch ≤ 0.21 占 %.1f%%'
      % (100 * (v.pitch >= 0.49).mean(), 100 * (v.pitch <= 0.21).mean()))

print('\n[C-2] 同一 |Δu| 桶内，高桨距组 vs 低桨距组 的 P_F（按住风的变化）')
Dc = D.dropna(subset=['du']).copy()
Dc['addu'] = Dc.du.abs()
Dc['ub'] = pd.cut(Dc.addu, [0, 0.05, 0.1, 0.2, 0.4, 0.8, 20])
Dc['pb_k'] = pd.qcut(Dc.pitch, 3, labels=['低桨距', '中', '高桨距'], duplicates='drop')
tb = Dc.groupby(['ub', 'pb_k'], observed=True).agg(P_F=('P_F', 'mean'), n=('P_F', 'size')).reset_index()
piv = tb.pivot(index='ub', columns='pb_k', values='P_F')
print(piv.round(4).to_string())
print('  （只要"高桨距"列的 P_F 一致地低于"低桨距"，就说明升桨距**确实**能降 P_F；若差异≈0，则无控制权）')

print('\n[C-3] 训练早/晚期，同一风速段的桨距与 P_F')
for lo, hi, tag in ((0, 50000, '早'), (500000, 568290, '晚')):
    s = D[(D.step >= lo) & (D.step < hi)]
    print('  %s期(step %d~%d): pitch均 %.4f | P_F均 %.3f | 违约率 %.3f%% | lam1均 %.3f'
          % (tag, lo, hi, s.pitch.mean(), s.P_F.mean(), 100 * s.viol.mean(), s.lam1.mean()))

print('\n[C-4] 价格/风速分层下的桨距（策略是否在"该降出力"的地方降了）')
D['ub2'] = pd.cut(D.u, [0, 7, 9, 11, 13, 40])
print(D.groupby(['ub2'], observed=True).agg(
    u=('u', 'mean'), P_F=('P_F', 'mean'), pitch=('pitch', 'mean'), viol=('viol', 'mean')).round(4).to_string())

D[['step', 'P_F', 'c1', 'lam1', 'd', 'pitch', 'viol']].to_csv('agentutil/_run87_slim.csv', index=False)
print('\n已落盘：agentutil/_run87_slim.csv（逐 1 步，4 列；供二次分析）')
