# -*- coding: utf-8 -*-
"""run57：H 层动作的**边际收益**（在线数据回归）—— 判断策略是否停在次优平台。

方法（不启 MATLAB，只用训练日志）
--------------------------------
`turbine_profit = w_wind·pr·ptur/60`，其中 `ptur = P_F(u, a)` 是被控对象的确定性函数。
策略冻结后（取末 25% 训练段，`[D]` 已证 rms(k) 标度律 α≈0.04），日志里的 (u, a, pr)
张开了工作点邻域 ⇒ 可以**同时控制 u 与 pr**、把 turbine_profit 回归到 5 维动作上，
得到各维的**条件边际收益** ∂(收益)/∂a_j（元/块/单位动作）。

判据
----
* 若某个 dim 的边际收益 **显著 > 0**（在策略当前落点上），而策略**不动** ⇒ RL 停在了
  未最优化的平台（∂V/∂a ≠ 0 却没有抬升动作）。
* 若边际收益 ≈ 0 ⇒ 该维已在最优（或该维对这一状态族无杠杆）。
* 逐风速分箱看符号：物理上低风速处限载的**机会成本趋于 0**（P ∝ u³），故低风速段
  边际收益应趋 0、甚至为负（把风留给下游）⇒ 若实测**相反**，说明策略的风速响应是反的。

用法：`python agentutil/_scratch_nolow_marginal.py [csv]`
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

CSV = sys.argv[1] if len(sys.argv) > 1 else 'output/TD3PERNOLOW-run57/df_td3pernolow.csv'

acc = []
for ch in pd.read_csv(CSV, usecols=['step', 'applied_action_H', 'u', 'pr', 'SOC',
                                    'turbine_profit', 'penalty_NN_H'],
                      chunksize=200000, encoding='utf-8-sig', low_memory=False):
    acc.append(ch)
df = pd.concat(acc, ignore_index=True)
A = np.array([[float(x) for x in t.strip().strip('[]').split()]
              for t in df['applied_action_H'].astype(str)])
for j in range(5):
    df['a%d' % (j + 1)] = A[:, j]
smax = df['step'].max()
df = df[df['step'] > 0.75 * smax].reset_index(drop=True)
print('末 25%% 训练段: %d 行, step %d~%d' % (len(df), df['step'].min(), df['step'].max()))

AC = ['a1', 'a2', 'a3', 'a4', 'a5']


def ols(X, y, names):
    XtX = X.T @ X
    beta = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    s2 = (resid ** 2).sum() / (n - k)
    cov = s2 * np.linalg.pinv(XtX)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, se, r2


def design(sub, extra=()):
    u = sub['u'].to_numpy(float)
    cols = [np.ones(len(sub)), u, u ** 2, u ** 3,
            sub['pr'].to_numpy(float) / 1000.0,
            sub['SOC'].to_numpy(float)]
    names = ['const', 'u', 'u^2', 'u^3', 'pr/1000', 'SOC']
    for c in AC:
        cols.append(sub[c].to_numpy(float))
        names.append(c)
    for c in extra:
        cols.append(np.asarray(c[1], dtype=float))
        names.append(c[0])
    return np.column_stack(cols), names


print('\n===== 全局 OLS（控 u 三次多项式 / pr / SOC）=====')
X, names = design(df)
y = df['turbine_profit'].to_numpy(float)
beta, se, r2 = ols(X, y, names)
print('  R² = %.4f   n = %d' % (r2, len(y)))
for i, nm in enumerate(names):
    if nm in AC or nm in ('u', 'u^2', 'u^3'):
        print('    %-8s β = %+10.4f  se = %7.4f  t = %+7.2f'
              % (nm, beta[i], se[i], beta[i] / se[i]))

print('\n===== 逐风速分箱（控 u^2/u^3 与 pr/SOC）=====')
bins = [(0, 7), (7, 9), (9, 11), (11, 13), (13, 16), (16, 30)]
print('  %-12s %6s %10s %10s %10s %10s %10s   %s'
      % ('u 区间', 'n', 'a1', 'a2', 'a3', 'a4', 'a5', 'R²'))
for lo, hi in bins:
    sub = df[(df['u'] >= lo) & (df['u'] < hi)]
    if len(sub) < 200:
        continue
    cols = [np.ones(len(sub)), sub['u'].to_numpy(float) - (lo + hi) / 2,
            sub['pr'].to_numpy(float) / 1000.0, sub['SOC'].to_numpy(float)]
    nm = ['const', 'du', 'pr/1000', 'SOC']
    for c in AC:
        cols.append(sub[c].to_numpy(float))
        nm.append(c)
    Xb = np.column_stack(cols)
    yb = sub['turbine_profit'].to_numpy(float)
    bb, ss, r2b = ols(Xb, yb, nm)
    print('  %-12s %6d %10.3f %10.3f %10.3f %10.3f %10.3f   %.3f'
          % ('[%d,%d)' % (lo, hi), len(sub),
             *[bb[nm.index(c)] for c in AC], r2b))

print('\n===== 边际收益的 t 值（逐风速分箱）=====')
for lo, hi in bins:
    sub = df[(df['u'] >= lo) & (df['u'] < hi)]
    if len(sub) < 200:
        continue
    cols = [np.ones(len(sub)), sub['u'].to_numpy(float) - (lo + hi) / 2,
            sub['pr'].to_numpy(float) / 1000.0, sub['SOC'].to_numpy(float)]
    nm = ['const', 'du', 'pr/1000', 'SOC']
    for c in AC:
        cols.append(sub[c].to_numpy(float))
        nm.append(c)
    bb, ss, _ = ols(np.column_stack(cols), sub['turbine_profit'].to_numpy(float), nm)
    print('  u∈[%2d,%2d)  %s' % (lo, hi,
          '  '.join('%s t=%+5.2f' % (c, bb[nm.index(c)] / ss[nm.index(c)])
                    for c in AC)))
print('\n  解读: |t|>3 且 β>0 ⇒ 该维"抬升动作还能多赚"而策略没动 ⇒ 未最优化的方向；')
print('        β<0 ⇒ 该维在当前落点上"已过度抬升"（应压低）。')
print('  物理预期: P ∝ u³ ⇒ 低风速段 ∂P/∂a 极小 ⇒ 限载机会成本小；'
      '若某维在低风速段 β≈0 而策略反而不动，则无损失。')
