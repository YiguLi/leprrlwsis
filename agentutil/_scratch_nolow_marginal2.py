# -*- coding: utf-8 -*-
"""run57：用**同风速严格配对**检验"动作→风机收益"的斜率（消掉 u 的一切泄漏）。

做法：把 u 取 3 位小数分箱（同一 u 内 u 完全相同），箱内做
  ① 一元 OLS  turbine_profit ~ a_j          ⇒ 原始斜率（含箱内其它维的混淆）
  ② 多元 OLS  turbine_profit ~ a_1..a_5     ⇒ 条件斜率（控制其余四维）
并给出箱内 n 与动作的箱内 sd（若箱内 a_j 几乎不动 ⇒ 该维不可识别，报 nan）。

对照：物理上限 ∂(turbine_profit)/∂a ≈ pr·dP/da/60；单机 P=κ·4a(1−a)²·u³、κ=0.00065
⇒ dP/da = κ·4(1−a)(1−3a)·u³ ⇒ 在 a=1/6 处最大值 = κ·4·(5/6)·0.5·u³ = 0.001083·u³ MW/单位
⇒ u=10 时 1.083 MW/单位 ⇒ ∂profit/∂a ≈ 580·1.083/60 = 10.5 元/块/单位（单机口径的上界）。
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
                                    'turbine_profit'],
                      chunksize=200000, encoding='utf-8-sig', low_memory=False):
    acc.append(ch)
df = pd.concat(acc, ignore_index=True)
A = np.array([[float(x) for x in t.strip().strip('[]').split()]
              for t in df['applied_action_H'].astype(str)])
for j in range(5):
    df['a%d' % (j + 1)] = A[:, j]
df = df[df['step'] > 0.75 * df['step'].max()].reset_index(drop=True)
df['uk'] = df['u'].round(3)
AC = ['a1', 'a2', 'a3', 'a4', 'a5']
print('样本 %d，唯一 u 值 %d（逐位匹配注定稀薄 ⇒ 主结果用"细箱 + 箱内控 u"）'
      % (len(df), df['uk'].nunique()))

# ---- 逐位匹配（作为可行性检查）----
n_ok = sum(1 for _, g in df.groupby('uk') if len(g) >= 5)
print('  逐位 u 箱中样本 ≥5 的: %d 个（若极少 ⇒ 该路线不可用）' % n_ok)

# ---- 细箱 + 箱内控 u ----
df['uk2'] = df['u'].round(1)
rows = []
for uk, g in df.groupby('uk2'):
    if len(g) < 40:
        continue
    X = g[AC].to_numpy(float)
    if (X.std(axis=0) < 0.005).any():
        continue
    uu = g['u'].to_numpy(float) - uk
    Z = np.column_stack([np.ones(len(g)), uu, X])
    y = g['turbine_profit'].to_numpy(float)
    b = np.linalg.lstsq(Z.T @ Z, Z.T @ y, rcond=None)[0]
    # 一元（只控 u）
    uni = []
    for j in range(5):
        Z1 = np.column_stack([np.ones(len(g)), uu, X[:, j]])
        b1 = np.linalg.lstsq(Z1.T @ Z1, Z1.T @ y, rcond=None)[0]
        uni.append(b1[2])
    rows.append(dict(u=uk, n=len(g),
                     **{'uni_a%d' % (j + 1): uni[j] for j in range(5)},
                     **{'mul_a%d' % (j + 1): b[2 + j] for j in range(5)}))
R = pd.DataFrame(rows)
print('可用 u 箱 %d 个，箱内样本中位 %d' % (len(R), int(R['n'].median())))
print('\n===== 箱内斜率的中位数（n 加权）=====')
w = R['n'].to_numpy(float)


def wmed(col):
    v = R[col].to_numpy(float)
    m = np.isfinite(v)
    if m.sum() == 0:
        return np.nan
    return float(np.average(v[m], weights=w[m]))


print('  %-6s %14s %14s' % ('dim', '一元(含混淆)', '五维条件'))
for j in range(5):
    print('  a%-5d %14.3f %14.3f'
          % (j + 1, wmed('uni_a%d' % (j + 1)), wmed('mul_a%d' % (j + 1))))
print('\n  物理上界参考（单机口径 ∂profit/∂a 的最大值）: u=10 → 10.5 元/块/单位；'
      'u=15 → 35.5')

print('\n===== 条件斜率按 u 分档（箱内多元，n 加权中位）=====')
df2 = df.copy()
df2['ub'] = pd.cut(df2['u'], [0, 7, 9, 11, 13, 16, 30])
for b, g in df2.groupby('ub', observed=True):
    if len(g) < 300:
        continue
    Z = np.column_stack([np.ones(len(g)), g[AC].to_numpy(float)])
    y = g['turbine_profit'].to_numpy(float)
    bb = np.linalg.lstsq(Z.T @ Z, Z.T @ y, rcond=None)[0][1:]
    print('  u∈%-10s n=%5d  %s' % (str(b), len(g),
          '  '.join('a%d %+8.3f' % (j + 1, bb[j]) for j in range(5))))

print('\n===== 策略落点 vs 物理/回归指向 =====')
print('  %-4s %8s %8s %14s %14s' % ('dim', 'mean(a)', 'sd(a)', '箱内条件斜率', '指向'))
for j in range(5):
    sl = wmed('mul_a%d' % (j + 1))
    print('  a%-3d %8.4f %8.4f %14.3f %14s'
          % (j + 1, df['a%d' % (j + 1)].mean(), df['a%d' % (j + 1)].std(), sl,
             '抬升' if sl > 1 else ('压低' if sl < -1 else '已平')))
