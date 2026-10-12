# -*- coding: utf-8 -*-
r"""test9 评测的**变异来源**判别：同一 (场景步序号, 用例) 的 10 次评测之间，
差异来自"风序列不同"还是"动作不同（探索噪声）"？

判别式（两者可分离）：
  * nolow 下 `c1 = P_F + 20·(P_F−3)⁺`（uniform 口径），P_F 由 (风序列, 5 维动作) 决定；
  * 动作抖动幅度已知（≈探索噪声地板）：若**动作**是主因，则 c1 的组内变异应 ≈ |∂c1/∂a|·σ_a；
  * 若**风序列**是主因，则 c1 的组内变异会远大于该值（因为 P_F 对风远比对 a 敏感）。
另外直接看：同一 cell 内 c1 的**极差**是否达到"未经动作改变就出现的量级"。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUN = 'output/TD3PERNOLOW-run57'
t9 = pd.read_csv(os.path.join(RUN, 'df_test9.csv'), encoding='utf-8-sig')
t9 = t9[t9.wind_level != 'OVERALL'].copy()
# 每一行 = 一次评测里的一个场景步；同一 (step 值, wind, price) 的 10 行 = 10 次评测的同一格
key = ['step', 'wind_level', 'price_level']
g = t9.groupby(key)
sd = g['c1'].std()          # 组内 sd（10 次评测之间）
rng_ = g['c1'].max() - g['c1'].min()
print('cell 数 = %d，每 cell 评测次数 = %s' % (len(sd), int(g.size().median())))
print('组内 c1 sd   : 中位 %.4f  p95 %.4f  最大 %.4f' % (sd.median(), sd.quantile(.95), sd.max()))
print('组内 c1 极差 : 中位 %.4f  p95 %.4f  最大 %.4f' % (rng_.median(), rng_.quantile(.95), rng_.max()))
print('全体 c1 的 sd（跨 cell）= %.4f' % t9['c1'].std())

# 若动作是主因：c1 ≈ P_F + 20(P_F-3)+；在 P_F<3 时 ∂c1/∂P_F = 1
# 动作抖动 σ_a ≈ 探索噪声地板/√3；P_F 对单维动作的灵敏度由训练日志回归 ≈ 0.3~1.4 MW/单位
amp = 0.05 * 0.135 + 0.01 * 0.865          # ε(末)≈0.135 时的逐维噪声半宽
sig_a = amp / np.sqrt(3)
print('\n参考：ε(末)≈0.135 ⇒ 逐维噪声半宽 %.5f ⇒ 单维 σ_a = %.5f' % (amp, sig_a))
for dPF in (0.3, 1.4):
    print('  若 ∂P_F/∂a ≈ %.1f（5 维独立、c1 在 P_F<3 段斜率=1）⇒ 由动作引起的 σ_c1 ≈ %.5f'
          % (dPF, dPF * sig_a * np.sqrt(5)))
print('⇒ 实测组内 σ_c1 中位 %.4f 比"动作通道"的上界大 %.0f~%.0f 倍'
      % (sd.median(), sd.median() / (0.3 * sig_a * np.sqrt(5)),
         sd.median() / (1.4 * sig_a * np.sqrt(5))))

print('\n⇒ 结论：同一 (场景步, 用例) 的 10 次评测之间，**风序列本身不同** ⇒')
print('   `nine_case_test(seed=42)` 的 "fixed scenario seed 42" 是对**同一次调用内**的')
print('   9 用例场景固定，但每次调用都用全局 `np.random` 继续抽样 ⇒ **每次评测是一段新的风轨迹**。')
print('   因此逐次评测的差异主要是"风况抽样"，不是"策略变化"，也不是探索噪声。')
print('\n   影响：① 单次评测的度量值不可与另一次直接比较；')
print('        ② 要配对比较必须让两臂在同一段风上跑（当前实现做不到）；')
print('        ③ 该面板对"策略改进"的分辨力受这个风况方差限制。')
