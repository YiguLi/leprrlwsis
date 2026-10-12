# -*- coding: utf-8 -*-
r"""run57 策略落点 vs **已知最优区间** [0.2, 0.2, 0.2, 0.2, 1/3] 的差，以及"起点/覆盖"诊断。

已知（用户提供）：大致最优区间 = 前四台 a=0.2、末台 a=1/3。
本脚本只做三件事（都可核对）：
  [1] 逐维算出"策略落点 → 目标"的位移，并用解析 Cp=4a(1−a)² 给出**每台机自身**的
      风功率系数变化 ΔCp（尾流通道不在内，故这是保守下界）。
  [2] 检查 warmup 的随机游走分布与 actor 的初始 bias 是否**覆盖**目标点
      （`hsrl_warmup.WarmupSchedule` 的真实实现，不猜）。
  [3] 用场总功率标度（WFSim u=10: a=0.20 → 3.7179 MW, a=1/3 → 3.4462 MW）把 ΔCp
      粗略换算成"每天多少钱"，给一个量级。
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

TARGET = np.array([0.2, 0.2, 0.2, 0.2, 1 / 3])
NAMES = ['a1', 'a2', 'a3', 'a4', 'a5']


def cp(a):
    return 4 * np.asarray(a, float) * (1 - np.asarray(a, float)) ** 2


# ---------------------------------------------------------------- [0] 策略落点
CSV = 'output/TD3PERNOLOW-run57/df_td3pernolow.csv'
acc = []
for ch in pd.read_csv(CSV, usecols=['step', 'applied_action_H', 'original_action_H',
                                    'u', 'wind_level', 'turbine_profit'],
                      chunksize=200000, encoding='utf-8-sig', low_memory=False):
    acc.append(ch)
df = pd.concat(acc, ignore_index=True)
A = np.array([[float(x) for x in t.strip().strip('[]').split()]
              for t in df['applied_action_H'].astype(str)])
df = df[df['step'] > 0.75 * df['step'].max()]
A = A[df.index]
pol = A.mean(axis=0)

print('=' * 100)
print('[0] 策略落点 vs 已知目标')
print('=' * 100)
print('  策略（末 25%% 训练段, n=%d, 含探索噪声）: %s'
      % (len(A), np.round(pol, 4)))
print('  目标（用户给出）                        : %s' % np.round(TARGET, 4))
print('  位移 Δ = 策略 − 目标                    : %s'
      % np.round(pol - TARGET, 4))

print('\n' + '=' * 100)
print('[1] 逐台的 ΔCp = 4a(1−a)²（自身风功率系数，不含尾流通道）')
print('=' * 100)
print('  %-4s %9s %9s %9s %9s %11s' %
      ('dim', '策略 a', '目标 a', 'Cp(策略)', 'Cp(目标)', 'ΔCp'))
cps, cpt = cp(pol), cp(TARGET)
for j in range(5):
    print('  %-4s %9.4f %9.4f %9.4f %9.4f %+11.4f'
          % (NAMES[j], pol[j], TARGET[j], cps[j], cpt[j], cpt[j] - cps[j]))
print('  %-4s %9s %9s %9.4f %9.4f %+11.4f'
      % ('Σ', '', '', cps.sum(), cpt.sum(), cpt.sum() - cps.sum()))
print('  ⇒ ΣΔCp = %+.4f（相对策略当前 ΣCp=%.4f 是 %+.1f%%）'
      % (cpt.sum() - cps.sum(), cps.sum(),
         100 * (cpt.sum() - cps.sum()) / cps.sum()))
print('  ⚠ 前四台降载（a→0.2）会**减弱尾流**，对下游是正贡献，未计入 ⇒ 这是保守下界；')
print('    末台 a5 无下游，ΔCp 是干净的（+%.4f）。' % (cpt[4] - cps[4]))

print('\n' + '=' * 100)
print('[2] warmup 与初始 bias 是否覆盖目标点')
print('=' * 100)
import hsrl_warmup as hw  # noqa: E402
print('  真源常量: WT_START=%s  WT_LO/HI=%s/%s  WT_STEP=%s'
      % (hw.WT_START, hw.WT_LO, hw.WT_HI, hw.WT_STEP))
sch = hw.WarmupSchedule(nb_turbine=5, seed=12345)
N = 1440                                    # warmup 固定 1440 步 ⇒ 288 块
blocks = []
for i in range(N // 5):
    blocks.append(sch.action())
B = np.array([b[:5] for b in blocks])
print('  warmup %d 块（每块 %d 步）动作统计:' % (len(B), 5))
for j in range(5):
    print('    %-4s mean %.4f sd %.4f  min %.2f max %.2f  取到目标值(±0.025)占比 %.2f%%'
          % (NAMES[j], B[:, j].mean(), B[:, j].std(), B[:, j].min(), B[:, j].max(),
             100 * (np.abs(B[:, j] - TARGET[j]) <= 0.025).mean()))
print('  ⇒ warmup 是**逐维独立**随机游走（每维各自 ±WT_STEP）⇒ 不携带"上游低/下游高"的')
print('    空间结构先验；每维单独看是覆盖 [%.2f,%.2f] 的。' % (B.min(), B.max()))
init_bias = np.random.uniform(hw.WT_LO, hw.WT_HI, 5)
print('\n  actor 初始 bias（init_wt_action_bias, U(%.2f,%.2f) 独立抽取）示例: %s'
      % (hw.WT_LO, hw.WT_HI, np.round(init_bias, 3)))
print('  ⇒ 初值在 [%.2f,%.2f] 内**随机**取值；目标向量 [%s] 不是初始化的一部分。'
      % (hw.WT_LO, hw.WT_HI, ', '.join('%.2f' % t for t in TARGET)))

print('\n' + '=' * 100)
print('[3] 换算成钱的量级（两个独立锚点）')
print('=' * 100)
print('  锚点 A: WFSim u=10 场总功率  均匀 a=0.20 → 3.7179 MW；均匀 a=1/3 → 3.4462 MW')
print('          ⇒ 每单位"场平均 Cp 当量" ≈ (3.7179−3.4462)/(Cp(1/3)·5−Cp(0.2)·5)')
d_cp = cp(1 / 3) * 5 - cp(0.2) * 5
d_p = 3.7179 - 3.4462
print('            = %.4f MW / %.4f = %.4f MW per ΣCp' % (d_p, d_cp, d_p / d_cp))
for price, lab in ((580, 'MED 电价'), (980, 'HIGH 电价')):
    yuan_per_block = abs(cpt.sum() - cps.sum()) * (d_p / d_cp) * price / 60 * 5
    print('  %s: |ΣΔCp|=%.4f → ≈ %.2f MW → ≈ %.1f 元/块（5 分钟）→ ≈ %.0f 元/天'
          % (lab, abs(cpt.sum() - cps.sum()),
             abs(cpt.sum() - cps.sum()) * d_p / d_cp,
             yuan_per_block, yuan_per_block * 288))
print('  ⚠ 该换算把"均匀族斜率"外推到异质动作，只给量级；')
print('    策略当前场总功率实测（9 用例 P_F 均值）≈ 2.41 MW ⇒ 上面的 MW 增量应按比例打折。')
pf_mean = 2.406
scale = pf_mean / 3.7179
print('    按 P_F 比例打折 (%.3f) 后: 净增益 ≈ %.3f MW → ≈ %.0f 元/块 → ≈ %.0f 元/天'
      % (scale, abs(cpt.sum() - cps.sum()) * d_p / d_cp * scale,
         abs(cpt.sum() - cps.sum()) * d_p / d_cp * scale * 580 / 60 * 5,
         abs(cpt.sum() - cps.sum()) * d_p / d_cp * scale * 580 / 60 * 5 * 288))
print('\n  策略当前实测日均（df_episode 48 集均值 69,113 元/集，1440 步/集）'
      '⇒ 上述增益占比 ≈ %.1f%%'
      % (100 * abs(cpt.sum() - cps.sum()) * d_p / d_cp * scale * 580 / 60 * 5 * 288
         / 69113))
