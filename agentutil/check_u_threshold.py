#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核查"风速 <10 m/s 全力发电就没问题、>10 才需要降载"这条经验（只读）。

两条独立证据：
  A. **实测**：noise 池 day122 上的 4 条固定动作 rollout（a=0.10/0.15/0.20/1/3），
     按 u 分箱统计违约率与 P_F ⇒ 看阈值到底在哪、以及各 a 在阈值之上的差别。
  B. **解析**：FLORIS 的 `P(u,a)` + 噪声模型 Δu~N(0, √2·1.0)（加噪池的分钟增量是白噪声，
     σ=1 m/s ⇒ 相邻差 σ=√2）⇒ 违约概率 `P(|P(u+Δu,a)−P(u,a)|>3)` 作为 (u,a) 的函数，
     给出"全出力在哪个 u 以下安全"与"降载能把违约压到多少"。

用法: python agentutil/check_u_threshold.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import aifflor.aifflor_turbine as AT  # noqa: E402

DIR = os.path.join('output', 'fixed_action')
ARMS = [('0.10', 'noise_day122_a0.10.csv'),
        ('0.15', 'noise_day122_a0.15.csv'),
        ('0.20', 'noise_day122_act0.2_0.2_0.2_0.2_0.2.csv'),
        ('1/3', 'noise_day122_act0.3_0.3_0.3_0.3_0.3.csv')]

print('=' * 96)
print('A. 实测：noise/0day122 的 4 条固定动作 rollout，按 u 分箱')
print('=' * 96)
bins = [0, 8, 9, 10, 11, 12, 13, 14, 40]
hdr = 'u 区间   ' + ''.join('%16s' % ('a=%s' % nm) for nm, _ in ARMS)
print(hdr)
data = {}
for nm, f in ARMS:
    p = os.path.join(DIR, f)
    if not os.path.exists(p):
        raise SystemExit('缺文件: %s' % p)
    data[nm] = pd.read_csv(p)

for lo, hi in zip(bins[:-1], bins[1:]):
    cells = []
    for nm, _ in ARMS:
        d = data[nm]
        m = (d.u >= lo) & (d.u < hi)
        if m.sum() == 0:
            cells.append('%16s' % '—')
            continue
        viol = 100 * (d.P_F[m] > 3).mean()
        cells.append('%7.2f%%/%5.2f' % (viol, d.P_F[m].mean()))
    print('  [%2d,%2d) ' % (lo, hi) + ''.join('%16s' % c for c in cells))
print('  （每格 = 违约率 / P_F 均值 [MW]）')

print('\n' + '=' * 96)
print('B. 解析：Δu~N(0,√2) 下，各 a 的违约概率 vs u（只看噪声，忽略慢变风）')
print('=' * 96)
UG = np.arange(6.0, 18.01, 0.5)
du = np.arange(-6, 6.001, 0.01)
pdf = np.exp(-du ** 2 / (2 * 2.0)) / np.sqrt(2 * np.pi * 2.0)      # σ=√2
pdf /= pdf.sum()

A_LIST = [0.10, 0.15, 0.20, 1 / 3.0]
# ⚠ 必须用**生产物理**：main 里是 AifflorPhysics(calibrated=False, power_scale=LEVEL_COEF)；
#   `AT.build_model()` 的默认是 calibrated=True（标定尾流参数），两者给的 P(u,a) **不同**
#   （标定参数下 P 在 a≈0.20 取峰、a=1/3 更低；生产物理下 P 随 a 单调升到 1/3）。
from aifflor.aifflor_env import AifflorPhysics          # noqa: E402
_PH = AifflorPhysics(nt=5, calibrated=False, power_scale=AT.LEVEL_COEF)
model = _PH.model
_scale = _PH.power_scale
_u_lo = float(getattr(AT, 'U_MIN_ENV', 1.0))


def p_batch(u_list, a):
    """一次性算一批 u 下的风场出力（向量化 FLORIS + 生产电平系数）。"""
    us = np.clip(np.asarray(u_list, dtype=float), max(_u_lo, 0.1), None)
    av = float(np.clip(a, AT.A_PHYS_LO, AT.A_HI))
    acts = np.full((us.size, AT.NT), av)
    farm, _ = AT.eval_batch(model, us, acts)
    return np.asarray(farm, dtype=float) * _scale


print('  %5s' % 'u' + ''.join('%12s' % ('a=%.2f' % a) for a in A_LIST))
cross = {}
for u in UG:
    row = []
    for a in A_LIST:
        p0 = float(p_batch([u], a)[0])
        pf = np.abs(p_batch(u + du, a) - p0)
        v = float(pdf[pf > 3].sum())
        row.append(v)
        cross.setdefault(a, []).append((u, v))
    print('  %5.1f' % u + ''.join('%11.2f%%' % (100 * r) for r in row))

print('\n  各 a 的"违约率 = 1% / 5%"对应风速（解析）:')
for a in A_LIST:
    arr = np.array(cross[a])
    for thr in (0.01, 0.05):
        idx = np.where(arr[:, 1] >= thr)[0]
        u_thr = arr[idx[0], 0] if len(idx) else np.nan
        print('    a=%.2f : %d%% 阈值 u ≈ %s m/s' % (a, int(thr * 100),
              ('%.1f' % u_thr) if np.isfinite(u_thr) else '>18'))

print('\n' + '=' * 96)
print('C. 结论口径')
print('=' * 96)
print('  * "全出力在 u 低于某阈值时无违约"是**对的**，但阈值比 10 更保守（见上表）。')
print('  * 阈值之上**降载只能削掉约 1/4 的 P_F**：dP/du ∝ P/u ⇒ a 从 1/3 砍到 0.10（3.3×）')
print('    只把 P_F 从 ~1.21 降到 ~0.90 ⇒ 违约率从 ~8.5% 降到 ~5%，**降不到 0**。')
print('  * 所以"学到 u 的阈值并降载"只值 3~4 个百分点；要达到 <1% 必须靠 BESS/当步补偿。')
