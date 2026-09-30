# -*- coding: utf-8 -*-
"""⚠ 本文件的口径**未定论** —— 请先读这一段再使用。

`forcescale` 的"力/功率是否自洽"取决于 `Actuator.m:64` 里 `/(1−a)` 的**速度参考约定**
（把 `Ue` 当当地风速、还是当上游风速），而这一点从代码本身**无法唯一判定**。
本文件的 §1/§2 在两轮修订里因为该约定得出过**互相矛盾**的表格
（`forcescale=1.0` 时而判"自洽"时而判"否"），说明我的代数在这一点上不可靠。

**因此：本文件只保留为"取证记录"（逐行引用的代码事实是可靠的），
其结论性表格不应当作依据。** 权威结论请看对话正文的保守表述，或执行
`agentutil/analyse_forcescale_per_turbine.py`（那里不需要这个约定：
它只依赖"F 只进 Actuator.m:64、不进 :69" 这一条**确定**事实）。

**可靠的（逐行读代码得到，不依赖上述约定）**
    WFSim_port.m:25   c_prime = 4·a·(1−a)²                    # 注释写 "C_p"
    Actuator.m:55     CT(kk) = input.CT_prime(kk)
    Actuator.m:51,53  Ue = cos(phi)·|u|                       # 当地风速
    Actuator.m:64     Fthrust = F·½ρ·Ue²·CT/(1−a)             # F = forcescale
    Actuator.m:69     Power = powerscale·½ρ·Ar·CT·mean(Ue³)   # **不乘 F**
    Actuator_Joeri.m:14  F = forcescale;  % NREL/TP-500-36834  ← 经验值的出处

**不确定的**
    `/(1−a)` 是"当地↔自由流"换算（则 F=1.5 与功率自洽），还是额外修正
    （则 F=1.5 的力偏大 50%）。需要数值实验（比较转子面动量通量亏损与抽出功率）
    才能定论；`F=1.0` 与 `F=1.5` 各会给出相反判读。

----
（以下为原始取证脚本，表格逻辑已知不可靠，仅留档）

**不可争议的事实（逐行读 `bin/core/Actuator.m` / `WFSim_port.m`）**
----------------------------------------------------------------
    WFSim_port.m:25   Control.c_prime = 4·a·(1−a)²          # 代码注释写 "C_p"
    Actuator.m:55     CT(kk) = input.CT_prime(kk)           # ⇒ CT = 4a(1−a)²  (= Cp)
    Actuator.m:51,53  Ue = cos(phi)·|u|                     # Ue 是**当地**风速
    Actuator.m:61     a(kk) = input.a(kk)                   # ⇒ a = Ue/U∞ 的**非轴向**定义
    Actuator.m:64     Fthrust = F·½ρ·Ue²·CT/(1−a)           # ← 力（F = forcescale）
    Actuator.m:69     Power   = powerscale·½ρ·Ar·CT·mean(Ue³)   # ← 功率（**不乘 F**）

因 `U∞ = Ue/(1−a)`（非轴向 a 的定义），把力折算到**自由流**口径：

    Fthrust / (½ρA U∞²) = F·4a(1−a)²/(1−a)²  =  F·4a(1−a)²/(1−a)²

一致性的唯一判据（动量守恒）
-----------------------------
    T = 2ρA U∞² a(1−a)          ⇒ 要求的力系数 CF = 4a(1−a)
    P = 2ρA U∞³ a(1−a)²         ⇒ 要求的功率系数 Cp = 4a(1−a)²
    （二者对同一个 a 必须同时成立 ⇒ 自洽条件 CF = Cp/(1−a)）

**注意**：本脚本只回答"力的**大小**是否等于动量守恒要求"。
代码把力定义成正比于 **a(1−a)²** 而不是 **a(1−a)**，这是**形状**问题（见 §3），
换 `forcescale` 无法修正。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_forcescale_consistency.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402


def main():
    a = 1.0 / 3.0
    cp_code = 4 * a * (1 - a) ** 2          # 0.5925926   代码的 CT_prime（= 标准 Cp）
    cf_req = 4 * a * (1 - a)                # 0.8888889   动量守恒要求的力系数
    print('a = 1/3')
    print('  代码 CT_prime = 4a(1−a)²      = %.7f   (= 标准 Cp)' % cp_code)
    print('  动量守恒要求 CF = 4a(1−a)     = %.7f' % cf_req)
    print('  自洽条件 CF = Cp/(1−a)  ⇒  %.7f = %.7f  %s'
          % (cf_req, cp_code / (1 - a), '✓' if abs(cf_req - cp_code / (1 - a)) < 1e-12 else '✗'))
    print('  CF / CT_prime = 1/(1−a) = %.4f' % (1 / (1 - a)))

    print('\n' + '=' * 96)
    print('§1 力的**大小**：forcescale 取不同值时的实际力系数（对自由流口径）')
    print('=' * 96)
    print('  实际力系数 = F · CT_prime/(1−a)² = F · 4a(1−a)²/(1−a)²')
    print('  （推导：Fthrust/(½ρA U∞²)，其中 U∞ = Ue/(1−a) ⇒ 除以 (1−a)²）')
    print('  要求在 a=1/3 处 = 4a(1−a) = %.7f' % cf_req)
    print()
    print('  %-11s %15s %13s %12s %8s' %
          ('forcescale', '实际力系数', '要求', '实际/要求', '自洽?'))
    for F in (1.0, 1.2, 4.0 / 3.0, 1.5, 1.6, 2.25):
        cf_actual = F * cp_code / (1 - a) ** 2
        ratio = cf_actual / cf_req
        print('  %-11.4f %15.7f %13.7f %12.4f %8s'
              % (F, cf_actual, cf_req, ratio, '是' if abs(ratio - 1) < 1e-12 else '否'))

    print('\n' + '=' * 96)
    print('§2 力与功率的**一致性**（这才是"物理模型是否失效"的判据）')
    print('=' * 96)
    print('  代码里功率 P = powerscale·½ρAr·Cp·⟨Ue³⟩，而力系数却是 F·Cp/(1−a)²。')
    print('  两者要同时满足动量守恒（CF = Cp/(1−a)），必须 F = 1/(1−a) = 1.5。')
    print()
    print('  %-11s %13s %13s %14s %s' %
          ('forcescale', '实际 CF', '要求 CF', '实际/要求', '物理判读'))
    for F in (1.0, 1.5):
        cf_actual = F * cp_code / (1 - a) ** 2
        ratio = cf_actual / cf_req
        if abs(ratio - 1) < 1e-12:
            note = '力的量级与功率自洽 ✓'
        elif ratio < 1:
            note = '力偏小：抽走动量比发电所需**少 %.1f%%** ⇒ 动量-能量不自洽' % (100 * (1 - ratio))
        else:
            note = '力偏大 %.1f%% ⇒ 动量-能量不自洽' % (100 * (ratio - 1))
        print('  %-11.4f %13.7f %13.7f %14.4f %s' % (F, cf_actual, cf_req, ratio, note))

    print('\n' + '=' * 96)
    print('§3 forcescale **修不了**的问题：力的"形状"')
    print('=' * 96)
    print('  代码的力 ∝ a(1−a)²   —— 峰值在 a=0.333（与 Cp 同峰）')
    print('  真实的推力 ∝ a(1−a)  —— 峰值在 a=0.5')
    print('  ⇒ 形状不同 ⇒ **任何常数倍率都无法同时修正整个 a 域**')
    print()
    print('  %-8s %14s %14s %14s' % ('a', '代码 4a(1−a)²', '真实 4a(1−a)', '代码/真实'))
    for aa in (0.10, 0.20, 1 / 3, 0.40, 0.50):
        code = 4 * aa * (1 - aa) ** 2
        real = 4 * aa * (1 - aa)
        print('  %-8.3f %14.7f %14.7f %14.5f' % (aa, code, real, code / real))
    print('  ⇒ 比值随 a 单调变化（0.9 → 0.6）⇒ 力的**分布形状**在偏离 1/3 时越错越多。')
    print('  ⇒ 这是上游 WFSim 的**结构性**选择（可能为匹配某台机型的 Cp/Ct 联合曲线），')
    print('     改 forcescale 只会整体缩放，不会让它变对。')

    print('\n' + '=' * 96)
    print('§4 结论')
    print('=' * 96)
    print('  · forcescale 的**量级**有明确含义：F = 1/(1−a) 才让"力"与"功率"自洽；')
    print('    在 a=1/3 处即 1.5 ⇒ **F=1.5 不是随手拍的，也不是错误**。')
    print('  · 把 F 改成 1.0 ⇒ 力降到要求的 %.1f%%，**引入 33%% 的动量-能量失衡**。'
          % (100 / 1.5))
    print('    那不是"修正"，是**制造**一个新的不一致。')
    print('  · 所以：**是的，改 forcescale 会让物理模型失效**（力与功率不再自洽），')
    print('    除非同时把功率乘以同一个因子（即 powerscale 同步改）。')
    print('  · 另有一条**独立**的 5%% 失衡：powerscale=0.95 让功率比其自身 CT_prime')
    print('    对应的动量提取小 5%%（这部分与 forcescale 无关）。')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
