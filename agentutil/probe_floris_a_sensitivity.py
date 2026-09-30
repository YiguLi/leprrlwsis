# -*- coding: utf-8 -*-
"""临时探针④：`a` 通道的**模型敏感度** —— 它到底能改变多少风场功率？

为什么需要它
------------
代码取证（`floris/core/solver.py:128,200` + `wake_velocity/gauss.py:46`）：
GCH 高斯尾流的亏损**只由 CT 决定**，构造函数收到的 `axial_induction_i`
在该函数体内**从未被使用**；`a` 唯一真正进入物理的路径是
`wake_turbulence.crespo_hernandez.py:98` 的 `wake_induced_mixing`（∝ a）。
⇒ **改 `a` 能改多少功率，完全取决于湍流模型的开关**。

本探针用有限差分量化 `∂P_farm/∂a`（相对口径），逐个湍流模型对比。

用法: F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_floris_a_sensitivity.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

import yawflor.yawflor_turbine as yt  # noqa: E402


def build(turb_model, combination='sosfs', grid_pts=3):
    m, _ = yt.build_model(turbulence_model=turb_model, combination_model=combination)
    m.set_param(['solver', 'turbine_grid_points'], grid_pts)
    return m


def main():
    u = 10.0
    print('=' * 96)
    print('A. 各湍流模型下, 把全场 a 从 1/3 降到 0.20 的**风场功率变化** (u=10, γ=0)')
    print('=' * 96)
    # 用原生 multidim 表做"全场统一 a"，避免逐台补丁的干扰
    from agentutil.probe_floris_a_authority import build_md, _tables
    csv = _tables()
    print('  %-22s %14s %14s %10s' % ('turbulence_model', 'P(a=1/3) MW', 'P(a=0.20) MW', 'Δ%'))
    for tm in ('none', 'crespo_hernandez', 'wake_induced_mixing'):
        try:
            m = build_md(csv)
            m.set_param(['wake', 'model_strings', 'turbulence_model'], tm)
            vals = {}
            for a in (1 / 3, 0.20):
                m.set(wind_speeds=[u], wind_directions=[270.0],
                      turbulence_intensities=[yt.TI], yaw_angles=np.zeros((1, 5)),
                      multidim_conditions={'ai': float(a)})
                m.run()
                vals[a] = float(m.get_farm_power()[0]) / 1e6
            print('  %-22s %14.6f %14.6f %+10.4f'
                  % (tm, vals[1 / 3], vals[0.20],
                     100 * (vals[0.20] / vals[1 / 3] - 1)))
        except Exception as e:
            print('  %-22s 失败: %s: %s' % (tm, type(e).__name__, e))

    print('\n' + '=' * 96)
    print('B. 关掉湍流模型(隔离 a 的通道)后, 逐台 a 是否还能改变下游功率')
    print('=' * 96)
    import agentutil.probe_floris_a_per_turbine as P
    for tm in ('none', 'crespo_hernandez'):
        m = P.build_patched()
        m.set_param(['wake', 'model_strings', 'turbulence_model'], tm)
        row = []
        for avec in ([1 / 3] * 5, [0.05, 0.10, 0.20, 1 / 3, 1 / 3],
                     [0.20, 0.3333, 0.3333, 0.3333, 0.3333]):
            pf = float(P.set_a(m, [u], avec)[0][0])
            row.append(pf)
        print('  turbulence=%-20s P(1/3)=%.6f | P(斜坡)=%.6f (%+.4f%%) | P(只降首台)=%.6f (%+.4f%%)'
              % (tm, row[0], row[1], 100 * (row[1] / row[0] - 1),
                 row[2], 100 * (row[2] / row[0] - 1)))

    print('\n' + '=' * 96)
    print('C. 对照: 偏航通道 (γ) 的同类敏感度 (u=10)')
    print('=' * 96)
    base, _ = yt.build_model()
    p0 = yt.farm_power(base, u, [0.0] * 5)
    print('  %-22s %14s %14s %10s' % ('', 'P(γ=0) MW', 'P(γ=25°) MW', 'Δ%'))
    print('  %-22s %14.6f %14.6f %+10.4f'
          % ('gauss/GCH 默认', p0, yt.farm_power(base, u, [25.0] * 5),
             100 * (yt.farm_power(base, u, [25.0] * 5) / p0 - 1)))


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
