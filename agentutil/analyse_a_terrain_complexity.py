# -*- coding: utf-8 -*-
"""把"地形复杂度"变成可测量：`a` 方向的曲率与"近平坦区"宽度。

立论（用户提出）
----------------
超参数能否跨引擎迁移，**不要求地形形状相同** —— 只要求**难度/复杂度相当**。
理由是：常数重标定 `P → k·P` (k>0) 是正比例变换，**保持 argmax 与全部水平集的形状**，
所以它不改变学习难度。

于是需要一个**可操作**的复杂度定义。本脚本用两个直接反映"搜索难度"的量：

  C1 **归一化曲率** `κ = |P''(a)| / P(a*)`  [1/a²]
     峰越尖 ⇒ 策略必须把 `a` 定得越准 ⇒ 越难。
  C2 **近平坦区宽度** `W(ε)` = `{a : P(a) ≥ (1−ε)·P(a*)}` 的长度 [a]
     越宽 ⇒ 同样的动作误差代价越小 ⇒ 越容易。

两者都由 **FLORIS 侧**在此算出（无需 MATLAB）；WFSim 侧的对应值由
`agentutil/probe_floris_a_authority.py --engines both`（待加）或 MATLAB 侧同法给出。

用法: F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_a_terrain_complexity.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

import yawflor.yawflor_turbine as YT  # noqa: E402


def floris_uniform_curve(us, a_grid):
    """FLORIS（原生 multidim 表）在均匀 a 下的风场功率。返回 (len(us), len(a_grid))。"""
    from agentutil.probe_floris_a_authority import build_md, _tables
    m = build_md(_tables())              # 表里已含 0~0.5 的 a 网格
    out = np.zeros((len(us), len(a_grid)))
    for j, a in enumerate(a_grid):
        n = len(us)
        m.set(wind_speeds=list(us), wind_directions=[YT.WIND_DIR_ALIGNED] * n,
              turbulence_intensities=[YT.TI] * n,
              yaw_angles=np.zeros((n, 5)),
              multidim_conditions={'ai': float(a)})
        m.run()
        out[:, j] = np.asarray(m.get_farm_power(), dtype=float) / 1e6
    return out


def curvature_and_flat(P, a):
    """返回 (a*, Pmax, κ = |P''|/Pmax, W(ε) 表, 相对半宽)."""
    i = int(np.argmax(P))
    a_star, Pmax = float(a[i]), float(P[i])
    # 中心差分二阶导（端点用单侧）
    if 0 < i < len(a) - 1:
        h = a[i + 1] - a[i - 1]
        d2 = (P[i + 1] - 2 * P[i] + P[i - 1]) / ((a[i + 1] - a[i]) ** 2)
    else:
        h = a[1] - a[0]
        d2 = (P[i] - 2 * P[i + 1] + P[i + 2]) / h ** 2 if i == 0 else \
             (P[i] - 2 * P[i - 1] + P[i - 2]) / h ** 2
    kappa = abs(float(d2)) / Pmax
    widths = {}
    for eps in (0.005, 0.01, 0.02, 0.05):
        thr = (1 - eps) * Pmax
        sel = np.flatnonzero(P >= thr)
        widths[eps] = float(a[sel[-1]] - a[sel[0]]) if sel.size > 1 else 0.0
    return a_star, Pmax, kappa, widths, float(d2)


def main():
    a_grid = np.round(np.arange(0.025, 0.5001, 0.025), 6)
    us = np.array([6.0, 8.0, 10.0, 12.0])
    print('=' * 92)
    print('FLORIS 侧：均匀 a 响应曲线的曲率与近平坦区（a 网格 %.3f~%.3f 步长 %.3f）'
          % (a_grid[0], a_grid[-1], a_grid[1] - a_grid[0]))
    print('=' * 92)
    C = floris_uniform_curve(us, a_grid)
    print('  %-8s %10s %12s %12s %12s %12s' %
          ('u', 'a*', 'P(a*) MW', 'κ=|P\'\'|/P', 'W(1%)', 'W(5%)'))
    store = {}
    for k, u in enumerate(us):
        a_star, Pmax, kappa, W, d2 = curvature_and_flat(C[k], a_grid)
        store[u] = dict(a_star=a_star, Pmax=Pmax, kappa=kappa, W=W, d2=d2)
        print('  %-8.1f %10.4f %12.6f %12.4f %12.4f %12.4f'
              % (u, a_star, Pmax, kappa, W[0.01], W[0.05]))

    print('\n  读法: κ 越大 ⇒ 峰越尖 ⇒ 策略越需要把 a 定准 ⇒ 越难;')
    print('        W 越宽   ⇒ 同样动作误差代价越小 ⇒ 越容易。')

    print('\n' + '=' * 92)
    print('归一化对比（把每个 u 的曲线各自除以它的最优点，看"形状"有多陡）')
    print('=' * 92)
    Pn = C / C.max(axis=1, keepdims=True)
    print('  %-10s %s' % ('a', ''.join('%9.3f' % x for x in a_grid[::2])))
    for k, u in enumerate(us):
        print('  u=%-8.1f %s' % (u, ''.join('%9.4f' % x for x in Pn[k][::2])))

    print('\n' + '=' * 92)
    print('把"复杂度"写成可比的三个数（供与 WFSim 对照）')
    print('=' * 92)
    ks = np.array([store[u]['kappa'] for u in us])
    ws = np.array([store[u]['W'][0.01] for u in us])
    ast = np.array([store[u]['a_star'] for u in us])
    print('  FLORIS:')
    print('    a*        = %s   (均值 %.4f, 极差 %.4f)'
          % (np.array2string(ast, precision=4), ast.mean(), ast.max() - ast.min()))
    print('    κ         = %s   (均值 %.4f)' % (np.array2string(ks, precision=4), ks.mean()))
    print('    W(1%%)     = %s   (均值 %.4f)' % (np.array2string(ws, precision=4), ws.mean()))
    print('\n  ⇒ 同法对 WFSim 跑一遍（均匀 a、同 u、同网格）即得可比的 (a*, κ, W)。')
    print('     判定: κ 同量级 且 W 同量级 ⇒ 两引擎"搜索难度相当" ⇒ 你的可迁移判据成立;')
    print('           若 κ 差一个量级或 W 差一个量级 ⇒ 难度不同 ⇒ 仍不能直接搬超参数。')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
