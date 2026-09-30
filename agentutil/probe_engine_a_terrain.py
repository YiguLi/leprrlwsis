# -*- coding: utf-8 -*-
"""**两引擎的 `a` 地形复杂度对比** —— 一次跑完，回答"难度是否相当"。

立论
----
超参数能否跨引擎迁移，**不要求地形形状相同**，只要求**难度/复杂度相当**：
常数重标定 `P → k·P`（k>0）保持 argmax 与全部水平集形状 ⇒ **不改变学习难度**。

本脚本为两引擎算出同一组**可比的复杂度量**（均匀 `a` 扫描，同一 u 与 a 网格）：

  `a*`            风场最优轴向因子
  `κ = |P''(a*)|/P(a*)`  归一化曲率 —— 峰越尖 ⇒ 动作要定得越准 ⇒ 越难
  `W(ε)`          `{a : P(a) ≥ (1−ε)P(a*)}` 宽度 —— 越宽 ⇒ 容错越大 ⇒ 越容易
  `P(a)/P(a*)`    归一化曲线（形状本身，仅作参考，**不是判据**）

WFSim 侧共 4 个风速 × 21 个 a 值 ≈ 84 次 `WFSim_port` 调用。

用法（普通终端；MATLAB Engine 需 named pipe）
--------------------------------------------
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_engine_a_terrain.py
    ... --us 6 8 10 12 --json output/engine_a_terrain.json
    ... --no-matlab          # 只出 FLORIS 侧（自检）
"""
import argparse
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

import yawflor.yawflor_turbine as YT  # noqa: E402

NT = 5
A_NOMINAL = 0.3333


# ---------------------------------------------------------------------------
def a_grid(lo=0.025, hi=0.5, step=0.025):
    return np.round(np.arange(lo, hi + 1e-9, step), 6)


def curvature_and_flat(P, a):
    i = int(np.argmax(P))
    a_star, Pmax = float(a[i]), float(P[i])
    if 0 < i < len(a) - 1:
        d2 = (P[i + 1] - 2 * P[i] + P[i - 1]) / (a[i + 1] - a[i]) ** 2
    elif i == 0:
        d2 = (P[i] - 2 * P[i + 1] + P[i + 2]) / (a[1] - a[0]) ** 2
    else:
        d2 = (P[i] - 2 * P[i - 1] + P[i - 2]) / (a[1] - a[0]) ** 2
    W = {}
    for eps in (0.005, 0.01, 0.02, 0.05):
        thr = (1 - eps) * Pmax
        sel = np.flatnonzero(P >= thr)
        W[eps] = float(a[sel[-1]] - a[sel[0]]) if sel.size > 1 else 0.0
    return dict(a_star=a_star, Pmax=Pmax, d2=float(d2),
                kappa=abs(float(d2)) / Pmax, W=W,
                norm=[float(x) for x in (P / Pmax)])


# ---------------------------------------------------------------------------
def floris_curve(us, ag):
    """FLORIS 原生 multidim 表（全场统一 a）；一次 set+run 算一个 a。"""
    from agentutil.probe_floris_a_authority import build_md, _tables
    m = build_md(_tables())
    n = len(us)
    C = np.zeros((len(us), len(ag)))
    for j, a in enumerate(ag):
        m.set(wind_speeds=list(us), wind_directions=[YT.WIND_DIR_ALIGNED] * n,
              turbulence_intensities=[YT.TI] * n, yaw_angles=np.zeros((n, NT)),
              multidim_conditions={'ai': float(a)})
        m.run()
        C[:, j] = np.asarray(m.get_farm_power(), dtype=float) / 1e6
    return C


def wfsim_curve(us, ag, verbose=True):
    import matlab
    import matlab.engine
    if verbose:
        print('[WFSim] 启动 MATLAB 引擎 ...', flush=True)
    t0 = time.time()
    eng = matlab.engine.start_matlab()
    if verbose:
        print('[WFSim] 就绪 (%.1f s)，MATLAB %s' % (time.time() - t0, eng.version()),
              flush=True)
    eng.cd(ROOT, nargout=0)
    C = np.full((len(us), len(ag)), np.nan)
    t0 = time.time()
    n_call = 0
    for k, u in enumerate(us):
        for j, a in enumerate(ag):
            res = eng.WFSim_port(matlab.double([float(u)]), matlab.double([0.0]),
                                 matlab.double([[float(a)] * NT]), matlab.double([0.0]))
            C[k, j] = float(res['out']) / 1e6
            n_call += 1
        if verbose:
            print('[WFSim] u=%5.1f 完成 (%d/%d 条机组调用, %.1f s)'
                  % (u, n_call, len(us) * len(ag), time.time() - t0), flush=True)
    eng.quit()
    if verbose:
        print('[WFSim] 完成，共 %d 次调用，用时 %.1f s' % (n_call, time.time() - t0),
              flush=True)
    return C


# ---------------------------------------------------------------------------
def report(tag, us, ag, C):
    print('\n' + '=' * 94)
    print('%s：均匀 a 响应曲线' % tag)
    print('=' * 94)
    print('  %-7s %9s %13s %12s %10s %10s %10s'
          % ('u', 'a*', 'P(a*) MW', "κ=|P''|/P", 'W(0.5%)', 'W(1%)', 'W(5%)'))
    res = {}
    for k, u in enumerate(us):
        r = curvature_and_flat(C[k], ag)
        res[float(u)] = r
        print('  %-7.1f %9.4f %13.6f %12.4f %10.4f %10.4f %10.4f'
              % (u, r['a_star'], r['Pmax'], r['kappa'], r['W'][0.005],
                 r['W'][0.01], r['W'][0.05]))
    return res


def main():
    ap = argparse.ArgumentParser(description='两引擎 a 地形复杂度对比',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument('--us', type=float, nargs='+', default=[6.0, 8.0, 10.0, 12.0])
    ap.add_argument('--lo', type=float, default=0.025)
    ap.add_argument('--hi', type=float, default=0.5)
    ap.add_argument('--step', type=float, default=0.025)
    ap.add_argument('--no-matlab', action='store_true')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    us = np.asarray(args.us, dtype=float)
    ag = a_grid(args.lo, args.hi, args.step)
    print('运行回执')
    print('  解释器 : %s' % sys.executable)
    print('  风速   : %s' % np.array2string(us, precision=2))
    print('  a 网格 : %.3f~%.3f 步长 %.3f（%d 点）' % (ag[0], ag[-1], args.step, len(ag)))
    print('  口径   : 单排 5 台 5D | γ≡0 | ρ=1.20 | shear=0.12 | 动作 = 全场统一 a')

    CF = floris_curve(us, ag)
    rF = report('FLORIS', us, ag, CF)
    CW = None
    rW = None
    if not args.no_matlab:
        try:
            CW = wfsim_curve(us, ag)
            rW = report('WFSim', us, ag, CW)
        except Exception as e:
            print('\n[!] WFSim 侧失败：%s: %s' % (type(e).__name__, e))
            traceback.print_exc()
            print('\n排查：MATLAB Engine 需能创建 named pipe。受限沙箱下会报 '
                  '"CreateFile failed: Access is denied" ⇒ 请在普通终端里跑。')
            return 2

    if rW is not None:
        print('\n' + '=' * 94)
        print('复杂度判据（你的立论：形状可以不同，难度必须相当）')
        print('=' * 94)
        print('  %-8s %-26s %-26s' % ('u', 'FLORIS (a*, κ, W1%)', 'WFSim (a*, κ, W1%)'))
        for u in us:
            a_, b_ = rF[float(u)], rW[float(u)]
            print('  %-8.1f (%6.3f, %7.3f, %6.3f)      (%6.3f, %7.3f, %6.3f)'
                  % (u, a_['a_star'], a_['kappa'], a_['W'][0.01],
                     b_['a_star'], b_['kappa'], b_['W'][0.01]))
        kF = np.array([rF[float(u)]['kappa'] for u in us])
        kW = np.array([rW[float(u)]['kappa'] for u in us])
        wF = np.array([rF[float(u)]['W'][0.01] for u in us])
        wW = np.array([rW[float(u)]['W'][0.01] for u in us])
        aF = np.array([rF[float(u)]['a_star'] for u in us])
        aW = np.array([rW[float(u)]['a_star'] for u in us])
        print('\n  汇总:')
        print('    a*      FLORIS %.4f±%.4f  |  WFSim %.4f±%.4f   ⇒ 最优相差 %.4f'
              % (aF.mean(), aF.std(), aW.mean(), aW.std(), abs(aF.mean() - aW.mean())))
        print('    κ       FLORIS %.4f  |  WFSim %.4f   ⇒ 比值 %.3f'
              % (kF.mean(), kW.mean(), kW.mean() / kF.mean()))
        print('    W(1%%)   FLORIS %.4f  |  WFSim %.4f   ⇒ 比值 %.3f'
              % (wF.mean(), wW.mean(), wW.mean() / wF.mean()))
        print('\n  判定（本脚本的预登记判据）:')
        print('    * |a*_F − a*_W| ≲ 0.05  且  κ 比值 ∈ [0.5, 2]  且  W 比值 ∈ [0.5, 2]')
        print('      ⇒ 难度相当 ⇒ 你的可迁移判据成立（形状不同无妨）。')
        print('    * 否则 ⇒ 难度不同 ⇒ 超参数仍需各自重标定。')

        # 归一化曲线并排（形状参考，非判据）
        print('\n  归一化曲线 P(a)/P(a*)（形状参考，不是判据）:')
        print('    %-12s %s' % ('a', ''.join('%8.3f' % x for x in ag[::2])))
        for u in us:
            print('    F u=%-8.1f %s' % (u, ''.join('%8.4f' % x
                                                    for x in rF[float(u)]['norm'][::2])))
            print('    W u=%-8.1f %s' % (u, ''.join('%8.4f' % x
                                                    for x in rW[float(u)]['norm'][::2])))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(dict(config=dict(us=[float(x) for x in us],
                                       a_grid=[float(x) for x in ag],
                                       nt=NT, a_nominal=A_NOMINAL,
                                       flo=floris_version()),
                           floris=dict(curve=[[float(v) for v in row] for row in CF],
                                       stats={str(u): rF[u] for u in rF}),
                           wfsim=(None if CW is None else
                                  dict(curve=[[float(v) for v in row] for row in CW],
                                       stats={str(u): rW[u] for u in rW}))),
                      f, ensure_ascii=False, indent=1)
        print('\nJSON 已写入 %s' % args.json)
    return 0


def floris_version():
    try:
        import floris
        return getattr(floris, '__version__', 'unknown')
    except Exception:
        return 'unknown'


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
