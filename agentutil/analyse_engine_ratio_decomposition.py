# -*- coding: utf-8 -*-
"""把实测比值 FLORIS/WFSim 拆解成「吞吐项 × 尾流项」。

分解（来源逐项可追溯）
----------------------
两边风场总功率都可写成   P = η_wake · N · ½ρA · Cp · u_eff³ · (1e-6 MW/W)

  WFSim : Cp_w = 4a(1−a)² = 0.59259259   (a=0.3333, WFSim_port.m:25 → Actuator.m:69)
  FLORIS: Cp_f = 0.95·4a(1−a)² = 0.56296296  (yawflor_turbine CP_EFF)
  u_eff : 由 wind_shear=0.12 决定。FLORIS 的 rotor-average 口径是
          u_eff/u_hub = cbrt(mean(u³)) —— 轮毂上下各一点取立方平均
          （yawflor/README.md §0 已实测该口径，比值 0.998516 @ 轮毂高 100 m，
           此处按 [u(50), u(100), u(150)] 三点复刻）。
          WFSim 侧用轮毂风速 u_hub（Actuator.m 取转子线平均，2D 无高度维）。

于是
      P_f/P_w = (Cp_f·u_eff_f³) / (Cp_w·u_hub³) · (η_f/η_w)
              = 吞吐项 → 已知解析式
              · 尾流项  → **由实测比值反解**（这是唯一未知，也是引擎差异所在）

反解出来的尾流项 = WFSim 相对 FLORIS 的**尾流亏损强度比**，可直接看它的风速依赖。

用法: F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_engine_ratio_decomposition.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

import yawflor.yawflor_turbine as yt  # noqa: E402

JSON_DEFAULT = os.path.join('output', 'engine_minute_compare.json')
ALPHA = yt.WIND_SHEAR        # 0.12
H_HUB = yt.HUB_HEIGHT        # 100
CP_W = 4.0 * (1 / 3) * (1 - 1 / 3) ** 2      # WFSim 的 CT_prime = Cp（不乘 0.95）
CP_F = yt.CP_EFF                              # FLORIS 表里的 Cp_eff


def throughput_ratio(u_hub):
    """(Cp_f·u_eff_f³) / (Cp_w·u_hub³)：FLORIS 吞吐 / WFSim 吞吐。

    u_eff 只用**切变形状**（与 u_hub 成正比）⇒ 比值仍随 u_hub 有 ~0.3% 的弱变化，
    因为 WFSim 侧用的是轮毂风速、FLORIS 侧用的是转子立方平均。
    """
    u_hub = np.asarray(u_hub, dtype=float)
    z = np.array([H_HUB - 50.0, H_HUB, H_HUB + 50.0])
    shape = (z / H_HUB) ** ALPHA                       # 剖面形状（=1 at hub）
    u_eff = u_hub * np.cbrt(np.mean(shape ** 3))       # 立方平均 = cbrt(mean(u³))
    return (CP_F * u_eff ** 3) / (CP_W * u_hub ** 3)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else JSON_DEFAULT
    if not os.path.exists(path):
        raise SystemExit('找不到 %s' % path)
    d = json.load(open(path, encoding='utf-8'))
    u = np.asarray(d['wind'], dtype=float)
    w = np.asarray(d['p_wfsim'], dtype=float)
    f = np.asarray(d['p_floris'], dtype=float)
    cfg = d.get('config', {})
    print('数据: %s' % path)
    print('  n=%d  源=%s  a≡%s  FLORIS %s'
          % (len(u), cfg.get('source'), cfg.get('a_nominal'), cfg.get('floris')))

    ratio = f / w
    thr = throughput_ratio(u)
    wake = ratio / thr                       # 反解出的尾流项 (η_f/η_w)
    m = w > 0.5

    print('\n' + '=' * 84)
    print('分解 P_f/P_w = 吞吐项 × 尾流项')
    print('  Cp_w(WFSim)=%.8f   Cp_f(FLORIS)=%.8f   Cp_f/Cp_w=%.6f  (即 −4.998%%)'
          % (CP_W, CP_F, CP_F / CP_W))
    print('=' * 84)
    print('  %8s %10s %12s %12s %12s' % ('u_hub', '实测比值', '吞吐项', '尾流项(反解)', '尾流占比'))
    print('  %8s %10s %12s %12s %12s' % ('', '', '(解析)', '= 实测/吞吐', '1−尾流项'))
    for lo, hi in [(3.5, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 11)]:
        s = m & (u >= lo) & (u < hi)
        if s.sum() == 0:
            continue
        print('  %4.1f-%-3.1f %10.4f %12.4f %12.4f %11.2f%%'
              % (lo, hi, ratio[s].mean(), thr[s].mean(), wake[s].mean(),
                 100 * (1 - wake[s].mean())))

    print('\n  全局:')
    print('    实测比值     中位 %.4f  均值 %.4f' % (np.median(ratio[m]), ratio[m].mean()))
    print('    吞吐项       中位 %.4f  均值 %.4f' % (np.median(thr[m]), thr[m].mean()))
    print('    尾流项(反解) 中位 %.4f  均值 %.4f' % (np.median(wake[m]), wake[m].mean()))
    print('    ⇒ 1.74× 中，约 %.2f 来自"CP/吞吐"（常数），约 %.2f 来自"尾流亏损更浅"'
          % (CP_F / CP_W, np.median(wake[m])))
    print('    尾流项风速相关性 corr(wake, u) = %.4f' % np.corrcoef(wake[m], u[m])[0, 1])

    # 若只按吞吐项重标定（不做任何尾流修正），残差多大？
    print('\n  单参数重标定的可行性:')
    for nm, k in (('k = Cp_f/Cp_w（纯吞吐，物理常数）', CP_F / CP_W),
                  ('k = 电量比', f.sum() / w.sum()),
                  ('k = 中位比值', np.median(ratio[m])),
                  ('k = 最小二乘 k*', float(np.dot(w[m], f[m]) / np.dot(w[m], w[m])))):
        r = (f[m] - k * w[m]) / f[m]
        print('    %-34s k=%.6f  残差RMS %.4f%%  最大|残差| %.4f%%'
              % (nm, k, 100 * np.sqrt((r ** 2).mean()), 100 * np.abs(r).max()))

    # 若加上"尾流项随 u 的线性修正"（两参数），残差多大？
    A = np.c_[w[m], w[m] * u[m]]
    beta, *_ = np.linalg.lstsq(A, f[m], rcond=None)
    r = (f[m] - A @ beta) / f[m]
    print('    %-34s 残差RMS %.4f%%  最大|残差| %.4f%%   (k1=%.5f, k2=%.6f)'
          % ('两参数 k1·w + k2·w·u', 100 * np.sqrt((r ** 2).mean()),
             100 * np.abs(r).max(), beta[0], beta[1]))


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
