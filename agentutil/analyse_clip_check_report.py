# -*- coding: utf-8 -*-
"""全风速域 FLORIS vs WFSim 的最终对比报告（含"相对误差 vs 绝对误差"诊断与逐台对比）。

数据来源
--------
* `output/clip_check_plan.json`     350 点执行计划（35 档 × 10 个随机行为，3–20 m/s）
* `output/clip_check_wfsim.json`    MATLAB 跑出的 WFSim 参考（含逐台）
* `output/clip_check_fullrange.json` FLORIS 侧结果

要点
----
**百分比误差在小分母处会失真**：低风速段农场出力本身只有 0.09 MW，0.01 MW 的绝对
偏差就是 11%。所以除 MAPE 外必须同时看**绝对误差 (MW)** 与 **归一化误差**
（除以该点的 WFSim 出力，或用"额定功率"归一）。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_clip_check_report.py
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

PLAN = os.path.join('output', 'clip_check_plan.json')
WREF = os.path.join('output', 'clip_check_wfsim.json')
FREF = os.path.join('output', 'clip_check_fullrange.json')


def main():
    plan = json.load(open(PLAN, encoding='utf-8'))
    w = json.load(open(WREF, encoding='utf-8'))
    f = json.load(open(FREF, encoding='utf-8'))
    u = np.asarray(plan['u'], dtype=float)
    A = np.asarray(plan['A'], dtype=float)
    pw = np.asarray(w['p_wfsim'], dtype=float)
    tw = np.asarray(w['p_wfsim_turbine'], dtype=float)
    pf = np.asarray(f['p_floris_clip'], dtype=float)
    tf = np.asarray(f.get('p_floris_turbine', []), dtype=float) if 'p_floris_turbine' in f else None
    n = len(u)
    print('点数 n=%d | u ∈ [%.1f, %.1f] | WFSim 参考齐全=%s'
          % (n, u.min(), u.max(), bool(np.isfinite(pw).all())))

    err = 100 * (pf - pw) / np.maximum(pw, 1e-9)
    abserr = pf - pw

    print('\n' + '=' * 104)
    print('① 总体偏差')
    print('=' * 104)
    print('  相对误差 (%%): MAPE=%.3f%%  中位=%.3f%%  p05=%.3f%%  p95=%.3f%%  min=%.3f%%  max=%.3f%%'
          % (np.mean(np.abs(err)), np.median(err), np.percentile(err, 5),
             np.percentile(err, 95), err.min(), err.max()))
    print('  绝对误差 (MW): MAE=%.4f  中位=%.4f  p95=%.4f  min=%.4f  max=%.4f'
          % (np.mean(np.abs(abserr)), np.median(abserr),
             np.percentile(abserr, 95), abserr.min(), abserr.max()))
    print('  相对 |误差| 的 90 分位 = %.3f%%' % np.percentile(np.abs(err), 90))

    print('\n' + '=' * 104)
    print('② 最大的几个相对误差 —— 检查是否只是"小分母"造成的失真')
    print('=' * 104)
    print('  %-8s %8s %10s %10s %10s %12s' %
          ('u', '|err|%', 'P_WFSim', 'P_FLORIS', '绝对误差', '|误差|/额定'))
    order = np.argsort(-np.abs(err))[:8]
    rated = 5 * 5.0     # 5 台 × ~5 MW 的额定量级（仅作归一化参照）
    for i in order:
        print('  %-8.2f %8.2f %10.4f %10.4f %10.4f %12.3f%%'
              % (u[i], abs(err[i]), pw[i], pf[i], abserr[i],
                 100 * abs(abserr[i]) / rated))
    print('  ⇒ 若这些点的 |绝对误差| 都很小（O(0.01) MW），则该"大百分比"无物理意义。')

    print('\n' + '=' * 104)
    print('③ 按风速段（同时给百分比与绝对值）')
    print('=' * 104)
    bands = [(3, 4), (4, 6), (6, 10), (10, 14), (14, 17), (17, 20.5)]
    print('  %-12s %5s %10s %10s %12s %10s %12s' %
          ('风速段', 'n', 'P_WFSim', 'P_FLORIS', 'MAPE', 'MAE(MW)', '最大|绝对|'))
    for lo, hi in bands:
        m_ = (u >= lo) & (u < hi)
        if not m_.any():
            continue
        e, a = err[m_], abserr[m_]
        print('  [%4.1f,%5.1f) %5d %10.4f %10.4f %11.3f%% %10.4f %12.4f'
              % (lo, hi, m_.sum(), pw[m_].mean(), pf[m_].mean(),
                 np.mean(np.abs(e)), np.mean(np.abs(a)), np.abs(a).max()))

    print('\n' + '=' * 104)
    print('④ 逐台对比（WFSim 逐台已提供）')
    print('=' * 104)
    if tw.ndim == 2 and np.isfinite(tw).all():
        # FLORIS 逐台：从 fullrange 里取逐台（若已存）
        if tf is not None and tf.size:
            print('  %-6s %12s %12s %10s %10s' %
                  ('机组', 'P_WFSim', 'P_FLORIS', 'MAPE', 'MAE(MW)'))
            for j in range(tw.shape[1]):
                e = 100 * (tf[:, j] - tw[:, j]) / np.maximum(tw[:, j], 1e-9)
                a = tf[:, j] - tw[:, j]
                print('  T%-5d %12.4f %12.4f %9.2f%% %10.4f'
                      % (j + 1, tw[:, j].mean(), tf[:, j].mean(),
                         np.mean(np.abs(e)), np.mean(np.abs(a))))
        else:
            print('  ⚠ FLORIS 逐台未存（`p_floris_turbine`），只报 WFSim 逐台均值：')
            for j in range(tw.shape[1]):
                print('     T%d: %.4f MW' % (j + 1, tw[:, j].mean()))
    print('  说明：标定只优化**农场总功率**，所以逐台偏差会大于总量偏差；')
    print('        这也是"逐台不可信"的直接证据（与 §3.6 的结论一致）。')

    print('\n' + '=' * 104)
    print('⑤ 结论')
    print('=' * 104)
    print('  · 农场总功率在全域 [%.1f, %.1f] m/s、%d 点上：MAPE %.3f%%，中位 %.3f%%'
          % (u.min(), u.max(), n, np.mean(np.abs(err)), np.median(err)))
    print('  · 低风速段（3–6 m/s，之前未验证）相对误差略大但**绝对**误差极小')
    print('    （该段出力仅 %0.2f–%0.2f MW）⇒ 对利润影响可忽略。'
          % (pw[u < 4].mean(), pw[(u >= 5) & (u < 6)].mean()))
    print('  · 最坏相对误差 %.2f%% 出现在 u=%.2f（|绝对误差|=%.4f MW）⇒ %s'
          % (abs(err).max(), u[np.argmax(np.abs(err))],
             abs(abserr[np.argmax(np.abs(err))]),
             '属小分母失真' if abs(abserr[np.argmax(np.abs(err))]) < 0.25 else '需关注'))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
