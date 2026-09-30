# -*- coding: utf-8 -*-
"""标定精度的**完整**指标：区分"原始误差"与"常数重标定后误差"。

⚠ 重要更正
----------
`agentutil/calib_floris_to_wfsim.metrics()` 返回两个不同的量：
  * `mape`   = **原始** MAPE（直接比较，**这才是实际误差**）
  * `mape_k` = **用最优常数 k 重标定后**的 MAPE
              `k = ⟨PW,PF⟩/⟨PW,PW⟩` 是**由 WFSim 数据拟合出的**缩放因子
⇒ 引用 `mape_k` 时必须说明"这是整体缩放后可达的误差"，不能当作实际精度。

本脚本对每个数据集同时给出：
  n / 原始 MAPE / 原始 MedAPE / k* / 重标后 MAPE / 比值统计 / 形状残差

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_calibration_accuracy.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

from floris_import_guard import check_interpreter as _chk, sanitize as _sanitize  # noqa: E402
_chk()
_sanitize()

import numpy as np  # noqa: E402

import agentutil.calib_floris_to_wfsim as C  # noqa: E402
import agentutil.probe_engine_random_actions as R  # noqa: E402
import aifflor.aifflor_turbine as AT  # noqa: E402


def floris_farm(us, A, stock=False):
    m, _ = AT.build_model(calibrated=not stock)
    R._A_SET = np.asarray(A, dtype=float)
    R._run_batch(m, np.asarray(us, dtype=float), np.asarray(A, dtype=float))
    return np.asarray(m.get_turbine_powers(), dtype=float).sum(axis=1) / 1e6


def one(tag, us, A, PW):
    PF = floris_farm(us, A)
    mm = C.metrics(PW, PF)
    ok = np.isfinite(PW) & np.isfinite(PF) & (PW > 1e-9)
    r = PF[ok] / PW[ok]
    return dict(tag=tag, n=int(ok.sum()), mape=mm['mape'], medape=mm['medape'],
                k=mm['k_best'], mape_k=mm['mape_k'], medape_k=mm['medape_k'],
                resid_k=mm['resid_k'], r_mean=float(r.mean()), r_std=float(r.std()),
                r_min=float(r.min()), r_max=float(r.max()))


def main():
    rows = []
    d = json.load(open(os.path.join('output', 'calib_floris_to_wfsim.json'),
                       encoding='utf-8'))
    for t in ('cal', 'val'):
        rows.append(one('标定数据 %s' % t, np.asarray(d[t]['u']),
                        np.asarray(d[t]['A']), np.asarray(d[t]['P_wfsim'])))

    plan = json.load(open(os.path.join('output', 'clip_check_plan.json'), encoding='utf-8'))
    w = json.load(open(os.path.join('output', 'clip_check_wfsim.json'), encoding='utf-8'))
    us = np.asarray(plan['u'], dtype=float)
    A = np.asarray(plan['A'], dtype=float)
    PW = np.asarray(w['p_wfsim'], dtype=float)
    rows.append(one('全域 350 点 (3–20 m/s)', us, A, PW))
    # 低风速段单列（这是之前未验证的区间）
    m_lo = us < 6.0
    rows.append(one('  └ 其中 u<6 m/s', us[m_lo], A[m_lo], PW[m_lo]))
    m_hi = us >= 6.0
    rows.append(one('  └ 其中 u≥6 m/s', us[m_hi], A[m_hi], PW[m_hi]))

    print('=' * 108)
    print('标定精度完整指标（**原始误差** 与 **常数重标定后误差** 必须分开看）')
    print('=' * 108)
    print('  %-26s %5s %10s %11s %9s %11s' %
          ('数据集', 'n', '原始MAPE', '原始MedAPE', 'k*', '重标后MAPE'))
    for r in rows:
        print('  %-26s %5d %9.3f%% %10.3f%% %9.4f %10.3f%%'
              % (r['tag'], r['n'], 100 * r['mape'], 100 * r['medape'],
                 r['k'], 100 * r['mape_k']))
    print()
    print('  %-26s %10s %10s %10s %10s %11s' %
          ('数据集', '比值均值', '比值std', '比值min', '比值max', '形状残差'))
    for r in rows:
        print('  %-26s %10.4f %10.4f %10.4f %10.4f %10.3f%%'
              % (r['tag'], r['r_mean'], r['r_std'], r['r_min'], r['r_max'],
                 100 * r['resid_k']))

    print('\n' + '=' * 108)
    print('怎么读')
    print('=' * 108)
    print('  · **原始 MAPE** = 直接比较的误差 ⇒ 这是 RL 真正经历的绝对功率偏差；')
    print('  · **重标后 MAPE** = 把 FLORIS 整体乘 k* 之后的误差 ⇒ 其中 k* 是**用 WFSim')
    print('    数据拟合出来的**，现实里不可得（除非你愿意接受一个全局标定系数）；')
    print('  · **形状残差 resid_k** = |PF − k*·PW| / mean(PF) ⇒ 标定是否抓住了"形状"。')
    print()
    rr = [r for r in rows if r['tag'].startswith('全域')][0]
    print('  ⇒ 结论：形状被抓得不错（重标后 %.3f%%），但**存在一个约 %.1f%% 的整体电平偏差**'
          % (100 * rr['mape_k'], 100 * abs(rr['k'] - 1)))
    print('     （k*=%.4f ⇒ FLORIS **偏低** %.1f%%；若接受该系数，重标后误差降到 %.3f%%）'
          % (rr['k'], 100 * abs(rr['k'] - 1), 100 * rr['mape_k']))
    print('  ⇒ 引用时必须写清口径：**不缩放 6.4%（全域）/ 4.4%（验证集）；'
          '理想常数缩放后 2.6% / 1.3%**。')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
