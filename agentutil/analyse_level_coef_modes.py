# -*- coding: utf-8 -*-
"""三种口径的全域对比：标定 / stock+电平系数 / 标定+电平系数。

回答"只用电平系数"这条路到底把功率和波动对齐到什么程度。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_level_coef_modes.py
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

NT = AT.NT


def farm(us, A, calibrated, scale):
    m, _ = AT.build_model(calibrated=calibrated)
    R._A_SET = np.asarray(A, dtype=float)
    R._run_batch(m, np.asarray(us, dtype=float), np.asarray(A, dtype=float))
    P = np.asarray(m.get_turbine_powers(), dtype=float).sum(axis=1) / 1e6
    return P * float(scale)


def line(tag, PW, PF):
    mm = C.metrics(PW, PF)
    r = PF / np.maximum(PW, 1e-9)
    print('  %-28s %9.3f%% %9.3f%% %8.4f %10.3f%% %9.4f  %.3f–%.3f'
          % (tag, 100 * mm['mape'], 100 * mm['medape'], mm['k_best'],
             100 * mm['mape_k'], r.std(), r.min(), r.max()))
    return mm


def main():
    plan = json.load(open(os.path.join('output', 'clip_check_plan.json'), encoding='utf-8'))
    w = json.load(open(os.path.join('output', 'clip_check_wfsim.json'), encoding='utf-8'))
    u = np.asarray(plan['u'], dtype=float)
    A = np.asarray(plan['A'], dtype=float)
    PW = np.asarray(w['p_wfsim'], dtype=float)
    c = AT.LEVEL_COEF

    print('=' * 112)
    print('① 全域 350 点（3–20 m/s，随机动作）   LEVEL_COEF = %.9f (= 1/%.4f)'
          % (c, AT.LEVEL_REF_K))
    print('=' * 112)
    print('  %-28s %10s %9s %8s %11s %9s  %s'
          % ('口径', '原始MAPE', 'MedAPE', 'k*', '重标后MAPE', '比值std', '比值范围'))
    # ⚠ 参数名必须与 `AT.build_model(calibrated=...)` 一致：
    #   第一版把这里写成 `farm(u, A, stock, sc)` 并传 `stock` 布尔量，导致标签与
    #   实际模型互换（真实性的独立锚点：stock 无系数 = 68.476% MAPE ⇔ §2 的
    #   k*=1.6696；标定 无系数 = 6.422% ⇔ 已登记的 6.4%）。
    modes = [
        ('FLORIS stock（原厂默认，无系数）', False, 1.0),
        ('FLORIS stock × 电平系数', False, c),
        ('FLORIS 标定（6 参数，无系数）', True, 1.0),
        ('FLORIS 标定 × 电平系数', True, c),
    ]
    got = {}
    for tag, cal, sc in modes:
        got[tag] = farm(u, A, cal, sc)
        line(tag, PW, got[tag])

    print('\n' + '=' * 112)
    print('② 100 步真实风速序列上的**波动**对照（标称动作）')
    print('=' * 112)
    d = json.load(open(os.path.join('output', 'engine_minute_compare.json'),
                       encoding='utf-8'))
    u1 = np.asarray(d['wind'], dtype=float)
    PW1 = np.asarray(d['p_wfsim'], dtype=float)
    a_nom = float(d['config'].get('a_nominal', AT.A_NOMINAL))
    A1 = np.tile(np.full(NT, a_nom), (len(u1), 1))
    LIM = 3.0
    print('  %-28s %9s %9s %9s %11s %9s %11s'
          % ('口径', 'mean', 'std', 'std/mean', '|ΔP|均值', '限幅后均值', 'vs WFSim'))
    ref_cl = np.maximum(0.0, np.abs(np.diff(PW1)) - LIM).mean()
    print('  %-28s %9.4f %9.4f %9.4f %9.4f %9.4f %11s'
          % ('WFSim（基准）', PW1.mean(), PW1.std(), PW1.std() / PW1.mean(),
             np.abs(np.diff(PW1)).mean(), ref_cl, '—'))
    for tag, cal, sc in modes:
        P = farm(u1, A1, cal, sc)
        raw = np.abs(np.diff(P)).mean()
        cl = np.maximum(0.0, np.abs(np.diff(P)) - LIM).mean()
        print('  %-28s %9.4f %9.4f %9.4f %9.4f %9.4f %+10.2f%%'
              % (tag, P.mean(), P.std(), P.std() / P.mean(), raw, cl,
                 100 * (cl - ref_cl) / max(ref_cl, 1e-12)))

    print('\n' + '=' * 112)
    print('③ 锚点')
    print('=' * 112)
    m, _ = AT.build_model(calibrated=False)
    base = AT.farm_power(m, 10.0, [1 / 3] * NT)
    m2, _ = AT.build_model(calibrated=True)
    base2 = AT.farm_power(m2, 10.0, [1 / 3] * NT)
    print('  u=10, a=1/3 :  stock %.6f | stock×系数 %.6f | 标定 %.6f | 标定×系数 %.6f'
          % (base, base * c, base2, base2 * c))
    print('  WFSim 参考（同点，§2 已登记）: 3.4462 MW')
    print()
    for tag, v in (('stock', base), ('stock×系数', base * c),
                   ('标定', base2), ('标定×系数', base2 * c)):
        print('  %-14s 相对 WFSim 参考: %+.2f%%' % (tag, 100 * (v / 3.4462 - 1)))

    print('\n' + '=' * 112)
    print('结论')
    print('=' * 112)
    best = min(got, key=lambda t: abs(C.metrics(PW, got[t])['mape']))
    print('  全域原始 MAPE 最小者: **%s** (%.3f%%)' % (best, 100 * C.metrics(PW, got[best])['mape']))
    print()
    print('  · 「stock × 系数」= 你指定的方案：物理用 FLORIS 原厂默认，只在出口乘一个常数。')
    print('  · 「stock × 系数」与「标定」谁的 MAPE 更小、谁的波动更准，见上面两表。')
    print('  · 两者都只引入**一处**配置差异（一个是 6 个尾流参数，一个是 1 个系数）。')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
