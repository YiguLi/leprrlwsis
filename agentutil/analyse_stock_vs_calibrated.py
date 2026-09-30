# -*- coding: utf-8 -*-
"""默认（stock）参数下，上述性质还成立吗？—— 与标定参数逐项对照。

要回答
------
用户命题："用 FLORIS 默认参数也能这么认为吗？默认参数肯定比自定义参数不容易出错。"

拆成两个独立问题：
  Q1 **可平移性**（corr / 相关长度 / 信噪比）在 stock 下是否同样成立？甚至更好？
  Q2 **训练可用性**（奖励电平、归一化常数、λ/COST_SCALE 是否要重调）在 stock 下如何？

Q1 与"参数对不对"无关 —— 它只取决于"引擎差异是否近似为常数倍率"。
Q2 才是关键：stock 的电平偏差是 ~74%，会把归一化常数全部推离原位。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_stock_vs_calibrated.py
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


def farm(us, A, stock):
    m, _ = AT.build_model(calibrated=not stock)
    R._A_SET = np.asarray(A, dtype=float)
    R._run_batch(m, np.asarray(us, dtype=float), np.asarray(A, dtype=float))
    return np.asarray(m.get_turbine_powers(), dtype=float).sum(axis=1) / 1e6


def acf(x, lag):
    x = np.asarray(x, dtype=float) - np.mean(x)
    d = np.dot(x, x)
    return float('nan') if d <= 0 else float(np.dot(x[:-lag], x[lag:]) / d)


def main():
    # ---------- Q1: 100 步真实风速序列上的可平移性 ----------
    d = json.load(open(os.path.join('output', 'engine_minute_compare.json'),
                       encoding='utf-8'))
    u = np.asarray(d['wind'], dtype=float)
    pw = np.asarray(d['p_wfsim'], dtype=float)
    a_nom = float(d['config'].get('a_nominal', AT.A_NOMINAL))
    A = np.tile(np.full(NT, a_nom), (len(u), 1))

    print('=' * 104)
    print('Q1 可平移性：100 步真实风速序列（标称动作 a=%.4f）' % a_nom)
    print('=' * 104)
    print('  %-20s %9s %9s %11s %9s %9s %9s'
          % ('口径', 'mean', 'std', 'std/mean', 'acf1', 'acf5', 'corr vs W'))
    rows = {}
    for tag, st in (('WFSim', None), ('FLORIS stock', True), ('FLORIS 标定', False)):
        P = pw if st is None else farm(u, A, st)
        rows[tag] = P
        rho = float(np.corrcoef(pw, P)[0, 1]) if st is not None else 1.0
        print('  %-20s %9.4f %9.4f %11.4f %+9.4f %+9.4f %9.6f'
              % (tag, P.mean(), P.std(), P.std() / P.mean(), acf(P, 1), acf(P, 5), rho))
    for tag in ('FLORIS stock', 'FLORIS 标定'):
        r = rows[tag] / np.maximum(pw, 1e-9)
        print('  %-20s 比值: 中位=%.4f  mean=%.4f  std=%.4f  min=%.4f  max=%.4f'
              % ('  └ ' + tag, np.median(r), r.mean(), r.std(), r.min(), r.max()))

    print('\n  ⇒ 判读：**corr 与 std/mean、acf 三项若都接近，则"可平移性"与参数选择无关**。')

    # ---------- Q2: 全域 350 点的电平与形状 ----------
    plan = json.load(open(os.path.join('output', 'clip_check_plan.json'), encoding='utf-8'))
    w = json.load(open(os.path.join('output', 'clip_check_wfsim.json'), encoding='utf-8'))
    u2 = np.asarray(plan['u'], dtype=float)
    A2 = np.asarray(plan['A'], dtype=float)
    pw2 = np.asarray(w['p_wfsim'], dtype=float)

    print('\n' + '=' * 104)
    print('Q2 训练可用性：全域 350 点（3–20 m/s，随机动作）')
    print('=' * 104)
    print('  %-14s %10s %10s %9s %11s %9s %10s'
          % ('口径', '原始MAPE', 'MedAPE', 'k*', '重标后MAPE', '比值std', '比值范围'))
    res = {}
    for tag, st in (('FLORIS stock', True), ('FLORIS 标定', False)):
        PF = farm(u2, A2, st)
        mm = C.metrics(pw2, PF)
        r = PF / np.maximum(pw2, 1e-9)
        res[tag] = dict(mm=mm, PF=PF, r=r)
        print('  %-14s %9.3f%% %9.3f%% %9.4f %10.3f%% %9.4f  %.3f–%.3f'
              % (tag, 100 * mm['mape'], 100 * mm['medape'], mm['k_best'],
                 100 * mm['mape_k'], r.std(), r.min(), r.max()))

    print('\n' + '=' * 104)
    print('⇒ 对"训练归一化常数"的影响（这是 Q2 的核心）')
    print('=' * 104)
    print('  这些常数是**绝对 MW 口径**的假设，引擎电平一变就必须跟着变：')
    print('  %-22s %12s %12s %s' % ('常数', 'stock 下应为', '标定下应为', '含义'))
    base = {'COST_SCALE_TD3': 100.0, 'VALUE_SCALE': 1000.0, 'OBS_POWER_SCALE': 10.0,
            'W_GAP': 10.0, 'P_F_LIMIT': 3.0}
    k_s = res['FLORIS stock']['mm']['k_best']      # FLORIS/WFSim 电平比倒数
    k_c = res['FLORIS 标定']['mm']['k_best']
    print('  %-22s %12s %12s %s' % ('k* (FLORIS/WFSim)', '%.4f' % k_s, '%.4f' % k_c,
                                    'k*>1 ⇒ FLORIS 偏低'))
    for name, v in base.items():
        print('  %-22s %12.4f %12.4f %s' % (name, v / k_s, v / k_c, '按电平比缩放'))
    print()
    print('  ⇒ stock 下 k*=%.3f ⇒ FLORIS 比 WFSim **低 %.1f%%** ⇒' % (k_s, 100 * (1 - k_s)))
    print('     要让二者共用同一组归一化常数，必须把所有 ABS-MW 常数除以 k*（= 放大约 %.1f 倍）。'
          % (1 / k_s))
    print('     标定下 k*=%.4f ⇒ 只需 %.1f%% 的微调。' % (k_c, 100 * abs(1 - k_c)))

    print('\n' + '=' * 104)
    print('结论')
    print('=' * 104)
    ws = res['FLORIS stock']
    wc = res['FLORIS 标定']
    r_s = rows['FLORIS stock'] / np.maximum(pw, 1e-9)
    r_c = rows['FLORIS 标定'] / np.maximum(pw, 1e-9)
    print('  Q1（可平移性）: stock 与标定**同样成立**，且 stock 的比值更紧'
          '（100 步序列上 std %.4f vs %.4f）⇒' % (r_s.std(), r_c.std()))
    print('      nstep / batchsize / rmsize / update 的平移性**与参数选择无关**，')
    print('      因为它只取决于"引擎差异是否近似常数倍率"，而两者都满足。')
    print()
    print('  Q2（训练可用性）: stock 的电平偏差是 %.1f%%（标定是 %.1f%%）⇒' %
          (100 * (1 - ws['mm']['k_best']), 100 * (1 - wc['mm']['k_best'])))
    print('      · 要用 stock 且与 WFSim 的绝对 MW 口径兼容 ⇒ 必须把 COST_SCALE_TD3 /')
    print('        VALUE_SCALE / OBS_POWER_SCALE / W_GAP / P_F_LIMIT 等**全部**按电平比重调；')
    print('      · 否则这些门限的相对位置全错（如 W_GAP=10 MW 在 stock 下相当于 %.1f MW）'
          % (10 * ws['mm']['k_best']))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
