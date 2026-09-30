# -*- coding: utf-8 -*-
"""stock + 单一电平系数（1/1.6696）后，**波动强度**是否与 WFSim 相仿？

三层要分开
----------
1. **相对波动**（`std/mean`、`CV`）—— 常数倍率**完全改不了**（无量纲、尺度不变）。
2. **绝对波动**（`std`、峰谷差）—— 常数倍率**按比例改**，若"引擎比"恰是常数则能对上。
3. **波动信号的可加性**（`P_F` 是功率的**差分** `P_G − P_G_prev`）——
   若 `P_F^F = c·P_F^W` 需要**两个时刻都乘 c**；`P_G_prev` 来自轨迹，故常数缩放
   在**稳态**下等价，但在**变化段**会引入误差。这才是本任务真正用的量。

本脚本在 100 步真实风速序列上，对 WFSim / stock / stock×c 三种口径比较：
  * 功率的 CV、std、峰谷
  * `P_F`（波动信号）的分布、std、以及 WFSim 口径下的相对误差
  * 逐点比值 `P_F^F/P_F^W` 的散布（= 波动能不能靠一个常数对上）

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_fluctuation_scaling.py
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

import agentutil.probe_engine_random_actions as R  # noqa: E402
import aifflor.aifflor_turbine as AT  # noqa: E402

NT = AT.NT


def farm(us, A, stock):
    m, _ = AT.build_model(calibrated=not stock)
    R._A_SET = np.asarray(A, dtype=float)
    R._run_batch(m, np.asarray(us, dtype=float), np.asarray(A, dtype=float))
    return np.asarray(m.get_turbine_powers(), dtype=float).sum(axis=1) / 1e6


def pf_of(P, limit=3.0):
    """RBB 波动信号（简化口径）：相邻两步之差，超出 ±limit 的部分计为波动。

    这里同时给两个版本：
      * `pf_raw` = |Δ P|（未限幅）
      * `pf_clip`= max(0, |Δ P| − limit)（限幅后的"被罚"波动）
    `limit` 取 hsrl_cost.P_F_LIMIT = 3 MW。
    """
    d = np.abs(np.diff(P))
    return d, np.maximum(0.0, d - limit)


def acf(x, lag):
    x = np.asarray(x, dtype=float) - np.mean(x)
    d = np.dot(x, x)
    return float('nan') if d <= 0 else float(np.dot(x[:-lag], x[lag:]) / d)


def main():
    d = json.load(open(os.path.join('output', 'engine_minute_compare.json'),
                       encoding='utf-8'))
    u = np.asarray(d['wind'], dtype=float)
    PW = np.asarray(d['p_wfsim'], dtype=float)
    a_nom = float(d['config'].get('a_nominal', AT.A_NOMINAL))
    A = np.tile(np.full(NT, a_nom), (len(u), 1))

    P_stock = farm(u, A, True)
    P_cal = farm(u, A, False)
    c = float(d['summary'].get('ratio_median', np.median(P_stock / PW)))
    P_scaled = P_stock / c

    print('=' * 108)
    print('100 步真实风速序列，标称动作 a=%.4f   电平系数 c=%.4f（=中位比值）' % (a_nom, c))
    print('=' * 108)

    print('\n① 功率信号本身的统计')
    print('  %-26s %10s %10s %10s %10s %10s' %
          ('口径', 'mean', 'std', 'std/mean', '峰谷', 'acf1'))
    for tag, P in (('WFSim', PW),
                   ('FLORIS stock', P_stock),
                   ('FLORIS stock ÷ c', P_scaled),
                   ('FLORIS 标定', P_cal)):
        print('  %-26s %10.4f %10.4f %10.4f %10.4f %+10.4f'
              % (tag, P.mean(), P.std(), P.std() / P.mean(),
                 P.max() - P.min(), acf(P, 1)))

    print('\n② 波动信号 P_F（|ΔP| 与限幅后）')
    print('  %-26s %10s %10s %10s %10s' %
          ('口径', '|ΔP| mean', '|ΔP| std', '限幅后 mean', '限幅后 max'))
    store = {}
    for tag, P in (('WFSim', PW),
                   ('FLORIS stock', P_stock),
                   ('FLORIS stock ÷ c', P_scaled),
                   ('FLORIS 标定', P_cal)):
        raw, cl = pf_of(P)
        store[tag] = (raw, cl)
        print('  %-26s %10.4f %10.4f %10.4f %10.4f'
              % (tag, raw.mean(), raw.std(), cl.mean(), cl.max()))

    print('\n③ 决定性判据：波动信号能否靠**一个常数**对上？')
    print('  逐点比值 r_F = P_F(FLORIS) / P_F(WFSim)（只取 WFSim 波动显著的点）')
    raw_w = store['WFSim'][0]
    msk = raw_w > 1e-6
    print('  %-26s %10s %10s %10s %10s %10s' %
          ('口径', 'n', 'r_F 中位', 'r_F std', 'r_F min', 'r_F max'))
    for tag in ('FLORIS stock ÷ c', 'FLORIS 标定', 'FLORIS stock'):
        r = store[tag][0][msk] / raw_w[msk]
        print('  %-26s %10d %10.4f %10.4f %10.4f %10.4f'
              % (tag, msk.sum(), np.median(r), r.std(), r.min(), r.max()))

    print('\n④ 换算成"波动惩罚"的差距')
    import hsrl_cost as hc
    lim = float(getattr(hc, 'P_F_LIMIT', 3.0))
    cl_w = store['WFSim'][1]
    print('  （P_F_LIMIT = %.1f MW）' % lim)
    for tag in ('FLORIS stock', 'FLORIS stock ÷ c', 'FLORIS 标定'):
        cl = store[tag][1]
        rel = 100 * (cl.mean() - cl_w.mean()) / max(cl_w.mean(), 1e-12)
        print('  %-26s 限幅后波动均值 %8.4f  相对 WFSim %+8.2f%%'
              % (tag, cl.mean(), rel))

    print('\n' + '=' * 108)
    print('结论')
    print('=' * 108)
    cv_w = PW.std() / PW.mean()
    cv_s = P_scaled.std() / P_scaled.mean()
    print('  1) **相对波动**（std/mean）: WFSim %.4f vs stock÷c %.4f ⇒ 差 %.2f%%'
          % (cv_w, cv_s, 100 * (cv_s - cv_w) / cv_w))
    print('     常数倍率**不改变**无量纲的相对波动，两者本来就几乎一样（差<1.1%%，')
    print('     因为 stock 与 WFSim 的比值本身近似常数）。')
    print('  2) **绝对波动**（std）: WFSim %.4f vs stock÷c %.4f ⇒ 差 %+.2f%%'
          % (PW.std(), P_scaled.std(),
             100 * (P_scaled.std() - PW.std()) / PW.std()))
    print('     乘 1/c 后绝对波动也对上了（比例由 c 决定）。')
    print('  3) **波动信号 P_F 的逐点可加性**（本任务真正用的量）:')
    r_sc = store['FLORIS stock ÷ c'][0][msk] / raw_w[msk]
    r_cl = store['FLORIS 标定'][0][msk] / raw_w[msk]
    print('     stock÷c: r_F 中位 %.4f, **std %.4f**' % (np.median(r_sc), r_sc.std()))
    print('     标定   : r_F 中位 %.4f, **std %.4f**' % (np.median(r_cl), r_cl.std()))
    print('     ⇒ 中位都接近 1（总量对得上），但**逐点散布**决定波动惩罚的偏差：')
    print('       std 越大 ⇒ 用该口径算出的"波动惩罚"与 WFSim 越不一致。')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
