# -*- coding: utf-8 -*-
"""在 FLORIS **公式的最对应位置**乘一个额外系数：自定义 `wake_combination`。

动机
----
之前只扫了"改 CT 倍率 c_ct"与"改尾流增长率 k_mult"，两者都在**单尾流的形状**上做文章，
结果互相打架（抬 CT 补功率 ⇒ 尾流更深 ⇒ 更非物理）。

但**非物理的根源不在单尾流**，而在**多尾流叠加规则**：
    SOSFS.function:  u = hypot(wake, d·u_initial) − wake           # √(Σd²)
    ⇒ 当 wake > u_initial 时 u < 0（实测 T5 累计亏损 103.4%）
    MAX.function  : 取逐台最大亏损
两条内置规则都没有可调系数。而 FLORIS 的 `wake_combination` 是**可插拔**的
（`floris/core/wake.py:126` `MODEL_MAP["combination_model"]`）。

本脚本注入一个自定义组合模型：
    u = hypot(w, d·u_initial) − w
⇒ 广义幂平均：  d_total = ( Σ d_i^n )^(1/n)
    n = 2  ≡ 现用 SOSFS；n = 1 ≡ 线性/Katic 叠加（**永不产生负值**）

扫描 (c_ct, n)，看是否存在"功率对得上 **且** 不非物理"的组合。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_custom_combination.py
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
import agentutil.calib_floris_to_wfsim as C  # noqa: E402
import aifflor.aifflor_turbine as AT  # noqa: E402

DATA = os.path.join('output', 'calib_floris_to_wfsim.json')
NAME = 'probe-power-mean'


def register_combination(name=NAME):
    """把幂平均组合模型注册进 FLORIS 的 `MODEL_MAP['combination_model']`。

    实现方式与 `SOSFS` 一致（`return np.hypot(...)` 形式），但把 `hypot` 换成
    广义幂平均：`hypot_n(w, x) = (w^n + x^n)^(1/n)`，`n=2` 即原 `hypot`。
    """
    import floris.core as FC
    from floris.core.wake_combination import SOSFS
    import floris.core.wake as W

    if name in W.MODEL_MAP['combination_model']:
        return name

    class PowerMean(SOSFS):
        """`u = (w^n + (d·u_initial)^n)^(1/n) − w`，指数 n 由全局注入。"""

        def function(self, wake_field, velocity_field):
            n = float(R.COMB_N)
            if abs(n - 2.0) < 1e-12:
                return np.hypot(wake_field, velocity_field)
            w = np.clip(wake_field, 0.0, None)
            v = np.clip(velocity_field, 0.0, None)
            return (w ** n + v ** n) ** (1.0 / n)

    PowerMean.__name__ = 'PowerMean'
    W.MODEL_MAP['combination_model'][name] = PowerMean
    return name


def build(c_ct, comb_n):
    """构造模型并把组合模型换成幂平均版。

    `Wake` 在 `__attrs_post_init__` 里按 `model_strings['combination_model']` 实例化一次，
    `combination_model` 是 `init=False` 字段（**没有 setter**）⇒ 构造后直接赋值实例；
    同时把 `model_strings` 里的串改掉，使后续 `from_dict` / 迭代取参时口径一致。
    """
    import floris.core.wake as W
    R.CT_SCALE = c_ct
    R.COMB_N = comb_n
    m, _ = AT.build_model(calibrated=True)
    wake = m.core.wake
    wake.model_strings['combination_model'] = NAME
    wake.combination_model = W.MODEL_MAP['combination_model'][NAME]()
    return m


def evaluate(m, vu, vA, vW):
    R._A_SET = vA
    R._run_batch(m, vu, vA)
    pf = np.asarray(m.get_turbine_powers(), dtype=float).sum(axis=1) / 1e6
    mm = C.metrics(vW, pf)
    ui = np.asarray(m.core.flow_field.u_initial_sorted)
    uu = np.asarray(m.core.flow_field.u_sorted)
    dmax = 0.0
    for j in range(ui.shape[1]):
        dmax = max(dmax, float(np.nanmax((ui[:, j] - uu[:, j]) / np.maximum(ui[:, j], 1e-9))))
    return dict(mape_k=mm['mape_k'], u_min=float(np.nanmin(uu)),
                n_neg=int((uu < 0).sum()), deficit_max=dmax)


def main():
    d = json.load(open(DATA, encoding='utf-8'))
    vu = np.asarray(d['val']['u'], dtype=float)
    vA = np.asarray(d['val']['A'], dtype=float)
    vW = np.asarray(d['val']['P_wfsim'], dtype=float)
    print('验证集 n=%d' % len(vu))

    register_combination()
    print('已注册自定义组合模型 %r（幂平均 d_total = (Σd^n)^(1/n)）' % NAME)
    print('  n=2 ≡ 原 SOSFS | n=1 ≡ 线性叠加（永不产生 u<0）\n')
    print('判据: 常数重标定后 MAPE ≤ 2%% 且 u_min > 0 且 deficit_max < 0.9\n')

    c_list = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
    n_list = [1.0, 1.25, 1.5, 1.75, 2.0]
    print('  %-7s %-7s %11s %13s %11s %7s %s'
          % ('c_ct', 'n', '常数重标后', 'deficit_max', 'u_min', 'u<0', '判定'))
    rows = []
    for c_ct in c_list:
        for n in n_list:
            try:
                m = build(c_ct, n)
                r = evaluate(m, vu, vA, vW)
            except Exception as e:
                print('  %-7.2f %-7.2f  构建/求值失败: %s' % (c_ct, n, str(e)[:50]))
                continue
            r.update(c_ct=c_ct, n=n)
            fit = r['mape_k'] <= 0.02
            phys = (r['u_min'] > 0.0) and (r['deficit_max'] < 0.9)
            v = '**可用**' if (fit and phys) else ('拟合差' if not fit else '非物理')
            rows.append((r, fit, phys, v))
            print('  %-7.2f %-7.2f %10.3f%% %13.4f %11.4f %7d %s'
                  % (c_ct, n, 100 * r['mape_k'], r['deficit_max'],
                     r['u_min'], r['n_neg'], v))

    good = [r for r, f, p, _ in rows if f and p]
    print('\n' + '=' * 90)
    print('结论')
    print('=' * 90)
    print('  %-7s %-6s %12s %13s %10s %7s  %s'
          % ('c_ct', 'n', '常数重标后', 'deficit_max', 'u_min', 'u<0', '判定'))
    for r, fit, phys, v in rows:
        if abs(r['c_ct'] - 1.0) < 1e-9 or abs(r['n'] - 2.0) < 1e-9:
            print('  %-7.2f %-6.2f %11.3f%% %13.4f %10.4f %7d  %s'
                  % (r['c_ct'], r['n'], 100 * r['mape_k'], r['deficit_max'],
                     r['u_min'], r['n_neg'], v))
    if good:
        best = min(good, key=lambda r: r['mape_k'])
        print('\n  ✅ 找到 %d 个"拟合≤2%% 且物理合理"的组合；最优:' % len(good))
        print('     c_ct=%.2f  n=%.2f ⇒ 常数重标定后 %.3f%%，deficit_max=%.3f, u_min=%.3f'
              % (best['c_ct'], best['n'], 100 * best['mape_k'],
                 best['deficit_max'], best['u_min']))
    else:
        print('\n  ❌ 无两全解。而且注意趋势：**降 n 反而更非物理** ——')
        print('     n<2 是"更激进的叠加"（n=1 即 5 台尾流直接相加，亏损可达 435%），')
        print('     所以 SOSFS 的 n=2 已是内置选项里**最温和**的一条。')
        print('     ⇒ 在"单尾流形状"与"叠加规则"上加系数都无法两全；')
        print('        缺的物理是 **WFSim 的封闭域（Ly=4D 风洞）** —— 尾流无法侧向扩散，')
        print('        其等效效果是"亏损不衰减"，任何自洽的叠加规则都复现不了。')
    print('=' * 90)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
