# -*- coding: utf-8 -*-
"""**能否"功率对得上"又"不产生负速度"？** —— 标定替代方案的参数空间扫描。

背景
----
现有标定（`agentutil/calib_floris_to_wfsim.py`）只优化**农场总功率 MAPE**
（验证集 4.448% / 常数重标定后 1.098%），但找到了一个**非物理**解：
尾流增长率 `k` 被压到默认的 1/5.3，逐台局地亏损累加到 **103.4%（T5 出现 u<0）**。
原因：**单标量目标有多个实现方式**，窄尾流只是其中一个。

本脚本系统扫描"实现方式"，看是否存在**功率同样对得上、但局地亏损物理合理**的解。

三个可独立调的自由度（都在自定义 operation model / wake 参数里）
------------------------------------------------------------------
1. **`c_ct`：CT 相对标定值的倍率**（`CT = clip(c_ct·4a(1−a), 1e-4, 0.9999)`）。
   ⚠ 这是**功率与动量解耦**——WFSim 自己也这么做（`Cp=4a(1−a)²` 而力用 `1.5×`
   ⇒ 其有效 `CT/(1−a)` 达 1.333）。**`c_ct` 不改变单机功率**（功率走 `Cp_eff` 表）。
2. **`k_mult`：尾流增长率倍率**（`k = k_mult·(ka·TI+kb)`）。
3. （只跑一个基准）标定参数本身 = `c_ct=1, k_mult=1`。

判据（三条同时满足才算"可用替代"）
----------------------------------
* **A 拟合**：验证集"常数重标定后 MAPE" ≤ 2%（与现有标定 1.098% 可比）；
* **B 物理**：流场最小速度 `u_min > 0`（无负速度）；
* **C 物理**：最大逐台局地亏损 `deficit_max < 0.9`（留 10% 余量）。

用法（不需 MATLAB；用已落盘的 WFSim 标定数据）
--------------------------------------------
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_power_match_alternatives.py
"""
import argparse
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


def install_ct_knob():
    """启用"CT 倍率"自由度。

    operation model 的 `thrust_coefficient` 本来就会读**模块全局 `R.CT_SCALE`**
    （默认 1.0）⇒ 本函数只需确保那个全局存在即可，无需替换方法。
    这实现的是"**功率/动量解耦**"：`CT` 被放大，而单机功率仍走 `Cp_eff` 表。
    """
    if not hasattr(R, 'CT_SCALE'):
        R.CT_SCALE = 1.0
    return R.CT_SCALE


def set_wake_spread(m, k_mult):
    """按倍率缩放尾流增长率：ka → k_mult·ka（kb 同步，保持"k = ka·TI+kb"的比值）。"""
    vm = m.core.wake.model_strings['velocity_model']
    base = dict(AT.CALIBRATED_WAKE_PARAMS)
    ka0 = base[('velocity_model', 'ka')]
    kb0 = base[('velocity_model', 'kb')]
    m.set_param(['wake', 'wake_velocity_parameters', vm, 'ka'], ka0 * k_mult)
    m.set_param(['wake', 'wake_velocity_parameters', vm, 'kb'], kb0 * k_mult)


def evaluate(m, us, A, W):
    """跑一批并给出 (功率指标, 物理指标)。"""
    R._A_SET = np.asarray(A, dtype=float)
    R._run_batch(m, np.asarray(us, dtype=float), np.asarray(A, dtype=float))
    pf = np.asarray(m.get_turbine_powers(), dtype=float).sum(axis=1) / 1e6
    mm = C.metrics(W, pf)
    ui = np.asarray(m.core.flow_field.u_initial_sorted)
    uu = np.asarray(m.core.flow_field.u_sorted)
    n_t = ui.shape[1]
    dmax = 0.0
    for j in range(n_t):
        d = (ui[:, j] - uu[:, j]) / np.maximum(ui[:, j], 1e-9)
        dmax = max(dmax, float(np.nanmax(d)))
    return dict(mape=mm['mape'], mape_k=mm['mape_k'], k_best=mm['k_best'],
                u_min=float(np.nanmin(uu)), deficit_max=dmax,
                n_neg=int((uu < 0).sum()))


def main():
    ap = argparse.ArgumentParser(description='功率对得上但不产生负速度的替代方案扫描')
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit('找不到 %s（标定数据）' % args.data)
    d = json.load(open(args.data, encoding='utf-8'))
    vu = np.asarray(d['val']['u'], dtype=float)
    vA = np.asarray(d['val']['A'], dtype=float)
    vW = np.asarray(d['val']['P_wfsim'], dtype=float)
    print('验证集 n=%d  (u 均值 %.2f, P_wfsim 均值 %.3f MW)'
          % (len(vu), vu.mean(), vW.mean()))

    install_ct_knob()
    print('\n扫描自由度: CT 倍率 c_ct × 尾流增长率倍率 k_mult')
    print('  ⚠ c_ct 只改**动量**（CT），不改单机功率（功率走 Cp_eff 表）⇒ 功率/动量解耦')
    print('  判据: 常数重标定后 MAPE ≤ 2%% 且 u_min > 0 且 deficit_max < 0.9\n')

    c_list = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
    k_list = [1.0, 2.0, 4.0, 8.0, 16.0]
    print('  %-7s %-7s %10s %12s %11s %9s %7s %s'
          % ('c_ct', 'k_mult', 'MAPE', '常数重标后', 'deficit_max', 'u_min', 'u<0', '判定'))
    rows = []
    for c_ct in c_list:
        for k_mult in k_list:
            R.CT_SCALE = c_ct
            m, _ = AT.build_model(calibrated=True)
            set_wake_spread(m, k_mult)
            r = evaluate(m, vu, vA, vW)
            r.update(c_ct=c_ct, k_mult=k_mult)
            fit = r['mape_k'] <= 0.02
            phys = (r['u_min'] > 0.0) and (r['deficit_max'] < 0.9)
            verdict = ('**可用**' if (fit and phys) else
                       ('拟合差' if not fit else ('非物理' if not phys else '?')))
            rows.append((r, fit, phys, verdict))
            print('  %-7.2f %-7.1f %9.3f%% %11.3f%% %11.4f %9.3f %7d %s'
                  % (c_ct, k_mult, 100 * r['mape'], 100 * r['mape_k'],
                     r['deficit_max'], r['u_min'], r['n_neg'], verdict))

    good = [r for r, f, p, _ in rows if f and p]
    print('\n' + '=' * 96)
    print('结论')
    print('=' * 96)
    print('  基准（现有标定 c_ct=1, k_mult=1）: 常数重标定后 %.3f%%，'
          'deficit_max=%.3f, u_min=%.3f, u<0 %d 点'
          % (100 * rows[0][0]['mape_k'], rows[0][0]['deficit_max'],
             rows[0][0]['u_min'], rows[0][0]['n_neg']))
    if good:
        best = min(good, key=lambda r: r['mape_k'])
        print('  ✅ 找到 %d 个"拟合≤2%% 且物理合理"的解；最优:' % len(good))
        print('     c_ct=%.2f  k_mult=%.1f  ⇒ 常数重标定后 %.3f%%，'
              'deficit_max=%.3f, u_min=%.3f'
              % (best['c_ct'], best['k_mult'], 100 * best['mape_k'],
                 best['deficit_max'], best['u_min']))
        print('     ⇒ 提高 CT（动量）＋放宽尾流即可两者兼顾，**不必用窄尾流**。')
    else:
        print('  ❌ 本网格内没有两全的解 ⇒ 需要换参数化（如改 turbulence/deflection 模型）')
    print('=' * 96)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump([r for r, _, _, _ in rows], open(args.json, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print('已写入 %s' % args.json)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
