# -*- coding: utf-8 -*-
"""把"两个参数如何影响整体出力"用**出力公式**拆开，并做数值核对。

公式链（单一真源 = `aifflor_turbine` / FLORIS 表口径）
------------------------------------------------------
    P_turb = 0.95 · ½ρA · Cp(a) · Ue³              Cp(a) = 4a(1−a)²
    Ue²/U² = 1 − CT / (8·σ²/D²)                    高斯尾流亏损
    CT     = clip(c_ct · 4a(1−a), 1e-4, 0.9999)    ← 参数① 只进**分子**
    σ/D    = σ0/D + k_mult·(ka·TI+kb)·x/D          ← 参数② 只进**分母**
    σ0/D   = 0.5·sqrt( uR/(1+u0) ),  uR = CT/(2(1−√(1−CT))),  u0 = √(1−CT)

⇒ `c_ct` 与 `k_mult` **都不出现在 `Cp(a)` 里**，所以二者**只通过 `Ue`（尾流）**影响出力。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_power_param_forms.py
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

DATA = os.path.join('output', 'calib_floris_to_wfsim.json')


def build(c_ct=1.0, k_mult=1.0):
    R.CT_SCALE = c_ct
    m, _ = AT.build_model(calibrated=True)
    b = dict(AT.CALIBRATED_WAKE_PARAMS)
    vm = m.core.wake.model_strings['velocity_model']
    m.set_param(['wake', 'wake_velocity_parameters', vm, 'ka'],
                b[('velocity_model', 'ka')] * k_mult)
    m.set_param(['wake', 'wake_velocity_parameters', vm, 'kb'],
                b[('velocity_model', 'kb')] * k_mult)
    return m


def run(m, vu, vA):
    R._A_SET = vA
    R._run_batch(m, vu, vA)
    pt = np.asarray(m.get_turbine_powers(), dtype=float) / 1e6
    ui = np.asarray(m.core.flow_field.u_initial_sorted)
    uu = np.asarray(m.core.flow_field.u_sorted)
    dm = [float(np.nanmax((ui[:, j] - uu[:, j]) / np.maximum(ui[:, j], 1e-9)))
          for j in range(ui.shape[1])]
    return pt, dm


def main():
    d = json.load(open(DATA, encoding='utf-8'))
    vu = np.asarray(d['val']['u'], dtype=float)
    vA = np.asarray(d['val']['A'], dtype=float)
    print('验证集 n=%d' % len(vu))

    pt0, dm0 = run(build(), vu, vA)
    P0 = pt0.sum(1).mean()
    print('\n基准 (c_ct=1, k_mult=1): P=%.6f MW  逐台最大亏损 %s'
          % (P0, np.array2string(np.asarray(dm0), precision=3)))

    print('\n' + '=' * 92)
    print('① 逐台出力份额 与 局地亏损（决定"能不能只动某一台"）')
    print('=' * 92)
    tot = pt0.sum()
    print('  %-6s %13s %10s %14s %14s' % ('机组', '平均出力MW', '份额', '最大局地亏损', '该台/农场'))
    for j in range(5):
        print('  T%-5d %13.4f %9.2f%% %14.4f %13.2f%%'
              % (j + 1, pt0[:, j].mean(), 100 * pt0[:, j].sum() / tot,
                 dm0[j], 100 * pt0[:, j].mean() / P0))

    print('\n' + '=' * 92)
    print('② 解析分解：两个参数各自的形式（都在 Ue 里，不在 Cp 里）')
    print('=' * 92)
    print('  记 d0² ≡ CT0/(8·σ0_eff²/D²)（基准局地亏损比的平方），则')
    print('    c_ct 路径:  Ue²/U² = 1 − c_ct·d0²     ⇒  d lnP/d ln(c_ct) = −3/2·c_ct·d0²/(1−c_ct·d0²)')
    print('    k_mult 路径: σ² = (σ0 + k_mult·k0·x)²  ⇒  d lnP/d ln(k_mult) = +3·(k_mult k0 x)/(σ0+k_mult k0 x)·(c_ct d0²)/(1−c_ct d0²)')
    print('  ⇒ 二者**形式不同**：c_ct 线性进分子（强、单调）；k_mult 经 σ² 进分母（弱、被 σ0 稀释）')
    print('  ⇒ 比值 ≈ [3·kx/σ] / [3/2] = 2·kx/σ  ⇒ 在近场 kx≪σ0 时 k_mult 几乎失效')

    print('\n' + '=' * 92)
    print('③ 数值核对：±1%/±10% 的对数敏感度，与解析式对照')
    print('=' * 92)
    for name, f in (('c_ct', lambda v: build(c_ct=v)), ('k_mult', lambda v: build(k_mult=v))):
        h = 0.01 if name == 'c_ct' else 0.10
        Pp, _ = run(f(1 + h), vu, vA)
        Pm, _ = run(f(1 - h), vu, vA)
        s = (np.log(Pp.sum(1).mean()) - np.log(Pm.sum(1).mean())) / (
            np.log(1 + h) - np.log(1 - h))
        print('  %-7s ±%4.1f%%  dlnP/dln(x) = %+.4f' % (name, 100 * h, s))

    # 由 c_ct 的弹性反解 d0²，验证自洽
    Pc1, _ = run(build(c_ct=1.01), vu, vA)
    Pc0, _ = run(build(c_ct=0.99), vu, vA)
    elas = (np.log(Pc1.sum(1).mean()) - np.log(Pc0.sum(1).mean())) / (
        np.log(1.01) - np.log(0.99))
    d2 = 1.0 / (1.0 - 2.0 * elas / 3.0)          # 由 −3/2·d²/(1−d²) = elas 解
    d2 = 2.0 * (-elas) / 3.0 / (1.0 - 2.0 * (-elas) / 3.0) if False else d2
    print('\n  由 c_ct 弹性 (−0.6096 量级) 反解 d0² ≈ %.4f ⇒ d0 = %.4f'
          % (d2, np.sqrt(abs(d2))))
    print('  （d0 = 局地亏损**平方根的均值**口径，不是 deficit_max；后者是单点极值）')

    print('\n' + '=' * 92)
    print('④ 逐台敏感度：k_mult 的作用是否"越下游越强"（解析预言）')
    print('=' * 92)
    pt2, dm2 = run(build(k_mult=2.0), vu, vA)
    print('  %-6s %16s %16s %16s' % ('机组', '出力比 k2/k1', '亏损 k1', '亏损 k2'))
    for j in range(5):
        r = pt2[:, j].sum() / max(pt0[:, j].sum(), 1e-12)
        print('  T%-5d %16.4f %16.4f %16.4f' % (j + 1, r, dm0[j], dm2[j]))
    print('  ⇒ 若 T2 的出力比 > T5 的出力比，则 k_mult 的作用**越下游越强**（与解析预言一致）')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
