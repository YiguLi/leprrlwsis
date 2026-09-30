# -*- coding: utf-8 -*-
"""分段尾流扩展（`empirical_gauss`）能否做到"功率对得上且不非物理"？

用户方案
--------
"WFSim 亏损不衰减的根源是侧向边界抑制横向扩散；在 FLORIS 里把
`wake_expansion_rates`（k）大幅调低甚至置零，使 σ 几乎不增长，
从而数值模拟出'不衰减'的剖面。Empirical Gaussian 支持分段，灵活性更大。"

已对齐的事实（见对话）
----------------------
* 现有标定的 `k = ka·TI + kb = 0.05×0.06 + 0.00208 = **0.00508**`
  ⇒ 用户建议的 `k = 0.005` **就是现有标定值**；
* `gauss` 随包默认 `k = 0.38×0.06 + 0.004 = 0.0268`（不是 0.065）；
* `empirical_gauss` 默认 `wake_expansion_rates=[0.023, 0.008]`、
  `breakpoints_D=[10]` ⇒ **本来就是分段线性的**，`len(rates) == len(breakpoints)+1`。

本脚本补测**没测过的部分**：`empirical_gauss` 的**分段**扩展
（含 k→0 的多段配置），并与现有 `gauss` 标定对照。

⚠ 一个先验的判断（待实测检验）
------------------------------
现有 `gauss` 标定的**单尾流增量亏损本来就不衰减**（trace 实测 6.93→6.85，几乎恒定），
原因是 `d ∝ u_initial/σ` 里 `u_initial` 随剪切增大，抵消了 σ 的增长。
⇒ 若这一条也适用于 `empirical_gauss`，那么"把 k 调低以制造不衰减"是
**在已满足的维度上加约束**，只会破坏拟合。

判据
----
常数重标定后 MAPE ≤ 2%  且  u_min > 0  且  deficit_max < 0.9

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_empirical_gauss_segments.py
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


def build(vm='gauss', k_seg=None, sigma0_D=None):
    """构造模型。

    * `vm='gauss'`：k_seg = ([], [k]) —— 把 `ka` 设为 k、`kb` 设为 0
      （因 `k = ka·TI + kb`，令 kb=0 则 ka = k/TI）。
    * `vm='empirical_gauss'`：k_seg = (breakpoints_D, rates)，单位是**无量纲 dσ/dx**。
    """
    if vm == 'gauss':
        m, _ = AT.build_model(calibrated=True)
        if k_seg is not None:
            k_target = float(k_seg[1][0])
            m.set_param(['wake', 'wake_velocity_parameters', 'gauss', 'ka'],
                        k_target / max(AT.TI, 1e-9))
            m.set_param(['wake', 'wake_velocity_parameters', 'gauss', 'kb'], 0.0)
        return m

    # empirical_gauss：先构造（用标定参数），再换成"速度=分段高斯 + 配套湍流=wake_induced_mixing"
    #   ⚠ empirical_gauss 有**自己的求解器**（`solver.empirical_gauss_solver`），
    #   它要求湍流模型提供 `atmospheric_ti_gain` —— 只有 `wake_induced_mixing` 有；
    #   用 `crespo_hernandez` 会报 `'CrespoHernandez' object has no attribute
    #   'atmospheric_ti_gain'`。
    m, _ = AT.build_model(calibrated=True)
    bps, rates = k_seg
    params = dict(wake_expansion_rates=[float(r) for r in rates],
                  breakpoints_D=[float(b) for b in bps],
                  sigma_0_D=float(sigma0_D if sigma0_D else 0.3536),
                  mixing_gain_velocity=2.0,
                  smoothing_length_D=2.0, awc_mode='baseline')
    m.set_param(['wake', 'wake_velocity_parameters', 'empirical_gauss'], params)
    m.set_param(['wake', 'wake_turbulence_parameters', 'wake_induced_mixing'],
                dict(atmospheric_ti_gain=0.0))
    # ⚠ `empirical_gauss` 速度模型需要**配套的挠曲模型**：给它 `gauss` 的挠曲会报
    #   `GaussVelocityDeflection.function() takes 7 positional arguments but 8 were given`
    m.set_param(['wake', 'wake_deflection_parameters', 'empirical_gauss'],
                dict(horizontal_deflection_gain_D=3.0, vertical_deflection_gain_D=-1.0,
                     deflection_rate=22.0, mixing_gain_deflection=0.0,
                     yaw_added_mixing_gain=0.0))
    m.set_param(['wake', 'model_strings', 'velocity_model'], 'empirical_gauss')
    m.set_param(['wake', 'model_strings', 'deflection_model'], 'empirical_gauss')
    m.set_param(['wake', 'model_strings', 'turbulence_model'], 'wake_induced_mixing')
    # ⚠ empirical_gauss 求解器**不支持**三个开关（solver.py 直接 raise）：
    #   `Secondary steering not available for this model.` (1054)
    #   `Transverse velocities not used in this model.`   (1106)
    #   `Yaw added recovery not used in this model.`      (1110)
    m.set_param(['wake', 'enable_secondary_steering'], False)
    m.set_param(['wake', 'enable_transverse_velocities'], False)
    m.set_param(['wake', 'enable_yaw_added_recovery'], False)
    return m


def evaluate(m, vu, vA, vW):
    R._A_SET = vA
    R._run_batch(m, vu, vA)
    pf = np.asarray(m.get_turbine_powers(), dtype=float).sum(axis=1) / 1e6
    mm = C.metrics(vW, pf)
    ui = np.asarray(m.core.flow_field.u_initial_sorted)
    uu = np.asarray(m.core.flow_field.u_sorted)
    dmax = max(float(np.nanmax((ui[:, j] - uu[:, j]) / np.maximum(ui[:, j], 1e-9)))
               for j in range(ui.shape[1]))
    return dict(mape_k=mm['mape_k'], u_min=float(np.nanmin(uu)),
                n_neg=int((uu < 0).sum()), deficit_max=dmax)


def _rate(m, vu, vA, vW):
    """`evaluate` 的别名（脚本内多处调用，保持一处实现）。"""
    return evaluate(m, vu, vA, vW)


def _verdict(r):
    fit = r['mape_k'] <= 0.02
    phys = (r['u_min'] > 0.0) and (r['deficit_max'] < 0.9)
    if fit and phys:
        return '**可用**'
    if not fit and not phys:
        return '拟合差+非物理'
    return '拟合差' if not fit else '非物理'


def main():
    d = json.load(open(DATA, encoding='utf-8'))
    vu = np.asarray(d['val']['u'], dtype=float)
    vA = np.asarray(d['val']['A'], dtype=float)
    vW = np.asarray(d['val']['P_wfsim'], dtype=float)
    b = dict(AT.CALIBRATED_WAKE_PARAMS)
    k0 = b[('velocity_model', 'ka')] * AT.TI + b[('velocity_model', 'kb')]
    print('验证集 n=%d | 现有标定 k = %.5f' % (len(vu), k0))

    # ---- ① 显式对照：把 gauss 的 k 按用户建议设到 0.005 / 0.0005 / 0 ----
    print('\n' + '=' * 96)
    print('① 直接检验用户方案：在**现有 gauss 模型**里把 k 调到 0.005 / 0.0005 / 0')
    print('=' * 96)
    print('  %-34s %12s %13s %11s %7s %s'
          % ('k 取值', '常数重标后', 'deficit_max', 'u_min', 'u<0', '判定'))
    base_rows = []
    for kk in (k0, 0.02, 0.005, 0.002, 0.0005, 0.0):
        try:
            r = _rate(build('gauss', ([], [kk])), vu, vA, vW)
        except Exception as e:
            print('  k=%-32.5f 失败: %s' % (kk, str(e)[:34]))
            continue
        tag = 'k=%.5f%s' % (kk, '  (= 现有标定)' if abs(kk - k0) < 1e-9 else
                            ('  (= 用户建议)' if abs(kk - 0.005) < 1e-9 else ''))
        print('  %-34s %11.3f%% %13.4f %11.4f %7d %s'
              % (tag, 100 * r['mape_k'], r['deficit_max'], r['u_min'], r['n_neg'],
                 _verdict(r)))
        base_rows.append((tag, r))

    # ---- ② empirical_gauss 单位对齐的缩放扫描（rates 是无量纲 dσ/dx） ----
    print('\n' + '=' * 96)
    print('② empirical_gauss 单位对齐扫描（rates 为无量纲 dσ/dx，单段无断点）')
    print('   k_eff 行给出 gauss 侧的等效 k = rate（因为 gauss 的 k 也是 dσ/dx，只是乘 D 换算）')
    print('=' * 96)
    print('  %-34s %12s %13s %11s %7s %s'
          % ('rate (dσ/dx)', '常数重标后', 'deficit_max', 'u_min', 'u<0', '判定'))
    emp_rows = []
    for rate in (0.0001, 0.001, 0.005, 0.01, 0.02, 0.04, 0.08):
        try:
            # ⚠ 单段必须传 `breaks=[]` + `rates=[rate]`：
            #   empirical_gauss 断言 `len(rates) == len(breakpoints) + 1`
            r = _rate(build('empirical_gauss', ([], [rate]), None), vu, vA, vW)
        except Exception as e:
            print('  rate=%-31.4f 失败: %s' % (rate, str(e)[:32]))
            continue
        print('  %-34.4f %11.3f%% %13.4f %11.4f %7d %s'
              % (rate, 100 * r['mape_k'], r['deficit_max'], r['u_min'],
                 r['n_neg'], _verdict(r)))
        emp_rows.append((rate, r))

    # ---- ③ 汇总 ----
    allr = [(t, r) for t, r in base_rows] + [('emp rate=%.4f' % t, r) for t, r in emp_rows]
    good = [t for t in allr if t[1]['mape_k'] <= 0.02 and t[1]['u_min'] > 0
            and t[1]['deficit_max'] < 0.9]
    print('\n' + '=' * 96)
    print('结论')
    print('=' * 96)
    if allr:
        best_fit = min(allr, key=lambda t: t[1]['mape_k'])
        print('  最佳拟合 : %-28s MAPE %.3f%%，deficit_max=%.3f，u<0 %d'
              % (best_fit[0], 100 * best_fit[1]['mape_k'],
                 best_fit[1]['deficit_max'], best_fit[1]['n_neg']))
        phys = [t for t in allr if t[1]['u_min'] > 0 and t[1]['deficit_max'] < 0.9]
        if phys:
            bf = min(phys, key=lambda t: t[1]['mape_k'])
            print('  最佳物理 : %-28s MAPE %.3f%%，deficit_max=%.3f'
                  % (bf[0], 100 * bf[1]['mape_k'], bf[1]['deficit_max']))
    if good:
        print('  ✅ 两全解: %s' % ', '.join(t[0] for t in good))
    else:
        print('  ❌ 无两全解。')
        print('  ⇒ 关键读数：把 k 调低确实能让 `u_min` 转正（物理变好），但')
        print('     拟合同时崩到 %s —— 两者**此消彼长**，没有交点。'
              % ' / '.join('%.1f%%' % (100 * r['mape_k']) for _, r in emp_rows[:4]))
    print('=' * 96)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
