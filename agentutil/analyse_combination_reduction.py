# -*- coding: utf-8 -*-
"""组合规则到底削减了多少单尾流亏损？—— 决定"自定义叠加规则"有没有腾挪空间。

要回答的问题
------------
用户思路：不要求叠加规则自洽，直接**定制**一个规则（例如"保留最大亏损、不衰减"）
来绕过限制。

关键判据（本脚本测量）
----------------------
若组合规则已经把"单尾流增量亏损"削减到接近其下限，那么**任何**自定义规则都无法再
降低下游亏损 ⇒ 只能靠加深**单尾流**来配平功率 ⇒ 与物理性的矛盾**结构性存在**。
反之若削减很多，则自定义规则有腾挪空间。

测量方式
--------
在标定模型 + 标定动作下，逐机组记录：
  * `w_in`  = 进入该机组迭代时的累积 wake 场（= 组合规则**累积结果**）
  * `d_in`  = 该机组（及其上游）的 velocity_deficit 场
  * `Ue_eff`= 该机组转子面上的**出力加权**等效风速（由 `flow_field.u_sorted` 算）
  * `u_inf` = 自由流（该机组 x 处的 u_initial）
并把它与"**若完全不叠加、只保留本台自身尾流**"的理想下限比较：
  单台自身亏损 = `d_new·u_initial`（本台刚产生的尾流，未经任何叠加）

⇒ 定义**削减比**
     R = (累积 wake 在该台转子面中心的值) / (单台自身亏损在该点的值)
   R ≈ 1  ⇒ 组合规则**没有**削减 ⇒ 自定义规则无空间
   R ≫ 1  ⇒ 累积远超单台 ⇒ 自定义规则若把 R 压回 1 可显著降低亏损

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_combination_reduction.py
"""
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
#: 逐机组记录：组合规则的输入/输出（由 monkey-patch 收集）
REC = []


def measure(u=10.0, a=1.0 / 3.0, c_ct=1.0, k_mult=1.0):
    """跑一个点，收集每一步组合的 (w_in, d_in) 与最终等效风速。"""
    import floris.core.wake as W
    from floris.core.wake_combination import SOSFS

    REC.clear()
    orig_fn = SOSFS.function

    def spy(self, wake_field, velocity_field):
        # ⚠ 必须在调用 orig_fn **之前** copy：求解器会**原地修改**这些数组
        #   （第一版在 orig_fn 之后 copy，记到的是被覆盖后的值 ⇒ v_new 全为 0）
        w_in = np.asarray(wake_field, dtype=float).copy()
        v_in = np.asarray(velocity_field, dtype=float).copy()
        out = orig_fn(self, wake_field, velocity_field)
        REC.append((w_in, v_in, np.asarray(out, dtype=float).copy()))
        return out

    W.MODEL_MAP['combination_model']['sosfs'].function = spy
    try:
        R.CT_SCALE = c_ct
        m, _ = AT.build_model(calibrated=True)
        if k_mult != 1.0:
            b = dict(AT.CALIBRATED_WAKE_PARAMS)
            vm = m.core.wake.model_strings['velocity_model']
            m.set_param(['wake', 'wake_velocity_parameters', vm, 'ka'],
                        b[('velocity_model', 'ka')] * k_mult)
            m.set_param(['wake', 'wake_velocity_parameters', vm, 'kb'],
                        b[('velocity_model', 'kb')] * k_mult)
        R._A_SET = np.full((1, NT), a)
        R._run_batch(m, [u], np.full((1, NT), a))
        uu = np.asarray(m.core.flow_field.u_sorted, dtype=float)      # (1,NT,gz,gy)
        ui = np.asarray(m.core.flow_field.u_initial_sorted, dtype=float)
        # 转子中心索引
        c = uu.shape[2] // 2, uu.shape[3] // 2
        return m, uu, ui, c
    finally:
        W.MODEL_MAP['combination_model']['sosfs'].function = orig_fn


def main():
    u0, a0 = 10.0, 1.0 / 3.0
    m, uu, ui, c = measure(u0, a0)
    print('=' * 100)
    print('组合规则的削减比测量   (u=%.1f, a=1/3, 标定参数)' % u0)
    print('=' * 100)
    print('  逐机组：组合规则**输入**的累积 wake(w) 与**本台自身**亏损(v)，都在转子中心取')
    print('  %-5s %11s %11s %11s %11s %11s %9s'
          % ('机组', 'w_in 中心', 'v_new 中心', 'w_out 中心', 'Ue_eff', 'u_inf', 'v_new/u_inf'))
    rows = []
    for j, (w_in, d_in, w_out) in enumerate(REC):
        wc = float(w_in[0, j, c[0], c[1]])
        vc = float(d_in[0, j, c[0], c[1]])
        oc = float(w_out[0, j, c[0], c[1]])
        uinf = float(ui[0, j, c[0], c[1]])
        ueff = float(np.cbrt(np.mean(np.clip(uu[0, j], 0, None) ** 3)))
        rows.append((j, wc, vc, oc, ueff, uinf))
        print('  T%-4d %11.4f %11.4f %11.4f %11.4f %11.4f %9.4f'
              % (j + 1, wc, vc, oc, ueff, uinf, vc / max(uinf, 1e-9)))

    print('\n' + '=' * 100)
    print('关键读数')
    print('=' * 100)
    print('  ① **单台自身**亏损占自由流的比例 v_new/u_inf：')
    for j, wc, vc, oc, ueff, uinf in rows:
        print('     T%d: %.4f  (等价于单台尾流就把风速压掉 %.1f%%)' % (j + 1, vc / uinf,
                                                                  100 * vc / uinf))
    print('\n  ② 累积 w_in 相对单台 v_new 的倍数（= 组合规则放大了多少）：')
    for j, wc, vc, oc, ueff, uinf in rows:
        r = wc / vc if abs(vc) > 1e-12 else float('nan')
        print('     T%d: w_in/v_new = %.4f' % (j + 1, r))
    print('\n  ③ 最终等效风速 Ue_eff/u_inf（这是决定出力的量）：')
    for j, wc, vc, oc, ueff, uinf in rows:
        print('     T%d: Ue/u_inf = %.4f  →  出力 ∝ (Ue/u_inf)^3 = %.4f'
              % (j + 1, ueff / uinf, (ueff / uinf) ** 3))

    print('\n' + '=' * 100)
    print('判据：下限测试')
    print('=' * 100)
    print('  若把组合规则改成"**完全不叠加**"（下游只看最近上游那一台），')
    print('  则 T_j 的亏损 = **单台自身** v_new = 上表 ① 的数值。这就是任何叠加')
    print('  规则能给出的**亏损下限**（因为规则再温和也不能低于单台自身的尾流）。')
    print()
    vmin = max(vc / uinf for _, _, vc, _, _, uinf in rows)
    print('  ⇒ 单台自身亏损的最大值 = %.4f（出现在最下游受尾流影响最大处）' % vmin)
    print('  ⇒ 若该值 ≥ 1，则**即使完全不叠加**，最下游也会出现 u<0 ⇒')
    print('     自定义叠加规则**无法**解决非物理问题。')

    # 直接验证：单台自身亏损是否已 ≥ 1
    print('\n' + '=' * 100)
    print('直接验证：把叠加关掉（只留最近上游一台）会怎样？')
    print('=' * 100)
    print('  用 q=1（完全丢弃累积，只用本台增量）跑一次，看 deficit_max / u_min：')
    import agentutil.calib_floris_to_wfsim as C
    import json
    d = json.load(open(os.path.join('output', 'calib_floris_to_wfsim.json'),
                       encoding='utf-8'))
    vu = np.asarray(d['val']['u'], dtype=float)
    vA = np.asarray(d['val']['A'], dtype=float)
    vW = np.asarray(d['val']['P_wfsim'], dtype=float)

    import agentutil.probe_persistent_combination as P
    P.register()
    R.CT_SCALE = 1.0
    R.PERSIST_Q, R.PERSIST_P = 1.0, 2.0     # q=1 ⇒ 只用本台增量，丢弃累积
    m2 = P.build(1.0)
    R._A_SET = vA
    R._run_batch(m2, vu, vA)
    uu2 = np.asarray(m2.core.flow_field.u_sorted, dtype=float)
    ui2 = np.asarray(m2.core.flow_field.u_initial_sorted, dtype=float)
    dmax2 = max(float(np.nanmax((ui2[:, j] - uu2[:, j]) / np.maximum(ui2[:, j], 1e-9)))
                for j in range(ui2.shape[1]))
    pf2 = np.asarray(m2.get_turbine_powers(), dtype=float).sum(axis=1) / 1e6
    mm2 = C.metrics(vW, pf2)
    print('  q=1（丢弃累积、只用本台增量）: 常数重标定后 MAPE %.3f%%，'
          'deficit_max=%.4f，u_min=%.4f，u<0 %d'
          % (100 * mm2['mape_k'], dmax2, float(np.nanmin(uu2)), int((uu2 < 0).sum())))
    print('  ⇒ 对比基准（q=0, 原 SOSFS）: MAPE 1.103%%，deficit_max=1.0873，u<0 22')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
