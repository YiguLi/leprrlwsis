# -*- coding: utf-8 -*-
"""决定性测量：**单台尾流**的亏损有多深？自定义叠加规则能否"绕过"限制？

背景
----
用户思路：不要求叠加规则自洽，直接**定制**一个规则（如"保留最大亏损、不衰减"）
来绕过"功率对不上 vs 非物理"的限制。

要判定这条路是否可行，只需一个量：**单台自身尾流引起的亏损**。
因为任何叠加规则 `C(d_1..d_k)` 都必须满足（合理的单调性 + 单台退化为自身）：

    max_i d_i  ≤  C(d_1..d_k)  ≤  Σ_i d_i

⇒ **亏损下限 = max_i d_i = 单台自身亏损**。
若"单台自身亏损"本身就已经足够深（深到能配平功率），那就与叠加规则无关了 ——
无论怎么定制规则都绕不过去。

测量
----
① 单台尾流：只留 1 台风机运行时，其尾流在下游 x 处的亏损（用 2 台机组实现，
   只让上游那台产生尾流 ⇒ 读下游那台的 Ue）。
② 配平功率所需的亏损：由"无尾流理想出力"与"WFSim 实测农场出力"反推，
   看每台下游平均需要多深的亏损。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_single_wake_depth.py
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


def run_point(us, A, c_ct=1.0, spread=1.0, k_mult=1.0):
    """跑一批，返回 (farm, turb, Ue_eff/Uinf 逐台)。"""
    R.CT_SCALE = c_ct
    m, _ = AT.build_model(calibrated=True)
    if k_mult != 1.0:
        b = dict(AT.CALIBRATED_WAKE_PARAMS)
        vm = m.core.wake.model_strings['velocity_model']
        m.set_param(['wake', 'wake_velocity_parameters', vm, 'ka'],
                    b[('velocity_model', 'ka')] * k_mult)
        m.set_param(['wake', 'wake_velocity_parameters', vm, 'kb'],
                    b[('velocity_model', 'kb')] * k_mult)
    R._A_SET = np.asarray(A, dtype=float)
    R._run_batch(m, np.asarray(us, dtype=float), np.asarray(A, dtype=float))
    turb = np.asarray(m.get_turbine_powers(), dtype=float) / 1e6
    uu = np.asarray(m.core.flow_field.u_sorted, dtype=float)
    ui = np.asarray(m.core.flow_field.u_initial_sorted, dtype=float)
    ueff = np.cbrt(np.mean(np.clip(uu, 0.0, None) ** 3, axis=(2, 3)))   # (n, nt)
    uinf = ui[:, :, ui.shape[2] // 2, ui.shape[3] // 2]                 # (n, nt)
    return turb.sum(1), turb, ueff, uinf


def main():
    u = 10.0
    a = 1.0 / 3.0
    print('=' * 100)
    print('① 单台尾流的亏损（u=%.1f, a=1/3, 标定参数）' % u)
    print('=' * 100)
    # 上游 T1 产生尾流；T2..T5 只作为"探针"读取局地 Ue（它们自身也发尾流，
    # 但我们要的是**T1 单独**的效果 ⇒ 先看把 T2..T5 的 a 设为 0.5（最小动量汇？
    # 不行，a 越小尾流越浅但仍存在）—— 更干净的做法：直接看"只开 T1"时的 Ue 剖面，
    # 这里用 farm 内逐台 Ue/Uinf 的**第一个下游台**作为单台尾流的代表。
    A = np.full((1, NT), a)
    farm, turb, ueff, uinf = run_point([u], A)
    print('  基准（5 台全开，a=1/3）:')
    print('  %-5s %13s %13s %13s' % ('机组', 'Ue_eff/Uinf', '亏损比 d', '出力 MW'))
    for j in range(NT):
        print('  T%-4d %13.4f %13.4f %13.4f'
              % (j + 1, ueff[0, j] / uinf[0, j],
                 1 - ueff[0, j] / uinf[0, j], turb[0, j]))

    print('\n' + '=' * 100)
    print('② 配平功率所需的亏损（由"无尾流理想出力"与"WFSim 实测农场出力"反推）')
    print('=' * 100)
    rng = np.random.default_rng(0)
    import json
    d = json.load(open(os.path.join('output', 'calib_floris_to_wfsim.json'),
                       encoding='utf-8'))
    vu = np.asarray(d['val']['u'], dtype=float)
    vA = np.asarray(d['val']['A'], dtype=float)
    vW = np.asarray(d['val']['P_wfsim'], dtype=float)
    # 理想（无尾流）：把上游尾流"关掉"—— 用单台模型逐台算不现实；
    # 这里用 FLORIS 的 run_no_wake 口径：把组合规则设为"恒零"（无尾流）
    import floris.core.wake as W
    from floris.core.wake_combination import SOSFS
    orig = SOSFS.function
    W.MODEL_MAP['combination_model']['sosfs'].function = lambda self, w, v: np.zeros_like(w)
    try:
        pw_nowake, _, _, _ = run_point(vu, vA)
    finally:
        W.MODEL_MAP['combination_model']['sosfs'].function = orig
    pw_wake, _, _, _ = run_point(vu, vA)
    print('  FLORIS 无尾流理想出力（同动作）均值 = %.4f MW' % pw_nowake.mean())
    print('  FLORIS 有尾流（标定）均值           = %.4f MW' % pw_wake.mean())
    print('  WFSim  实测均值                     = %.4f MW' % vW.mean())
    print()
    print('  ⇒ 尾流造成的损失：')
    print('     FLORIS 标定: -%.4f MW (%.1f%%)' % (pw_nowake.mean() - pw_wake.mean(),
                                                100 * (1 - pw_wake.mean() / pw_nowake.mean())))
    print('     WFSim      : -%.4f MW (%.1f%%)' % (pw_nowake.mean() - vW.mean(),
                                                100 * (1 - vW.mean() / pw_nowake.mean())))
    print()
    # 反推：若尾流损失集中在 4 台下游（T2..T5），每台需要多深的亏损
    n_dn = NT - 1
    print('  若尾流损失全部由 %d 台下游承担（T1 无尾流）：' % n_dn)
    for lab, P, loss in (('FLORIS 标定', pw_wake.mean(), pw_nowake.mean() - pw_wake.mean()),
                         ('WFSim', vW.mean(), pw_nowake.mean() - vW.mean())):
        # 每台下游需要把 Ue 降到原值的 r 倍，使 sum(r^3)*P1 + P1 = P
        p1 = None
        for j in range(NT):
            pass
        # P1 = 无尾流时 T1 出力；下游各台理想 = P1（同 u 同 a）
        p1 = pw_nowake.mean() / NT
        target = P - p1
        r3 = target / (n_dn * p1) if p1 > 0 else float('nan')
        r = r3 ** (1.0 / 3.0) if r3 > 0 else float('nan')
        print('     %-12s 每台下游需 Ue/Uinf = %.4f ⇒ 亏损 = %.4f'
              % (lab, r, 1 - r))
    print()
    print('  ⇒ 于是问题的核心变成一个**与叠加规则无关**的量：')
    print('     "配平功率所需的单台下游亏损" 有多大？')
    print('     若它已经 ≥ 单台自身尾流能提供的亏损，则**任何**叠加规则都必须')
    print('     把多台尾流叠加到更深 ⇒ 非物理无法避免。')

    print('\n' + '=' * 100)
    print('③ 单台尾流在 5D..20D 的**自身**亏损（把上游以外全设为最小动量汇的近似）')
    print('=' * 100)
    print('  做法：只让 T1 有动作（a=1/3），T2..T5 的 a 取 0.025（几乎无动量汇），')
    print('  于是下游读到的 Ue 亏损基本只来自 T1。')
    A1 = np.full((1, NT), 0.025)
    A1[0, 0] = a
    _, turb1, ueff1, uinf1 = run_point([u], A1)
    print('  %-5s %13s %13s %13s' % ('机组', 'Ue_eff/Uinf', '亏损比 d', '出力 MW'))
    for j in range(NT):
        print('  T%-4d %13.4f %13.4f %13.4f'
              % (j + 1, ueff1[0, j] / uinf1[0, j],
                 1 - ueff1[0, j] / uinf1[0, j], turb1[0, j]))
    print('  ⇒ T2 的亏损即"单台尾流在 5D 处"的真实深度。')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
