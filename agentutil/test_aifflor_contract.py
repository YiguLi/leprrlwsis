# -*- coding: utf-8 -*-
"""`aifflor` 环境契约测试（**只读、不需要 MATLAB**）。

断言"`aifflor` 就是 `td3per` 换了引擎"，逐条把"相同"与"不同"都变成可执行判据。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\test_aifflor_contract.py
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

FAIL = []


def ok(cond, msg):
    print('  %s %s' % ('PASS' if cond else 'FAIL', msg))
    if not cond:
        FAIL.append(msg)


def main():
    import aifflor.aifflor_turbine as AT
    from aifflor.aifflor_env import AifflorPhysics, WindBatteryAIF
    from env import WindBattery
    import td3per.td3per_myfunc as MF
    import aifflor.aifflor_myfunc as AMF

    print('=' * 90)
    print('A. 逐字复用的部分（myfunc / ddpg 必须与 td3per 逐位相同）')
    print('=' * 90)
    for name in ('CheckAction', 'Normalize', 'Normalize_H', 'Cal_Rew_RB',
                 'gap_penalty', 'soc_target_from_signal', 'lower_reward',
                 'lower_cost_soc_track', 'violation_derate', 'build_L_obs_distilled'):
        a, b = getattr(MF, name, None), getattr(AMF, name, None)
        if a is None or b is None:
            ok(False, '%s 在一边缺失' % name)
            continue
        src_a = a.__code__.co_code if hasattr(a, '__code__') else a
        src_b = b.__code__.co_code if hasattr(b, '__code__') else b
        ok(src_a == src_b or a == b, '%s 逐位相同（代码对象一致）' % name)
    ok(MF.PENALTY_NN_H_COEF == AMF.PENALTY_NN_H_COEF,
       'PENALTY_NN_H_COEF 相同（= %r）' % MF.PENALTY_NN_H_COEF)
    ok((MF.VIOL_DERATE_DIMS, getattr(MF, '_ACT_HI', None), getattr(MF, '_ACT_LO', None))
       == (AMF.VIOL_DERATE_DIMS, getattr(AMF, '_ACT_HI', None),
           getattr(AMF, '_ACT_LO', None)),
       'violation_amount 动作带相同（前 5 维 [0,0.5]）')

    print('\n' + '=' * 90)
    print('B. 动作语义：aifflor 与 td3per 相同（≠ yawflor）')
    print('=' * 90)
    ok((AT.A_LO, AT.A_HI) == (0.0, 0.5), '动作带 = [0, 0.5]（轴向因子）')
    ok(abs(AT.A_NOMINAL - 0.3333) < 1e-12, '标称动作 = 0.3333（= td3per）')
    ok(WindBatteryAIF.nominal_action_value == WindBattery.nominal_action_value,
       'WindBatteryAIF 的标称动作**未被覆写**（= %.4f）'
       % WindBatteryAIF.nominal_action_value)
    ok(WindBatteryAIF.nominal_action is None and WindBattery.nominal_action is None,
       'nominal_action 均为 None（标量铺满，与 td3per 同）')

    print('\n' + '=' * 90)
    print('C. 物理后端：FLORIS + 标定参数，且不启 MATLAB')
    print('=' * 90)
    ph = AifflorPhysics(nt=5)
    ok(ph.calibrated is True, '默认施加标定尾流参数')
    th = ph.info.get('theta')
    ok(th is not None and abs(th[0] - 0.05) < 1e-9 and abs(th[4] - 1.65503) < 1e-6,
       '标定 θ 已生效（ka=%.5f, ai=%.5f）' % (th[0], th[4]))
    ok('matlab' not in sys.modules, '导入 aifflor 全链路**未**引入 matlab 模块')
    p13 = ph.power(10.0, [1 / 3] * 5)
    ok(abs(p13 - 3.294116302293012) < 1e-9,
       'u=10, a=1/3 ⇒ %.9f MW（标定后现值）' % p13)
    p20 = ph.power(10.0, [0.20] * 5)
    ok(p20 > p13, 'u=10 时 a=0.20 (%.4f) > a=1/3 (%.4f)：标定后最优点在 0.375 侧' % (p20, p13))
    # a 夹带
    ok(np.allclose(ph.a_clip([-1.0, 9.0, 0.25, 0.25, 0.25]), [0.0, 0.5, 0.25, 0.25, 0.25]),
       'a_clip 正确夹到 [0, 0.5]')

    print('\n' + '=' * 90)
    print('D. env 契约：观测布局 / 围栏 / 预取 / 接口同形（不启 MATLAB）')
    print('=' * 90)
    e = WindBatteryAIF(5.0, 5, upload=False, engine=None, physics=AifflorPhysics(nt=5))
    obs = e.reset()
    ok(np.shape(obs) == (4,), 'reset() 观测 = 4 维 [pr, u, E, P_last]（与 td3per 同形）')
    ok(3.5 <= float(obs[1]) <= 25.0, 'reset() 风速在围栏内 (%.3f)' % obs[1])
    ok(abs(float(obs[3]) - e.PowerOutput(float(obs[1]), e.nominal())) < 1e-9,
       'obs[3] == PowerOutput(u, nominal)（与 env.py 口径一致）')
    ok(e.PowerOutput(3.0, [1 / 3] * 5) == 0.0, 'u<3.5 时出力 0（env 守卫，未调引擎）')
    ok(e.PowerOutput(26.0, [1 / 3] * 5) == 0.0, 'u>25 时出力 0（env 守卫）')
    n_direct_before = e.physics.n_direct
    e.PowerOutput(3.0, [1 / 3] * 5)
    ok(e.physics.n_direct == n_direct_before, '守卫命中时**不调用物理引擎**（与旧路径一致）')
    e.TestReset(0)
    ok(np.shape(e.TestReset(0)) == (4,), 'TestReset() 返回 4 维观测')
    # step 接口
    a_h = np.array([0.3333] * 5 + [0.0])
    pw = e.PowerOutput(e.u, e.nominal())
    out = e.step(a_h, np.array([0.0]), pw)
    ok(len(out) == 7, 'step() 返回 7 元组（与 env.py 同形）')
    ok(hasattr(e, 'summary') and 'aifflor-physics' in e.summary(),
       'summary() 可打印物理统计')
    # 预取确实发生（第二个 env，避免上面 TestReset 的干扰）
    e2 = WindBatteryAIF(5.0, 5, upload=False, engine=None, physics=AifflorPhysics(nt=5))
    e2.reset()
    npf = e2.n_prefetch
    ok(npf >= 1, 'reset 路径触发过预取（n_prefetch=%d）' % npf)
    a0 = np.array([0.30] * 5 + [0.0])
    v1 = e2.PowerOutput(float(e2.u), a0)
    v2 = e2.PowerOutput(float(e2.u), a0)
    ok(abs(v1 - v2) < 1e-12, '同 (u, a) 两次调用逐位一致（缓存不改数值）')

    print('\n' + '=' * 90)
    print('E. 与 yawflor 的边界：aifflor 不做偏航')
    print('=' * 90)
    ok(not hasattr(AT, 'GAMMA_MAX'), 'aifflor_turbine 没有偏航常量 GAMMA_MAX')
    ok(AT.WIND_DIR_ALIGNED == 270.0, '风向仍沿 +x（270°），对应 WFSim 的 wind_angle=0')

    print('\n' + '=' * 90)
    if FAIL:
        print('结果: **%d 项失败** ❌' % len(FAIL))
        for f in FAIL:
            print('   - %s' % f)
        return 1
    print('结果: 全部通过 ✅')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
