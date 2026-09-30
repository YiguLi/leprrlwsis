# -*- coding: utf-8 -*-
"""临时探针③：**逐台**轴向因子的价值 vs 全场统一 a —— 决定"是否值得改造引擎"。

背景
----
`probe_floris_a_authority.py` 已证：FLORIS 原生 `multidim_condition` **只能**做
全场统一 a（该通道全域单调：降载 ⇒ 降波动，但没有增益；相对 a=1/3 只有下探）。

本探针回答"是否值得为逐台 a 改引擎"必须先回答的问题：
  逐台 a（上游降载留风 + 下游推到单机最优）相对**全场统一 a**，
  在**同一电量**下能否把波动压得更低？即"逐台"是否带来**真实的帕累托改进**。

实现（不修改安装的 floris，零写盘）
----------------------------------
FLORIS 的 turbine 函数**不是**调用模块级 `turbine.power()`，而是调用
`Farm.turbine_power_functions[turbine_type]`——构造时从**操作模型类**绑定的静态方法
（取证 `agentutil/_locate_power_hook.py`：生产路径命中 `Cosine.power/ct/ai`）。

因此本探针：
  1. 在 `operation_models.TURBINE_MODEL_MAP["operation_model"]` 里注册一个
     **自定义操作模型** `PerTurbineInductionTurbine`（类身份，非实例属性），
     其 `power/thrust_coefficient/axial_induction` 读模块全局 `A_OVERRIDE`；
  2. 用它构造本探针自己的模型，**不碰任何既有模型**（生产 `yawflor` 基线在同一
     进程里测得，且**符号约定**由"逐台 a 全 1/3 须与基线逐位一致"把守）；
  3. `A_OVERRIDE is None` 时逐位转发给 `SimpleTurbine`/`CosineLossTurbine`（保底）。

⚠ 符号约定：solver 对 `axial_induction()` 的返回值**再取负**后才喂给尾流模型
（`solver.py:128` `−aI`），故本模型返回 `−a`。
⚠ 索引约定：`turbine.py` 在调用操作模型**之前**已按 `ix_filter` 过滤了
`velocities`/`power_setpoints`/`turbine_type_map`（取证 `agentutil/_locate_shapes.py`），
故本模块一律按 `np.shape(velocities)[0:2]` 回形。

用法: F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_floris_a_per_turbine.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

import yawflor.yawflor_turbine as yt  # noqa: E402
from agentutil.probe_floris_a_authority import a_curve, build_md, _tables  # noqa: E402

A_OVERRIDE = None          # None = 逐位转发原模型; ndarray (findex, n_turbines) = 逐台 a
CUR_I = None               # 当前正在推进的机组序号（由 solver 层注入）
POWER_TRACE = None         # 非 None 时记录每次 power() 的 (velocities, a, P)
OPNAME = 'probe-per-turbine-induction'


def _a_sel(velocities):
    """取出应当作用到当前（已被引擎过滤过的）velocities 上的 a。

    引擎的 `turbine.py` 在调用操作模型**之前**已按 ix_filter 过滤 velocities，
    而 operation model 拿不到 ix_filter ⇒ 由 solver 层把当前机组序号写进 `CUR_I`。
    """
    a = np.asarray(A_OVERRIDE, dtype=float)
    tgt = np.shape(velocities)[0:2]
    if CUR_I is not None and a.ndim == 2 and a.shape[1] > int(CUR_I):
        a = a[:, int(CUR_I):int(CUR_I) + 1]
    return np.broadcast_to(a, tgt).copy()


def _cp(a):
    a = np.asarray(a, dtype=float)
    return 4.0 * a * (1.0 - a) ** 2


def _ct(a):
    a = np.asarray(a, dtype=float)
    return 4.0 * a * (1.0 - a)


# 生产功率口径：**整数风速网格上的表 + 线性插值**（`SimpleTurbine.power` 用
# `interp1d(wind_speed, power)`），不是连续解析式。两者在网格点上相等，
# 网格之间差 0.05%(u≈10)~1.8%(u≈5.7) —— 这是**表本身的口径**，必须逐点复刻
# 才能让"逐台 a 全 1/3"与生产基线逐位一致（实测取证见本文件顶部说明）。
_U_GRID = np.arange(0.0, 41.0, 1.0)


def _power_table(a):
    """按 a 生成与生产同口径的功率表（kW，在 `_U_GRID` 上采样）。"""
    cp = 4.0 * a * (1.0 - a) ** 2
    return yt.POWER_SCALE * 0.5 * yt.RHO * yt.AREA * cp * _U_GRID ** 3 / 1e3


def _power_of_u(a, u):
    """逐台 a、逐点 u → 功率 [W]，复刻 `SimpleTurbine.power` 的插值口径。"""
    from scipy.interpolate import interp1d
    a = np.asarray(a, dtype=float)
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u, dtype=float)
    for av in np.unique(a):
        m = (a == av)
        f = interp1d(_U_GRID, _power_table(float(av)), fill_value=0.0, bounds_error=False)
        out[m] = np.asarray(f(u[m]), dtype=float) * 1e3
    return out


# ---------------------------------------------------------------------------
# 1. 自定义操作模型（注册进 TURBINE_MODEL_MAP，靠**类身份**被绑定）
# ---------------------------------------------------------------------------
def register_operation_model():
    from floris.core.turbine import operation_models as OM
    from floris.core.turbine import turbine as TT
    from floris.core.rotor_velocity import average_velocity
    if OPNAME in TT.TURBINE_MODEL_MAP['operation_model']:
        return OM

    class PerTurbineInductionTurbine(OM.BaseOperationModel):
        """γ≡0 固定；Cp_eff(a)=0.95·4a(1−a)²，CT(a)=4a(1−a)，a 由 A_OVERRIDE 逐台给。"""

        @staticmethod
        def power(velocities=None, turbulence_intensities=None, air_density=None,
                  yaw_angles=None, tilt_angles=None, power_setpoints=None, awc_modes=None,
                  awc_amplitudes=None, tilt_interp=None, average_method='cubic-mean',
                  cubature_weights=None, correct_cp_ct_for_tilt=False, **_):
            if A_OVERRIDE is None:
                return OM.CosineLossTurbine.power(
                    velocities=velocities, turbulence_intensities=turbulence_intensities,
                    air_density=air_density, yaw_angles=yaw_angles,
                    tilt_angles=tilt_angles, power_setpoints=power_setpoints,
                    awc_modes=awc_modes, awc_amplitudes=awc_amplitudes,
                    tilt_interp=tilt_interp, average_method=average_method,
                    cubature_weights=cubature_weights,
                    correct_cp_ct_for_tilt=correct_cp_ct_for_tilt)
            from floris.core.rotor_velocity import rotor_velocity_air_density_correction
            u = average_velocity(velocities, method=average_method,
                                 cubature_weights=cubature_weights)
            u = rotor_velocity_air_density_correction(u, air_density, yt.RHO)
            a = _a_sel(velocities)
            p = _power_of_u(a, u)
            if POWER_TRACE is not None:
                POWER_TRACE.append(dict(vel_shape=np.shape(velocities),
                                        vel=np.array(velocities, copy=True),
                                        u=np.array(u, copy=True),
                                        a=np.array(a, copy=True),
                                        P=np.array(np.broadcast_to(p, np.shape(u)), copy=True)))
            return np.broadcast_to(p, np.shape(u)).copy()

        @staticmethod
        def thrust_coefficient(velocities=None, turbulence_intensities=None, air_density=None,
                               yaw_angles=None, tilt_angles=None, power_setpoints=None,
                               awc_modes=None, awc_amplitudes=None, tilt_interp=None,
                               average_method='cubic-mean', cubature_weights=None,
                               correct_cp_ct_for_tilt=False, **_):
            if A_OVERRIDE is None:
                return OM.CosineLossTurbine.thrust_coefficient(
                    velocities=velocities, turbulence_intensities=turbulence_intensities,
                    air_density=air_density, yaw_angles=yaw_angles,
                    tilt_angles=tilt_angles, power_setpoints=power_setpoints,
                    awc_modes=awc_modes, awc_amplitudes=awc_amplitudes,
                    tilt_interp=tilt_interp, average_method=average_method,
                    cubature_weights=cubature_weights,
                    correct_cp_ct_for_tilt=correct_cp_ct_for_tilt)
            # 引擎已在调用前按 ix_filter 过滤过 ⇒ a 也必须同样过滤
            return _ct(_a_sel(velocities))

        @staticmethod
        def axial_induction(velocities=None, turbulence_intensities=None, air_density=None,
                            yaw_angles=None, tilt_angles=None, power_setpoints=None,
                            awc_modes=None, awc_amplitudes=None, tilt_interp=None,
                            average_method='cubic-mean', cubature_weights=None,
                            correct_cp_ct_for_tilt=False, **_):
            if A_OVERRIDE is None:
                return OM.CosineLossTurbine.axial_induction(
                    velocities=velocities, turbulence_intensities=turbulence_intensities,
                    air_density=air_density, yaw_angles=yaw_angles,
                    tilt_angles=tilt_angles, power_setpoints=power_setpoints,
                    awc_modes=awc_modes, awc_amplitudes=awc_amplitudes,
                    tilt_interp=tilt_interp, average_method=average_method,
                    cubature_weights=cubature_weights,
                    correct_cp_ct_for_tilt=correct_cp_ct_for_tilt)
            return _a_sel(velocities)

    PerTurbineInductionTurbine.__name__ = 'PerTurbineInductionTurbine'
    TT.TURBINE_MODEL_MAP['operation_model'][OPNAME] = PerTurbineInductionTurbine
    return OM


def _solver_hook():
    """临时包装 `floris.core.solver.thrust_coefficient`（solver 模块内的名字绑定）：

    solver 逐台推进（`solver.py:99` `ix_filter=[i]`），但 operation model 拿不到
    机组序号（`turbine.py:310` 在调用前已过滤 velocities）⇒ 在此把序号写进 `CUR_I`。
    """
    import contextlib

    import floris.core.solver as S

    orig_ts = S.thrust_coefficient

    @contextlib.contextmanager
    def ctx():
        global CUR_I

        def counting_ts(*args, **kwargs):
            global CUR_I
            ix = kwargs.get('ix_filter')
            CUR_I = int(ix[0]) if ix is not None else None
            return orig_ts(*args, **kwargs)

        S.thrust_coefficient = counting_ts
        try:
            yield
        finally:
            S.thrust_coefficient = orig_ts
            CUR_I = None

    return ctx


def build_patched(nt=5):
    register_operation_model()
    from floris import FlorisModel
    m = FlorisModel('defaults')
    m.set(layout_x=[i * 500.0 for i in range(nt)], layout_y=[0.0] * nt)
    m.set(turbine_type=[dict(
        turbine_type='probe_a_per_turbine',
        hub_height=100.0, rotor_diameter=100.0, TSR=8.0,
        operation_model=OPNAME,
        power_thrust_table=dict(
            ref_air_density=yt.RHO, ref_tilt=0.0,
            cosine_loss_exponent_yaw=yt.COSINE_LOSS_EXPONENT_YAW,
            cosine_loss_exponent_tilt=yt.COSINE_LOSS_EXPONENT_YAW,
            wind_speed=[float(x) for x in np.arange(0.0, 41.0, 1.0)],
            power=[0.0] * 41,
            thrust_coefficient=[yt.CT_EFF] * 41,
        ),
    )])
    m.set(wind_shear=0.12, reference_wind_height=100.0, air_density=yt.RHO)
    for key, val in (('velocity_model', 'gauss'), ('deflection_model', 'gauss'),
                     ('turbulence_model', 'crespo_hernandez'),
                     ('combination_model', 'sosfs')):
        m.set_param(['wake', 'model_strings', key], val)
    for k in ('enable_secondary_steering', 'enable_yaw_added_recovery',
              'enable_transverse_velocities'):
        m.set_param(['wake', k], True)
    m.set_param(['wake', 'enable_active_wake_mixing'], False)
    m.set_param(['solver', 'type'], 'turbine_grid')
    m.set_param(['solver', 'turbine_grid_points'], 3)
    return m


def set_a(m, us, a_vec):
    """逐台 a → (风场总功率 [MW], 逐台功率 [MW])。"""
    global A_OVERRIDE
    us = np.atleast_1d(np.asarray(us, dtype=float))
    a_vec = np.asarray(a_vec, dtype=float).reshape(-1)
    n, nt = us.size, a_vec.size
    A_OVERRIDE = np.tile(a_vec, (n, 1))
    try:
        m.set(wind_speeds=list(us), wind_directions=[270.0] * n,
              turbulence_intensities=[yt.TI] * n, yaw_angles=np.zeros((n, nt)))
        with _solver_hook()():
            m.run()
        farm = np.asarray(m.get_farm_power(), dtype=float) / 1e6
        turb = np.asarray(m.get_turbine_powers(), dtype=float) / 1e6
    finally:
        A_OVERRIDE = None
    return farm, turb


def main():
    m = build_patched()
    prod, _ = yt.build_model()
    u0 = 10.0
    p_prod = yt.farm_power(prod, u0, [0.0] * 5)
    prod.set(wind_speeds=[u0], wind_directions=[270.0], turbulence_intensities=[0.06],
             yaw_angles=np.zeros((1, 5)))
    prod.run()
    pw_prod = prod.get_turbine_powers()[0] / 1e6

    farm, turb = set_a(m, [u0], [1 / 3] * 5)
    print('=' * 96)
    print('0. 自校验 (u=10, 逐台 a 全 = 1/3)')
    print('   生产 yawflor        = %.12f MW' % p_prod)
    print('   逐台 a 自定义操作模型 = %.12f MW   相对差 %.3e'
          % (farm[0], abs(farm[0] - p_prod) / p_prod))
    print('   逐台 生产 = %s' % np.array2string(pw_prod, precision=8))
    print('   逐台 本探针 = %s' % np.array2string(turb[0], precision=8))
    ok = (abs(farm[0] - p_prod) / p_prod < 1e-12
          and np.allclose(turb[0], pw_prod, rtol=0, atol=1e-12))
    print('   => %s' % ('逐台 a 通道可信 (与生产基线逐位一致)' if ok else
                        '⚠ 未复现基线 ⇒ 逐台 a 通道不可信, 后续数字无效'))

    print('\n' + '=' * 96)
    print('1. 逐台 a 的空间结构 (坐标上升, 目标 = 风场功率, 步长 0.025)')
    print('   ⚠ a=0 会让 GCH 高斯尾流除零 (uR 分母) ⇒ 网格从 0.025 起')
    print('=' * 96)
    grid = np.round(np.arange(0.025, 0.5001, 0.025), 6)
    for u in (8.0, 10.0, 12.0):
        a = np.full(5, 1.0 / 3.0)
        p0 = float(set_a(m, [u], a)[0][0])
        for _it in range(3):
            for j in range(5):
                cand = [(float(set_a(m, [u], np.where(np.arange(5) == j, g, a))[0][0]), g)
                        for g in grid]
                a[j] = max(cand)[1]
        p1 = float(set_a(m, [u], a)[0][0])
        print('  u=%4.1f 均匀 1/3 = %.6f MW | 逐台最优 a = %s -> %.6f MW (%+.4f MW, %+.3f%%)'
              % (u, p0, np.array2string(a, precision=4), p1, p1 - p0, 100 * (p1 / p0 - 1)))

    print('\n' + '=' * 96)
    print('2. 同一电量下的波动: 全场统一 a  vs  逐台斜坡 (真实风速序列)')
    print('=' * 96)
    u0s = np.clip(_load_seq(), 3.5, 25.0)
    print('  序列 n=%d 均值 %.3f 范围 [%.2f, %.2f]' % (len(u0s), u0s.mean(), u0s.min(), u0s.max()))

    m_uniform = build_md(_tables())
    us = np.round(np.arange(4.0, 25.0001, 0.5), 3)
    print('\n  [A 全场统一 a]  (原生 multidim_condition)')
    print('  %-24s %11s %10s %10s' % ('a', '电量MWh', '|dP|均值', 'dP_std'))
    for a in (1 / 3, 0.30, 0.25, 0.20, 0.15, 0.10):
        Pf = np.interp(u0s, us, a_curve(m_uniform, us, a))
        d = np.abs(np.diff(Pf))
        print('  %-24.4f %11.3f %10.4f %10.4f' % (a, Pf.sum() / 60, d.mean(), d.std()))

    print('\n  [B 逐台斜坡] a_vec = 上游低 / 下游高')
    print('  %-34s %11s %10s %10s' % ('a_vec', '电量MWh', '|dP|均值', 'dP_std'))
    ramps = [
        ('1/3 全体', [1 / 3] * 5),
        ('[.05 .10 .20 1/3 1/3]', [0.05, 0.10, 0.20, 1 / 3, 1 / 3]),
        ('[.05 .15 .25 1/3 1/3]', [0.05, 0.15, 0.25, 1 / 3, 1 / 3]),
        ('[.10 .20 .30 1/3 .40]', [0.10, 0.20, 0.30, 1 / 3, 0.40]),
        ('[.15 .25 1/3 .40 .45]', [0.15, 0.25, 1 / 3, 0.40, 0.45]),
        ('[.20 1/3 1/3 1/3 1/3]', [0.20, 1 / 3, 1 / 3, 1 / 3, 1 / 3]),
        ('[1/3 .20 .20 .20 .20]', [1 / 3, 0.20, 0.20, 0.20, 0.20]),
    ]
    import time
    for name, r in ramps:
        t0 = time.time()
        Pf = np.asarray(set_a(m, u0s, r)[0])
        dt = time.time() - t0
        d = np.abs(np.diff(Pf))
        print('  %-34s %11.3f %10.4f %10.4f  (%.1f s)'
              % (name, Pf.sum() / 60, d.mean(), d.std(), dt))

    print('\n  ⇒ 判据: 找电量相近的 A/B 两行比较 dP_std。')
    print('     B 在同电量下 dP_std 明显更低 ⇒ 逐台 a 有真实价值（值得扩引擎）；')
    print('     若 A 已覆盖同一前沿 ⇒ 全场统一 a 就够，不必动引擎。')

    # ------------------------------------------------------------------
    # 3. 真正公平的对比: **优化**逐台 a，看能否压过"全场统一 a"的前沿
    # ------------------------------------------------------------------
    print('\n' + '=' * 96)
    print('3. 公平对比: 对每个电量目标**优化**逐台 a（坐标上升最小化 |dP| 均值）')
    print('   目标 J = |dP|均值 + w·max(0, E_target − E)  (惩罚少发电)')
    print('=' * 96)
    uni = {}
    for a in (1 / 3, 0.30, 0.25, 0.20, 0.15, 0.10):
        Pf = np.interp(u0s, us, a_curve(m_uniform, us, a))
        uni[round(a, 4)] = (Pf.sum() / 60, np.abs(np.diff(Pf)).mean())
    print('  [A 全场统一 a 前沿]')
    for a, (E, D) in sorted(uni.items(), key=lambda kv: -kv[1][0]):
        print('     a=%.4f  E=%.2f MWh  |dP|均值=%.4f' % (a, E, D))

    g = np.round(np.arange(0.05, 0.4601, 0.05), 6)

    def obj(avec, E_target, w=0.5):
        Pf = np.asarray(set_a(m, u0s, avec)[0])
        E = Pf.sum() / 60
        D = np.abs(np.diff(Pf)).mean()
        return D + w * max(0.0, E_target - E), E, D

    print('\n  [B 逐台 a 前沿 (7 维候选, 坐标上升 2 轮)]')
    print('  %10s | %10s %10s | %10s %10s | %s'
          % ('E_target', 'E_uniform', 'D_uniform', 'E_per', 'D_per', 'a_vec'))
    for E_target in (122.5, 121.4, 117.7, 112.9, 105.4, 91.6):
        # 统一 a 的对照行（取电量最接近者）
        au, (Eu, Du) = min(uni.items(), key=lambda kv: abs(kv[1][0] - E_target))
        # 逐台优化：从均匀解起步
        best = None
        for start in ([1 / 3] * 5, [0.10, 0.20, 0.30, 0.40, 0.45]):
            cur = list(start)
            j, E, D = obj(cur, E_target)
            for _it in range(2):
                for k in range(5):
                    for gv in g:
                        cand = list(cur)
                        cand[k] = gv
                        jc, Ec, Dc = obj(cand, E_target)
                        if jc < j - 1e-12:
                            j, E, D, cur = jc, Ec, Dc, cand
            if best is None or j < best[0]:
                best = (j, E, D, list(cur))
        _, E, D, avec = best
        print('  %10.1f | %10.2f %10.4f | %10.2f %10.4f | %s'
              % (E_target, Eu, Du, E, D, np.array2string(np.asarray(avec), precision=3)))
    print('\n  读法: 同一 E_target 下 D_per < D_uniform ⇒ 逐台 a 有真实价值。')


def _load_seq():
    import csv as _csv
    import glob
    cands = []
    for pat in ('FFC_speed_1.csv', 'fixed_day_sequence.csv', 'data_piecewise/*.csv'):
        cands += glob.glob(pat)
    for f in cands:
        try:
            with open(f, encoding='utf-8', errors='replace') as fh:
                rd = list(_csv.reader(fh))
            if len(rd) < 1000:
                continue
            vals = []
            for row in rd:
                for cell in row:
                    try:
                        v = float(cell)
                    except ValueError:
                        continue
                    if 0.0 <= v <= 40.0:
                        vals.append(v)
                        break
            if len(vals) >= 1440:
                return np.asarray(vals[:1440], dtype=float)
        except Exception:
            continue
    rng = np.random.default_rng(0)
    return 10.0 + 2.0 * np.sin(np.arange(1440) / 180.0) + rng.normal(0, 0.4, 1440)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
