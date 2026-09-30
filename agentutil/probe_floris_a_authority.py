# -*- coding: utf-8 -*-
"""临时探针②：轴向因子 a 通道的**调控权限**与**可行性**（对照偏航通道）。

问题
----
现有 `yawflor` 已实测：偏航通道 γ∈[−25°,25°] 的增益 **+39%**，但绝对波动**同比放大**，
相对工作点 γ* 的下探余量只有 **≈30%**，且 γ≡0 时"削减"只有 2.6%
⇒ 能量与波动只能折中。

本探针回答：把被控量换成**轴向因子 a**（γ≡0 固定）后，
  ① a 通道能提供多大的**降功率**权限（相对停机/最优点的下探量）
  ② a 通道的降功率是否**同时**降低绝对波动（|ΔP| 的 std），即是否存在"帕累托前沿"
  ③ 对比 γ 通道（同一组功率电平下的波动）

只用 FLORIS 的 `multidim_condition` 多维表（**全场统一 a**，原生支持、零引擎改动）。
逐台 a 需要引擎扩展，另见结论段。

用法: F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_floris_a_authority.py
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

OUT = os.path.join(ROOT, 'output', '_probe_a')
U_TABLE_MAX = 40.0
A_GRID = np.round(np.arange(0.0, 0.5001, 0.025), 6)


def _tables():
    """(a, u) -> (Cp_eff, CT)，以及多维表 CSV。"""
    us = np.arange(0.0, U_TABLE_MAX + 1e-9, 1.0)
    lines = ['ai,ws,power,thrust_coefficient']
    for a in A_GRID:
        cp = 4.0 * a * (1.0 - a) ** 2
        ct = 4.0 * a * (1.0 - a)
        for u in us:
            pk = yt.POWER_SCALE * 0.5 * yt.RHO * yt.AREA * cp * u ** 3 / 1e3
            lines.append('%.17g,%.17g,%.17g,%.17g' % (a, u, pk, ct))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'betz_a_grid_multidim.csv')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def build_md(csv_path, nt=5, operation_model='cosine-loss'):
    from floris import FlorisModel
    m = FlorisModel('defaults')
    m.set(layout_x=[i * 500.0 for i in range(nt)], layout_y=[0.0] * nt)
    m.set(turbine_type=[dict(
        turbine_type='probe_a_grid',
        hub_height=100.0, rotor_diameter=100.0, TSR=8.0,
        operation_model=operation_model,
        multi_dimensional_cp_ct=True,
        power_thrust_table=dict(
            power_thrust_data_file=csv_path,
            ref_air_density=yt.RHO, ref_tilt=0.0,
            cosine_loss_exponent_yaw=yt.COSINE_LOSS_EXPONENT_YAW,
            cosine_loss_exponent_tilt=yt.COSINE_LOSS_EXPONENT_YAW,
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


def a_curve(m, us, a):
    """给定均匀 a，返回逐风速的风场总功率 [MW]。"""
    n = len(us)
    m.set(wind_speeds=list(us), wind_directions=[270.0] * n,
          turbulence_intensities=[yt.TI] * n, yaw_angles=np.zeros((n, 5)),
          multidim_conditions={'ai': float(a)})
    m.run()
    return np.asarray(m.get_farm_power(), dtype=float) / 1e6


def main():
    csv = _tables()
    m = build_md(csv)
    us = np.round(np.arange(4.0, 25.0001, 0.5), 3)

    print('=' * 96)
    print('① 均匀轴向因子 a 的风场功率曲线 P_farm(u)  —— "a 通道有多少降功率权限"')
    print('=' * 96)
    curves = {}
    for a in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 1 / 3, 0.40, 0.45, 0.50):
        curves[round(float(a), 6)] = a_curve(m, us, a)
    ref = curves[round(1 / 3, 6)]
    print('  %-8s %s' % ('a', ''.join('%9.1f' % u for u in us[::4])))
    for a, c in curves.items():
        print('  %-8.4f %s   均值 %.4f MW' % (a, ''.join('%9.4f' % v for v in c[::4]), c.mean()))
    print('\n  相对 a=1/3 的电量比:')
    for a, c in curves.items():
        print('    a=%.4f  ΣP/ΣP(1/3) = %.4f   (单机 Cp/Cp_max = %.4f)'
              % (a, c.sum() / ref.sum(), (4 * a * (1 - a) ** 2) / (4 / 27)))

    print('\n' + '=' * 96)
    print('② 调控权限: 在真实风速序列下, a 通道能否"降功率且降波动"')
    print('=' * 96)
    # 与 yawflor README 表 B/C 同口径: 用一条真实风速日序列
    seq = _load_sequence()
    if seq is None:
        print('  (没找到风速序列文件, 退化为合成序列)')
        rng = np.random.default_rng(0)
        seq = 10.0 + 2.0 * np.sin(np.arange(1440) / 180.0) + rng.normal(0, 0.4, 1440)
    u0 = np.clip(seq, 3.5, 25.0)
    print('  风速序列: n=%d, 均值 %.3f, 范围 [%.2f, %.2f]' % (len(u0), u0.mean(), u0.min(), u0.max()))

    print('\n  a 通道 (全场统一 a):')
    print('  %-8s %11s %9s %9s %9s %9s' % ('a', '电量MWh', '增益%', '|ΔP|均值', 'ΔP_std', 'max|ΔP|'))
    rows_a = []
    for a, c in sorted(curves.items()):
        Pf = np.interp(u0, us, c)
        d = np.abs(np.diff(Pf))
        rows_a.append((a, Pf.sum() / 60.0, d.mean(), d.std(), d.max()))
    base_wh = rows_a[[r[0] for r in rows_a].index(round(1 / 3, 6))][1]
    for a, wh, dm, ds, dx in rows_a:
        print('  %-8.4f %11.3f %+9.2f %9.4f %9.4f %9.4f'
              % (a, wh, 100 * (wh / base_wh - 1), dm, ds, dx))

    print('\n  γ 通道 (单维表, 固定 a=1/3; 序列同口径):')
    base_model, _ = yt.build_model()
    gammas = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    print('  %-8s %11s %9s %9s %9s %9s' % ('γ_up', '电量MWh', '增益%', '|ΔP|均值', 'ΔP_std', 'max|ΔP|'))
    rows_g = []
    for g in gammas:
        Pf = np.asarray(yt.eval_batch(base_model, u0, np.full((len(u0), 5), g))[0])
        d = np.abs(np.diff(Pf))
        rows_g.append((g, Pf.sum() / 60.0, d.mean(), d.std(), d.max()))
    g_base = rows_g[0][1]
    for g, wh, dm, ds, dx in rows_g:
        print('  %-8.1f %11.3f %+9.2f %9.4f %9.4f %9.4f'
              % (g, wh, 100 * (wh / g_base - 1), dm, ds, dx))

    print('\n  ⇒ 判据: 若某 a 档的"电量"与某 γ 档相近, 而 |ΔP| 明显更低 ⇒ a 通道在')
    print('     "同电量下更平"这一点上**优于** γ 通道; 反之亦然。')


def _load_sequence():
    import glob
    for pat in ('output/_probe_a/*.csv', 'data_piecewise/*.csv', 'fixed_day_sequence.csv',
                'FFC_speed_1.csv'):
        for f in glob.glob(pat):
            try:
                import csv as _csv
                with open(f, encoding='utf-8', errors='replace') as fh:
                    rd = list(_csv.reader(fh))
                if len(rd) < 100:
                    continue
                vals = []
                for row in rd[1:]:
                    for cell in row:
                        try:
                            v = float(cell)
                        except ValueError:
                            continue
                        if 0.0 <= v <= 40.0:
                            vals.append(v)
                            break
                if len(vals) >= 1000:
                    return np.asarray(vals[:1440], dtype=float)
            except Exception:
                continue
    return None


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
