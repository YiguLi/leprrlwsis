#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_floris_yaw_structure.py —— FLORIS 偏航环境（新方法）的**功率结构与可行性探针**。

背景
----
拟新建一个方法：环境换成 FLORIS，动作从"5 维轴向因子"改为"5 维偏航角"，
轴向因子**固定为 a = 0.33**（单机 Betz 点）；风机/布局/电价/储能与现有 env 完全相同。
本脚本**只读**，不改任何生产代码，用来回答"这个环境合不合理、能不能用"：

1. **单机标定是否精确** —— 隔离机组（无尾流）的 FLORIS 功率 vs 解析式
   `P = 0.95·½ρA·Cp(a)·U³`，`Cp(a) = 4a(1−a)²`（与 `WFSim_port.m` 同口径）。
2. **偏航对风场总功率的结构** —— 均匀族扫描（5 个风速 × 7 个偏航角）。
3. **风场最优是不是"上游偏航、下游归零"的斜坡** —— 逐台串行精化搜索（serial refine）。
4. **H 层的动作权限** —— 在 ±25° 带内可达的功率区间 [P_min, P_max]（含"削减"能力）。
5. **不变量** —— 镜像对称 P(+γ) = P(−γ)、可重复性（无记忆/确定性）。
6. **调用耗时** —— 单条 vs 批量向量化，用于估算训练/评测的物理调用预算。
7. **模型口径敏感性** —— 尾流模型组合、偏航损失指数 `cosine_loss_exponent_yaw`。
8. **真实风速序列下的波动量级** —— 用 `FFC_speed_noise_1.csv` 的 1440 分钟序列
   算 |ΔP_w|(γ=0)，与并网波动限值 3 MW 对比，判断"风侧平抑"这一项在新环境里的分量。

物理口径（对齐 `WFSim_port.m` / `bin/core/meshing.m` 的 windTunnel 场景）
------------------------------------------------------------------------
* 转子直径 D = 100 m；单排 5 台，间距 5D（x = 0/500/1000/1500/2000，y = 0）；
* rho = 1.2；`powerscale = 0.95`；a ≡ 1/3 ⇒ Cp = 4a(1−a)² = 0.59259、Cp_eff = 0.56296；
* 推力系数：WFSim 里 `Fthrust ∝ CT_prime/(1−a)` 且 `CT_prime = Cp(a)`
  ⇒ 有效 CT = Cp/(1−a) = 8/9 = 0.88889；
* **无额定功率上限**（WFSim 口径，`Power ∝ U³` 无截断）；env 侧沿用 `3.5 ≤ u ≤ 25` 守卫。
* 风向固定沿机组排布方向（WFSim 侧 `wind_angle = 0`）⇒ FLORIS 的 `wind_directions = 270`
  （FLORIS 用气象"来向"约定：0=N、90=E；270 = 西风 = 沿 +x）。

运行
----
    python agentutil/probe_floris_yaw_structure.py
    python agentutil/probe_floris_yaw_structure.py --skip-opt --skip-series
    python agentutil/probe_floris_yaw_structure.py --json output/probe_floris_yaw.json

前置：`pip install floris`（本项目实测 4.6.6；需要 numpy≥2、pandas 2~3）。
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np

# Windows 控制台默认 GBK，报告里的 "−"/"⇒" 等字符会让 print 直接抛 UnicodeEncodeError。
# 统一把 stdout 切到 UTF-8（errors='replace' 兜底），并用 --log 落一份 UTF-8 报告。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                            # pragma: no cover
    pass


class _Tee:
    """把 stdout 同时写进一个 UTF-8 文件（便于归档探针证据、规避控制台编码）。"""

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = io.open(path, 'w', encoding='utf-8')

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def close(self):
        self.fh.close()

# ---- 路径 bootstrap（可被任意 CWD 调用）------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

# matplotlib 会尝试写 `~/.matplotlib` 字体缓存；在受限沙箱/只读 HOME 下会报权限错
# （实测 `Could not save font_manager cache ... Permission denied`）。指到 output/ 下。
os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, 'output', '_mplcache'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# ---- 物理常量（单一真源：全部对齐 WFSim 现有口径，见模块 docstring）----------
D_ROTOR = 100.0
RHO = 1.20
AREA = np.pi * (D_ROTOR / 2.0) ** 2
A_IND = 1.0 / 3.0
CP_A = 4.0 * A_IND * (1.0 - A_IND) ** 2          # 0.59259
POWER_SCALE = 0.95                                # meshing.m: powerscale
CP_EFF = CP_A * POWER_SCALE                       # 0.56296
CT_EFF = CP_A / (1.0 - A_IND)                     # 0.88889 = 8/9
NT = 5
SPACING = 5.0 * D_ROTOR                           # 500 m
LAYOUT_X = [i * SPACING for i in range(NT)]
LAYOUT_Y = [0.0] * NT
WIND_DIR_ALIGNED = 270.0                          # 沿 +x 吹（对齐排布）
TI = 0.06
GAMMA_MAX = 25.0                                  # ±25°：FLORIS 主流的 wake steering 动作域
US_DEFAULT = (6.0, 8.0, 10.0, 12.0, 14.0)         # 与 probe_a_farm_structure.py 同组
U_MIN_ENV, U_MAX_ENV = 3.5, 25.0                  # env.py::PowerOutput 的守卫
U_TABLE_MAX = 40.0                                # 功率表上界：必须 > U_MAX_ENV，
                                                  # 否则 interp1d(fill_value=0) 会把高风速截成 0


def cp_analytic(u, a=A_IND):
    """单机功率 [MW]（自由流、无尾流），与 WFSim_port.m 的 Cp=4a(1−a)² 同口径。"""
    cp = 4.0 * a * (1.0 - a) ** 2
    return POWER_SCALE * 0.5 * RHO * AREA * cp * np.asarray(u, dtype=float) ** 3 / 1e6


def build_turbine_dict(cosine_loss_exponent_yaw=1.88, ref_tilt=0.0):
    """固定 a=1/3 的风机：常数 Cp_eff / CT_eff 的 power_thrust_table。

    FLORIS v4 的表列是 `wind_speed` / `power`(kW, 绝对电功率) / `thrust_coefficient`；
    不接受 `cp`/`ct` 列，也没有 `cut_in/out_wind_speed`（用 0 行表达）。
    """
    u = np.arange(0.0, U_TABLE_MAX + 1e-9, 1.0)
    power_kw = cp_analytic(u) * 1e3                   # MW → kW
    return dict(
        turbine_type='wfsim_betz_a033',
        hub_height=D_ROTOR,                           # 2D 模型无轮毂高度；取 D，且置 wind_shear=0
        rotor_diameter=D_ROTOR,
        TSR=8.0,
        operation_model='cosine-loss',
        power_thrust_table=dict(
            ref_air_density=RHO,
            ref_tilt=ref_tilt,
            cosine_loss_exponent_yaw=cosine_loss_exponent_yaw,
            cosine_loss_exponent_tilt=cosine_loss_exponent_yaw,
            wind_speed=[float(x) for x in u],
            power=[float(x) for x in power_kw],
            thrust_coefficient=[float(CT_EFF)] * len(u),
        ),
    )


def build_model(velocity_model='gauss', deflection_model='gauss',
                turbulence_model='crespo_hernandez', combination_model='sosfs',
                cosine_loss_exponent_yaw=1.88, nt=NT, wake_flags=None):
    """构造 FlorisModel；返回 (model, info)。

    `wake_flags` 用于覆盖 `wake:` 下的布尔开关（`enable_secondary_steering` 等）——
    实测 `empirical_gauss` 与 `secondary_steering` 不兼容（`NotImplementedError`），
    所以换模型时必须能关掉它。
    """
    from floris import FlorisModel

    f = FlorisModel('defaults')
    # 顺序要求（实测）：必须先定 layout 再给 turbine_type —— 否则
    # farm.check_turbine_type 会拿默认布局的 3 台去比 "nt 个 turbine_type" 而报错。
    f.set(layout_x=[float(x) for x in LAYOUT_X[:nt]],
          layout_y=[float(y) for y in LAYOUT_Y[:nt]])
    f.set(turbine_type=[build_turbine_dict(cosine_loss_exponent_yaw)])
    f.set(wind_shear=0.0, reference_wind_height=D_ROTOR, air_density=RHO)
    for key, val in (('velocity_model', velocity_model),
                     ('deflection_model', deflection_model),
                     ('turbulence_model', turbulence_model),
                     ('combination_model', combination_model)):
        f.set_param(['wake', 'model_strings', key], val)
    for key, val in (wake_flags or {}).items():
        f.set_param(['wake', key], bool(val))
    info = dict(floris_version=__import__('floris').__version__,
                velocity_model=velocity_model, deflection_model=deflection_model,
                turbulence_model=turbulence_model, combination_model=combination_model,
                cosine_loss_exponent_yaw=cosine_loss_exponent_yaw,
                enable_secondary_steering=f.get_param(['wake', 'enable_secondary_steering']),
                enable_yaw_added_recovery=f.get_param(['wake', 'enable_yaw_added_recovery']),
                enable_transverse_velocities=f.get_param(['wake', 'enable_transverse_velocities']),
                air_density=RHO, ti=TI, wind_direction=WIND_DIR_ALIGNED,
                layout_x=list(LAYOUT_X[:nt]), layout_y=list(LAYOUT_Y[:nt]),
                rotor_diameter=D_ROTOR, cp_a033=CP_A, cp_eff=CP_EFF, ct_eff=CT_EFF,
                a_induced=A_IND, power_scale=POWER_SCALE,
                table_max_u=U_TABLE_MAX, gamma_max=GAMMA_MAX)
    return f, info


def eval_batch(f, us, yaws):
    """一次向量化调用算多组 (u, γ) 的风场总功率。

    Parameters
    ----------
    us : array (n,)
    yaws : array (n, nt)

    Returns
    -------
    (farm_mw (n,), turb_mw (n, nt))
    """
    us = np.atleast_1d(np.asarray(us, dtype=float))
    yaws = np.atleast_2d(np.asarray(yaws, dtype=float))
    n = int(us.size)
    if yaws.shape[0] != n:
        raise ValueError('us 与 yaws 行数不一致: %d vs %d' % (n, yaws.shape[0]))
    keep = np.clip(us, 1e-6, None)                    # FLORIS 不接受 u=0 的 0 功率条件
    f.set(wind_speeds=list(keep), wind_directions=[WIND_DIR_ALIGNED] * n,
          turbulence_intensities=[TI] * n, yaw_angles=yaws)
    f.run()
    farm = np.asarray(f.get_farm_power(), dtype=float) / 1e6
    turb = np.asarray(f.get_turbine_powers(), dtype=float) / 1e6
    farm = np.where(us >= U_MIN_ENV, farm, 0.0)       # 与 env.py 的守卫同口径
    return farm, turb


def hdr(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


# ---------------------------------------------------------------------------
# 1. 单机标定
# ---------------------------------------------------------------------------
def sec_calibration():
    hdr('[1] 单机标定：隔离机组的 FLORIS 功率 vs 解析式 0.95·½ρA·Cp(a)·U³')
    f, info = build_model(nt=1)
    us = np.array([6.0, 8.0, 10.0, 12.0, 14.0])
    farm, _ = eval_batch(f, us, np.zeros((us.size, 1)))
    ref = cp_analytic(us)
    print('  u [m/s]   FLORIS [MW]   解析 [MW]    相对误差')
    worst = 0.0
    for u, got, want in zip(us, farm, ref):
        rel = abs(got - want) / want
        worst = max(worst, rel)
        print('  %6.1f   %11.6f  %11.6f   %8.2e' % (u, got, want, rel))
    print('  ⇒ 最大相对误差 %.2e  %s' % (worst, 'OK（逐位级）' if worst < 1e-9 else '⚠ 有偏差'))
    return dict(us=us.tolist(), floris=farm.tolist(), analytic=ref.tolist(),
                max_rel_err=float(worst))


# ---------------------------------------------------------------------------
# 2. 均匀族扫描
# ---------------------------------------------------------------------------
def sec_uniform_sweep(us, gammas):
    hdr('[2] 均匀偏航族扫描（全场同角）—— 功率随偏航的结构')
    f, _ = build_model()
    rows = []
    print('  γ\\u      ' + ''.join('%11.1f' % u for u in us))
    for g in gammas:
        y = np.tile(np.array([g] * NT), (len(us), 1))
        farm, _ = eval_batch(f, us, y)
        rows.append((g, farm))
        print(('  %+6.1f°  ' + ''.join('%11.4f' % p for p in farm)) % g)
    base = dict(rows)[0.0] if 0.0 in dict(rows) else None
    if base is not None:
        print('\n  相对 γ=0 的变化 [%]：')
        print('  γ\\u      ' + ''.join('%11.1f' % u for u in us))
        for g, farm in rows:
            print('  %+6.1f°  ' % g + ''.join('%+11.2f' % (100 * (p / b - 1))
                                              for p, b in zip(farm, base)))
    return dict(us=list(us), gammas=[float(g) for g in gammas],
                farm_mw={str(g): list(map(float, farm)) for g, farm in rows})


# ---------------------------------------------------------------------------
# 3. 风场最优形态（串行精化搜索）
# ---------------------------------------------------------------------------
def serial_refine(f, u, passes=((0, 5, 15), (0, 1.25, 2.5, 5.0)), gamma_max=GAMMA_MAX):
    """逐台串行精化：每轮对每台机组试一组候选角，取风场总功率最大者。

    与 FLORIS 官方 `YawOptimizationSR` 同思路（此处为独立实现，不依赖 floris.optimization）。
    返回 (best_yaws, best_power_mw, trace)。
    """
    y = np.zeros(NT)
    trace = []
    for pi, deltas in enumerate(passes):
        step = 1.0 if pi == 0 else 0.25
        for i in range(NT):
            cand = np.unique(np.clip(y[i] + step * np.asarray(deltas, dtype=float),
                                     -gamma_max, gamma_max))
            Y = np.tile(y, (cand.size, 1))
            Y[:, i] = cand
            farm, _ = eval_batch(f, [u] * cand.size, Y)
            k = int(np.argmax(farm))
            y = Y[k].copy()
            trace.append(dict(pass_=pi, turbine=int(i), gamma=float(y[i]),
                              farm_mw=float(farm[k])))
    farm, _ = eval_batch(f, [u], y[None, :])
    return y, float(farm[0]), trace


def sec_optimum(us, do_opt=True):
    hdr('[3] 风场最优偏航形态（上游转向、下游归零？）')
    f, _ = build_model()
    out = {}
    if not do_opt:
        print('  （--skip-opt：跳过）')
        return out
    print('  u [m/s]  γ=0 基线   最优总功率   增益 %    最优偏航角 [deg]（1→5）')
    for u in us:
        base, _ = eval_batch(f, [u], np.zeros((1, NT)))
        y, best, _ = serial_refine(f, u)
        gain = 100 * (best / base[0] - 1)
        out[float(u)] = dict(base_mw=float(base[0]), opt_mw=float(best),
                             gain_pct=float(gain), yaws=[float(v) for v in y])
        print('  %6.1f   %10.4f   %10.4f  %+7.2f    %s'
              % (u, base[0], best, gain, np.round(y, 2).tolist()))
    return out


# ---------------------------------------------------------------------------
# 4. H 层动作权限：可达功率区间
# ---------------------------------------------------------------------------
def greedy_extreme(f, u, maximize=True, passes=((0, 6.25, 12.5, 18.75, 25),), gamma_max=GAMMA_MAX):
    """逐台贪心找带内的**最大/最小**风场总功率（`maximize=False` 时是"削减能力"）。

    与 `serial_refine` 同结构，只是目标相反；候选角在 `[-γ_max, +γ_max]` 内。
    """
    y = np.zeros(NT)
    for deltas in passes:
        for i in range(NT):
            cand = np.unique(np.clip(y[i] + np.asarray(deltas, dtype=float),
                                     -gamma_max, gamma_max))
            Y = np.tile(y, (cand.size, 1))
            Y[:, i] = cand
            farm, _ = eval_batch(f, [u] * cand.size, Y)
            y = Y[int(np.argmax(farm) if maximize else np.argmin(farm))].copy()
    farm, _ = eval_batch(f, [u], y[None, :])
    return y, float(farm[0])


def sec_authority(us):
    hdr('[4] H 层动作权限：±25° 带内可达的功率区间（含"削减"能力）')
    f, _ = build_model()
    out = {}
    print('  u [m/s]   γ=0 [MW]  带内最大 [MW] (最优角)      带内最小 [MW] (最劣角)      可增 %   可减 %')
    for u in us:
        base, _ = eval_batch(f, [u], np.zeros((1, NT)))
        ymax, pmax = greedy_extreme(f, u, maximize=True)
        ymin, pmin = greedy_extreme(f, u, maximize=False)
        up = 100 * (pmax / base[0] - 1)
        dn = 100 * (pmin / base[0] - 1)
        out[float(u)] = dict(base_mw=float(base[0]), pmax_mw=float(pmax), pmin_mw=float(pmin),
                             up_pct=float(up), down_pct=float(dn),
                             yaws_max=[float(v) for v in ymax],
                             yaws_min=[float(v) for v in ymin])
        print('  %6.1f   %9.4f  %9.4f %-18s %9.4f %-18s %+6.2f  %+6.2f'
              % (u, base[0], pmax, str(np.round(ymax, 1).tolist()), pmin,
                 str(np.round(ymin, 1).tolist()), up, dn))
    print('  注：可减 % > 0 表示"最劣偏航"仍比 γ=0 出力高 ⇒ 该带内**没有削减能力**。')
    return out


# ---------------------------------------------------------------------------
# 5. 不变量：镜像对称 + 可重复性
# ---------------------------------------------------------------------------
def sec_invariants():
    hdr('[5] 不变量：镜像对称 P(+γ)=P(−γ) 与可重复性（无记忆/确定性）')
    f, _ = build_model()
    rng = np.random.RandomState(0)
    Y = rng.uniform(-GAMMA_MAX, GAMMA_MAX, size=(6, NT))
    Y = np.vstack([Y, np.zeros((1, NT)), np.full((1, NT), GAMMA_MAX)])
    u = np.full(Y.shape[0], 10.0)
    p_pos, _ = eval_batch(f, u, Y)
    p_neg, _ = eval_batch(f, u, -Y)
    d = np.max(np.abs(p_pos - p_neg))
    print('  6 组随机偏航模式 + 全 0 + 全 25°：max|P(γ) − P(−γ)| = %.3e MW  %s'
          % (d, 'OK' if d < 1e-9 else '⚠ 不对称'))
    p_again, _ = eval_batch(f, u, Y)
    print('  同一批条件重复调用：逐位相同 = %s' % np.array_equal(p_pos, p_again))
    return dict(mirror_max_abs_diff_mw=float(d),
                repeat_bitwise_equal=bool(np.array_equal(p_pos, p_again)))


# ---------------------------------------------------------------------------
# 6. 调用耗时
# ---------------------------------------------------------------------------
def sec_timing():
    hdr('[6] 调用耗时（训练/评测的物理预算）')
    f, _ = build_model()
    y = np.zeros((1, NT))
    eval_batch(f, [10.0], y)                                     # warmup（含首次编译）
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        eval_batch(f, [10.0], y)
    t_single = (time.perf_counter() - t0) / n
    us = np.linspace(6.0, 14.0, 200)
    Y = np.zeros((us.size, NT))
    t0 = time.perf_counter()
    for _ in range(10):
        eval_batch(f, us, Y)
    t_batch = (time.perf_counter() - t0) / 10
    print('  单条 (set+run+get, 5 台)：%.2f ms/次' % (t_single * 1e3))
    print('  批量 200 条一次调用   ：%.2f ms/批 ⇒ %.3f ms/条'
          % (t_batch * 1e3, t_batch / us.size * 1e3))

    # 拆解：set(wind_speeds) / set_operation(yaw) / run / get 各占多少 ——
    # 决定 env 的调用路径怎么写（每次重建 Core 很贵）。
    yaw_m = np.zeros((1, NT))
    t_set_u = t_set_op = t_run = t_get = 0.0
    for _ in range(n):
        t = time.perf_counter()
        f.set(wind_speeds=[10.0], wind_directions=[WIND_DIR_ALIGNED],
              turbulence_intensities=[TI])
        t_set_u += time.perf_counter() - t
        t = time.perf_counter()
        f.set_operation(yaw_angles=yaw_m)
        t_set_op += time.perf_counter() - t
        t = time.perf_counter()
        f.run()
        t_run += time.perf_counter() - t
        t = time.perf_counter()
        _ = f.get_farm_power()
        t_get += time.perf_counter() - t
    print('  拆解（每次都改风速 + 偏航）：set(u) %.2f / set_operation(yaw) %.2f / '
          'run %.2f / get %.2f ms'
          % (t_set_u / n * 1e3, t_set_op / n * 1e3, t_run / n * 1e3, t_get / n * 1e3))
    # 只改偏航（风速不变）的路径
    f.set(wind_speeds=[10.0], wind_directions=[WIND_DIR_ALIGNED], turbulence_intensities=[TI])
    t0 = time.perf_counter()
    for _ in range(n):
        f.set_operation(yaw_angles=yaw_m)
        f.run()
        _ = f.get_farm_power()
    t_yaw_only = (time.perf_counter() - t0) / n
    print('  只改偏航（风速不动）：%.2f ms/次' % (t_yaw_only * 1e3))
    print('  一个 episode = 1440 步、每步 ≥1 次物理调用 ⇒ 单条口径 %.1f s/episode'
          % (t_single * 1440))
    return dict(single_ms=float(t_single * 1e3), batch200_ms=float(t_batch * 1e3),
                per_call_in_batch_ms=float(t_batch / us.size * 1e3),
                breakdown_ms=dict(set_u=float(t_set_u / n * 1e3),
                                  set_operation=float(t_set_op / n * 1e3),
                                  run=float(t_run / n * 1e3), get=float(t_get / n * 1e3)),
                yaw_only_ms=float(t_yaw_only * 1e3),
                est_s_per_episode_single=float(t_single * 1440))


# ---------------------------------------------------------------------------
# 6b. 结构诊断：排布/朝向是否如预期、镜像不对称从哪来
# ---------------------------------------------------------------------------
def sec_diagnostics():
    hdr('[6b] 结构诊断：排布/朝向 + 镜像不对称的来源')
    print('  逐台功率 [MW]（u=10, γ=0）：')
    for n in (1, 2, 3, 5):
        f, _ = build_model(nt=n)
        farm, turb = eval_batch(f, [10.0], np.zeros((1, n)))
        print('    同排 n=%d (风向 270°, 沿 +x)：%s  总 %.4f'
              % (n, np.round(turb[0], 4).tolist(), farm[0]))
    f, _ = build_model(nt=5)
    f.set(wind_speeds=[10.0], wind_directions=[0.0], turbulence_intensities=[TI],
          yaw_angles=np.zeros((1, 5)))
    f.run()
    print('    横风 n=5 (风向 0°)         ：%s'
          % np.round(np.asarray(f.get_turbine_powers())[0] / 1e6, 4).tolist())
    f, _ = build_model(nt=5)
    f.set(wind_speeds=[10.0], wind_directions=[270.0], turbulence_intensities=[TI],
          yaw_angles=np.zeros((1, 5)))
    f.run_no_wake()
    print('    无尾流（run_no_wake, 270°） ：%s  总 %.4f'
          % (np.round(np.asarray(f.get_turbine_powers())[0] / 1e6, 4).tolist(),
             float(np.asarray(f.get_farm_power())[0]) / 1e6))
    print('    layout =', f.get_turbine_layout())

    print('\n  镜像不对称 vs wake 开关（前 4 台 ±20°、末台 0°）：')
    out = {}
    cases = [('默认(GCH 三个开关全开)', {}),
             ('secondary_steering off', {'enable_secondary_steering': False}),
             ('yaw_added_recovery off', {'enable_yaw_added_recovery': False}),
             ('transverse_velocities off', {'enable_transverse_velocities': False}),
             ('三个开关全关', {'enable_secondary_steering': False,
                               'enable_yaw_added_recovery': False,
                               'enable_transverse_velocities': False})]
    Y = np.array([[20.0, 20.0, 20.0, 20.0, 0.0], [-20.0, -20.0, -20.0, -20.0, 0.0]])
    for name, flags in cases:
        f, _ = build_model(wake_flags=flags)
        p, _ = eval_batch(f, [10.0, 10.0], Y)
        d = p[0] - p[1]
        out[name] = dict(p_plus=float(p[0]), p_minus=float(p[1]), diff_mw=float(d),
                         diff_pct=float(100 * (p[0] / p[1] - 1)))
        print('    %-28s P(+20)=%.4f  P(-20)=%.4f  Δ=%.4f (%+.2f%%)'
              % (name, p[0], p[1], d, 100 * (p[0] / p[1] - 1)))
    return out


# ---------------------------------------------------------------------------
# 7. 模型口径敏感性
# ---------------------------------------------------------------------------
def sec_sensitivity(us):
    hdr('[7] 口径敏感性：尾流模型组合 / 偏航损失指数')
    cases = [
        ('gch: gauss+gauss', dict(velocity_model='gauss', deflection_model='gauss')),
        ('empirical_gauss', dict(velocity_model='empirical_gauss',
                                 deflection_model='empirical_gauss',
                                 turbulence_model='wake_induced_mixing',
                                 wake_flags={'enable_secondary_steering': False,
                                             'enable_yaw_added_recovery': False})),
        ('jensen+jimenez', dict(velocity_model='jensen', deflection_model='jimenez',
                                turbulence_model='crespo_hernandez',
                                combination_model='max')),
        ('turboparkgauss+gauss', dict(velocity_model='turboparkgauss',
                                      deflection_model='gauss')),
        ('gch + cos^3.0', dict(velocity_model='gauss', deflection_model='gauss',
                               cosine_loss_exponent_yaw=3.0)),
    ]
    out = {}
    pat = np.array([[GAMMA_MAX, GAMMA_MAX, GAMMA_MAX, GAMMA_MAX, 0.0]])
    print('  口径                      u=10 基线 [MW]   u=10 固定模式* [MW]   增益 %')
    print('  * 固定模式 = 前 4 台 +25°、末台 0°（不重做优化，保证可比）')
    for name, kw in cases:
        try:
            f, _ = build_model(**kw)
            base, _ = eval_batch(f, [10.0], np.zeros((1, NT)))
            yawed, _ = eval_batch(f, [10.0], pat)
            gain = 100 * (yawed[0] / base[0] - 1)
            out[name] = dict(base_mw=float(base[0]), yawed_mw=float(yawed[0]),
                             gain_pct=float(gain))
            print('  %-24s %13.4f   %18.4f   %+7.2f' % (name, base[0], yawed[0], gain))
        except Exception as exc:                                  # noqa: BLE001
            out[name] = dict(error='%s: %s' % (type(exc).__name__, exc))
            print('  %-24s 失败：%s: %s' % (name, type(exc).__name__, exc))
    return out


# ---------------------------------------------------------------------------
# 8b. 真实一天的最优偏航轨迹 → 速率限制的代价（回答"5 min 一次 / 一次能调多少度"）
# ---------------------------------------------------------------------------
def sec_day_trajectory(delta_max_list=(2.5, 5.0, 10.0, 20.0, 1e9), col=1, n_steps=1440,
                       block=5, yaw_rate_deg_s=0.3, block_seconds=300.0,
                       return_traj=False):
    """按"每个 5 min 块重新求一次最优偏航"造出一天的最优轨迹，量 |Δγ*| 的分布，
    再对若干 Δmax 做速率限制、比较全天电量。

    这是"5 min 调一次现不现实 / 一次最多调多少度"的定量依据：
    * **执行器上限**（速率口径）= `yaw_rate × block_seconds`；取 NREL 5MW 参考机的
      yaw rate 0.3 °/s（见本函数参数）⇒ 300 s 内可转 **90°**，远超 ±25° 带宽。
    * 于是现实约束**不是速率**，而是偏航载荷/占空比（每小时偏航次数）。真正要问的是
      "**最优轨迹本身需要多快移动**"：|Δγ*| 小 ⇒ 速率限制不咬人。
    """
    import pandas as pd

    series = pd.read_csv('FFC_speed_noise_1.csv').iloc[:n_steps, col].values.astype(float)
    nb = n_steps // block
    u_blk = series[::block]
    f, _ = build_model()

    ys = np.zeros((nb, NT))
    for i, u in enumerate(u_blk):
        y, _, _ = serial_refine(f, float(u),
                                passes=((0, 6.25, 12.5, 18.75, 25), (0, 1.5, 3.0, 6.25)))
        ys[i] = y

    def day_power(Yblk):
        p, _ = eval_batch(f, series, np.repeat(Yblk, block, axis=0))
        return p

    def energy_mwh(p):
        return float(np.sum(p) / 60.0)          # 1 min 一步 ⇒ MW·(1/60 h)

    p0 = day_power(np.zeros((nb, NT)))
    ps = day_power(ys)
    d = np.abs(np.diff(ys, axis=0))                          # (nb-1, NT)
    dmax_blk = d.max(axis=1)                                 # 每块里最大的一维跳变

    print('  执行器速率口径：%.2f °/s × %.0f s = **%.0f°**/块（本环境带宽只有 %.0f°）'
          % (yaw_rate_deg_s, block_seconds, yaw_rate_deg_s * block_seconds, 2 * GAMMA_MAX))
    print('  逐块最优偏航的跳变 |Δγ*|（%d 块 × %d 维）：' % (nb - 1, NT))
    print('    全体      p50 %.2f  p90 %.2f  p99 %.2f  max %.2f  mean %.3f 度'
          % (np.percentile(d, 50), np.percentile(d, 90), np.percentile(d, 99),
             d.max(), d.mean()))
    print('    每块最大维 p50 %.2f  p90 %.2f  p99 %.2f  max %.2f 度'
          % (np.percentile(dmax_blk, 50), np.percentile(dmax_blk, 90),
             np.percentile(dmax_blk, 99), dmax_blk.max()))
    print('    需要动（|Δγ|>1°）的块占比 %.1f%%；>5° 的块占比 %.1f%%；>10° 的 %.1f%%'
          % (100 * (dmax_blk > 1).mean(), 100 * (dmax_blk > 5).mean(),
             100 * (dmax_blk > 10).mean()))

    print('\n  速率限制的代价（全天电量，%d 步 = 24 h）：' % n_steps)
    print('    γ≡0 基线            %8.3f MWh' % energy_mwh(p0))
    print('    逐块最优(无限制)     %8.3f MWh   %+.2f%%' % (energy_mwh(ps),
                                                          100 * (energy_mwh(ps) / energy_mwh(p0) - 1)))
    out = dict(yaw_rate_deg_s=float(yaw_rate_deg_s),
               actuator_max_per_block_deg=float(yaw_rate_deg_s * block_seconds),
               base_mwh=energy_mwh(p0), unconstrained_mwh=energy_mwh(ps),
               unconstrained_gain_pct=float(100 * (energy_mwh(ps) / energy_mwh(p0) - 1)),
               dgamma_p50=float(np.percentile(d, 50)), dgamma_p90=float(np.percentile(d, 90)),
               dgamma_p99=float(np.percentile(d, 99)), dgamma_max=float(d.max()),
               dgamma_mean=float(d.mean()),
               frac_blocks_move_gt_1deg=float((dmax_blk > 1).mean()),
               frac_blocks_move_gt_5deg=float((dmax_blk > 5).mean()),
               limits={})
    for dmax in delta_max_list:
        Y = np.zeros((nb, NT))
        prev = np.zeros(NT)
        for i in range(nb):
            Y[i] = np.clip(ys[i], prev - dmax, prev + dmax)
            prev = Y[i]
        p = day_power(Y)
        e = energy_mwh(p)
        n_hit = int((np.abs(Y - ys).max(axis=1) > 1e-9).sum())
        need_s = dmax / yaw_rate_deg_s
        tag = '∞（=无限制）' if dmax > 1e8 else '%.1f°/块' % dmax
        print('    Δmax = %-11s %8.3f MWh   %+.2f%% (相对无限制 %+.2f%%)  '
              '受限块数 %3d   转过该角度需 %.0f s（预算 %.0f s）'
              % (tag, e, 100 * (e / energy_mwh(p0) - 1), 100 * (e / energy_mwh(ps) - 1),
                 n_hit, need_s, block_seconds))
        out['limits'][tag] = dict(mwh=e, gain_vs_base_pct=float(100 * (e / energy_mwh(p0) - 1)),
                                  loss_vs_unconstrained_pct=float(100 * (e / energy_mwh(ps) - 1)),
                                  n_blocks_clipped=n_hit, actuator_seconds=float(need_s))
    return (out, ys) if return_traj else out


# ---------------------------------------------------------------------------
# 10. 准稳态假设的代价：偏航的"渐变"FLORIS 不算 —— 用输运延迟做量化上界
# ---------------------------------------------------------------------------
def sec_delay_proxy(ys, col=1, n_steps=1440, block=5):
    """量化"FLORIS 是稳态、不建模偏航渐变/尾流输运延迟"这一假设的**电量代价**。

    物理时间尺度（本配置：D=100 m、单排 5D 间距、行中点 = 10D = 1000 m）：
    * **执行器**：0.3 °/s ⇒ 实测最大块内跳变 5.5° 只需 ~18 s（占 300 s 的 6%）——可忽略；
    * **尾流输运**：(i−1)·5D/U 秒，第 5 台的入流要等上游的偏航改变**输运过去**：
      U=6/8/10/14 m/s → 第 5 台滞后 **333/250/200/143 s**（5.6/4.2/3.3/2.4 min），
      与 5 min 决策周期**同量级**。FLORIS 对这个滞后**一无所知**（源码里 0 处 yaw rate/lag）。

    代理实验（**保守上界**）：把整条逐分钟偏航命令按行中点输运时间（10D/U）整体后移后再
    评估——真实情况里第 1 台没有上游、本应即时响应，故这样估出来的差值是**偏大**的。
    同时给出按第 5 台输运时间（20D/U）的更大上界。
    """
    import pandas as pd

    series = pd.read_csv('FFC_speed_noise_1.csv').iloc[:n_steps, col].values.astype(float)
    f, _ = build_model()
    Y = np.repeat(ys, block, axis=0)                      # (1440,5) 逐分钟命令

    def energy_mwh(p):
        return float(np.sum(p) / 60.0)

    p_cmd, _ = eval_batch(f, series, Y)
    p0, _ = eval_batch(f, series, np.zeros((n_steps, NT)))
    out = dict(base_mwh=energy_mwh(p0), steady_cmd_mwh=energy_mwh(p_cmd), delays={})

    print('  尾流输运时间 (i−1)·5D/U [s]（第 2/3/4/5 台）：')
    for u in (6.0, 8.0, 10.0, 14.0):
        t = [(i * SPACING) / u for i in range(1, NT)]
        print('    U=%4.1f m/s → %s   （行中点 10D/U = %.0f s = %.1f min）'
              % (u, ' '.join('%5.0f' % x for x in t), 10 * D_ROTOR / u,
                 (10 * D_ROTOR / u) / 60.0))
    print('  FLORIS 侧的对应实现：**没有**（源码 grep `yaw.?rate|lag|slew|rate_limit` = 0 命中；'
          '`core.steady_state_atmospheric_condition()` 逐条件求稳态解）')

    print('\n  延迟代理（保守上界）：')
    print('    γ≡0 基线                      %8.3f MWh' % energy_mwh(p0))
    print('    逐分钟命令（稳态假设=FLORIS） %8.3f MWh' % energy_mwh(p_cmd))
    for tag, nd in (('10D/U（行中点）', 10.0), ('20D/U（第 5 台）', 20.0)):
        tau_min = (nd * D_ROTOR / np.maximum(series, 1.0)) / 60.0
        idx = np.clip(np.arange(n_steps) - np.round(tau_min).astype(int), 0, n_steps - 1)
        p_del, _ = eval_batch(f, series, Y[idx])
        e = energy_mwh(p_del)
        print('    命令后移 %-16s %8.3f MWh  （相对稳态假设 %+.2f%%，平均滞后 %.2f min）'
              % (tag, e, 100 * (e / energy_mwh(p_cmd) - 1), float(tau_min.mean())))
        out['delays'][tag] = dict(mwh=e,
                                  delta_vs_steady_pct=float(100 * (e / energy_mwh(p_cmd) - 1)),
                                  mean_lag_min=float(tau_min.mean()))
    return out


# ---------------------------------------------------------------------------
# 11. 波动 vs 能量：偏航作为"块级工作点"旋钮的折中曲线（本项目的主目标）
# ---------------------------------------------------------------------------
def sec_fluct_tradeoff(ys, col=1, n_steps=1440, block=5,
                       s_levels=(0.0, 0.25, 0.5, 0.75, 1.0), pf_limit=3.0,
                       battery_mw=10.0, battery_mwh=6.0):
    """把偏航当"转向档位 s"（γ = s·γ*），量 **波动控制 vs 盈利** 的折中。

    动机：本项目主目标是"**波动控制条件下的盈利最大化**"。新环境里这两个目标**直接对抗**
    —— 转向增益按比例放大绝对波动。本节的三个量：

    1. **档位曲线**：s ∈ [0,1] 下的全天电量、|ΔP_w| 分位、超过并网限值 3 MW 的分钟占比；
    2. **带端点**：±25° 带内可达的 [P_lo(u), P_hi(u)]（第 4 节实测两端模式与 u 无关）；
    3. **"偏航单独能做到的平滑地板"**：假设 H 层每分钟都把 P_w 拉向上一分钟的 P_w
       （目标 = 项目 c1 的定义 `|P_G(t)−P_G(t−1)|`），残差 = 目标到 [lo,hi] 的距离。
       它回答"**多少比例的分钟步，偏航单独就能完全抹平**"，其余才需要电池。
       （注：这是理想 H 层上界，不是可实现策略；H 每 5 min 才动一次、且不知道未来，实际更低。）
    """
    import pandas as pd

    series = pd.read_csv('FFC_speed_noise_1.csv').iloc[:n_steps, col].values.astype(float)
    f, _ = build_model()

    def energy_mwh(p):
        return float(np.sum(p) / 60.0)

    def stats(p):
        d = np.abs(np.diff(p))
        return dict(energy_mwh=energy_mwh(p), d_mean=float(d.mean()),
                    d_p95=float(np.percentile(d, 95)), d_p99=float(np.percentile(d, 99)),
                    d_max=float(d.max()), frac_over=float((d > pf_limit).mean()))

    # ---- 带端点（第 4 节实测：最劣 = 只转末台 25°，最优 = 上游满转）----
    pat_hi = np.tile(np.array([GAMMA_MAX, GAMMA_MAX, 12.5, 6.25, 0.0]), (n_steps, 1))
    pat_lo = np.tile(np.array([0.0, 0.0, 0.0, 0.0, GAMMA_MAX]), (n_steps, 1))
    p_hi, _ = eval_batch(f, series, pat_hi)
    p_lo, _ = eval_batch(f, series, pat_lo)
    p_zero, _ = eval_batch(f, series, np.zeros((n_steps, NT)))

    print('  一、带端点（可达工作点范围）')
    print('     γ≡0 基线        mean %.3f MW   全天 %7.3f MWh'
          % (p_zero.mean(), energy_mwh(p_zero)))
    print('     带内最优(γ*)    mean %.3f MW   全天 %7.3f MWh  (%+.1f%%)'
          % (p_hi.mean(), energy_mwh(p_hi), 100 * (energy_mwh(p_hi) / energy_mwh(p_zero) - 1)))
    print('     带内最劣        mean %.3f MW   全天 %7.3f MWh  (%+.1f%%)'
          % (p_lo.mean(), energy_mwh(p_lo), 100 * (energy_mwh(p_lo) / energy_mwh(p_zero) - 1)))
    print('     ⇒ 相对工作点 γ* 的**下探余量**：%.1f%%（均值 %.3f → %.3f MW）'
          % (100 * (1 - p_lo.mean() / p_hi.mean()), p_hi.mean(), p_lo.mean()))

    print('\n  二、转向档位 s（γ = s·γ*）的能量-波动折中')
    print('     s      全天电量[MWh]   增益%    |ΔP|均值  p95    p99    max    >3MW占比')
    out = dict(band=dict(base_mwh=energy_mwh(p_zero), hi_mwh=energy_mwh(p_hi),
                         lo_mwh=energy_mwh(p_lo),
                         turndown_pct=float(100 * (1 - p_lo.mean() / p_hi.mean()))),
               levels={})
    Ystar = np.repeat(ys, block, axis=0)
    for s in s_levels:
        p, _ = eval_batch(f, series, s * Ystar)
        st = stats(p)
        st['gain_vs_base_pct'] = float(100 * (st['energy_mwh'] / energy_mwh(p_zero) - 1))
        out['levels']['%.2f' % s] = st
        print('     %.2f   %11.3f   %+6.2f   %7.3f  %5.2f  %5.2f  %5.2f   %5.2f%%'
              % (s, st['energy_mwh'], st['gain_vs_base_pct'], st['d_mean'],
                 st['d_p95'], st['d_p99'], st['d_max'], 100 * st['frac_over']))

    # ---- 三、偏航单独能做到的平滑地板（理想 H）----
    target = np.concatenate([[p_hi[0]], p_hi[:-1]])     # 目标 = 上一分钟的 P_w（γ* 轨迹）
    residual = np.maximum(0.0, np.maximum(target - p_hi, p_lo - target))
    p_hat = np.clip(target, p_lo, p_hi)
    d_hat = np.abs(np.diff(p_hat))
    d_raw = np.abs(np.diff(p_hi))
    print('\n  三、偏航单独能做到的平滑地板（目标 = 上一分钟的 P_w，与 c1 定义一致）')
    print('     γ* 轨迹原始    |ΔP_w| 均值 %.3f  p99 %.3f  max %.3f MW  超 3 MW 占比 %.2f%%'
          % (d_raw.mean(), np.percentile(d_raw, 99), d_raw.max(),
             100 * (d_raw > pf_limit).mean()))
    print('     理想 H 用偏航   |ΔP_w| 均值 %.3f  p99 %.3f  max %.3f MW  超 3 MW 占比 %.2f%%'
          % (d_hat.mean(), np.percentile(d_hat, 99), d_hat.max(),
             100 * (d_hat > pf_limit).mean()))
    print('     偏航**完全抹平**的分钟步占比 %.1f%%；残留 >3 MW 的步占比 %.1f%%'
          % (100 * (residual <= 1e-9).mean(), 100 * (residual > pf_limit).mean()))
    print('     残留（偏航管不了、必须交给电池）：均值 %.3f  p99 %.3f  max %.3f MW'
          % (residual.mean(), np.percentile(residual, 99), residual.max()))
    print('     电池能力对照：额定 %.0f MW / 容量 %.0f MWh（满功率仅 %.0f min）'
          % (battery_mw, battery_mwh, battery_mwh / battery_mw * 60))
    out['floor'] = dict(
        raw=dict(d_mean=float(d_raw.mean()), d_p99=float(np.percentile(d_raw, 99)),
                 d_max=float(d_raw.max()), frac_over=float((d_raw > pf_limit).mean())),
        ideal_h=dict(d_mean=float(d_hat.mean()), d_p99=float(np.percentile(d_hat, 99)),
                     d_max=float(d_hat.max()), frac_over=float((d_hat > pf_limit).mean())),
        frac_fully_absorbable=float((residual <= 1e-9).mean()),
        residual=dict(mean=float(residual.mean()), p99=float(np.percentile(residual, 99)),
                      max=float(residual.max())),
        battery=dict(mw=battery_mw, mwh=battery_mwh))

    # ---- 三b、现实口径：H 每 5 min 只能选一次"转向档位 s"（块内恒定）----
    # 贪心代理：每个块选一个 s，使**块首功率**尽量贴住上一分钟的实际功率（对块内波动的
    # 缩放效应则由 s 的大小间接体现）。这是 H 层 5-min 粒度的能力上界近似。
    S = np.linspace(0.0, 1.0, 11)
    P = np.stack([eval_batch(f, series, s * Ystar)[0] for s in S])       # (11, n_steps)
    nb = n_steps // block
    p_blk = np.zeros(n_steps)
    s_sel = np.zeros(nb)
    for b in range(nb):
        sl = slice(b * block, (b + 1) * block)
        tgt = p_blk[b * block - 1] if b > 0 else P[-1, 0]
        k = int(np.argmin(np.abs(P[:, b * block] - tgt)))
        s_sel[b] = S[k]
        p_blk[sl] = P[k, sl]
    d_blk = np.abs(np.diff(p_blk))
    print('     H 层 5-min 粒度（每块一个 s，块内恒定；贪心贴住上一分钟）：')
    print('       |ΔP_w| 均值 %.3f  p99 %.3f  max %.3f MW  超 3 MW 占比 %.2f%%   '
          '全天电量 %.3f MWh（%+.1f%%）'
          % (d_blk.mean(), np.percentile(d_blk, 99), d_blk.max(),
             100 * (d_blk > pf_limit).mean(), energy_mwh(p_blk),
             100 * (energy_mwh(p_blk) / energy_mwh(p_zero) - 1)))
    print('       选中的档位 s：均值 %.2f  p05 %.2f  p95 %.2f（0=不转向, 1=满转向）'
          % (s_sel.mean(), np.percentile(s_sel, 5), np.percentile(s_sel, 95)))
    out['floor']['block_greedy'] = dict(
        d_mean=float(d_blk.mean()), d_p99=float(np.percentile(d_blk, 99)),
        d_max=float(d_blk.max()), frac_over=float((d_blk > pf_limit).mean()),
        energy_mwh=energy_mwh(p_blk),
        gain_vs_base_pct=float(100 * (energy_mwh(p_blk) / energy_mwh(p_zero) - 1)),
        s_mean=float(s_sel.mean()), s_p05=float(np.percentile(s_sel, 5)),
        s_p95=float(np.percentile(s_sel, 95)))
    return out


def sec_wind_series(col=1, n_steps=1440):
    hdr('[8] 真实风速序列（FFC_speed_noise_1.csv 第 %d 列，前 %d 步）下的波动量级'
        % (col, n_steps))
    import pandas as pd

    series = pd.read_csv('FFC_speed_noise_1.csv').iloc[:n_steps, col].values.astype(float)
    f, _ = build_model()
    Y0 = np.zeros((series.size, NT))
    p0, _ = eval_batch(f, series, Y0)                     # 基线（γ=0）
    d0 = np.abs(np.diff(p0))
    # "最大转向"侧：前 4 台 ±25°、末台 0°（用同一固定模式，避免逐分钟做优化）
    pat = np.tile(np.array([GAMMA_MAX, GAMMA_MAX, GAMMA_MAX, GAMMA_MAX, 0.0]),
                  (series.size, 1))
    ps, _ = eval_batch(f, series, pat)
    ds = np.abs(np.diff(ps))
    print('  风速：min %.2f / mean %.2f / max %.2f m/s' % (series.min(), series.mean(), series.max()))
    print('  风场总功率（γ=0）：mean %.4f MW  min %.4f  max %.4f'
          % (p0.mean(), p0.min(), p0.max()))
    print('  |ΔP_w|（γ=0，逐分钟）：mean %.4f  p90 %.4f  p99 %.4f  max %.4f MW'
          % (d0.mean(), np.percentile(d0, 90), np.percentile(d0, 99), d0.max()))
    print('  |ΔP_w|（固定转向模式）：mean %.4f  p90 %.4f  p99 %.4f  max %.4f MW'
          % (ds.mean(), np.percentile(ds, 90), np.percentile(ds, 99), ds.max()))
    print('  固定转向模式的出力比：mean %.3f（即 %+.1f%% 出力）'
          % ((ps / np.maximum(p0, 1e-9)).mean(), 100 * ((ps / np.maximum(p0, 1e-9)).mean() - 1)))
    print('  并网波动限值 3 MW：γ=0 时超限比例 %.2f%%；转向模式下 %.2f%%'
          % (100 * (d0 > 3.0).mean(), 100 * (ds > 3.0).mean()))
    return dict(n=int(series.size), u_min=float(series.min()), u_mean=float(series.mean()),
                u_max=float(series.max()),
                p_mean_mw=float(p0.mean()), p_min_mw=float(p0.min()), p_max_mw=float(p0.max()),
                dP_mean_mw=float(d0.mean()), dP_p99_mw=float(np.percentile(d0, 99)),
                dP_max_mw=float(d0.max()), frac_over_3mw=float((d0 > 3.0).mean()),
                steering_power_ratio=float((ps / np.maximum(p0, 1e-9)).mean()))


def main():
    ap = argparse.ArgumentParser(description='FLORIS 偏航环境功率结构探针（只读）')
    ap.add_argument('--us', default=','.join('%.1f' % u for u in US_DEFAULT),
                    help='风速列表，逗号分隔')
    ap.add_argument('--gammas', default=('-60,-50,-40,-30,-25,-20,-15,-10,-5,0,'
                                         '5,10,15,20,25,30,40,50,60'),
                    help='均匀偏航角列表（度），逗号分隔')
    ap.add_argument('--skip-opt', action='store_true', help='跳过逐台最优搜索（较慢）')
    ap.add_argument('--skip-series', action='store_true', help='跳过真实风速序列一节')
    ap.add_argument('--skip-sens', action='store_true', help='跳过口径敏感性一节')
    ap.add_argument('--skip-diag', action='store_true', help='跳过结构诊断一节')
    ap.add_argument('--skip-day', action='store_true', help='跳过"一天最优轨迹/速率限制"一节')
    ap.add_argument('--skip-delay', action='store_true', help='跳过"输运延迟代理"一节')
    ap.add_argument('--skip-fluct', action='store_true', help='跳过"波动-能量折中"一节')
    ap.add_argument('--yaw-rate', default=0.3, type=float,
                    help='偏航角速度 [°/s]，默认 0.3（NREL 5MW 参考机口径）')
    ap.add_argument('--json', default=None, help='把全部结果写入该 JSON 路径')
    ap.add_argument('--log', default=None, help='把报告（UTF-8）同时写入该文本路径')
    args = ap.parse_args()

    us = np.array([float(x) for x in args.us.split(',') if x.strip()])
    gammas = [float(x) for x in args.gammas.split(',') if x.strip()]

    tee = None
    if args.log:
        log_path = args.log if os.path.isabs(args.log) else os.path.join(ROOT, args.log)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        tee = _Tee(sys.stdout, log_path)
        sys.stdout = tee

    t_start = time.perf_counter()
    print('FLORIS 偏航环境探针 —— 物理口径对齐 WFSim windTunnel 场景')
    print('  D=%.0f m, rho=%.2f, a=1/3, Cp=%.5f, Cp_eff=%.5f, CT_eff=%.5f (=8/9)'
          % (D_ROTOR, RHO, CP_A, CP_EFF, CT_EFF))
    print('  排布 x=%s, y=%s, TI=%.2f, 风向=%.0f°(=沿 +x)'
          % (np.round(LAYOUT_X, 1).tolist(), np.round(LAYOUT_Y, 1).tolist(), TI,
             WIND_DIR_ALIGNED))
    print('  动作域 gamma ∈ [-%.0f°, +%.0f°]^5；a 固定 1/3' % (GAMMA_MAX, GAMMA_MAX))

    res = dict(calibration=sec_calibration(),
               uniform=sec_uniform_sweep(us, gammas),
               optimum=sec_optimum(us, do_opt=not args.skip_opt),
               authority=sec_authority(us),
               invariants=sec_invariants(),
               timing=sec_timing())
    if not args.skip_diag:
        res['diagnostics'] = sec_diagnostics()
    if not args.skip_sens:
        res['sensitivity'] = sec_sensitivity(us)
    if not args.skip_series:
        res['wind_series'] = sec_wind_series()
    if not args.skip_day:
        day_out, day_ys = sec_day_trajectory(yaw_rate_deg_s=args.yaw_rate,
                                             return_traj=True)
        res['day_trajectory'] = day_out
        if not args.skip_delay:
            res['delay_proxy'] = sec_delay_proxy(day_ys)
        if not args.skip_fluct:
            res['fluct_tradeoff'] = sec_fluct_tradeoff(day_ys)
    res['elapsed_s'] = time.perf_counter() - t_start

    hdr('汇总')
    print(json.dumps({k: v for k, v in res.items() if k not in ('uniform',)},
                     ensure_ascii=False, indent=2, default=float)[:4000])
    print('\n总耗时 %.1f s' % res['elapsed_s'])

    if args.json:
        out = args.json if os.path.isabs(args.json) else os.path.join(ROOT, args.json)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2, default=float)
        print('已写出 %s' % out)
    if tee is not None:
        sys.stdout = tee.stream
        tee.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
