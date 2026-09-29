# -*- coding: utf-8 -*-
"""`yawflor` 风机 / 功率表 / 口径的**单一真源**（FLORIS 4.6.6 后端）。

本模块与 `agentutil/probe_floris_yaw_structure.py` 的口径**逐项一致**
（探针是只读证据，本模块是生产入口；契约测试 `test_yawflor_config.py` 与
`agentutil/test_yaw_env_contract.py` 都对本模块断言）。

口径来源（详见 `yawflor/README.md` §2，三条规则：①WFSim 给了值 → 用 WFSim 的
②否则用 FLORIS 随包默认 ③两者都没给 → 官方示例/文献惯例）
------------------------------------------------------------------------
* 转子 `D = 100 m`、`hub_height = 100 m`、`TSR = 8`      —— 规则①（`meshing.m` Drotor）
* 单排 5 台、**5D** 间距（x = 0/500/1000/1500/2000, y = 0）—— 规则①（`meshing.m` Crx）
* `air_density = 1.20`（含 `table.ref_air_density`）      —— 规则①（`meshing.m` Rho）
* `a ≡ 1/3` 固定 ⇒ `Cp = 4a(1−a)² = 0.59259`、`powerscale = 0.95` ⇒
  `Cp_eff = 0.56296`；`CT_eff = Cp/(1−a) = 8/9 = 0.88889`  —— 规则①
* **无额定上限**（`power ∝ u³` 不截断）；env 侧沿用旧守卫 `3.5 ≤ u ≤ 25 m/s`
* 尾流 = GCH（`gauss`/`gauss`/`crespo_hernandez`/`sosfs` + 三个开关全 true）
                                                          —— 规则②（随包默认）
* `turbine_grid_points = 3`（纯数值参数，一阶，论文必须报告）—— 规则②
* `cosine_loss_exponent_yaw = 1.88`（随包全部参考机型同值）  —— 规则②
* `wind_shear = 0.12`、`TI = 0.06`、`wind_directions = 270°` —— 规则②/③
* 动作 `γ ∈ [−25°, +25°]`（网络用无量纲 `n ∈ [−1,1]`，env 内 ×25）—— 规则③

**为什么 `wind_directions = 270`**：FLORIS 用气象"来向"约定（0=北、90=东）；
WFSim 侧 `wind_angle = 0` 表示沿 +x 吹 ⇒ FLORIS 等价写法是 270（西风 = 吹向 +x）。

`build_model()` 是本项目**唯一的模型构造入口**（探针与契约测试亦复用其口径）。
若要整体切换官方参考机型（README §2.3(c) 的"逃生门"），只需传
`turbine_type='nrel_5MW'` —— 但代价是 §1 的锚表全部要重测。
"""
import os

import numpy as np

# ---------------------------------------------------------------------------
# matplotlib 的字体缓存目录：`floris` 在 import 时会间接 import matplotlib，
# 后者默认要写 `~/.matplotlib`，在受限沙箱/只读 HOME 下报
# `Could not save font_manager cache ... Permission denied`。
# **必须在 import floris 之前**设好（本模块是唯一持有 floris import 的生产模块）。
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))            # yawflor/
ROOT = os.path.dirname(_HERE)                                 # 仓库根
os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, 'output', '_mplcache'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# ===========================================================================
# 1. 物理常量（单一真源；改动任何一项都会被口径契约测试抓到）
# ===========================================================================
D_ROTOR = 100.0                       # m
HUB_HEIGHT = 100.0                    # m（2D 模型无高度维，取 D）
RHO = 1.20                            # kg/m³（WFSim meshing.m）
AREA = np.pi * (D_ROTOR / 2.0) ** 2   # m²
A_IND = 1.0 / 3.0                     # 固定轴向诱导因子（Betz 单机最优点）
CP_A = 4.0 * A_IND * (1.0 - A_IND) ** 2       # 0.59259
POWER_SCALE = 0.95                    # meshing.m: powerscale
CP_EFF = CP_A * POWER_SCALE           # 0.56296
CT_EFF = CP_A / (1.0 - A_IND)         # 0.88889 = 8/9
NT = 5                                # 机组台数
SPACING_D = 5.0                       # 间距（以 D 为单位）
SPACING = SPACING_D * D_ROTOR         # 500 m
LAYOUT_X = [i * SPACING for i in range(NT)]
LAYOUT_Y = [0.0] * NT
WIND_DIR_ALIGNED = 270.0              # 沿 +x 吹（对齐排布）
TI = 0.06                             # 官方 gch.yaml
WIND_SHEAR = 0.12                     # 官方 gch.yaml / default_inputs.yaml
GAMMA_MAX = 25.0                      # deg，动作域 ±25°
U_MIN_ENV, U_MAX_ENV = 3.5, 25.0      # env.py::PowerOutput 的守卫（旧环境同值）
U_TABLE_MAX = 40.0                    # 功率表上界；**必须 > U_MAX_ENV**，
                                      # 否则 interp1d(fill_value=0) 把高风速静默截成 0
COSINE_LOSS_EXPONENT_YAW = 1.88       # 随包全部参考机型同值
REF_TILT = 0.0                        # 2D 无倾角

# 模型口径（随包 default_inputs.yaml: name=GCH）
VELOCITY_MODEL = 'gauss'
DEFLECTION_MODEL = 'gauss'
TURBULENCE_MODEL = 'crespo_hernandez'
COMBINATION_MODEL = 'sosfs'
WAKE_FLAGS = dict(enable_secondary_steering=True,
                  enable_yaw_added_recovery=True,
                  enable_transverse_velocities=True,
                  enable_active_wake_mixing=False)
SOLVER_TYPE = 'turbine_grid'
TURBINE_GRID_POINTS = 3

# ===========================================================================
# 2. 执行器现实性常量（README §4；**默认全部关**，位置留好给对照臂）
# ===========================================================================
YAW_RATE_MAX_DEG_PER_BLOCK = None     # None = 不限制（实测最优轨迹跳变 p99 = 5.5°；
                                      # 按 Wang et al. 2026 开此臂并报告执行器占空比）
YAW_TAU_S = 0.0                       # 一阶延迟（0 = 关）；5.5° @0.3°/s 仅需 18 s
YAW_TRANSPORT_DELAY = False           # 输运延迟（False = 关）；开启会使 env **有记忆**，
                                      # 破坏 test9 的确定性判据 ⇒ 只能作对照臂
YAW_SYMMETRIZE = False                # R2 对照臂：P_sym = ½[P(+γ) + P(−γ)]
                                      # 主臂保持 FLORIS 原口径（不对称化）


# ===========================================================================
# 3. 动作换算（网络无量纲 n ↔ 物理角度 γ）
# ===========================================================================
def cp_analytic(u, a=A_IND):
    """单机功率 [MW]（自由流、无尾流），与 `WFSim_port.m` 的 Cp = 4a(1−a)² 同口径。"""
    cp = 4.0 * a * (1.0 - a) ** 2
    return POWER_SCALE * 0.5 * RHO * AREA * cp * np.asarray(u, dtype=float) ** 3 / 1e6


def yaw_deg_from_norm(n):
    """无量纲动作 `n ∈ [−1,1]` → 偏航角 [deg]（未夹带：夹带由调用方负责）。"""
    return GAMMA_MAX * np.asarray(n, dtype=float)


def norm_from_yaw_deg(gamma):
    """偏航角 [deg] → 无量纲动作。"""
    return np.asarray(gamma, dtype=float) / GAMMA_MAX


def limit_yaw_rate(gamma_prev, gamma_new, max_deg=None):
    """按块速率限制夹带偏航角（默认 `None` ⇒ 原样返回，逐位不变）。

    用途 = README §7/R6 的"速率感知对照臂"。逐元素夹带到
    `[γ_prev − max_deg, γ_prev + max_deg]`，并保持 ±GAMMA_MAX 带内。
    """
    g = np.asarray(gamma_new, dtype=float)
    if max_deg is None:
        return g
    gp = np.asarray(gamma_prev, dtype=float)
    return np.clip(g, gp - float(max_deg), gp + float(max_deg))


# ===========================================================================
# 4. 风机定义与功率表
# ===========================================================================
def build_turbine_dict(cosine_loss_exponent_yaw=COSINE_LOSS_EXPONENT_YAW,
                       ref_tilt=REF_TILT, rotor_diameter=D_ROTOR,
                       hub_height=HUB_HEIGHT, air_density=RHO,
                       u_table_max=U_TABLE_MAX):
    """固定 a = 1/3 的常数 Cp/CT 风机（理想 Betz 执行器盘 + `powerscale`）。

    FLORIS v4 的表列是 `wind_speed` / `power`(kW) / `thrust_coefficient`；
    不接受 `cp`/`ct` 列，也没有 `cut_in/out_wind_speed`（用 0 行表达）。
    `ref_air_density` 与流场 `air_density` 必须相等，否则功率被静默 ×ρ/ρ_ref。
    """
    u = np.arange(0.0, float(u_table_max) + 1e-9, 1.0)
    power_kw = cp_analytic(u) * 1e3                   # MW → kW
    return dict(
        turbine_type='wfsim_betz_a033',
        hub_height=float(hub_height),
        rotor_diameter=float(rotor_diameter),
        TSR=8.0,
        operation_model='cosine-loss',
        power_thrust_table=dict(
            ref_air_density=float(air_density),
            ref_tilt=float(ref_tilt),
            cosine_loss_exponent_yaw=float(cosine_loss_exponent_yaw),
            cosine_loss_exponent_tilt=float(cosine_loss_exponent_yaw),
            wind_speed=[float(x) for x in u],
            power=[float(x) for x in power_kw],
            thrust_coefficient=[float(CT_EFF)] * len(u),
        ),
    )


# ===========================================================================
# 5. 模型构造（**唯一入口**）
# ===========================================================================
def build_model(nt=NT, rotor_diameter=D_ROTOR, spacing_m=None, spacing_d=SPACING_D,
                turbine_type=None, hub_height=None, air_density=None,
                wind_shear=WIND_SHEAR, ti=TI, wind_direction=WIND_DIR_ALIGNED,
                cosine_loss_exponent_yaw=COSINE_LOSS_EXPONENT_YAW, ref_tilt=REF_TILT,
                velocity_model=VELOCITY_MODEL, deflection_model=DEFLECTION_MODEL,
                turbulence_model=TURBULENCE_MODEL, combination_model=COMBINATION_MODEL,
                wake_flags=None, allow_density_mismatch=False, u_table_max=U_TABLE_MAX):
    """构造 `FlorisModel`；返回 `(model, info)`。

    Parameters
    ----------
    nt : int                    机组台数（主臂 5）
    rotor_diameter : float      转子直径 [m]（主臂 100）
    spacing_m : float or None   绝对间距 [m]；None ⇒ `spacing_d · rotor_diameter`
    turbine_type : None or str   None ⇒ 本项目常数 Cp 表；否则取 FLORIS 机型库
                                （如 `'nrel_5MW'`，官方示例复现用）
    air_density : float or None 流场密度；**默认 None = 跟随功率表的 `ref_air_density`**
                                （本项目表 = 1.20 WFSim 口径；库内机型 = 其自带值）。
                                显式给别的值时，若与表不等则本函数报错（陷阱②：
                                P 被静默 ×ρ/ρ_ref），除非 `allow_density_mismatch=True`
                                （README §2.2(e) 的 `air_density 1.225` 敏感性行就走这条）
    wind_shear : float | None   None ⇒ 用模块常量 WIND_SHEAR
    wake_flags : dict or None   覆盖 `wake:` 下的布尔开关（换模型族时用）

    实现注记（实测踩坑）：
    * **必须先定 layout 再给 turbine_type**，否则 `farm.check_turbine_type`
      会拿默认布局的 3 台去比 "nt 个 turbine_type" 而报错；
    * `reference_wind_height` 必须**显式**写成轮毂高（本项目）或 `-1`
      （库内机型 ⇒ 用其轮毂高），否则 `turbine_type` 变更后它仍停在默认 90 m
      ⇒ 切变下整场电平静默偏移（契约测试的陷阱①）。
    """
    from floris import FlorisModel

    nt = int(nt)
    d = float(rotor_diameter)
    spacing = float(spacing_m) if spacing_m is not None else float(spacing_d) * d
    rho_req = None if air_density is None else float(air_density)
    ws = WIND_SHEAR if wind_shear is None else float(wind_shear)
    flags = dict(WAKE_FLAGS)
    if wake_flags:
        flags.update({k: bool(v) for k, v in wake_flags.items()})

    model = FlorisModel('defaults')
    model.set(layout_x=[i * spacing for i in range(nt)], layout_y=[0.0] * nt)

    if turbine_type is None:
        # 本项目常数 Cp/CT 表：流场密度 **就是** 表的 ref（同一次构造 ⇒ 结构上相等）
        rho = RHO if rho_req is None else rho_req
        tdict = build_turbine_dict(cosine_loss_exponent_yaw, ref_tilt, d,
                                   hub_height if hub_height is not None else d,
                                   rho, u_table_max)
        model.set(turbine_type=[tdict])
        table_ref_rho = float(tdict['power_thrust_table']['ref_air_density'])
    else:
        tlist = list(turbine_type) if isinstance(turbine_type, (list, tuple)) else [turbine_type]
        model.set(turbine_type=tlist)
        # ⚠ `get_param(['farm','turbine_type'])` 对**库内机型**返回的是字符串名
        #   （如 'nrel_5MW'），解析后的完整定义在 `core.farm.turbine_definitions`。
        table_ref_rho = float(
            model.core.farm.turbine_definitions[0]
            ['power_thrust_table']['ref_air_density'])
        rho = table_ref_rho if rho_req is None else rho_req

    # 陷阱②（静默失效）：流场密度 ≠ 表参考密度 ⇒ 功率静默 ×ρ/ρ_ref
    if abs(rho - table_ref_rho) > 1e-12 and not allow_density_mismatch:
        raise ValueError(
            'air_density({}) != 功率表 ref_air_density({}) ⇒ 功率会被静默缩放 ρ/ρ_ref；'
            '要么把 air_density 设成表值，要么显式 allow_density_mismatch=True'
            .format(rho, table_ref_rho))

    ref_h = -1.0 if turbine_type is not None else float(
        hub_height if hub_height is not None else d)
    model.set(wind_shear=ws, reference_wind_height=ref_h, air_density=rho)

    for key, val in (('velocity_model', velocity_model),
                     ('deflection_model', deflection_model),
                     ('turbulence_model', turbulence_model),
                     ('combination_model', combination_model)):
        model.set_param(['wake', 'model_strings', key], val)
    for key, val in flags.items():
        model.set_param(['wake', key], bool(val))

    import floris
    td = model.core.farm.turbine_definitions[0]
    pt = td['power_thrust_table']
    info = dict(
        floris_version=getattr(floris, '__version__', 'unknown'),
        velocity_model=velocity_model, deflection_model=deflection_model,
        turbulence_model=turbulence_model, combination_model=combination_model,
        enable_secondary_steering=model.get_param(['wake', 'enable_secondary_steering']),
        enable_yaw_added_recovery=model.get_param(['wake', 'enable_yaw_added_recovery']),
        enable_transverse_velocities=model.get_param(['wake', 'enable_transverse_velocities']),
        enable_active_wake_mixing=model.get_param(['wake', 'enable_active_wake_mixing']),
        solver_type=model.get_param(['solver', 'type']),
        turbine_grid_points=int(model.get_param(['solver', 'turbine_grid_points'])),
        turbine_type=(turbine_type if turbine_type is None else
                      (turbine_type[0] if isinstance(turbine_type, (list, tuple))
                       else turbine_type)),
        cosine_loss_exponent_yaw=float(pt['cosine_loss_exponent_yaw']),
        ref_tilt=float(pt['ref_tilt']),
        air_density=rho, table_ref_air_density=table_ref_rho,
        reference_wind_height=float(model.get_param(['flow_field', 'reference_wind_height'])),
        wind_shear=ws, ti=float(ti), wind_direction=float(wind_direction),
        nt=nt, rotor_diameter=float(td['rotor_diameter']),
        hub_height=float(td['hub_height']), TSR=float(td.get('TSR', 8.0)),
        layout_x=list(model.get_param(['farm', 'layout_x'])),
        layout_y=list(model.get_param(['farm', 'layout_y'])),
        spacing_m=spacing, cp_a033=CP_A, cp_eff=CP_EFF, ct_eff=CT_EFF,
        a_induced=A_IND, power_scale=POWER_SCALE,
        table_max_u=float(u_table_max), gamma_max=GAMMA_MAX,
    )
    return model, info


# ===========================================================================
# 6. 向量化求值（批量 ⇒ 逐块预取的底座）
# ===========================================================================
def eval_batch(f, us, yaws, wind_direction=WIND_DIR_ALIGNED, ti=TI,
               u_min=U_MIN_ENV, symmetrize=YAW_SYMMETRIZE):
    """一次向量化调用算多组 `(u, γ)` 的风场总功率。

    Parameters
    ----------
    f : FlorisModel
    us : array (n,)
    yaws : array (n, nt)    **物理角度 [deg]**（不是无量纲 n）
    symmetrize : bool       True ⇒ 同时算 P(−γ) 并返回两者平均（R2 对照臂）

    Returns
    -------
    (farm_mw (n,), turb_mw (n, nt))

    注：`u < u_min`（env 守卫）⇒ 出力 0，与 `env.py::PowerOutput` 同口径；
    `u = 0` 会被夹到 1e-6（FLORIS 不接受 0 风速的 0 功率条件）。
    """
    us = np.atleast_1d(np.asarray(us, dtype=float))
    yaws = np.atleast_2d(np.asarray(yaws, dtype=float))
    n = int(us.size)
    if yaws.shape[0] != n:
        raise ValueError('us 与 yaws 行数不一致: %d vs %s' % (n, yaws.shape))
    keep = np.clip(us, 1e-6, None)

    def _run(yy):
        f.set(wind_speeds=list(keep), wind_directions=[float(wind_direction)] * n,
              turbulence_intensities=[float(ti)] * n, yaw_angles=yy)
        f.run()
        return (np.asarray(f.get_farm_power(), dtype=float),
                np.asarray(f.get_turbine_powers(), dtype=float))

    farm_w, turb_w = _run(yaws)
    if symmetrize:
        _, turb_m = _run(-yaws)
        farm_w = 0.5 * (farm_w + np.asarray(f.get_farm_power(), dtype=float))
        turb_w = 0.5 * (turb_w + turb_m)
    farm_mw = farm_w / 1e6
    turb_mw = turb_w / 1e6
    farm_mw = np.where(us >= float(u_min), farm_mw, 0.0)
    return farm_mw, turb_mw


def farm_power(f, u, yaw_deg, **kw):
    """单点便捷入口：返回风场总功率 [MW]（标量）。"""
    p, _ = eval_batch(f, [float(u)], np.asarray(yaw_deg, dtype=float).reshape(1, -1), **kw)
    return float(p[0])


def declared_scope():
    """范围声明（README §2；论文环境章节开头一字不改地引用）。"""
    return ('本文采用准稳态尾流模型（FLORIS 4.6.6）：每个控制周期内假定偏航已到位、'
            '尾流已平衡，不建模偏航执行器动力学、尾流输运延迟与结构载荷/疲劳代价。'
            '因此结论适用于稳态功率与并网波动的优化，不适用于载荷、寿命与快速暂态。')
