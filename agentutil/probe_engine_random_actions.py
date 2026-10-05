# -*- coding: utf-8 -*-
"""随机行为下 WFSim vs FLORIS 的**风场出力**对比（动作 = 逐台 a ∈ [0, 0.5] 随机扫描）。

为什么要做这个（而不是只用 a≡1/3）
-----------------------------------
a≡1/3 的固定动作只能测出**一个工作点**的比值（1.74×）。RL 要在这个动作空间里搜索，
所以真正决定"能否换引擎"的是：**比值在整个动作空间上是不是常数？**

本脚本用**同一组**固定种子的随机动作（5 维逐台 `a ∈ [0, 0.5]`）与随机风速，逐条喂给
两个引擎，回答：

  ① 比值的**全局**分布（均值/分位/极差/标准差）；
  ② 比值是否依赖**动作绝对水平** `mean(a)`（⇒ 能否用常数重标定）；
  ③ 比值是否依赖**动作空间结构** `std(a)`（⇒ FLORIS 是否更/更不奖励空间差异化）；
  ④ 比值是否依赖**风速**（⇒ 重标定要不要加风速项）。

对齐保证：逐条 `(u, a₁…a₅)` **逐位相同**，唯一差异是引擎。

FLORIS 侧如何拿到**逐台** a
---------------------------
原生 `multidim_conditions` 是**全场标量**（`core/flow_field.py:33`，形状按
`(n_wd, n_ws)` 校验）⇒ 逐台 a 必须走**自定义 operation model**（本文件内注册，
与 `agentutil/probe_floris_a_per_turbine.py` 同法，但此处**独立实现**以便单文件复跑）。
功率按"整数风速网格表 + 线性插值"复刻 `SimpleTurbine.power` 的口径，
`Cp_eff(a)=0.95·4a(1−a)²`、`CT(a)=4a(1−a)`（= WFSim 的 `CT_prime/(1−a)`）。
自校验：逐台 a 全取 1/3 时必须与 `yawflor` 生产基线**逐位一致**。

用法（普通终端；MATLAB Engine 需要 named pipe）
-----------------------------------------------
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_engine_random_actions.py
    ... --n 100 --mode lhs --u-mode unif --seed 0
    ... --mode edge                     # 结构化边界样本（补角点覆盖）
    ... --u-mode rated                  # 风速集中 8±1.5（尾流影响最大的一段）
    ... --json output/engine_random_actions.json
"""
import argparse
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

# ---- 必须在任何 floris/matplotlib import 之前 ----
# ① 解释器不对 ⇒ 立刻给出可照抄的命令（否则只会看到 pyexpat 的 DLL 报错）
# ② 清掉"其它 Python 安装"注入 sys.path 的路径
from floris_import_guard import check_interpreter as _chk, sanitize as _sanitize  # noqa: E402
_chk()
_SANITIZED = _sanitize()
if _SANITIZED:
    print('[guard] 已剔除污染 sys.path 的 %d 条路径（基础 conda 环境）: %s'
          % (len(_SANITIZED), _SANITIZED))

import numpy as np  # noqa: E402

NT = 5
# ⚠ 下界取 0.025 而非 0：`wake_velocity/gauss.py:76` 的
# `uR = u·ct/(2(1−√(1−ct)))` 在 CT→0 时是 **0/0** ⇒ 单条 NaN 会经 FLORIS 的
# **批量**调用污染整批（已实测）。env 语义上界 0.5，下界留 0.025 作引擎安全边距。
A_LO, A_HI = 0.025, 0.5
U_LO, U_HI = 3.5, 25.0         # env.py::PowerOutput 的围栏

_U_GRID = np.arange(0.0, 41.0, 1.0)
_A_SET = None                  # 逐台 a（(n, NT)）由主流程注入
_CUR_I = None                  # 当前机组序号（solver 层注入）

#: CT 夹带区间 —— **必须与 FLORIS 原厂口径一致**
#: （`floris/core/turbine/operation_models.py:134/:250` 的 `np.clip(ct, 0.0001, 0.9999)`）。
#: 下界 0.0001 把高斯尾流的可去奇点 `uR = u·CT/(2(1−√(1−CT)))`（CT=0 时 0/0）挡在门外。
CT_LO, CT_HI = 0.0001, 0.9999

#: **CT 倍率旋钮**（默认 1.0 = 用 `CT = 4a(1−a)`）。
#: ⚠ 这是"**功率/动量解耦**"的显式开关：功率走独立的 `Cp_eff` 表，所以
#: 调大 `CT_SCALE` 会**加深尾流但不改变单机功率** —— 这正是 WFSim 自己做的事
#: （` Cp = 4a(1−a)²` 而力用 `1.5×`，其有效 `CT/(1−a)` 达 1.333）。
#: 供 `agentutil/probe_power_match_alternatives.py` 扫描"功率对得上但尾流合理"的替代解。
CT_SCALE = 1.0

#: **多尾流组合的幂平均指数**（`agentutil/probe_custom_combination.py` 用）。
#: `n = 2` ≡ FLORIS 现用的 SOSFS（`√(Σd²)`）；`n = 1` ≡ 线性/Katic 叠加。
#: 这是"在 FLORIS 公式最对应位置乘系数"的落点 —— 非物理负速度的根源在**组合规则**
#: （`u = hypot(wake, d·u_initial) − wake`，`wake > u_initial` 时 `u < 0`），
#: 而不在单尾流形状。
COMB_N = 2.0

#: **"持久性"叠加**的两个旋钮（`agentutil/probe_persistent_combination.py` 用）。
#: `PERSIST_Q ∈ [0,1]`：把上游已累积的亏损按权重原样保留（0 ⇒ 退化为原 SOSFS）；
#: `PERSIST_P ∈ [1,2]`：幂平均指数（2 ⇒ 欧氏，1 ⇒ 线性）。
#: 物理含义：`Q → 1` 时亏损**不再随下游衰减**，对应 WFSim 封闭风洞（Ly=4D）的等效行为。
PERSIST_Q = 0.0
PERSIST_P = 2.0


# ===========================================================================
# 1. 采样（固定种子；两引擎共用同一组）
# ===========================================================================
def sample_design(n, mode, seed):
    rng = np.random.default_rng(seed)
    if mode == 'edge':
        pts = [np.full(NT, A_LO), np.full(NT, A_HI), np.full(NT, (A_LO + A_HI) / 2),
               np.array([A_HI] + [A_LO] * (NT - 1)),
               np.array([A_LO] + [A_HI] * (NT - 1)),
               np.array(([A_HI, A_LO] * 3)[:NT]),
               np.array(([A_LO, A_HI] * 3)[:NT]),
               np.linspace(A_LO, A_HI, NT), np.linspace(A_HI, A_LO, NT),
               np.array([A_LO, A_LO, (A_LO + A_HI) / 2, A_HI, A_HI])]
        while len(pts) < n:
            pts.append(rng.uniform(A_LO, A_HI, NT))
        return np.clip(np.array(pts[:n]), A_LO, A_HI)
    if mode == 'lhs':
        try:
            from scipy.stats import qmc
            return A_LO + qmc.LatinHypercube(d=NT, seed=seed).random(n) * (A_HI - A_LO)
        except Exception as e:
            print('[采样] LHS 不可用(%s)，退回均匀随机' % type(e).__name__)
    return rng.uniform(A_LO, A_HI, size=(n, NT))


def sample_wind(n, u_mode, seed):
    rng = np.random.default_rng(seed + 10007)
    if u_mode == 'unif':
        return rng.uniform(U_LO, U_HI, n)
    if u_mode == 'rated':
        return np.clip(rng.normal(8.0, 1.5, n), U_LO, U_HI)
    if u_mode == 'seq':
        import pandas as pd
        return pd.read_csv('FFC_speed_noise_1.csv').iloc[:n, 1].to_numpy(dtype=float)
    raise SystemExit('未知 --u-mode %s' % u_mode)


# ===========================================================================
# 2. FLORIS：注册"逐台 a"自定义 operation model（与生产基线逐位可比）
# ===========================================================================
def _power_table(a):
    """按 a 生成与生产同口径的功率表 [kW]（整数风速网格上采样）。"""
    cp = 4.0 * a * (1.0 - a) ** 2
    import yawflor.yawflor_turbine as YT
    return YT.POWER_SCALE * 0.5 * YT.RHO * YT.AREA * cp * _U_GRID ** 3 / 1e3


_INTERP_CACHE = {}                 # a(round 12) -> interp1d（标定时会被调用上万次）


def _interp_for(a_val):
    """按 `a` 取（并缓存）功率插值器 —— 复刻 `SimpleTurbine.power` 的 `interp1d` 口径。

    缓存是**纯性能**优化：同一个 `a` 的插值器与逐次重建逐位等价（同样的
    `_U_GRID` 与表值），但标定时 `floris_eval` 会被调上百次 × 每次上百样本
    ⇒ 不缓存会反复重建同一个插值器。
    """
    from scipy.interpolate import interp1d
    key = round(float(a_val), 12)
    f = _INTERP_CACHE.get(key)
    if f is None:
        f = interp1d(_U_GRID, _power_table(float(key)), fill_value=0.0,
                     bounds_error=False)
        _INTERP_CACHE[key] = f
    return f


def _power_of_u(a, u, ref_rho):
    """逐台 a、逐点 u → 功率 [W]，复刻 SimpleTurbine.power 的插值口径。"""
    a = np.asarray(a, dtype=float)
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u, dtype=float)
    for av in np.unique(a):
        sel = (a == av)
        out[sel] = np.asarray(_interp_for(av)(u[sel]), dtype=float) * 1e3
    return out


def register_op_model():
    from floris.core.turbine import operation_models as OM
    from floris.core.turbine import turbine as TT
    from floris.core.rotor_velocity import (average_velocity,
                                            rotor_velocity_air_density_correction)
    import yawflor.yawflor_turbine as YT
    NAME = 'probe-per-turbine-a'
    if NAME in TT.TURBINE_MODEL_MAP['operation_model']:
        return NAME

    def _a_sel(velocities):
        """按 `velocities` 的**实际形状**取出对应的逐台 `a`。

        ⚠ 为什么不能只看全局 `_CUR_I`（这里修过一个真 bug）：
        引擎在两条路径上调用本操作模型，形状不同：
          * `power(velocities=(n, nt, g, g))`   —— **不筛选**，累加器形状 (n, nt)
            （`floris_model._get_turbine_powers` 直接调，无 `ix_filter`）；
          * `thrust_coefficient` / `axial_induction` —— solver 逐台循环，
            进本函数时 `velocities` **已被按 `ix_filter` 缩到 (n, 1, g, g)**，
            累加器形状 (n, 1)。
        若无条件用 `_CUR_I` 切片，`power` 路径会拿到**上一次 `run()` 残留的陈旧索引**
        （`_CUR_I` 只在 `run()` 期间有效，`get_turbine_powers()` 是之后才调用的）
        ⇒ 只算一台的功率。实测症状：逐台 `a=[0,1/3,1/3,1/3,1/3]` 时风场功率
        给出 3.237 MW（= 只有 T1），而"全场 a=0"给出 0.000 MW —— 两者自相矛盾。
        修法：`tgt = shape(velocities)[0:2]`，只按 `tgt[1]` 选列 ⇒ 与调用点无关。
        """
        a = np.asarray(_A_SET, dtype=float)
        tgt = np.shape(velocities)[0:2]
        if a.ndim == 1:
            a = a.reshape(1, -1)
        nt = int(tgt[1]) if len(tgt) > 1 else int(a.shape[1])
        if a.shape[1] != nt:
            if _CUR_I is not None and a.shape[1] > int(_CUR_I):
                a = a[:, int(_CUR_I):int(_CUR_I) + 1]
            if a.shape[1] > nt:
                a = a[:, :nt]
        return np.broadcast_to(a, tgt).copy()

    class PerTurbineA(OM.BaseOperationModel):
        @staticmethod
        def power(velocities=None, air_density=None, average_method='cubic-mean',
                  cubature_weights=None, **_):
            u = average_velocity(velocities, method=average_method,
                                 cubature_weights=cubature_weights)
            u = rotor_velocity_air_density_correction(u, air_density, YT.RHO)
            u = np.clip(np.asarray(u, dtype=float), 0.0, None)
            return _power_of_u(_a_sel(velocities), u, YT.RHO)

        @staticmethod
        def thrust_coefficient(velocities=None, **_):
            """CT = 4a(1−a)，并**按 FLORIS 原厂口径夹到 [CT_LO, CT_HI]**。

            ⚠ 为什么必须夹（2026 实测，`aifflor` 训练崩 `PER assert priority>0` 的根因）：
            FLORIS 随包路径在 `operation_models.py:134/:250` 有
                `np.clip(thrust_coefficient, 0.0001, 0.9999)`
            ⇒ 原厂 CT **永远不会恰好是 0**。而本算子为做**逐台 `a`** 自己算 CT，
            若不夹，`a = 0 ⇒ CT = 0`，就会命中高斯尾流的可去奇点：
                `wake_velocity/gauss.py:76`  `uR = u·CT/(2(1−√(1−CT)))` 是 **0/0**
                （`wake_deflection/gauss.py:143` 同款）
            ⇒ NaN → `ptur` → 观测 → TD 误差 → PER 优先级 → 崩溃（实测坏点率 4.37%）。
            夹到 0.0001 后：`uR/u = 0.999975`、`u0/u = 0.99995`、`σ0 → 0.3536D`，
            即"几乎无尾流"的**正确**极限（解析式验证：`uR/u ≡ 1−a` 在 CT≥0.0001 逐位成立）。
            """
            a = _a_sel(velocities)
            ct = CT_SCALE * 4.0 * a * (1.0 - a)
            ct = np.clip(ct, CT_LO, CT_HI)          # 与 FLORIS 原厂同口径
            return np.broadcast_to(ct, np.shape(velocities)[0:2]).copy()

        @staticmethod
        def axial_induction(velocities=None, **_):
            """引擎内的 `a`（由 CT 反解，恒等式上等于动作 `a`）。

            ⚠ **符号取 `+a`**：`agentutil/probe_floris_a_per_turbine.py` 的自校验
            （"逐台 a≡1/3 必须与生产 `yawflor` 基线逐位一致"，实测相对差 1.5e-16）
            是在返回 `+a` 下通过的；返回 `−a` 会让尾流恢复过快、对不上基线。
            这里用 CT 反解（而不是直接回传动作 `a`）是为了与被夹过的 CT 自洽：
            `a_eng = ½(1−√(1−CT_clipped))`。
            """
            a = _a_sel(velocities)
            ct = np.clip(4.0 * a * (1.0 - a), CT_LO, CT_HI)
            a_eng = 0.5 * (1.0 - np.sqrt(np.clip(1.0 - ct, 0.0, None)))
            return np.broadcast_to(a_eng, np.shape(velocities)[0:2]).copy()

    PerTurbineA.__name__ = 'PerTurbineA'
    TT.TURBINE_MODEL_MAP['operation_model'][NAME] = PerTurbineA
    return NAME


def _solver_hook():
    """临时包装 `floris.core.solver.thrust_coefficient`，把当前机组序号写进 `_CUR_I`。"""
    import contextlib
    import floris.core.solver as S
    orig = S.thrust_coefficient

    @contextlib.contextmanager
    def ctx():
        global _CUR_I

        def counting(*a, **kw):
            global _CUR_I
            ix = kw.get('ix_filter')
            _CUR_I = int(ix[0]) if ix is not None else None
            return orig(*a, **kw)

        S.thrust_coefficient = counting
        try:
            yield
        finally:
            S.thrust_coefficient = orig
            _CUR_I = None

    return ctx


def _switch_to_empirical_gauss(m, verbose=False):
    """把 `m` 从 GCH(`gauss`) 换成 **empirical_gauss** 族（四个配套项必须一起换）。

    实测踩坑（`probe_empirical_gauss_segments.py:75-105` 首次记录，此处为单一真源）：
      ① **挠曲也必须是 `empirical_gauss`** —— 给它 `gauss` 挠曲会报
         `GaussVelocityDeflection.function() takes 7 positional arguments but 8 were given`；
      ② **湍流必须是 `wake_induced_mixing`** —— `empirical_gauss` 自带求解器要求湍流模型提供
         `atmospheric_ti_gain`，只有它和 `wake_induced_mixing` 有；用 `crespo_hernandez` 报
         `'CrespoHernandez' object has no attribute 'atmospheric_ti_gain'`；
      ③ **GCH 三开关必须全 False** —— empirical_gauss 求解器对三者直接 raise
         （`Secondary steering not available for this model.` 等）；
      ④ 参数：`len(wake_expansion_rates) == len(breakpoints_D) + 1`（分段线性 σ(x)）。

    这里显式写入的值 = FLORIS 随包 `empirical_gauss` 的默认值（写出来是为了可审计）。
    """
    params = dict(wake_expansion_rates=[0.023, 0.008], breakpoints_D=[10.0],
                  sigma_0_D=0.3536, mixing_gain_velocity=2.0,
                  smoothing_length_D=2.0, awc_mode='baseline')
    m.set_param(['wake', 'wake_velocity_parameters', 'empirical_gauss'], params)
    m.set_param(['wake', 'wake_turbulence_parameters', 'wake_induced_mixing'],
                dict(atmospheric_ti_gain=0.0))
    m.set_param(['wake', 'wake_deflection_parameters', 'empirical_gauss'],
                dict(horizontal_deflection_gain_D=3.0, vertical_deflection_gain_D=-1.0,
                     deflection_rate=22.0, mixing_gain_deflection=0.0,
                     yaw_added_mixing_gain=0.0))
    m.set_param(['wake', 'model_strings', 'velocity_model'], 'empirical_gauss')
    m.set_param(['wake', 'model_strings', 'deflection_model'], 'empirical_gauss')
    m.set_param(['wake', 'model_strings', 'turbulence_model'], 'wake_induced_mixing')
    for flag in ('enable_secondary_steering', 'enable_transverse_velocities',
                 'enable_yaw_added_recovery'):
        m.set_param(['wake', flag], False)
    if verbose:
        print('[wake-model] empirical_gauss 族已生效：velocity/deflection=empirical_gauss、'
              'turbulence=wake_induced_mixing、GCH 三开关全 False')
    return m


def build_floris_model(stock=False, verbose=False, wake_model='gauss'):
    """构造"逐台 a"的 FLORIS 模型。

    Parameters
    ----------
    stock : bool
        `False`（**默认**）⇒ 施加 `floris_calibrated_defaults` 里标定好的尾流参数
            （对 WFSim 的验证集误差：常数重标定后 1.098%）；
        `True` ⇒ 用 FLORIS **随包默认**参数（对它自己的基线，如
            `analyse_calibrated_cost_scale.py` 的"默认 vs 标定"对照）。
    wake_model : {'gauss', 'empirical_gauss'}
        尾流**速度模型族**（默认 `'gauss'`）：
        * `'gauss'` = Bastankhah 经典高斯 + `gauss` 挠曲 + `crespo_hernandez` 湍流 +
          `sosfs` 叠加 + GCH 三开关（= 官方 `default_inputs.yaml` 的 GCH 口径）；
        * `'empirical_gauss'` = 随包 empirical 高斯（LES 拟合的**分段**扩展率），
          配套项见 `_switch_to_empirical_gauss()`。
        ⚠ `'empirical_gauss'` **只允许配 `stock=True`**：标定表里的 `ka/kb/alpha/beta`
        是 **gauss 族**参数，不可跨族平移。
    """
    import yawflor.yawflor_turbine as YT
    from floris import FlorisModel
    from floris_calibrated_defaults import (apply_calibrated_wake_params,
                                            describe)
    if wake_model not in ('gauss', 'empirical_gauss'):
        raise ValueError("wake_model 只支持 'gauss' / 'empirical_gauss'，收到 %r" % (wake_model,))
    if wake_model == 'empirical_gauss' and not stock:
        raise ValueError("标定参数 (ka/kb/alpha/beta) 只属于 gauss 族，不可平移 ⇒ "
                         "wake_model='empirical_gauss' 必须配 stock=True")
    name = register_op_model()
    m, _ = YT.build_model(turbine_type=None)
    m.set(turbine_type=[dict(
        turbine_type=name, hub_height=YT.HUB_HEIGHT, rotor_diameter=YT.D_ROTOR,
        TSR=8.0, operation_model=name,
        power_thrust_table=dict(
            ref_air_density=YT.RHO, ref_tilt=0.0,
            cosine_loss_exponent_yaw=YT.COSINE_LOSS_EXPONENT_YAW,
            cosine_loss_exponent_tilt=YT.COSINE_LOSS_EXPONENT_YAW,
            wind_speed=[float(x) for x in _U_GRID],
            power=[0.0] * len(_U_GRID),
            thrust_coefficient=[float(YT.CT_EFF)] * len(_U_GRID))),
    ])
    m.set(wind_shear=YT.WIND_SHEAR, reference_wind_height=YT.HUB_HEIGHT,
          air_density=YT.RHO)
    if wake_model == 'empirical_gauss':
        _switch_to_empirical_gauss(m, verbose=verbose)
    if not stock:
        apply_calibrated_wake_params(m)
        if verbose:
            print(describe())
    else:
        if verbose:
            print('[stock] 使用 FLORIS 随包默认尾流参数（仅用于对照；对 WFSim 差 66.9%）')
    # 适用域守卫：标定参数只在"5 台 5D 单排 + a 通道 + γ≡0"下有效
    if not stock and (m.core.farm.n_turbines != NT):
        raise ValueError(
            '标定后的 FLORIS 参数只在 %d 台 5D 单排 + a 通道 + γ≡0 下标定过；'
            '当前 n_turbines=%d ⇒ 请用 stock=True 或重新标定。'
            % (NT, m.core.farm.n_turbines))
    return m


def _run_batch(m, us, A):
    """一次向量化调用；返回 (farm, turb)，NaN 原样带出（由调用方处置）。"""
    import yawflor.yawflor_turbine as YT
    global _A_SET
    n = len(us)
    _A_SET = np.asarray(A, dtype=float)
    m.set(wind_speeds=list(np.asarray(us, dtype=float)),
          wind_directions=[YT.WIND_DIR_ALIGNED] * n,
          turbulence_intensities=[YT.TI] * n, yaw_angles=np.zeros((n, NT)))
    with _solver_hook()():
        m.run()
    turb = np.asarray(m.get_turbine_powers(), dtype=float) / 1e6
    return turb.sum(axis=1), turb


def run_floris(us, A, verbose=True, stock=False):
    import yawflor.yawflor_turbine as YT
    m = build_floris_model(stock=stock)
    info = dict(floris_version=_floris_version(), stock=bool(stock))
    if verbose:
        print('[FLORIS] %s 就绪（逐台 a 自定义 operation model，参数=%s），算 %d 条 ...'
              % (info['floris_version'], '随包默认' if stock else '标定后',
                 len(us)), flush=True)

    # ---- 自校验 ----
    # 仅对 **stock** 模型有意义：它应与生产 `yawflor`（同样随包默认）逐位一致。
    # 标定后的模型**故意**不等于 yawflor 基线 —— 那正是标定要做的事
    # （对 WFSim：66.9% → 1.1%）。此时改做"逐位可复现性"自校验：
    # 标定模型上显式施加同一组 θ 后，应逐位复现 stock 模型 + 该 θ。
    if stock:
        prod, _ = YT.build_model()
        ref = YT.farm_power(prod, float(us[0]), [0.0] * NT)
        farm0, _ = _run_batch(m, us, np.full((len(us), NT), 1.0 / 3.0))
        chk = float(farm0[0])
        rel = abs(chk - ref) / ref if np.isfinite(chk) else float('inf')
        if verbose:
            print('[FLORIS] 自校验(stock) a≡1/3: 本模型 %.9f vs yawflor %.9f  '
                  '相对差 %.2e  %s' % (chk, ref, rel,
                                       'OK' if rel < 1e-12 else '**不一致**'),
                  flush=True)
    else:
        from floris_calibrated_defaults import calibrated_theta
        th = np.asarray(calibrated_theta(), dtype=float)
        A33 = np.full((len(us), NT), 1.0 / 3.0)
        # ⚠ 两条路径都必须走 `_run_batch`（它内部装 `_solver_hook` 维护 `_CUR_I`）。
        #   若其中一条改走 `C.floris_eval`（自己装 hook），另一条会拿到未更新的
        #   `_CUR_I`，`_a_sel` 就会返回未切片的全宽数组 ⇒ 形状广播报错。
        f_cal, _ = _run_batch(m, us, A33)                       # 标定模型
        m_stock = build_floris_model(stock=True)
        # stock 模型 + 同一组 θ ⇒ 应与标定模型逐位相同（验证"默认值 = 那组 θ"这条链）
        from agentutil.calib_floris_to_wfsim import _apply_theta as _at
        _at(m_stock, th)
        f_ref, _ = _run_batch(m_stock, us, A33)
        d = float(np.max(np.abs(f_cal - f_ref)))
        if verbose:
            print('[FLORIS] 自校验(标定) 逐位可复现性: 与 stock+同一 θ 的 max|Δ| = %.3e  %s'
                  % (d, 'OK' if d < 1e-12 else '**不一致**'), flush=True)
            print('[FLORIS] 提示: 标定模型**故意**不等于 yawflor 基线'
                  '（对 WFSim 的验证集误差 66.9% → 1.1%）。', flush=True)

    # ---- 正式：逐台随机 a（整批 → 对 NaN 逐条重试 → 仍 NaN 则丢弃） ----
    t0 = time.time()
    farm, turb = _run_batch(m, us, A)
    bad = ~np.isfinite(farm)
    if bad.any():
        if verbose:
            print('[FLORIS] ⚠ 整批出现 %d 条非有限值 ⇒ 逐条重试' % int(bad.sum()), flush=True)
        for i in np.flatnonzero(bad):
            f1, t1 = _run_batch(m, [us[i]], A[i:i + 1])
            farm[i], turb[i] = f1[0], t1[0]
        still = ~np.isfinite(farm)
        if still.any() and verbose:
            print('[FLORIS] ⚠ 逐条重试后仍有 %d 条非有限值（已置 NaN，分析时会剔除）: %s'
                  % (int(still.sum()), np.flatnonzero(still).tolist()), flush=True)
    if verbose:
        print('[FLORIS] 完成，用时 %.3f s' % (time.time() - t0), flush=True)
    return farm, turb, info


def _floris_version():
    try:
        import floris
        return getattr(floris, '__version__', 'unknown')
    except Exception:
        return 'unknown'


# ===========================================================================
# 3. WFSim
# ===========================================================================
def run_wfsim(us, A, verbose=True):
    import matlab
    import matlab.engine
    if verbose:
        print('[WFSim] 启动 MATLAB 引擎 ...', flush=True)
    t0 = time.time()
    eng = matlab.engine.start_matlab()
    if verbose:
        print('[WFSim] 引擎就绪 (%.1f s)，MATLAB %s' % (time.time() - t0, eng.version()),
              flush=True)
    eng.cd(ROOT, nargout=0)

    pw = np.zeros(len(us), dtype=float)
    tw = np.full((len(us), NT), np.nan, dtype=float)
    t0 = time.time()
    for i in range(len(us)):
        res = eng.WFSim_port(matlab.double([float(us[i])]), matlab.double([0.0]),
                             matlab.double([list(map(float, A[i]))]), matlab.double([0.0]))
        pw[i] = float(res['out']) / 1e6
        try:
            tw[i] = np.asarray(res['test'][0][4]['turbine']['power'],
                               dtype=float).reshape(-1)[:NT] / 1e6
        except Exception:
            pass
        if verbose and ((i + 1) % 20 == 0 or i == 0):
            print('[WFSim] %3d/%d u=%7.3f a=%s P=%9.6f MW (%.1f s)'
                  % (i + 1, len(us), us[i], np.array2string(A[i], precision=3),
                     pw[i], time.time() - t0), flush=True)
    eng.quit()
    if verbose:
        print('[WFSim] 完成，用时 %.1f s（逐台回读 %d/%d 条）'
              % (time.time() - t0, int(np.isfinite(tw).all(axis=1).sum()), len(us)),
              flush=True)
    return pw, tw


# ===========================================================================
# 4. 对比
# ===========================================================================
def compare(us, A, pw, pf, tw, tf, info, mode, seed):
    m = (us >= U_LO) & (us <= U_HI) & (pw > 1e-9)
    ratio = np.divide(pf, pw, out=np.full_like(pf, np.nan), where=pw > 1e-9)
    am, asd = A.mean(axis=1), A.std(axis=1)
    W = 96
    print('\n' + '=' * W)
    print('随机行为下的风场出力对比  |  mode=%s n=%d seed=%d  |  FLORIS %s'
          % (mode, len(us), seed, info.get('floris_version')))
    print('  动作 a∈[%.1f,%.1f]^%d   mean(a)∈[%.3f,%.3f]   std(a)∈[%.3f,%.3f]'
          % (A_LO, A_HI, NT, am.min(), am.max(), asd.min(), asd.max()))
    print('=' * W)
    print('  %-34s %14s %14s' % ('', 'WFSim', 'FLORIS'))
    print('  %-34s %14.4f %14.4f' % ('ΣP/60 [MWh]（等权，非物理电量）',
                                     pw.sum() / 60, pf.sum() / 60))
    print('  %-34s %14.6f %14.6f' % ('平均出力 [MW]', pw[m].mean(), pf[m].mean()))
    print('  %-34s %14.6f %14.6f' % ('出力标准差 [MW]', pw[m].std(), pf[m].std()))
    print('  %-34s %14.6f %14.6f' % ('出力 min', pw[m].min(), pf[m].min()))
    print('  %-34s %14.6f %14.6f' % ('出力 max', pw[m].max(), pf[m].max()))

    print('\n  >>> 比值 FLORIS/WFSim（核心）:')
    print('      均值 %.4f | 中位 %.4f | std %.4f | min %.4f | max %.4f | 极差 %.4f'
          % (ratio[m].mean(), np.median(ratio[m]), ratio[m].std(), ratio[m].min(),
             ratio[m].max(), ratio[m].max() / ratio[m].min()))
    print('      p05 %.4f | p25 %.4f | p50 %.4f | p75 %.4f | p95 %.4f'
          % tuple(np.percentile(ratio[m], [5, 25, 50, 75, 95])))
    print('      ⇒ 对比 a≡1/3 单点时的 1.7420（std 0.0086）')

    print('\n  [②] 比值 vs 动作绝对水平 mean(a)：')
    print('      %-16s %6s %14s %14s' % ('mean(a)', 'n', '比值均值', '比值std'))
    for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5001)]:
        s = m & (am >= lo) & (am < hi)
        if s.sum():
            print('      %4.2f-%-10.2f %6d %14.4f %14.4f'
                  % (lo, hi, s.sum(), ratio[s].mean(), ratio[s].std()))
    print('      corr(比值, mean(a)) = %+.4f' % np.corrcoef(ratio[m], am[m])[0, 1])

    print('\n  [③] 比值 vs 空间结构 std(a)：')
    if asd[m].std() > 1e-9:
        print('      corr(比值, std(a))  = %+.4f' % np.corrcoef(ratio[m], asd[m])[0, 1])
    sf, sd = m & (asd < 0.05), m & (asd > 0.15)
    if sf.sum():
        print('      近均匀 (std<0.05, n=%d): 比值均值 %.4f' % (sf.sum(), ratio[sf].mean()))
    if sd.sum():
        print('      强差异化(std>0.15, n=%d): 比值均值 %.4f' % (sd.sum(), ratio[sd].mean()))

    print('\n  [④] 比值 vs 风速：')
    print('      corr(比值, u) = %+.4f' % np.corrcoef(ratio[m], us[m])[0, 1])
    for lo, hi in [(3.5, 6), (6, 9), (9, 12), (12, 18), (18, 25.1)]:
        s = m & (us >= lo) & (us < hi)
        if s.sum():
            print('      u %4.1f-%-5.1f n=%3d  比值均值 %.4f  std %.4f'
                  % (lo, hi, s.sum(), ratio[s].mean(), ratio[s].std()))

    rho = float(np.corrcoef(pw[m], pf[m])[0, 1])
    k = float(np.dot(pw[m], pf[m]) / np.dot(pw[m], pw[m]))
    rms = float(np.sqrt(((pf[m] - k * pw[m]) ** 2).mean()))
    print('\n  形状 / 常数重标定:')
    print('    corr(P_WFSim, P_FLORIS) = %.6f' % rho)
    print('    最优常数 k*              = %.6f' % k)
    print('    常数重标定后相对残差      = %.3f%%   （a≡1/3 单点时为 0.465%%）'
          % (100 * rms / pf[m].mean()))

    if np.isfinite(tf).any() and np.isfinite(tw).any():
        print('\n  逐台平均出力 [MW]（上游→下游）:')
        mw, mf = np.nanmean(tw, axis=0), np.nanmean(tf, axis=0)
        print('    WFSim  : %s' % np.array2string(mw, precision=4))
        print('    FLORIS : %s' % np.array2string(mf, precision=4))
        print('    比值   : %s' % np.array2string(mf / mw, precision=4))

    print('\n  明细（按 mean(a) 排序，抽 25 条）:')
    order = np.argsort(am)
    for i in order[::max(1, len(us) // 25)]:
        print('  u=%7.3f am=%.3f asd=%.3f  %10.6f %10.6f  %8.4f'
              % (us[i], am[i], asd[i], pw[i], pf[i], ratio[i]))

    return dict(n=int(len(us)), mode=mode, seed=int(seed),
                floris_version=info.get('floris_version'),
                ratio_mean=float(ratio[m].mean()),
                ratio_median=float(np.median(ratio[m])),
                ratio_std=float(ratio[m].std()),
                ratio_min=float(ratio[m].min()), ratio_max=float(ratio[m].max()),
                ratio_p05=float(np.percentile(ratio[m], 5)),
                ratio_p95=float(np.percentile(ratio[m], 95)),
                corr_ratio_amean=float(np.corrcoef(ratio[m], am[m])[0, 1]),
                corr_ratio_astd=(float(np.corrcoef(ratio[m], asd[m])[0, 1])
                                 if asd[m].std() > 1e-9 else None),
                corr_ratio_u=float(np.corrcoef(ratio[m], us[m])[0, 1]),
                corr_power=rho, k_best=k, resid_rel=float(rms / pf[m].mean()),
                mean_wfsim_mw=float(pw[m].mean()), mean_floris_mw=float(pf[m].mean()))


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description='随机行为下 WFSim vs FLORIS 风场出力对比',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--mode', choices=['lhs', 'unif', 'edge'], default='lhs')
    ap.add_argument('--u-mode', choices=['unif', 'rated', 'seq'], default='unif')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-matlab', action='store_true')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    A = sample_design(args.n, args.mode, args.seed)
    us = sample_wind(len(A), args.u_mode, args.seed)
    print('运行回执')
    print('  解释器 : %s' % sys.executable)
    print('  动作   : mode=%s n=%d  a∈[%.1f,%.1f]^%d  seed=%d'
          % (args.mode, len(A), A_LO, A_HI, NT, args.seed))
    print('  风速   : u-mode=%s n=%d  范围 [%.3f,%.3f]  均值 %.3f'
          % (args.u_mode, len(us), us.min(), us.max(), us.mean()))
    print('  对齐   : 逐条 (u, a1..a5) 逐位相同，唯一差异是引擎')

    pf, tf, info = run_floris(us, A)
    if args.no_matlab:
        print('\n--no-matlab: FLORIS ΣP/60=%.4f MWh 均值 %.6f MW'
              % (pf.sum() / 60, pf.mean()))
        _json(args.json, us, A, None, pf, tf, info, args, None)
        return 0

    try:
        pw, tw = run_wfsim(us, A)
    except Exception as e:
        print('\n[!] WFSim 侧失败：%s: %s' % (type(e).__name__, e))
        traceback.print_exc()
        print('\n排查：MATLAB Engine 需要能创建 named pipe。受限沙箱/服务账号下会报 '
              '"CreateFile failed: Access is denied" ⇒ 请在普通终端里跑。')
        return 2

    res = compare(us, A, pw, pf, tw, tf, info, args.mode, args.seed)
    _json(args.json, us, A, pw, pf, tw, tf, info, args, res)
    return 0


def _json(path, us, A, pw, pf, tf=None, tw=None, info=None, args=None, res=None):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    obj = dict(config=dict(n=len(us), mode=args.mode, u_mode=args.u_mode, seed=args.seed,
                           a_lo=A_LO, a_hi=A_HI, nt=NT,
                           floris=info.get('floris_version'), interpreter=sys.executable),
               wind=[float(x) for x in us],
               actions=[[float(v) for v in r] for r in A],
               p_floris=[float(x) for x in pf],
               p_floris_turbine=[[float(v) for v in r] for r in tf],
               summary=res)
    if pw is not None:
        obj['p_wfsim'] = [float(x) for x in pw]
    if tw is not None and np.isfinite(tw).any():
        obj['p_wfsim_turbine'] = [[None if not np.isfinite(v) else float(v) for v in r]
                                  for r in tw]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print('\nJSON 已写入 %s' % path)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
