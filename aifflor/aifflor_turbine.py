# -*- coding: utf-8 -*-
"""`aifflor` 物理常量 / 模型构造 / 向量化求值（**FLORIS 引擎的单一真源**）。

定位
----
`aifflor` = `td3per` 把**风场仿真器从 WFSim 换成 FLORIS**，其余完全一致。
因此本模块要保证：**动作语义与 `td3per` 逐字相同，只有引擎不同**。

两套仿真器的输入输出映射（本次任务的核心，逐项可追溯）
------------------------------------------------------
| 维度 | WFSim（`td3per` 原用） | FLORIS（`aifflor`） |
|---|---|---|
| 入口 | `WFSim_port(u, 0.0, a[5], 0.0)` | `Farm_power(u, a[5])`（向量化） |
| 动作 ∈ ℝ⁵ | **轴向因子 a**，`env.py` 夹到 **[0, 0.5]** | **同一个 a**（逐台），夹带同上 |
| 动作 → 物理 | `CT_prime = 4a(1−a)²`(=`Cp`)；力 = `1.5·½ρUe²·CT_prime/(1−a)` | `CT = clip(4a(1−a), 1e-4, 0.9999)`、`Cp_eff = 0.95·4a(1−a)²` |
| 航向 | `phi = 0`（恒） | `yaw_angles ≡ 0`（恒） |
| 端口 | 无端口维（2D 单排） | 第五维（y=0 单排） |
| 第 6 维动作 | SOC 目标信号，**不进风机** | 同（`env.PowerOutput` 已截到前 5 维） |
| 输出 | 风场**总有功** [MW]（`out/1e6`） | 风场**总有功** [MW]（Σ 逐台） |
| 围栏 | `3.5 ≤ u ≤ 25` ⇒ 否则 0 | **同**（`WindBattery` 基类守卫） |
| 单机口径 | `P = 0.95·½ρA·Cp·Ue³` | `P = 0.95·½ρA·Cp·u_eff³`（整数风速表 + 插值） |
| 布局 | 单排 5 台 5D、D=100 m、ρ=1.20、shear=0.12 | **同** |

⇒ **动作语义零改动、输出语义零改动**，因此 `td3per_myfunc.py` 可以**逐字复用**
（动作带 `[0,0.5]`、`PENALTY_NN_H_COEF=400`、`violation_amount` 边界全部不变）。

引擎差异（声明，不试图抹平）
----------------------------
1. **尾流参数必须标定**：FLORIS 随包默认对 WFSim 的 MAPE 是 **66.9%**；用
   `agentutil/floris_calibrated_defaults.py` 里标定后的参数后降到 **1.1%**
   （见 `documentation/floris_calibration_to_wfsim.md`）。本模块**默认施加标定参数**。
2. **适用域**：标定只针对 **5 台 5D 单排 + `a` 通道 + `γ≡0`** ⇒ 本模块会断言。
3. 两引擎的绝对电平仍然不同（例如 u=10、a=1/3：FLORIS 3.294 MW vs WFSim 3.446 MW）
   ⇒ **跨引擎的 profit 不可直接比较**，只能同轨比较或归一化后比较。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))            # aifflor/
_ROOT = os.path.dirname(_HERE)                                # 仓库根
for _p in (_ROOT, os.path.join(_ROOT, 'agentutil')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ===========================================================================
# 1. 口径常量（与 td3per / yawflor 同源；改动任何一项都会被契约测试抓到）
# ===========================================================================
from yawflor import yawflor_turbine as _YT                      # noqa: E402
from floris_calibrated_defaults import (                        # noqa: E402
    apply_calibrated_wake_params, calibrated_theta, describe,
    CALIBRATED_WAKE_PARAMS)

D_ROTOR = _YT.D_ROTOR            # 100 m
HUB_HEIGHT = _YT.HUB_HEIGHT      # 100 m
RHO = _YT.RHO                    # 1.20 kg/m³（WFSim meshing.m）
AREA = _YT.AREA
A_IND = _YT.A_IND                # 1/3（单机 Cp 最优点）
POWER_SCALE = _YT.POWER_SCALE    # 0.95
CP_EFF = _YT.CP_EFF              # 0.56296
CT_EFF = _YT.CT_EFF              # 8/9

#: **出口电平系数**（`FLORIS 电场出力 → 与 WFSim 对齐的常数倍率`）。
#:
#: 由来（实测，`agentutil/analyse_stock_vs_calibrated.py`）：用 **FLORIS 随包默认**
#: 尾流参数时，FLORIS 的电场出力系统性**低于** WFSim 约 40%，而且这个偏差几乎是一个
#: **纯常数倍率**（100 步真实风速序列上 `corr = 0.999980`、比值 `std = 0.0103`）。
#: ⇒ 用**一个**显式系数即可对齐电平和波动，**不需要**改任何尾流参数。
#:
#: 取值 = `1/k*`，其中 `k* = ⟨P_W, P_F⟩/⟨P_W, P_W⟩` 由全域 350 点
#: （3–20 m/s、0.5 m/s 粒度、每档 10 个随机行为）拟合：
#:     `k* = 1.6696`  ⇒  `1/k* = 0.598947`
#:
#: ⚠ 适用域（必须声明）：该系数在**全域平均**意义上最优，但逐点比值范围是
#: **1.311–2.149**，且分歧**由动作强度驱动**（`corr(r, min(a)) = −0.658`）
#: ⇒ 在**强降载 / 动作异质**区的偏差最大。详见 `aifflor/README.md` §3.7.1。
LEVEL_REF_K = 1.6696
LEVEL_COEF = 1.0 / LEVEL_REF_K   # ≈ 0.5989471
NT = _YT.NT                      # 5
SPACING_D = _YT.SPACING_D        # 5D
WIND_DIR_ALIGNED = _YT.WIND_DIR_ALIGNED   # 270°
TI = _YT.TI                      # 0.06
WIND_SHEAR = _YT.WIND_SHEAR      # 0.12

#: 动作先验带（**与 td3per 逐字相同**：轴向因子 [0, 0.5]）—— 这是**学习带**，
#: `td3per_myfunc.CheckAction` 就是按它做夹带与越界罚，故 aifflor 不改它。
A_LO, A_HI = 0.0, 0.5
A_NOMINAL = 0.3333               # env.WindBattery.nominal_action_value
U_MIN_ENV, U_MAX_ENV = _YT.U_MIN_ENV, _YT.U_MAX_ENV           # 3.5 / 25

#: **引擎安全下界**：`a` 必须严格 > 0。
#: ⚠ 实测（`agentutil/probe_aifflor_nan_map.py`）：**恰好 `a = 0` 时 100% 出 NaN**，
#: 而 `a ≥ 0.025` 全部干净。原因是高斯尾流 `wake_velocity/gauss.py:76` 的
#:     `uR = u·ct / (2·(1 − √(1 − ct)))`，`ct = 4a(1−a) → 0` 时是 **0/0**。
#: 因为 `CheckAction` 把动作夹到 `[0, 0.5]`，策略只要把某台推到 `a ≤ 0`，就会被夹成
#: **恰好 0** ⇒ 命中死点（aifflor 训练实测坏点率 **4.37%**，且逐条重试救回 0）。
#: ⇒ 物理层只接受 `a ≥ A_PHYS_LO`；`A_LO = 0` 与它之间那一条窄带在物理上按
#: `A_PHYS_LO` 计（见 `agentutil/probe_aifflor_nan_map.py` 与 aifflor/README.md）。
A_PHYS_LO = 0.025

#: 上一次 `eval_batch` 里"逐条重试仍失败、退回标称动作"的条数（模块级，供上层累计）
_LAST_N_FALLBACK = 0
#: 累计统计：整批坏点数 / 逐条重试成功数 / 退回标称数（诊断用，见 `stats()`）
STATS = dict(n_eval=0, n_bad_in_batch=0, n_retry_ok=0, n_fallback=0)


def last_n_fallback():
    """上一次 `eval_batch` 里退回标称动作的条数。"""
    return _LAST_N_FALLBACK


def stats():
    """累计的 NaN 处理统计（诊断用）。"""
    return dict(STATS)


# ===========================================================================
# 2. 模型构造（**唯一入口**）
# ===========================================================================
def build_model(nt=NT, calibrated=True, verbose=False, wake_model='gauss'):
    """构造"逐台 a"的 FLORIS 模型；返回 `(model, info)`。

    Parameters
    ----------
    nt : int            机组台数（主臂 5；≠5 时若 `calibrated=True` 会报错）
    calibrated : bool   `True`（默认）施加 `floris_calibrated_defaults` 的标定尾流参数；
                        `False` 用 FLORIS 随包默认（**"只用电平系数"模式**用这个；
                        此时必须配 `AifflorPhysics(power_scale=LEVEL_COEF)` 才与 WFSim 对齐）
    wake_model : {'gauss', 'empirical_gauss'}
        尾流速度模型族；`'gauss'`（默认）= Bastankhah 经典高斯（GCH 口径），
        `'empirical_gauss'` = 随包 empirical 高斯（分段扩展率，配套项见
        `probe_engine_random_actions._switch_to_empirical_gauss`）。
        ⚠ `'empirical_gauss'` 只允许 `calibrated=False`（标定参数是 gauss 族、不可平移）。
    """
    import agentutil.probe_engine_random_actions as _R
    if wake_model not in ('gauss', 'empirical_gauss'):
        raise ValueError("wake_model 只支持 'gauss' / 'empirical_gauss'，收到 %r" % (wake_model,))
    if wake_model != 'gauss' and calibrated:
        raise ValueError("尾流族 %r 没有对应的标定参数（标定表只覆盖 gauss 族）⇒ 必须 "
                         "calibrated=False" % (wake_model,))
    _R.register_op_model()
    m = _R.build_floris_model(stock=not calibrated, wake_model=wake_model, verbose=verbose)
    info = dict(
        engine='floris',
        floris_version=_R._floris_version(),
        calibrated=bool(calibrated),
        wake_model=str(wake_model),
        model_strings=dict(m.core.wake.model_strings),
        nt=int(nt),
        a_lo=A_LO, a_hi=A_HI, a_nominal=A_NOMINAL,
        u_min=U_MIN_ENV, u_max=U_MAX_ENV,
        layout='single_row_5D', rotor_diameter=D_ROTOR, hub_height=HUB_HEIGHT,
        rho=RHO, wind_shear=WIND_SHEAR, ti=TI, wind_direction=WIND_DIR_ALIGNED,
        cp_eff=CP_EFF, ct_eff=CT_EFF, power_scale=POWER_SCALE,
        level_ref_k=LEVEL_REF_K, level_coef=LEVEL_COEF,
        wake_params=(dict(CALIBRATED_WAKE_PARAMS) if calibrated else None),
        theta=(calibrated_theta() if calibrated else None),
    )
    if int(m.core.farm.n_turbines) != int(nt):
        raise ValueError('n_turbines=%d 与 nt=%d 不一致' % (m.core.farm.n_turbines, nt))
    if verbose:
        print(describe() if calibrated else
              '[aifflor] 使用 FLORIS 随包默认尾流参数（非生产配置）')
    return m, info


# ===========================================================================
# 3. 向量化求值
# ===========================================================================
def eval_batch(m, us, acts):
    """一次向量化调用算多组 `(u, a₁…a₅)` 的风场总功率 [MW]。

    Parameters
    ----------
    m : FlorisModel
    us : array (n,)          风速 [m/s]
    acts : array (n, nt)     **轴向因子**（不是偏航角；与 `td3per` 同语义）

    Returns
    -------
    (farm_mw (n,), turb_mw (n, nt))

    非有限值处理（**必需**，不是可选）
    ----------------------------------
    标定后的窄尾流在极端动作组合下会让 FLORIS 报
    `WARNING Some velocities at the rotor are negative` 并**返回 NaN**；
    而 FLORIS 是**批量**调用 ⇒ 一条坏点会污染整批。故这里：

      1. 整批调用 → 检出非有限项；
      2. 对每个坏点**逐条重试**（单点调用常能给出有限值）；
      3. 仍非有限 ⇒ 退回**标称动作 `a = 1/3`**（实测该点尾流最宽、必有限），
         并把该条记入 `_LAST_N_FALLBACK`（模块级计数，供 `AifflorPhysics` 累计）。

    ⚠ 这一层是**必需**的：历史 bug（aifflor 训练 step≈3386 崩在
    `PER_buffer.update_priorities: assert priority > 0`）的根因就是
    `prefetch` 缓存里存进了 NaN，且缓存命中路径**绕过**了上层守卫 ⇒ NaN 一路
    变成 `ptur` → 观测 → TD 误差 → PER 优先级。修在这里之后，所有调用方
    （直算 / 预取 / 批量）都拿到有限值。
    """
    global _LAST_N_FALLBACK
    us = np.atleast_1d(np.asarray(us, dtype=float))
    acts = np.atleast_2d(np.asarray(acts, dtype=float))
    n = int(us.size)
    if acts.shape[0] != n:
        raise ValueError('us 与 acts 行数不一致: %d vs %s' % (n, acts.shape))
    keep = np.clip(us, 1e-6, None)

    import agentutil.probe_engine_random_actions as _R
    farm, turb = _R._run_batch(m, keep, acts)
    farm = np.asarray(farm, dtype=float)
    turb = np.asarray(turb, dtype=float)

    n_fb = 0
    STATS['n_eval'] += n
    bad = ~np.isfinite(farm)
    if bad.any():
        STATS['n_bad_in_batch'] += int(bad.sum())
        for i in np.flatnonzero(bad):
            f1, t1 = _R._run_batch(m, [keep[i]], acts[i:i + 1])
            farm[i], turb[i] = float(np.ravel(f1)[0]), np.ravel(t1)
        ok = np.isfinite(farm)
        STATS['n_retry_ok'] += int((bad & ok).sum())
        still = ~ok
        for i in np.flatnonzero(still):
            a_nom = np.full((1, acts.shape[1]), float(A_IND))
            f2, t2 = _R._run_batch(m, [keep[i]], a_nom)
            farm[i], turb[i] = float(np.ravel(f2)[0]), np.ravel(t2)
            n_fb += 1
        STATS['n_fallback'] += n_fb
    _LAST_N_FALLBACK = int(n_fb)

    # `env.py` 的守卫：u < 3.5 时出力 0（与 `yawflor_turbine.eval_batch` 同口径）
    farm = np.where(us >= U_MIN_ENV, farm, 0.0)
    return farm, turb


def farm_power(m, u, act):
    """单点便捷入口：返回风场总功率 [MW]（标量）。"""
    a = np.asarray(act, dtype=float).reshape(-1)[:NT]
    if a.size < NT:
        a = np.full(NT, float(a.reshape(-1)[0]) if a.size else A_NOMINAL)
    p, _ = eval_batch(m, [float(u)], a.reshape(1, -1))
    return float(p[0])


def declared_scope():
    """范围声明（论文环境章节可一字不改引用）。"""
    return ('本文的 `aifflor` 环境采用准稳态尾流模型（FLORIS）：每个控制周期内假定'
            '轴向因子已到位、尾流已平衡，不建模变桨/转矩执行器动力学与尾流输运延迟。'
            'FLORIS 的尾流参数已按 WFSim 标定（验证集相对误差 66.9% → 1.1%），'
            '但两引擎的绝对功率电平仍不相同，跨引擎的收益不可直接比较。')
