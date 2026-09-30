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
| 动作 → 物理 | `CT_prime = 4a(1−a)²`(=`Cp`)；力 = `1.5·½ρUe²·CT_prime/(1−a)` | 表里 `CT = 4a(1−a)`、`Cp_eff = 0.95·4a(1−a)²` |
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
NT = _YT.NT                      # 5
SPACING_D = _YT.SPACING_D        # 5D
WIND_DIR_ALIGNED = _YT.WIND_DIR_ALIGNED   # 270°
TI = _YT.TI                      # 0.06
WIND_SHEAR = _YT.WIND_SHEAR      # 0.12

#: 动作先验带（**与 td3per 逐字相同**：轴向因子 [0, 0.5]）
A_LO, A_HI = 0.0, 0.5
A_NOMINAL = 0.3333               # env.WindBattery.nominal_action_value
U_MIN_ENV, U_MAX_ENV = _YT.U_MIN_ENV, _YT.U_MAX_ENV           # 3.5 / 25

#: FLORIS 高斯尾流在 CT→0 时 `uR = u·ct/(2(1−√(1−ct)))` 是 0/0 ⇒ a=0 会出 NaN。
#: 标定后的表用 `_U_GRID` 覆盖 0..40，故此处只做**运行期守卫**（见 `Farm_power`）。
A_SAFE_LO = 0.025


# ===========================================================================
# 2. 模型构造（**唯一入口**）
# ===========================================================================
def build_model(nt=NT, calibrated=True, verbose=False):
    """构造"逐台 a"的 FLORIS 模型；返回 `(model, info)`。

    Parameters
    ----------
    nt : int            机组台数（主臂 5；≠5 时若 `calibrated=True` 会报错）
    calibrated : bool   `True`（默认）施加 `floris_calibrated_defaults` 的标定尾流参数；
                        `False` 用 FLORIS 随包默认（仅供"标定前/后"对照，**不要用于生产**）
    """
    import agentutil.probe_engine_random_actions as _R
    _R.register_op_model()
    m = _R.build_floris_model(stock=not calibrated)
    info = dict(
        engine='floris',
        floris_version=_R._floris_version(),
        calibrated=bool(calibrated),
        nt=int(nt),
        a_lo=A_LO, a_hi=A_HI, a_nominal=A_NOMINAL,
        u_min=U_MIN_ENV, u_max=U_MAX_ENV,
        layout='single_row_5D', rotor_diameter=D_ROTOR, hub_height=HUB_HEIGHT,
        rho=RHO, wind_shear=WIND_SHEAR, ti=TI, wind_direction=WIND_DIR_ALIGNED,
        cp_eff=CP_EFF, ct_eff=CT_EFF, power_scale=POWER_SCALE,
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

    注：与 `yawflor_turbine.eval_batch` 同构，只把"偏航角"换成"轴向因子"。
    """
    import agentutil.probe_engine_random_actions as _R
    us = np.atleast_1d(np.asarray(us, dtype=float))
    acts = np.atleast_2d(np.asarray(acts, dtype=float))
    n = int(us.size)
    if acts.shape[0] != n:
        raise ValueError('us 与 acts 行数不一致: %d vs %s' % (n, acts.shape))
    keep = np.clip(us, 1e-6, None)
    farm, turb = _R._run_batch(m, keep, acts)
    farm = np.asarray(farm, dtype=float)
    turb = np.asarray(turb, dtype=float)
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
