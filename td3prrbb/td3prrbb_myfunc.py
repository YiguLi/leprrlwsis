"""
PRRBB — Wind × Price Segmented Reward (机会/风险分离版)
===================================================
Decomposes PG revenue into turbine (ptur) and BESS (pb) components.
机会侧（利润）用收益权重编码，风险侧（波动/平抑可行性/SOC）全部由 c1/c2 编码，
不再混入收益权重。

Upper (H) r1（机会侧，乘法分解）:
  r1 = wT * (pr * ptur * time)           ← 风机收益, wT = wT_u(u) × wT_p(pr)
     + wB * (-pr * pbrule * time)         ← BESS 收益, wB ≡ 1.0 (分段系数已去掉)
      - BESS_DEG_RMB_PER_MWH * max(-pbrule, 0) * time    ← 退化 (H 看 BESS 净收益)
      - penalty_NN_H

  Weight tables:
    wT_u:   LOW=1.2  MEDIUM=1.0  HIGH=0.9    (高风轻度降载倾向)
    wT_p:   LOW=1.0  MEDIUM=1.0  HIGH=1.05   (高价时段轻微加码发电)
    wB_p:   恒 1.0 —— 【本轮变更】BESS 收益项分段系数已去掉
    wB_soc: 恒 1.0 —— SOC 贴边折减同步取消 ⇒ wB ≡ 1.0
            (BESS 由 SOC + RuleBasedBESS 硬窗口调控, 基本不存在安全问题)

  Wind boundaries: LOW < 7.0 m/s, MEDIUM [7.0, 11.0), HIGH >= 11.0 m/s

Upper (H) c1（风险侧）:
  c1 = r_u × [10*(P_F-3)+ + 0.1*P_F] + W_GAP * unfeasible_gap
    r_u:   恒 1.0 —— 【本轮变更】波动风险项分段系数已去掉
           (风力变化本身已证明控制完成, 无需按风档加减风险罚)
    W_GAP = 50.0  (平抑可行性缺口 unfeasible_gap 的直接代价, 单位 MW)

Lower (L):
  r2 = −1 · min(clip_amount, 20)              (命令与实际之差 (两段 clip 合计); 影子套利项随信号语义删除)

    a6 → soc_target = clip(3+5·a6, 0.5, 5.5)   (SOC 目标, 块级固定)
  c2 = 1 · trackerr [MW]             (两区间分治: Soctrack范围 ∩ 平抑窗口;
                                      空交集 ⇒ 最近可达端点, 见 hsrl_soctarget)
"""

import numpy as np
import hsrl_cost as hc
import hsrl_soctarget as hst   # 下层 SOC 目标单一真源
import pandas as pd
from copy import deepcopy

from util import *


# =============================================================================
# 1. Price Classification (unchanged)
# =============================================================================

PRICE_HIGH   = 980
PRICE_MEDIUM = 580
PRICE_LOW    = 300


def get_price_level(pr):
    """Classify electricity price into 3 levels."""
    if pr >= PRICE_HIGH:
        return 'HIGH'
    elif pr <= PRICE_LOW:
        return 'LOW'
    else:
        return 'MEDIUM'


# =============================================================================
# 2. Wind Speed Classification (thresholds selectable at startup)
# =============================================================================
# 默认固定阈值 7.0/11.0；main 启动时可按 --seg 用 set_wind_thresholds()
# 切换为 percentile / kmeans / expert 三种分段边界（见 data_piecewise/boundaries.py）。

WIND_LOW_MAX  = 7.0   # 默认: u < WIND_LOW_MAX  → LOW wind
WIND_HIGH_MIN = 11.0  # 默认: u >= WIND_HIGH_MIN → HIGH wind, else MEDIUM

_wind_low_max  = WIND_LOW_MAX
_wind_high_min = WIND_HIGH_MIN


def get_wind_thresholds():
    """Return current (wind_low_max, wind_high_min)."""
    return _wind_low_max, _wind_high_min


def set_wind_thresholds(low_max, high_min):
    """Set wind thresholds with validity clamping (same constraints as PRLE).

    Constraints enforced:
      - low_max ∈ [3.0, high_min - 1.0]
      - high_min ∈ [low_max + 1.0, 22.0]
      - Always: low_max < high_min (at least 1.0 m/s gap)

    Returns (actual_low_max, actual_high_min) after clamping.
    """
    global _wind_low_max, _wind_high_min
    low_max = float(np.clip(low_max, 3.0, high_min - 1.0))
    high_min = float(np.clip(high_min, low_max + 1.0, 22.0))
    _wind_low_max = low_max
    _wind_high_min = high_min
    return _wind_low_max, _wind_high_min


def get_wind_level(u):
    """Classify wind speed into LOW / MEDIUM / HIGH using current thresholds."""
    if u < _wind_low_max:
        return 'LOW'
    elif u >= _wind_high_min:
        return 'HIGH'
    else:
        return 'MEDIUM'


# =============================================================================
# 3. Segmented Weight Lookup (multiplicative decomposition)
# =============================================================================
# 机会侧（利润）与风险侧（c1/c2）分离：
#   wT = wT_u(u) × wT_p(pr)                 — 风机发电收益权重
#   wB ≡ 1.0 — BESS 套利收益权重 (原 wB_p(pr) × wB_soc; 【本轮变更】分段系数已去掉)
# 风险全部进入 c1（波动+不可行缺口）与 c2（SOC 安全带），不再混入收益权重。

# 风机收益：低风鼓励多发（机会大、风险小），高风轻度降载倾向
WT_WIND = {'LOW': 1.2, 'MEDIUM': 1.0, 'HIGH': 0.9}
# 风机收益：高价时段轻微加码（降载的机会成本更大）
WT_PRICE = {'LOW': 1.0, 'MEDIUM': 1.0, 'HIGH': 1.05}
# BESS 收益：分段系数已取消 ⇒ wB ≡ 1.0（原 LOW=0.8 / MEDIUM=1.0 / HIGH=1.2）
#   理由: BESS 由 SOC + RuleBasedBESS 硬窗口调控, 基本不存在安全问题。
#   改回原三档值即一键恢复。
WB_PRICE = {'LOW': 1.0, 'MEDIUM': 1.0, 'HIGH': 1.0}
# BESS 收益：SOC 贴边折减同步取消（原 0.7）⇒ get_wB 恒返回 1.0
WB_SOC_OUTSIDE = 1.0

# c1 风险系数：分段系数已取消 ⇒ 恒 1.0（原 LOW=0.8 / MEDIUM=1.0 / HIGH=1.2）
#   理由: 风力变化本身已证明控制完成, 无需再按风档加/减波动风险罚。
C1_RISK = {'LOW': 1.0, 'MEDIUM': 1.0, 'HIGH': 1.0}
# unfeasible_gap（平抑可行性缺口, MW）进入 c1 的系数
W_GAP = 50.0


def get_wT(wind_level, price_level):
    """Turbine revenue weight: multiplicative wind × price decomposition."""
    return WT_WIND[wind_level] * WT_PRICE[price_level]


def get_wB(price_level, wind_level, SOC):
    """BESS revenue weight —— 【本轮变更】恒为 1.0（分段系数已去掉）。

    历史机制保留在下方代码里（一键恢复）: wB = WB_PRICE[price] × w_soc,
    w_soc = 1.0（SOC 在 SOC_SAFE_RANGES[wind] 带内）/ WB_SOC_OUTSIDE（带外）。
    现 WB_PRICE 三档均 1.0 且 WB_SOC_OUTSIDE = 1.0 ⇒ 两条分支都返回 1.0。
    恢复旧行为: 把 WB_PRICE / WB_SOC_OUTSIDE 改回原值即可（签名与调用点不变）。
    """
    soc_min, soc_max = SOC_SAFE_RANGES[wind_level]
    w_soc = 1.0 if soc_min <= SOC <= soc_max else WB_SOC_OUTSIDE
    return WB_PRICE[price_level] * w_soc


# =============================================================================
# 4. Upper-Level (H): r1 with wind × price segmented profit decomposition
# =============================================================================

# ---- 越界乘性惩罚 (--viol_derate) -------------------------------------------
# 动机 (agentutil/probe_q_profile_boundary.py 的实测):
#   线性罚 200/单位在 critic 回归空间只有 200/s_p = 0.2, 而框内经济斜率实测
#   0.27~4.5 ⇒ 跨界几乎不花钱。一维 Q(a_j) 剖面显示 Q 峰就贴在约束边界上
#   (dim4 峰 a=0.05; dim5 峰甚至在界外 -0.60) ⇒ 平滑 actor 逼近角点解时必有一
#   半状态落界外。且典型越界(深度~0.09)的罚贡献 = 200*0.09/1000 = 0.018,
#   比 critic 自身 TD 误差 0.64 小 30+ 倍 ⇒ 边界位置落在 critic 噪声里。
# 做法: 给 **风机收益** 乘一个自缩放系数 f = exp(-c·x), x = Σ_{j=0..4} 越界量
#   (只算机组限功率维; dim5=signal 不计 —— 按用户决定)。
#   * 乘在 turbine_profit 上而非净利润: turbine_profit = w·pr·ptur·dt ≥ 0 恒成立
#     (实测 100% 的步), 结构上排除"亏损步上 f<1 反而鼓励越界"的符号陷阱;
#     且其量级(49.9/步)大于 r1(29.9/步), 界外斜率更强。
#   * 与收益同尺度 ⇒ 界面斜率 = c·(该步风机收益), 随风速/电价自动缩放。
#   * r1 单调不增: f<=1 且 turbine_profit>=0 ⇒ 越界在任何情况下都不会让 r1 变大。
#   * 只动 r1; env 物理量、revenue/degradation/turbine_profit/profit 等**记录列全部不变**。
VIOL_DERATE_COEF = 0.0     # c; <=0 关闭 (返回 1.0 ⇒ 与历史路径逐位相同)
#   ⚠ **默认 0 = 关闭**: c=5.0 试行过一次 (run14) 并**实测有害** ——
#     乘性罚让奖励电平依赖"策略自身的越界率" ⇒ 电平非平稳 (块奖励 117->29,
#     净 r1 掉到负), 而 critic 的 Qbar 因 warmup 基线 + 固定 s_p 不跟随
#     ⇒ 自洽比 0.4~0.8 变 1.25~1.78 (高估约 2x) ⇒ actor 追虚高 Q ⇒ 越界更多
#     ⇒ 正反馈 (越界率 18%~78% 振荡, 策略抖动 3 倍)。要重新启用必须**先**解决
#     "奖励电平非平稳" (见 README 的 --viol_derate 条目)。
VIOL_DERATE_DIMS = 5       # 前 N 维 = 机组限功率; dim5(signal) 不计入 x
_ACT_HI, _ACT_LO = 0.5, 0.0   # 与 CheckAction 的前 5 维边界一致


def violation_amount(action, n_dims=None):
    """x = 前 n_dims 维 (机组限功率, 边界 [0,0.5]) 的越界量之和。"""
    if n_dims is None:
        n_dims = VIOL_DERATE_DIMS
    a = np.asarray(action, dtype=float).reshape(-1)
    n = min(int(n_dims), a.size)
    if n <= 0:
        return 0.0
    v = np.maximum(np.maximum(a[:n] - _ACT_HI, _ACT_LO - a[:n]), 0.0)
    return float(v.sum())


def violation_derate(action, coef=None, n_dims=None):
    """乘性越界系数 f = exp(-c·x)。

    coef<=0 或 x==0 ⇒ 恰好返回 1.0 (IEEE754 下 `tp*1.0 == tp` 逐位相等),
    因此 `--viol_derate 0` 与改动前的训练路径**逐位一致**。
    """
    c = VIOL_DERATE_COEF if coef is None else float(coef)
    if c <= 0.0:
        return 1.0
    x = violation_amount(action, n_dims)
    if x <= 0.0:
        return 1.0
    return float(np.exp(-c * x))


def Cal_r1_PRRBB(pr, ptur, pbrule, penalty_NN_H, wind_level, price_level,
                 SOC=3.0, time=1.0 / 60, turbine_derate=1.0):
    """
    PRRBB upper-level r1 — 机会/风险分离的分段收益分解.

    r1 = wT * (pr * ptur * time)           ← 风机收益, wT = wT_u(u)×wT_p(pr)
       + wB * (-pr * pbrule * time)        ← BESS 收益, wB ≡ 1.0 (【本轮变更】分段系数已去掉)
       - BESS_DEG_RMB_PER_MWH * max(-pbrule, 0.0) * time  ← 退化 (H 看 BESS 净收益)
       - penalty_NN_H

    风险（波动/平抑可行性/SOC 越界）由 c1/c2 编码，不再混入收益权重。

    Parameters
    ----------
    pr : float
        Electricity price (RMB/MWh).
    ptur : float
        Turbine power output (MW).
    pbrule : float
        Rule-based BESS power after RBB clipping (MW). Positive = charging.
    penalty_NN_H : float
        Upper action boundary penalty.
    wind_level : str
        'LOW' | 'MEDIUM' | 'HIGH'.
    price_level : str
        'LOW' | 'MEDIUM' | 'HIGH'.
    SOC : float
        Battery SOC at decision time (MWh). 当前 wB ≡ 1.0 不再用它; 保留参数以便把 WB_SOC_OUTSIDE 改回 0.7 时一键恢复。
    time : float
        Step duration in hours (default 1/60).

    Returns
    -------
    r1 : float
    """
    w_wind = get_wT(wind_level, price_level)
    w_bess = get_wB(price_level, wind_level, SOC)

    turbine_profit = w_wind * pr * ptur * time
    bess_profit    = w_bess * (-pr * pbrule * time)
    degradation    = BESS_DEG_RMB_PER_MWH * max(-pbrule, 0.0) * time

    return (turbine_profit * turbine_derate + bess_profit - degradation
            - penalty_NN_H)


# =============================================================================
# 5. Lower-Level (L): wind-segmented SOC safety ranges (c2 拉格朗日约束项)
# =============================================================================

# SOC safe ranges by wind level (risk level):
#   LOW wind    → low risk  → widest range  [0.5, 5.5]
#   MEDIUM wind → med risk  → medium range  [1.0, 5.0]
#   HIGH wind   → high risk → narrowest     [1.5, 4.5]
SOC_SAFE_RANGES = {
    'LOW':    (0.5, 5.5),
    'MEDIUM': (1.0, 5.0),
    'HIGH':   (1.5, 4.5),
}


def Cal_soc_deviation(SOC, wind_level='MEDIUM'):
    """
    Wind-segmented SOC deviation.

    Returns distance outside the safe band for the given wind risk level.
    0 if SOC is inside the band.
    """
    soc_min, soc_max = SOC_SAFE_RANGES[wind_level]
    dev = 0.0
    if SOC < soc_min:
        dev = soc_min - SOC
    elif SOC > soc_max:
        dev = SOC - soc_max
    return dev


def Cal_r2_PER_baseline(pbrule, shadow_price, SOC, wind_level='MEDIUM'):
    """
    LEGACY helper (kept for backward compatibility, no longer called in
    Cal_Rew_PRRBB). The current lower-level reward/cost are computed inline:
      r2 = 10.0 * shadow_profit - 100.0 * clip_amount
      c2 = 700.0 * soc_deviation   (SOC-safe 项进入拉格朗日约束)

    Wind-segmented lower-level r2.

    r2 = 10.0 * shadow_profit - 700.0 * soc_deviation

    SOC safe range tightens with wind risk level:
      LOW wind    → [0.5, 5.5]
      MEDIUM wind → [1.0, 5.0]
      HIGH wind   → [1.5, 4.5]

    shadow_profit = -shadow_price * pbrule
      shadow_price > 0 → encourage discharge (pb < 0 → shadow_profit > 0)
      shadow_price < 0 → encourage charge    (pb > 0 → shadow_profit > 0)
    """
    shadow_profit = -shadow_price * pbrule
    soc_dev = Cal_soc_deviation(SOC, wind_level)
    return 10.0 * shadow_profit - 700.0 * soc_dev


# =============================================================================
# 6. Combined Reward Function
# =============================================================================

def Cal_Rew_PRRBB(observation, observation2, action_rbb, rbb_info,
                   penalty_NN_H=0.0, time=1.0 / 60, ptur=None, n_rem=None, turbine_derate=1.0):
    """
    PRRBB reward: 机会/风险分离的分段奖励.
    Upper r1 = wT*pr*ptur*dt + wB*(-pr*pbrule*dt) - 退化 - penalty_NN_H
             （wT = wT_u×wT_p; wB ≡ 1.0 —— 【本轮变更】BESS 分段系数已去掉）
    Upper c1 = r_u(u)*[10*(P_F-3)+ + 0.1*P_F] + W_GAP*unfeasible_gap
               (legacy 原式; uniform 默认走 hc.c1_apply ⇒ P_F + K_V*(P_F-3)+, K_V=100)
    Lower r2 = −1·min(clip_amount, 20) [MW]  (SOC 目标语义; 影子套利项已删除)
    Lower c2 = +1·trackerr    [MW]  参考集合 = Soctrack范围 ∩ 平抑窗口;
              空交集 ⇒ **无范围, 只有纯粹一个竭力值 r*** (顶到才不罚)
    (LEGACY 基线 Cal_r2_PER_baseline 另用 10·shadow_profit − 700·soc_deviation)

    Parameters
    ----------
    observation  : [pr, u, E, P_G_last]
    observation2 : [pr', u', E', P_G]
    action_rbb   : [a1..a5, soc_signal, pbrule]
    rbb_info     : dict from RuleBasedBESS.compute_pbrule()
    penalty_NN_H : float, upper action boundary penalty
    time         : step duration in hours
    ptur         : float, turbine power output (MW). Must be provided.

    Returns
    -------
    r1, c1, r2, c2, details
    """
    if ptur is None:
        raise ValueError("Cal_Rew_PRRBB: ptur must be provided")

    pr_old = float(observation[0])
    u = float(observation[1])
    P_G_new = float(observation2[-1])
    P_G_old = float(observation[-1])
    SOC_new = float(observation2[2])

    pbrule = float(rbb_info['pbrule'])
    unfeasible_gap = float(rbb_info['unfeasible_gap'])
    clip_amount = float(rbb_info['clip_amount'])
    P_F = abs(P_G_new - P_G_old)

    soc_signal = float(action_rbb[5])
    soc_target = hst.soc_target_from_signal(soc_signal)   # SOC 目标 (块级固定)

    wind_level = get_wind_level(u)
    price_level = get_price_level(pr_old)
    SOC_decision = float(observation[2])  # 决策时刻 SOC（wB ≡ 1.0 后仅供签名/回退用）
    w_wind = get_wT(wind_level, price_level)
    w_bess = get_wB(price_level, wind_level, SOC_decision)

    # ==== Upper Level (H) ====
    r1 = Cal_r1_PRRBB(pr_old, ptur, pbrule, penalty_NN_H,
                       wind_level, price_level, SOC=SOC_decision, time=time,
                       turbine_derate=turbine_derate)

    # c1: fluctuation penalty
    #   越限部分 10.0*(P_F-3.0) 仅在 P_F > 3.0 时生效；
    #   平滑项 0.1*P_F 对任意波动值 P_F（不仅限于越限值）都生效。
    #   风险系数三档: LOW×0.8, MEDIUM×1.0, HIGH×1.2。
    #   unfeasible_gap（平抑可行性缺口, MW）进入 c1: H 制造了物理上
    #   无法平抑的波动时直接付费（风速/SOC/轴向因子/BESS额定 的综合体现）。
    c1_risk_coeff = C1_RISK[wind_level]
    if P_F > 3.0 + 1e-4:
        c1 = c1_risk_coeff * (10.0 * (P_F - 3.0) + 0.1 * P_F)
    else:
        c1 = c1_risk_coeff * (0.1 * P_F)
    c1 = c1 + W_GAP * unfeasible_gap
    c1 = hc.c1_apply(c1, P_F)   # uniform: 波动当量替换 (legacy: 原样)

    # ==== Lower Level (L) —— 两区间分治 (单一真源 hsrl_soctarget) ====
    #   r2 = −1·min(clip_amount, 20) [MW]  命令与实际之差 (两段 clip 合计)
    #   c2 = +1·trackerr    [MW]  实际施加 pb 到"Soctrack范围 ∩ 平抑窗口"的距离
    #                             (空交集 ⇒ 无范围, 只有纯粹一个竭力值 r*: 顶到才不罚)
    n_rem = n_rem if n_rem is not None else rbb_info.get('n_rem')
    if n_rem is None:
        raise ValueError('Cal_Rew_PRRBB: 需要 n_rem (块内剩余步数 = 5 − step%5)')
    soc_dev = Cal_soc_deviation(SOC_new, wind_level)   # 仅诊断: 原安全带偏离
        # r2 = −1·min(clip_amount,20); 不可行窗口下 RBB 忽略命令(见 hst.lower_reward 的 ⚠), 本口径保留
    r2 = hst.lower_reward(clip_amount)
    c2, c2_info = hst.lower_cost_soc_track(
        soc_target, float(observation[2]), SOC_new,
        rbb_info['pbrulemin'], rbb_info['pbrulemax'], n_rem,
        pbrule=float(rbb_info['pbrule']),   # 放电退化罚 (L_DEG_COEF)
        return_info=True)
    slack = c2_info['slack']

    # ==== Details ====
    turbine_profit = w_wind * pr_old * ptur * time
    bess_profit    = w_bess * (-pr_old * pbrule * time)

    details = {
        # Segmentation info
        'wind_level': wind_level,
        'price_level': price_level,
        'w_wind': w_wind,
        'w_bess': w_bess,
        # Price / Wind
        'pr': pr_old,
        'u': u,
        # Reward components
        'turbine_profit': turbine_profit,
        'bess_profit': bess_profit,
        'revenue': turbine_profit + bess_profit,
        'degradation': BESS_DEG_RMB_PER_MWH * max(-pbrule, 0.0) * time,
        'soc_signal': soc_signal,
        'soc_target': soc_target,
        'soc_slack': slack,
        'soc_deviation': soc_dev,   # 仅诊断 (原 SOC 安全带偏离)
        # Power
        'ptur': ptur,
        'pbrule': pbrule,
        'P_F': P_F,
        'SOC': SOC_new,
        # RBB
        'unfeasible_gap': unfeasible_gap,
        'clip_amount': clip_amount,
        # Penalties
        'penalty_NN_H': penalty_NN_H,
        'c1_fluct': c1,
    }

    return r1, c1, r2, c2, details


# =============================================================================
# 7. Action Checking (PER convention, unchanged)
# =============================================================================

PENALTY_NN_H_COEF = 400.0  # [元/单位] H 层动作越界罚 (机组限功率 [0,0.5]×5 + 信号 [±0.5])
                         #   原 200 -> 400: 越界仍未压住, 先整体加倍。
                         #   ⚠ s_p=1000 ⇒ critic 空间 0.4/单位, 而框内经济斜率实测 0.27~4.5
                         #   ⇒ 仍然同量级, 不改变"边界位置落在 critic 噪声里"的结论。
PENALTY_NN_L_COEF = 200.0  # [元/单位] 下层 1 维动作罚 —— **保持 200 不变** (仅诊断, 不进 r2 公式)


def CheckAction(action, charge=0.5):
    """Clip actions to feasible ranges. WT: [0,0.5], Signal: [-0.5,0.5], Bat: [-charge,charge]."""
    penalty = 0
    SIGNAL_MIN, SIGNAL_MAX = -0.5, 0.5

    if len(action) == 7:
        for j in action[:5]:
            ma, mi = max(0.5, j), min(0, j)
            penalty += PENALTY_NN_H_COEF *((ma - 0.5) + (0. - mi))
        sig = action[5]
        if sig > SIGNAL_MAX:    penalty += PENALTY_NN_H_COEF *(sig - SIGNAL_MAX)
        elif sig < SIGNAL_MIN:  penalty += PENALTY_NN_H_COEF *(SIGNAL_MIN - sig)
        for j in action[6:]:
            ma, mi = max(charge, j), min(-charge, j)
            penalty += PENALTY_NN_L_COEF *((ma - charge) + (-charge - mi))
        a_H = np.append(np.clip(action[:5], 0., 0.5),
                        np.clip(action[5], SIGNAL_MIN, SIGNAL_MAX))
        a_L = np.clip(action[6:], -charge, charge)
        return np.concatenate((a_H, a_L)), penalty

    elif len(action) == 6:
        for j in action[:5]:
            ma, mi = max(0.5, j), min(0, j)
            penalty += PENALTY_NN_H_COEF *((ma - 0.5) + (0. - mi))
        sig = action[5]
        if sig > SIGNAL_MAX:    penalty += PENALTY_NN_H_COEF *(sig - SIGNAL_MAX)
        elif sig < SIGNAL_MIN:  penalty += PENALTY_NN_H_COEF *(SIGNAL_MIN - sig)
        applied = np.append(np.clip(action[:5], 0., 0.5),
                            np.clip(action[5], SIGNAL_MIN, SIGNAL_MAX))
        return applied, penalty

    elif len(action) == 5:
        for j in action[:5]:
            ma, mi = max(0.5, j), min(0, j)
            penalty += PENALTY_NN_H_COEF *((ma - 0.5) + (0. - mi))
        return np.clip(action[:5], 0., 0.5), penalty

    elif len(action) == 1:
        for j in action:
            ma, mi = max(charge, j), min(-charge, j)
            penalty += PENALTY_NN_L_COEF *((ma - charge) + (-charge - mi))
        return np.clip(action, -charge, charge), penalty

    return action, penalty


# =============================================================================
# 8. Normalization (unchanged)
# =============================================================================

def Normalize(obs, max_E=0.5):
    return np.array([obs[0] / 1000, obs[1] / 10, obs[2] / 0.5, obs[3] / 5.0])


def Original(obs, max_E=0.5):
    return np.array([obs[0] * 1000, obs[1] * 10, obs[2] * 0.5, obs[3] * 5.0])


def build_L_obs_distilled(observation, pw_tru, applied_action_H, rbb_info, n_rem=None):
    """下层观测 (5 维) —— **规则的原配料** (单一真源 hsrl_soctarget.build_L_obs):

        [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]

    * `pg(t−1) = observation[-1]` 与 `ptur` 给出平抑窗口的中心;
    * SOC 与目标同刻度 ⇒ δ = 10·(obs[3] − obs[2]); 加 `n_rem/5` ⇒ 摊还需求 `need` 可算;
    * 因此规则动作所需的状态量都在观测里 (窗口的平抑部分 = pg(t−1)/ptur; 物理限幅部分
      由 SOC + 已知常量决定), 不必再从两个近似相等的
      /10 数里做差 (旧观测里 δ 分辨率只有 ~0.015)。
    丢掉 u (风速) 与窗口边界 (可由 ptur/pg_prev + 已知限幅推出)。

    `n_rem` = 含当前步的块内剩余步数 (5 - step %% 5) —— **必填**。
    `rbb_info` 仅为与其它项目同名函数保持签名一致 (本函数不再使用)。
    """
    if n_rem is None:
        raise ValueError('build_L_obs_distilled: 需要 n_rem (块内剩余步数 = 5 - step mod 5)')
    soc_target = hst.soc_target_from_signal(applied_action_H[-1])
    return hst.build_L_obs(float(observation[-1]), float(pw_tru),
                           float(observation[2]), soc_target, n_rem)


# =============================================================================
# 9. Test Function
# =============================================================================

# ⚠ 历史遗留: `prrbb_S_test` 当前**无任何调用者**（评测走 test_five_days.py 的 run_day，
#   训练内测试走 nine_case_test）。它只会构造 4 维 H 观测：若将来要启用，必须像
#   nine_case_test 那样补上相位（`Normalize_H(obs, phase_minute_of_day(step), phase_mode,
#   phase_table)`）并给它加 phase_mode/phase_table 参数，否则与 6 维 H 网络形状不符。
def prrbb_S_test(agent_H, agent_L, env_test, rbb_system,
                  num_iterations=5, test_csv=False, output=None, day=0):
    """PRRBB test with per-step detail recording."""
    obs = env_test.TestReset(day)
    norm_obs = Normalize(obs)
    time = 1.0 / 60
    agent_H.is_training = False
    agent_L.is_training = False
    step = 0
    episode_r1 = episode_c1 = episode_r2 = episode_c2 = 0.0
    episode_reward = episode_cost = 0.0
    episode_action = []
    df = pd.DataFrame()
    data_list = []
    cvm = 0

    while step < num_iterations:
        if step % 10 == 0:
            print(f"step: {step}")

        if step % 5 == 0:
            action_H = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H)
            pw_tru = env_test.PowerOutput(obs[1], action_H)
            sum_r1 = 0.0
            sum_c1 = 0.0

        # L 层观测蒸馏 (5维): [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]
        _, rbb_info_obs = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=float(pw_tru),
            pg_prev=float(obs[-1]), E=float(obs[2]))
        norm_obs_L = build_L_obs_distilled(obs, pw_tru, action_H, rbb_info_obs, 5 - (step % 5))
        action_L = agent_L.select_action(norm_obs_L)
        action_L, penalty_NN_L = CheckAction(action_L)

        action = np.concatenate((action_H, action_L))

        pb_original = float(action[-1])
        ptur = float(pw_tru)
        pg_prev = float(obs[-1])
        E_current = float(obs[2])

        pbrule, rbb_info = rbb_system.compute_pbrule(
            pb_original=pb_original, ptur=ptur,
            pg_prev=pg_prev, E=E_current,
        )

        action_rbb = action.copy()
        action_rbb[-1] = pbrule

        pw_tru2, obs2, action_L_env, penalty_L, done, done_L, info = \
            env_test.TestStep(action_rbb[:6], action_rbb[6:], pw_tru)

        action_rbb[6:] = action_L_env
        episode_action.append(action_rbb)
        norm_obs2 = Normalize(obs2)

        r1, c1, r2, c2, details = Cal_Rew_PRRBB(
            obs, obs2, action_rbb, rbb_info,
            penalty_NN_H=penalty_NN_H, time=time,
            ptur=ptur, n_rem=5 - (step % 5),
        )

        sum_r1 += r1
        sum_c1 += c1
        if c1 > 0:
            cvm += 1

        if test_csv:
            df = pd.concat([df, pd.DataFrame([{
                'step': step,
                'obs': obs, 'norm_obs_L': norm_obs_L,
                'action': action_rbb, 'action_L': action_L_env,
                'r1': r1, 'c1': c1, 'r2': r2, 'c2': c2,
                'r': r1 + r2, 'c': c1,
                'sum_r1': sum_r1, 'sum_c1': sum_c1,
                'pb_original': rbb_info['pb_original'],
                'pbrule': rbb_info['pbrule'],
                'clip_amount': rbb_info['clip_amount'],
                'unfeasible_gap': rbb_info['unfeasible_gap'],
                'soc': rbb_info['soc'],
                'wind_level': details['wind_level'],
                'price_level': details['price_level'],
                'w_wind': details['w_wind'],
                'w_bess': details['w_bess'],
                'pr': details['pr'],
                'u': details['u'],
                'turbine_profit': details['turbine_profit'],
                'bess_profit': details['bess_profit'],
                'soc_deviation': details['soc_deviation'],
                'soc_slack': details['soc_slack'],
                'soc_target': details['soc_target'],
                'revenue': details['revenue'],
                'degradation': details['degradation'],
                'P_F': details['P_F'],
                'SOC': details['SOC'],
                'penalty_NN_H': penalty_NN_H,
                'penalty_L': penalty_L,
                'penalty_NN_L': penalty_NN_L,
            }])], ignore_index=True)

            data = {
                'step': step, 'obs': obs, 'obs2': obs2, 'done': done,
                'r1': r1, 'c1': c1, 'r2': r2, 'c2': c2,
                'details': details, 'rbb_info': rbb_info,
            }
            data_list.append(data)
            if step % 50 == 0:
                df.to_csv(f'{output}/test_s_prrbb.csv')
                np.save(f'{output}/test_s_prrbb.npy', data_list)

        episode_reward += (r1 + r2)
        episode_cost += c1
        episode_r1 += r1; episode_r2 += r2
        episode_c1 += c1; episode_c2 += c2

        norm_obs = norm_obs2
        obs = obs2
        pw_tru = pw_tru2
        step += 1

    agent_H.is_training = True
    agent_L.is_training = True
    if test_csv:
        df.to_csv(f'{output}/test_s_prrbb.csv')
        np.save(f'{output}/test_s_prrbb.npy', data_list)

    return (episode_r1, episode_c1, episode_r2, episode_c2,
            episode_reward, episode_cost, action_rbb, cvm)


# =============================================================================
# 10. Nine-Case Segmented Test (45 steps, every 480 training steps)
# =============================================================================

try:
    from test_scenario import build_9case_scenario
except ImportError:
    import sys as _sys, os as _os
    _HSRL_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _HSRL_ROOT not in _sys.path:
        _sys.path.insert(0, _HSRL_ROOT)
    from test_scenario import build_9case_scenario


# test9 存货结算参考价 —— **外生常数**（= 名义电价 hc.PR_REF, MED=580, 避免裸数字）。
# 为什么必须外生: 若取"当时电价", 每步的充放收益逐项抵消 (ΔSOC·(pr_ref − pr) ≡ 0)
#   ⇒ 电池对目标的贡献恒为 0 ⇒ 套利激励归零 (退化成没有电池)。
# 为什么取 580: 把效率算进去后, 充电闸门 = η_c·pr_ref = 568.4, 放电闸门 = pr_ref/η_d = 591.8,
#   而场景三价是 LOW 300 / MED 580 / HIGH 980 ⇒ **MED 恰好落在中性带 [568.4, 591.8] 内**
#   ⇒ 该充电时充电、该放电时放电、平价时段不动作, 与物理一致。
SOC_SETTLE_PR = float(hc.PR_REF)


# =============================================================================
# 24h 相位特征 (自 td3per/td3perrb/td3prrlrb 逐字复制, 六方法统一口径)
#   --phase_feat 0/1/2 -> 追加 0/2/5 维; 训练侧与 test9 侧必须用同一构造。
# =============================================================================
PHASE_DIM = {0: 0, 1: 2, 2: 5}
DAY_MINUTES = 1440

_PHASE_TABLE_CACHE = {}


def day_price_schedule():
    """与 env.CalPrice 一致的 1440 分钟电价日程表 (LOW 300 / MED 580 / HIGH 980)。"""
    m = np.arange(DAY_MINUTES)
    p = np.full(DAY_MINUTES, 980.0)
    p[(m < 7 * 60) | (m >= 23 * 60)] = 300.0
    p[((m >= 11 * 60) & (m < 17 * 60)) | ((m >= 21 * 60) & (m < 23 * 60))] = 580.0
    return p


def phase_daily_table():
    """日程表逐分钟的 [段内剩余比例, 剩余小时/8, 下一段电价/1000] (循环日程)。"""
    key = 'daily'
    if key in _PHASE_TABLE_CACHE:
        return _PHASE_TABLE_CACHE[key]
    pr = day_price_schedule()
    n = DAY_MINUTES
    t = np.zeros((n, 3))
    for i in range(n):
        j = i
        while (j - i) < n - 1 and pr[(j + 1) % n] == pr[i]:
            j += 1
        k = i
        while (i - k) < n - 1 and pr[(k - 1) % n] == pr[i]:
            k -= 1
        steps = (j - i) + 1               # 本段剩余步数 (含当前步)
        seg_len = (j - k) + 1
        t[i] = [steps / seg_len, min(steps, 480) / 480.0, pr[(j + 1) % n] / 1000.0]
    _PHASE_TABLE_CACHE[key] = t
    return t


def phase_features(minute, mode=1, table=None):
    """相位特征向量; mode<=0 返回空数组。minute 为一天中的第几分钟 (0..1439)。"""
    if mode <= 0:
        return np.zeros(0)
    m = int(minute) % DAY_MINUTES
    ang = 2.0 * np.pi * m / DAY_MINUTES
    feats = [np.sin(ang), np.cos(ang)]
    if mode >= 2:
        t = phase_daily_table() if table is None else table
        feats.extend([t[m, 0], t[m, 1], t[m, 2]])
    return np.asarray(feats, dtype=np.float64)


def phase_minute_of_day(episode_steps):
    return int(episode_steps) % DAY_MINUTES


def Normalize_H(obs, minute=0, mode=1, table=None):
    """H 层观测归一化: 4 维基础观测 + 相位特征 (mode 0 时即原 Normalize)。"""
    base = Normalize(obs)
    if mode <= 0:
        return base
    return np.concatenate([base, phase_features(minute, mode, table)])


# 9-case 测试用: 把 case 的电价档映射到该档"代表性时刻"(取该段中点),
# 这样测试时的相位与训练同分布 (场景电价每 5 步就变, 直接用 wall-clock 相位会 OOD)。
REP_MINUTE = {'LOW': 210, 'MEDIUM': 840, 'HIGH': 540}


def rep_minute_for_price_level(level):
    """电价档 → 该档的"代表性时刻"(分钟)。同时接受档位名('LOW')与数值电价(300/580/980)。

    ⚠️ 历史 bug: 9-case 场景的 price_seq 是**数值**电价
    (test_scenario.build_9case_scenario 返回 int), 而早期实现只按档位名查表
    ⇒ str(300)='300' 查不到 ⇒ 45 步全部回退 210 (LOW)
    ⇒ MEDIUM/HIGH 的 case 用"凌晨 3:30 的相位"评估, 相位与电价自相矛盾 (OOD)。
    这里先按档位名查, 查不到再按数值电价分档 (get_price_level), 两种调用都正确。
    """
    key = str(level).strip().upper()
    if key not in REP_MINUTE:
        try:
            key = get_price_level(float(level))
        except (TypeError, ValueError):
            pass
    return REP_MINUTE.get(key, 210)


def nine_case_test(agent_H, agent_L, env_test, rbb_system,
                   scenario=None, test_csv=False, output=None, seed=None,
                   phase_mode=0, phase_table=None, viol_derate_coef=None):
    """
    PRRBB 45-step 9-case in-training test.

    Wind bands: LOW < 7.0, MEDIUM [7.0, 11.0), HIGH >= 11.0 m/s.
    Layout: wind LOW->MEDIUM->HIGH; within each wind segment price
    LOW->MEDIUM->HIGH; each case = 5 steps = one upper-level period.

    Returns
    -------
    case_metrics : list of dict (length 9)
        Per-case 5-step mean r1/c1/r2/c2/r/c plus violation count cvm;
        'profit' = the case's 5-step profit sum
        (profit_step = P_G*pr*dt - BESS_DEG_RMB_PER_MWH*max(-pbrule,0)*dt).
    overall : dict
        Mean over the 9 case means (cvm = total count);
        'profit' = the 45-step total profit sum.
    """
    if scenario is None:
        wind_seq, price_seq, case_ids, blocks = build_9case_scenario(seed=seed)
    else:
        wind_seq, price_seq, case_ids, blocks = scenario

    obs = env_test.ScenarioReset(wind_seq, price_seq)
    # 相位: 场景电价每 5 步就变一次, 直接用 wall-clock 相位会 OOD;
    # 这里按 case 的电价档映射到该档"代表性时刻"(段中点), 使测试与训练同分布,
    # 且跨 run 可比 (相同 case 相同相位)。phase_mode=0 时 Normalize_H 退化为 4 维。
    _pminute = {i: rep_minute_for_price_level(price_seq[i]) for i in range(len(price_seq))}
    norm_obs = Normalize_H(obs, _pminute[0], phase_mode, phase_table)
    time = 1.0 / 60
    agent_H.is_training = False
    agent_L.is_training = False
    step = 0
    num_iterations = len(wind_seq)
    n_per_case = num_iterations // len(blocks)

    per_case = {i: {'r1': 0.0, 'c1': 0.0, 'r2': 0.0, 'c2': 0.0,
                    'r': 0.0, 'c': 0.0, 'cvm': 0, 'profit': 0.0,
                    'soc_start': None, 'soc_end': None,
                    'turbine_profit': 0.0, 'derate_loss': 0.0, 'penalty_NN_H': 0.0}   # 存货结算用 (见 SOC_SETTLE_PR)
                for i in range(len(blocks))}
    df = pd.DataFrame()

    while step < num_iterations:
        if step % 5 == 0:
            action_H_raw = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H_raw)
            _derate = violation_derate(action_H_raw, viol_derate_coef)
            pw_tru = env_test.PowerOutput(obs[1], action_H)

        # L 层观测蒸馏 (5维): [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]
        _, rbb_info_obs = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=float(pw_tru),
            pg_prev=float(obs[-1]), E=float(obs[2]))
        norm_obs_L = build_L_obs_distilled(obs, pw_tru, action_H, rbb_info_obs, 5 - (step % 5))
        action_L = agent_L.select_action(norm_obs_L)
        action_L, penalty_NN_L = CheckAction(action_L)

        action = np.concatenate((action_H, action_L))

        pb_original = float(action[-1])
        ptur = float(pw_tru)
        pg_prev = float(obs[-1])
        E_current = float(obs[2])

        pbrule, rbb_info = rbb_system.compute_pbrule(
            pb_original=pb_original, ptur=ptur,
            pg_prev=pg_prev, E=E_current,
        )

        action_rbb = action.copy()
        action_rbb[-1] = pbrule

        pw_tru2, obs2, action_L_env, penalty_L, done, done_L, info = \
            env_test.ScenarioStep(action_rbb[:6], action_rbb[6:], pw_tru)

        action_rbb[6:] = action_L_env
        norm_obs2 = Normalize_H(obs2, _pminute[min(step + 1, num_iterations - 1)],
                                phase_mode, phase_table)

        r1, c1, r2, c2, details = Cal_Rew_PRRBB(
            obs, obs2, action_rbb, rbb_info,
            penalty_NN_H=penalty_NN_H, time=time,
            ptur=ptur, n_rem=5 - (step % 5),
            turbine_derate=_derate,
        )

        bi = step // 5
        pc = per_case[bi]
        if step % 5 == 0:
            pc['soc_start'] = float(obs[2])      # 本块决策时刻的储存电量
        pc['soc_end'] = float(obs2[2])           # 末次写入 = 本块末的储存电量
        pc['r1'] += r1
        # r1 分解三项 (审计: profit/N - r1 == (罚 + 乘性削减)/N)
        pc['turbine_profit'] += details['turbine_profit']
        pc['derate_loss'] += details['turbine_profit'] * (1.0 - _derate)
        pc['penalty_NN_H'] += penalty_NN_H
        pc['c1'] += c1
        pc['r2'] += r2
        pc['c2'] += c2
        pc['r'] += (r1 + r2)
        pc['c'] += c1
        if c1 > 0:
            pc['cvm'] += 1

        # profit: 上层利润 = P_G * pr * dt - BESS_DEG_RMB_PER_MWH*max(-pb,0)*dt（与训练 episode_profit 同口径）
        profit_step = obs2[-1] * obs[0] * time - BESS_DEG_RMB_PER_MWH * max(-float(pbrule), 0.0) * time
        pc['profit'] += profit_step

        if test_csv:
            df = pd.concat([df, pd.DataFrame([{
                'step': step,
                'case_wind': case_ids[step][0],
                'case_price': case_ids[step][1],
                'wind_level': details['wind_level'],
                'price_level': details['price_level'],
                'u': details['u'],
                'pr': details['pr'],
                'r1': r1, 'c1': c1, 'r2': r2, 'c2': c2,
                'r': r1 + r2, 'c': c1,
                'profit': profit_step,
                'pbrule': rbb_info['pbrule'],
                'clip_amount': rbb_info['clip_amount'],
                'action': action_rbb,
            }])], ignore_index=True)

        norm_obs = norm_obs2
        obs = obs2
        pw_tru = pw_tru2
        step += 1

    agent_H.is_training = True
    agent_L.is_training = True
    if test_csv and output:
        df.to_csv(f'{output}/test_s_prrbb_9case.csv', index=False)

    case_metrics = []
    for i, b in enumerate(blocks):
        pc = per_case[i]
        _dsoc = float(pc['soc_end']) - float(pc['soc_start'])
        case_metrics.append({
            'wind_level': b['wind_level'],
            'price_level': b['price_level'],
            'r1': pc['r1'] / n_per_case,
            'c1': pc['c1'] / n_per_case,
            'r2': pc['r2'] / n_per_case,
            'c2': pc['c2'] / n_per_case,
            'r': pc['r'] / n_per_case,
            'c': pc['c'] / n_per_case,
            'cvm': pc['cvm'],
            'profit': pc['profit'],  # 该 case 5 步 profit 之和（**现金流口径**）
            'turbine_profit': pc['turbine_profit'],
            'derate_loss': pc['derate_loss'],
            'penalty_NN_H': pc['penalty_NN_H'],
            # ---- 存货结算（口径修正，不改训练）----
            # profit 只记现金流；储存在电池里的电其成本已扣（充电时 P_G 减少、收入减少），
            # 价值却未认回 ⇒ ΔSOC>0（净充电）时 profit 低估、ΔSOC<0（净放电）时高估
            # （把期初存量卖掉记成了利润）。按外生账面价 SOC_SETTLE_PR 结算。
            'soc_start': float(pc['soc_start']),
            'soc_end': float(pc['soc_end']),
            'dsoc': _dsoc,
            'profit_econ': float(pc['profit']) + _dsoc * SOC_SETTLE_PR,
        })

    # 45 步总量口径（与下面 profit 是"9 个 case 之和"一致；逐 case 的 dsoc 望远镜求和
    # 恰等于首末 SOC 之差，故这里直接取首 case 的起点与末 case 的终点）
    _soc0 = float(case_metrics[0]['soc_start'])
    _socN = float(case_metrics[-1]['soc_end'])
    _profit_tot = float(sum(m['profit'] for m in case_metrics))
    overall = {
        'r1': float(np.mean([m['r1'] for m in case_metrics])),
        'c1': float(np.mean([m['c1'] for m in case_metrics])),
        'r2': float(np.mean([m['r2'] for m in case_metrics])),
        'c2': float(np.mean([m['c2'] for m in case_metrics])),
        'r': float(np.mean([m['r'] for m in case_metrics])),
        'c': float(np.mean([m['c'] for m in case_metrics])),
        'cvm': int(sum(m['cvm'] for m in case_metrics)),
        'profit': _profit_tot,  # 45 步 profit 之和（现金流口径）
        'turbine_profit': float(sum(m['turbine_profit'] for m in case_metrics)),
        'derate_loss': float(sum(m['derate_loss'] for m in case_metrics)),
        'penalty_NN_H': float(sum(m['penalty_NN_H'] for m in case_metrics)),
        'soc_start': _soc0,
        'soc_end': _socN,
        'dsoc': _socN - _soc0,
        'profit_econ': _profit_tot + (_socN - _soc0) * SOC_SETTLE_PR,
    }
    return case_metrics, overall
