"""
TD3LEPRRLRB — Piecewise Reward + Rule-Based BESS + LLM 分段微调
==============================================================
= td3prrlrb (Piecewise Reward RL + 规则 BESS 下层) + PRLE 的 LLM 元控制器层。

与 td3prrlrb 的差异: ① 风速分段阈值改为可变模块状态 (LLM 在线微调);
② P0.2 罚项不封顶; ③ P0.3 新增 24h 电价相位特征。

Upper layer (H, RL):
  r1 = wT*(pr*ptur*time) + wB*(-pr*pbrule*time)
       - 490*max(-pbrule,0)*time            (电池退化, 由命令方 H 付费)
       - penalty_NN_H                        (【P0.2】不封顶)
       wT = wT_u(u) × wT_p(pr)
       wB ≡ 1.0  【本轮变更】BESS 收益项分段系数已去掉
                 (BESS 由 SOC + RuleBasedBESS 硬窗口调控, 基本不存在安全问题)

  c1 = 1.0 * [ 10*(P_F-3)+ + 0.1*P_F ]                (波动越限 + 平滑)
       + W_GAP * unfeasible_gap                       (平抑可行性缺口)
       + [块末结算] gap_penalty                        (SOC 目标可行性罚)
       【本轮变更】波动风险项分段系数已去掉 ⇒ 系数恒 1.0
       (风力变化本身已证明控制完成)

  a6 = SOC 目标信号 (-0.5 ~ 0.5)  →  soc_target = 3 + 5*a6 ∈ [0.5, 5.5]
       块级(每 5 步)固定; 上层为"不可行的目标"付费。

  H 观测 = Normalize(obs) ⊕ 相位特征 (见 §8b / --phase_feat): 4 / 6 / 9 维

Lower layer (L, 规则): SocTracker (见 td3leprrlrb_rule_bess.py)
  在 RBB 可行窗口内以最大能力跟踪 soc_target:
    * 物理可行 + 平抑波动 = 窗口硬保证 → clip_amount ≡ 0
    * **无跟踪死区** (tracker soc_tol 默认 0): 残差 0.01 MWh 也追;
      δ=0 时只做窗口允许的最小被迫吞吐。
  因此下层无需 RL; 块末残差 |soc_target - SOC| 近似等于"不可行量"。

【死区归属】0.1 MWh 的豁免**只属于上层罚项** (`SOC_TOL`, 见 gap_penalty):
  H 只对超过 0.1 MWh 的目标残差付费, 下层则尽力追到 0 —— 不对称方向安全
  (L 能消的残差 ⊇ H 豁免的), 不会产生幽灵税。

【阈值依赖面】本轮变更后, 风速档 wind_level(u) 只剩**一处**进入奖励:
  (a) 收益权重 wT = WT_WIND[wind]×WT_PRICE[price]
已取消的两处: wB (原经 SOC_SAFE_RANGES[wind]) 与 c1 的 C1_RISK[wind]
  ⇒ wB ≡ 1.0, c1 的波动项系数 ≡ 1.0 (见 §2 的 WB_PRICE / WB_SOC_OUTSIDE / C1_RISK)。
因此阈值一动, 只有 wT 那一项会让历史 r1 的口径变化 → relabel 仍然必要
(本模块提供两个 relabel_*_substep 辅助函数, 只重算阈值相关的乘法因子)。

Constants
---------
PRICE 分档: ≥980 HIGH, ≤300 LOW, 其余 MEDIUM
WIND  分档: 动态 —— 初值 7.0/11.0, 由 --seg 覆盖, 之后 LLM 在 ±1 m/s 内微调
相位  分档: REP_MINUTE=1440; 日程 0:00-7:00 & 23:00-24:00=300, 11:00-17:00 &
           21:00-23:00=580, 其余 980 (与 env.CalPrice 的判档一致)
"""

import numpy as np
import pandas as pd

from util import *


# =============================================================================
# 0. Dynamic wind thresholds (mutable module-level state)
# =============================================================================
# 模块默认值 7.0/11.0 (与 td3prrlrb 写死的常量一致); main 启动时按 --seg
# (percentile / kmeans / expert) 用 set_wind_thresholds() 设定起点, 之后由
# LLM 元控制器每 N 步在【原设定 ±1 m/s】内在线微调。

WIND_LOW_MAX  = 7.0   # u < 7.0   → LOW wind
WIND_HIGH_MIN = 11.0  # u >= 11.0 → HIGH wind, else MEDIUM

_wind_low_max  = WIND_LOW_MAX
_wind_high_min = WIND_HIGH_MIN


def get_wind_thresholds():
    """Return the current (wind_low_max, wind_high_min) in m/s."""
    return _wind_low_max, _wind_high_min


def set_wind_thresholds(low_max, high_min):
    """Set the wind thresholds with physical validity clamping.

    Constraints enforced (与 td3prrbb/td3prle 一致):
      - low_max  ∈ [3.0, high_min - 1.0]
      - high_min ∈ [low_max + 1.0, 22.0]
      - always low_max < high_min (至少 1.0 m/s 间隔)

    Returns
    -------
    (actual_low_max, actual_high_min) : (float, float)
        Values after clamping (may differ from the request).
    """
    global _wind_low_max, _wind_high_min
    low_max = float(np.clip(low_max, 3.0, high_min - 1.0))
    high_min = float(np.clip(high_min, low_max + 1.0, 22.0))
    _wind_low_max = low_max
    _wind_high_min = high_min
    return _wind_low_max, _wind_high_min


# =============================================================================
# 1. Price / Wind classification
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


def get_wind_level(u):
    """Classify wind speed into LOW / MEDIUM / HIGH using CURRENT thresholds."""
    if u < _wind_low_max:
        return 'LOW'
    elif u >= _wind_high_min:
        return 'HIGH'
    else:
        return 'MEDIUM'


# =============================================================================
# 2. Segmented weight lookup (multiplicative decomposition)
# =============================================================================
# 机会侧（利润）与风险侧（c1）分离: wT = WT_WIND(u)×WT_PRICE(pr), wB ≡ 1.0
# 【本轮变更】BESS 收益项分段系数彻底去掉: WB_PRICE 三档均 1.0 且
# WB_SOC_OUTSIDE = 1.0 ⇒ wB ≡ 1.0（不再按电价档/风档/SOC 位置加权）。

WT_WIND = {'LOW': 1.2, 'MEDIUM': 1.0, 'HIGH': 0.9}
WT_PRICE = {'LOW': 1.0, 'MEDIUM': 1.0, 'HIGH': 1.05}
# BESS 收益：分段系数已取消 ⇒ wB ≡ 1.0（原 LOW=0.8 / MEDIUM=1.0 / HIGH=1.2）
#   理由: BESS 由 SOC + RuleBasedBESS 硬窗口调控, 基本不存在安全问题。
WB_PRICE = {'LOW': 1.0, 'MEDIUM': 1.0, 'HIGH': 1.0}
# BESS 收益: SOC 贴边折减（原 0.7 → 1.0）⇒ get_wB 恒返回 1.0
#   注意: relabel_r1_substep 仍按位置接收 SOC 并缓存决策时刻 SOC ——
#   即使当前 wB 与 SOC 无关, 也不要删掉该字段 (恢复旧权重时 relabel 需要它)。
WB_SOC_OUTSIDE = 1.0

# c1 风险系数：分段系数已取消 ⇒ 恒 1.0（原 LOW=0.8 / MEDIUM=1.0 / HIGH=1.2）
#   理由: 风力变化本身已证明控制完成, 无需再按风档加/减波动风险罚。
C1_RISK = {'LOW': 1.0, 'MEDIUM': 1.0, 'HIGH': 1.0}
# unfeasible_gap（平抑可行性缺口, MW）进入 c1 的系数
W_GAP = 10.0

# 【上层】SOC 目标 gap 罚的豁免带 (MWh): 0.1 MWh = 6 MW × 1 min
#   —— 这是"H 不必为零头付费"的记账约定, 只作用于上层罚项;
#      下层的 SocTracker 默认没有死区 (尽力追随), 因此不会出现
#      "规则停住、上层却照罚"的幽灵税 (不对称方向是安全的)。
#      仅当把 --tracker_dead_zone 调成 > 本值时, 幽灵税才会回归。
SOC_TOL = 0.1
# SOC 目标映射: a6 ∈ [-0.5, 0.5] → soc_target ∈ [0.5, 5.5]
SOC_TARGET_CENTER = 3.0
SOC_TARGET_SPAN = 5.0
SOC_TARGET_MIN, SOC_TARGET_MAX = 0.5, 5.5

# 退化系数 (元/MWh): 放电侧; 充电侧可以取更小 (k_c = 0 表示只在放电计退化)
DEG_COST_DISCHARGE = 490.0
DEG_COST_CHARGE = 0.0


def get_wT(wind_level, price_level):
    """Turbine revenue weight: multiplicative wind × price decomposition."""
    return WT_WIND[wind_level] * WT_PRICE[price_level]


def get_wB(price_level, wind_level, SOC):
    """BESS revenue weight —— 【本轮变更】恒为 1.0（分段系数已去掉）。

    WB_PRICE 三档均 1.0 且 WB_SOC_OUTSIDE = 1.0 ⇒ 两条分支都返回 1.0,
    r1 的 BESS 项退化为物理净收益 -pr*pbrule*time。
    历史机制保留在下方代码里: 把 WB_PRICE / WB_SOC_OUTSIDE 改回原值即一键恢复。
    wind_level / SOC 参数继续保留以保证调用签名不变 (relabel 也按位置传参)。
    """
    soc_min, soc_max = SOC_SAFE_RANGES[wind_level]
    w_soc = 1.0 if soc_min <= SOC <= soc_max else WB_SOC_OUTSIDE
    return WB_PRICE[price_level] * w_soc


# =============================================================================
# 3. SOC 安全带 (仅用于日志指标; 不参与任何成本)
# =============================================================================
# 注意: SOC 贴边折减取消后, 这条安全带【只】用于 Cal_soc_deviation 日志指标。
# 唯一的例外是 get_wB: 它仍按安全带分支取值, 但两分支在 WB_SOC_OUTSIDE=1.0 下
# 等价 —— 保留该分支是为了"改回 0.7"的一键回退。SOC 的硬约束由 RBB 可行窗口
# (min_E=0 / max_E=6) 保证, 软约束由 H 的块末 gap 罚承担。
SOC_SAFE_RANGES = {
    'LOW':    (0.5, 5.5),
    'MEDIUM': (1.0, 5.0),
    'HIGH':   (1.5, 4.5),
}


def Cal_soc_deviation(SOC, wind_level='MEDIUM'):
    """Wind-segmented SOC deviation outside the safe band (log metric only)."""
    soc_min, soc_max = SOC_SAFE_RANGES[wind_level]
    dev = 0.0
    if SOC < soc_min:
        dev = soc_min - SOC
    elif SOC > soc_max:
        dev = SOC - soc_max
    return dev


# =============================================================================
# 4. Upper-level r1
# =============================================================================

def Cal_r1_PRRBB(pr, ptur, pbrule, penalty_NN_H, wind_level, price_level,
                 SOC=3.0, time=1.0 / 60):
    """
    r1 = wT * (pr * ptur * time)             ← 风机收益
       + wB * (-pr * pbrule * time)          ← BESS 收益 (正=pbrule<0 放电)
       - 490 * max(-pbrule, 0) * time        ← 退化
       - penalty_NN_H
    """
    w_wind = get_wT(wind_level, price_level)
    w_bess = get_wB(price_level, wind_level, SOC)

    turbine_profit = w_wind * pr * ptur * time
    bess_profit    = w_bess * (-pr * pbrule * time)
    degradation    = (DEG_COST_DISCHARGE * max(-pbrule, 0.0)
                      + DEG_COST_CHARGE * max(pbrule, 0.0)) * time

    return turbine_profit + bess_profit - degradation - penalty_NN_H


# =============================================================================
# 4b. Reward / cost relabeling helpers (LLM 阈值变动后重算历史转移)
# =============================================================================
# 阈值改动只影响两处乘法因子:
#   (a) 收益权重 wT = WT_WIND[wind(u)] × WT_PRICE[price]
#                 wB = WB_PRICE[price] × w_soc   (w_soc≡1.0, 见 WB_SOC_OUTSIDE)
#   (b) 风险系数恒 1.0 —— 【本轮变更】原 C1_RISK[wind(u)] 分段系数已去掉
# 其余项 (pr·ptur·dt, −pr·pbrule·dt, 退化, penalty_NN_H, P_F 项, W_GAP·gap)
# 都与阈值无关, 因此缓存"与阈值无关的原料"(raw) + 少数上下文标量即可精确重算。
#
# 【关于 SOC】折减取消后 wB 已不依赖 SOC, 但 relabel 仍然接收并使用缓存的
# SOC: 一是保持与 td3prle 的 relabel 签名一致; 二是让"把 WB_SOC_OUTSIDE 改回
# 0.7"这条一键回退路径依然精确 (回退后 wB 重新依赖 SOC 与 wind 档)。因此
# context 里的 'SOC' 字段必须保留。

def relabel_r1_substep(u, pr, turbine_raw, bess_raw, degradation, penalty_NN_H,
                       SOC=3.0):
    """用【当前】阈值重算单个子步的 r1.

    Parameters
    ----------
    u : float                     该子步风速 (m/s)
    pr : float                    该子步电价 (RMB/MWh)
    turbine_raw : float           pr * ptur * dt        (与阈值无关)
    bess_raw : float              -pr * pbrule * dt     (与阈值无关)
    degradation : float           退化成本             (与阈值无关)
    penalty_NN_H : float          上层动作越界罚        (与阈值无关)
    SOC : float                   决策时刻 SOC (MWh). 当前 WB_SOC_OUTSIDE = 1.0,
                                  该参数【不影响结果】(wB 只由电价档决定);
                                  保留它是为了兼容 td3prle 的调用签名, 以及让
                                  恢复 0.7 折减后 relabel 仍然精确 (那时 wB 会
                                  重新依赖 SOC 落在哪一档安全带上)。

    Returns
    -------
    r1 : float
    """
    wind_level = get_wind_level(u)
    price_level = get_price_level(pr)
    w_wind = get_wT(wind_level, price_level)
    w_bess = get_wB(price_level, wind_level, SOC)
    return w_wind * turbine_raw + w_bess * bess_raw - degradation - penalty_NN_H


def relabel_c1_substep(u, P_F, unfeasible_gap=0.0, gap_pen=0.0):
    """用【当前】阈值重算单个子步的 c1.

    c1 = 1.0 · [10·(P_F−3)⁺ + 0.1·P_F]   (【本轮变更】C1_RISK 分段系数已去掉)
         + W_GAP·unfeasible_gap     (与阈值无关, 原样带回)
         + gap_pen                  (块末 SOC 目标罚, 与阈值无关; 仅块内最后
                                     一个子步非零 —— 见 main 的 context 记账)

    parameters 中 unfeasible_gap / gap_pen 都从存档 context 原样读回, 保证
    relabel 前后成本口径逐项一致 (td3prle 曾因漏存块末跟踪项而在 relabel 时
    静默丢掉该项, 本项目已修正)。
    """
    coeff = C1_RISK[get_wind_level(u)]
    if P_F > 3.0 + 1e-4:
        base = coeff * (10.0 * (P_F - 3.0) + 0.1 * P_F)
    else:
        base = coeff * (0.1 * P_F)
    return base + W_GAP * unfeasible_gap + float(gap_pen)


# =============================================================================
# 5. SOC 目标映射 与 上层可行性(gap)惩罚
# =============================================================================

def soc_target_from_signal(a6):
    """上层信号 → SOC 目标: 3 + 5*a6, 并夹在 [0.5, 5.5] (合法目标带)."""
    return float(np.clip(SOC_TARGET_CENTER + SOC_TARGET_SPAN * float(a6),
                         SOC_TARGET_MIN, SOC_TARGET_MAX))


def gap_penalty(soc_target, SOC_end, tol=SOC_TOL):
    """上层可行性罚 (块末结算): 10 * max(0, |soc_target - SOC_end| - tol).

    由于下层规则是"最大能力跟踪 + 无死区", 块末残差 ≈ 不可行量;
    tol 是**上层自己的记账豁免带** (--soc_tol, 默认 0.1): 残差不超过它就不收费,
    与下层跟踪死区无关 (下层默认 0, 尽力追到零)。
    ⚠ 只罚真正的不可行部分 —— 若残差里混入了"风阶跃迫使的吞吐"造成的过冲,
      那部分也在此被收费, 属口径的已知不精确处 (见 SocTracker 文档)。
    """
    return W_GAP * max(0.0, abs(float(soc_target) - float(SOC_end)) - tol)


# =============================================================================
# 6. Combined reward (H 层)
# =============================================================================

def Cal_Rew_RLRB(observation, observation2, action_rbb, rbb_info,
                 penalty_NN_H=0.0, time=1.0 / 60, ptur=None,
                 pb_cmd=None):
    """
    TD3PRRLRB 上层奖励/成本.

    Parameters
    ----------
    observation  : [pr, u, E, P_G_last]
    observation2 : [pr', u', E', P_G]
    action_rbb   : [a1..a5, soc_signal, pbrule]
    rbb_info     : dict from RuleBasedBESS.compute_pbrule()
    penalty_NN_H : 上层动作越界罚
    ptur         : 风机功率 (MW)
    pb_cmd       : 下层规则输出的功率指令 (MW) —— 用于记录额定裁剪 (规则下应为 0)

    Returns
    -------
    r1, c1_step, details      # 注意: 不含块末 gap 罚 (由 main 在块末累加)
    """
    if ptur is None:
        raise ValueError("Cal_Rew_RLRB: ptur must be provided")

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
    soc_target = soc_target_from_signal(soc_signal)

    wind_level = get_wind_level(u)
    price_level = get_price_level(pr_old)
    SOC_decision = float(observation[2])

    # ==== r1 (H 机会侧) ====
    r1 = Cal_r1_PRRBB(pr_old, ptur, pbrule, penalty_NN_H,
                      wind_level, price_level, SOC=SOC_decision, time=time)

    # ==== c1 (H 风险侧) ====
    c1_risk_coeff = C1_RISK[wind_level]
    if P_F > 3.0 + 1e-4:
        c1 = c1_risk_coeff * (10.0 * (P_F - 3.0) + 0.1 * P_F)
    else:
        c1 = c1_risk_coeff * (0.1 * P_F)
    c1 = c1 + W_GAP * unfeasible_gap

    # ==== details ====
    w_wind = get_wT(wind_level, price_level)
    w_bess = get_wB(price_level, wind_level, SOC_decision)
    turbine_profit = w_wind * pr_old * ptur * time
    bess_profit = w_bess * (-pr_old * pbrule * time)
    soc_dev = Cal_soc_deviation(SOC_new, wind_level)

    details = {
        'wind_level': wind_level,
        'price_level': price_level,
        'w_wind': w_wind,
        'w_bess': w_bess,
        'pr': pr_old,
        'u': u,
        'turbine_profit': turbine_profit,
        'bess_profit': bess_profit,
        'revenue': turbine_profit + bess_profit,
        'degradation': (DEG_COST_DISCHARGE * max(-pbrule, 0.0)
                        + DEG_COST_CHARGE * max(pbrule, 0.0)) * time,
        'soc_signal': soc_signal,
        'soc_target': soc_target,
        'soc_delta': SOC_new - soc_target,
        'soc_tol': SOC_TOL,
        # 决策时刻 SOC: 目前 wB 已不依赖它 (贴边折减取消), 但仍随 context 缓存,
        # 以便把 WB_SOC_OUTSIDE 改回 0.7 时 relabel 依然精确
        'SOC_decision': SOC_decision,
        'soc_deviation': soc_dev,
        'ptur': ptur,
        'pbrule': pbrule,
        'pb_cmd': float(pb_cmd) if pb_cmd is not None else np.nan,
        # 规则下层自身已落在窗口内 ⇒ 该差值 = RBB 兜底接管量 (仅不可行步 >0)
        'pb_rbb_clip': (abs(float(pb_cmd) - pbrule) if pb_cmd is not None
                        else 0.0),
        'P_F': P_F,
        'SOC': SOC_new,
        'unfeasible_gap': unfeasible_gap,
        'clip_amount': clip_amount,
        'is_infeasible': rbb_info.get('is_infeasible', False),
        'inevitable_violation': rbb_info.get('inevitable_violation', 0.0),
        'penalty_NN_H': penalty_NN_H,
        'c1_fluct': c1,
    }

    return r1, c1, details


# =============================================================================
# 7. Action checking
# =============================================================================

def CheckAction(action, charge=10.):
    """Clip actions to feasible ranges. WT: [0,0.5], Signal: [-0.5,0.5], Bat: [-charge,charge].

    上层 (6 维): WT 越界与信号越界按 100/MW 计罚, penalty_NN_H **不封顶**。
    下层 (1 维): 规则控制器已保证落在 RBB 窗口内, 这里只做额定功率安全限幅。

    【P0.2 变更】原实现有 `penalty = min(penalty, 50.0)` 的总封顶, 已删除。
      理由 (run159/160 实测): 封顶后"越界区"在【收益通道 (r1 被 clip 成平的)】与
      【成本通道 (c1 被 min 封成平的)】两侧梯度同时为 0 ⇒ 越界维度失去任何恢复力,
      被共享隐层顺带拖成随机游走: run160 实测 a2/a3/a4 漂到 −3.9/−4.7/−6.4 (三台机
      永久停机, ΣCp 仅 30.8%), 而真实越界罚 1600~2400 只被记成 50。
      删掉封顶后, 每单位越界恒定产生 100 的恢复梯度, 任意深度都不衰减 (实测同一动作族
      在不同越界深度下罚值 1590 → 1730 → 2130)。

    代价: 越界期 r1/c1 会出现千级尖峰 → λ1 会短暂顶格、PER 会优先重放这些样本
      (见 main 的 --PER_alpha 说明); 这是"恢复力换稳定性"的自觉取舍。
    """
    penalty = 0
    SIGNAL_MIN, SIGNAL_MAX = -0.5, 0.5

    if len(action) == 6:
        for j in action[:5]:
            ma, mi = max(0.5, j), min(0, j)
            penalty += 100 * ((ma - 0.5) + (0. - mi))
        sig = action[5]
        if sig > SIGNAL_MAX:
            penalty += 100 * (sig - SIGNAL_MAX)
        elif sig < SIGNAL_MIN:
            penalty += 100 * (SIGNAL_MIN - sig)
        applied = np.append(np.clip(action[:5], 0., 0.5),
                            np.clip(action[5], SIGNAL_MIN, SIGNAL_MAX))
        # 【P0.2】不封顶 —— 保留完整的"拉回界内"梯度
        return applied, penalty

    elif len(action) == 5:
        for j in action[:5]:
            ma, mi = max(0.5, j), min(0, j)
            penalty += 200 * ((ma - 0.5) + (0. - mi))
        return np.clip(action[:5], 0., 0.5), penalty

    elif len(action) == 1:
        for j in action:
            ma, mi = max(charge, j), min(-charge, j)
            penalty += 200 * ((ma - charge) + (-charge - mi))
        return np.clip(action, -charge, charge), penalty

    return action, penalty


# =============================================================================
# 8. Normalization
# =============================================================================

def Normalize(obs, max_E=6.):
    return np.array([obs[0] / 1000, obs[1] / 10, obs[2] / 10, obs[3] / 10])


def Original(obs, max_E=6.):
    return np.array([obs[0] * 1000, obs[1] * 10, obs[2] * 10, obs[3] * 10])


# =============================================================================
# 8b. 24h 电价相位特征 (P0.3)
# =============================================================================
# 动机: 套利里的"方向/持留"只需当前电价即可判断 (run159 的"持留—倾泻"就是相位盲学
# 出来的), 但"择时"—— 在低价窗的最后一刻把电充满 —— 必须知道"本段还剩多久、下一段
# 涨跌"。算术: 下层规则速率 ≈0.245 MWh/块 ⇒ 充满 5 MWh 需 1.7 h, 而低价窗长达 7~8 h。
# 相位盲的"见 300 就充"要提前 ~7 h 决策, 折算贴现需 γ ≥ 0.988; 而"最后 1.7 h 才充"
# 只需 γ ≥ 0.951。给出相位 = 把"择时"从不可学变成可学。
#
# 三个可选档位 (--phase_feat):
#   0: 无相位 (4 维, 复现旧 run)
#   1: [sin(2π·m/1440), cos(2π·m/1440)]                        (+2 → 6 维, 默认)
#   2: 1 的全部 + [段内剩余比例, 剩余小时/8, 下一段电价/1000]      (+3 → 9 维)
#
# 为什么是 sin/cos 而不是 1 维锯齿: 锯齿跨零点不连续, 同结构网络拟合 MSE 0.047 vs
# 0.0008 (差 60 倍), 判对率 82.5% vs 99.7%。

REP_MINUTE = 1440          # 一天 1440 分钟 = episode 长度 = 相位周期

# 日内电价档位 (与 env.WindBattery.CalPrice 的判档规则完全一致):
#   0:00-7:00 与 23:00-24:00 → 300 (LOW)
#   11:00-17:00 与 21:00-23:00 → 580 (MEDIUM)
#   其余                      → 980 (HIGH)
_PRICE_SEGMENTS = (        # (起始分钟, 结束分钟[不含], 电价)
    (0, 420, 300), (420, 660, 980), (660, 1020, 580),
    (1020, 1260, 980), (1260, 1380, 580), (1380, 1440, 300),
)

# 各价格档的"代表相位"分钟, 供 9-case 使用 (避免把 case 的价格档与相位档配错 → OOD)
PHASE_REP_MINUTE = {300: 210, 580: 840, 980: 540}


def day_price_schedule():
    """日内电价日程表: 长度 1440, index = 当日第几分钟 (人类时钟, 0:00 起)."""
    sched = [0] * REP_MINUTE
    for a, b, p in _PRICE_SEGMENTS:
        for m in range(a, b):
            sched[m] = p
    return sched


def _cyclic_price_segments():
    """把档位按【循环】合并成段 (相邻同价合并, 跨零点也合并).

    因此 23:00-24:00 与 0:00-7:00 两段 (同为 300) 合并成一段 8 h,
    尾段区间可超过 1440 (表示跨零点)。
    """
    merged = []
    for a, b, p in _PRICE_SEGMENTS:
        if merged and merged[-1][2] == p:
            merged[-1][1] = b
        else:
            merged.append([a, b, p])
    if len(merged) > 1 and merged[0][2] == merged[-1][2]:
        first = merged.pop(0)
        merged[-1][1] += (first[1] - first[0])      # 尾段延伸跨过零点
    return [tuple(s) for s in merged]


def phase_daily_table():
    """对每个分钟 m 给出 [段内剩余比例, 剩余小时/8, 下一段电价/1000]."""
    segs = _cyclic_price_segments()
    table = []
    for m in range(REP_MINUTE):
        cur_i = None
        for i, (s, e, p) in enumerate(segs):
            if s <= m < e:
                cur_i = i
                break
        if cur_i is None:
            # m 落在"跨零点的尾段"里 (尾段区间 > 1440), 换算到同一坐标系
            cur_i = len(segs) - 1
            s, e, p = segs[cur_i]
            s, e = s - REP_MINUTE, e - REP_MINUTE
        else:
            s, e, p = segs[cur_i]
        total = float(e - s)
        remain = float(e - m)
        nxt = segs[(cur_i + 1) % len(segs)][2]
        table.append([remain / total, remain / 60.0 / 8.0, nxt / 1000.0])
    return table


_PHASE_DAILY_TABLE = phase_daily_table()      # 启动时算一次 (1440×3, 纯查表)

PHASE_DIM = {0: 0, 1: 2, 2: 5}


def phase_dim(phase_feat):
    """相位特征维数 (0→0, 1→2, 2→5)."""
    return PHASE_DIM[int(phase_feat)]


def phase_features(phase_minute, mode=1):
    """给定当日分钟, 返回相位特征向量 (长度为 phase_dim(mode))."""
    mode = int(mode)
    if mode <= 0:
        return np.zeros(0, dtype=np.float64)
    m = int(phase_minute) % REP_MINUTE
    ang = 2.0 * np.pi * m / REP_MINUTE
    feats = [np.sin(ang), np.cos(ang)]
    if mode >= 2:
        feats.extend(_PHASE_DAILY_TABLE[m])
    return np.asarray(feats, dtype=np.float64)


def phase_log_fields(phase_minute):
    """相位诊断用字段 (与 --phase_feat 无关, 永远可算), 供 CSV/TB 记录."""
    m = int(phase_minute) % REP_MINUTE
    ang = 2.0 * np.pi * m / REP_MINUTE
    frac, hours8, nxt = _PHASE_DAILY_TABLE[m]
    return {
        'phase_minute': m,
        'phase_sin': float(np.sin(ang)),
        'phase_cos': float(np.cos(ang)),
        'seg_remain_frac': float(frac),
        'seg_remain_hours': float(hours8) * 8.0,
        'next_price': float(nxt) * 1000.0,
    }


def representative_phase_minute(price):
    """把某个价格档映射到该档内的代表相位分钟 (9-case 用)."""
    return PHASE_REP_MINUTE.get(int(round(float(price))), 210)


def Normalize_H(obs, phase_minute=0, phase_feat=1, max_E=6.):
    """H 层观测 = Normalize(obs) ⊕ 相位特征 (phase_feat=0 时退化为原来的 4 维)."""
    base = Normalize(obs, max_E=max_E)
    if int(phase_feat) <= 0:
        return base
    return np.concatenate((base, phase_features(phase_minute, phase_feat)))


# =============================================================================
# 9. 9-case 测试 (下层为规则控制器)
# =============================================================================

try:
    from test_scenario import build_9case_scenario
except ImportError:
    import sys as _sys, os as _os
    _HSRL_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _HSRL_ROOT not in _sys.path:
        _sys.path.insert(0, _HSRL_ROOT)
    from test_scenario import build_9case_scenario


def nine_case_test(agent_H, tracker, env_test, rbb_system,
                   scenario=None, test_csv=False, output=None, seed=None,
                   phase_feat=0, soc_tol_upper=SOC_TOL):
    """
    TD3LEPRRLRB 45-step 9-case in-training test (下层 = 规则控制器).

    Layout: wind LOW->MEDIUM->HIGH; within each wind segment price
    LOW->MEDIUM->HIGH; each case = 5 steps = one upper-level period.

    【动态阈值】场景本身按【固定设计档】(LOW<7.0 / MEDIUM[7.0,11.0) / HIGH≥11.0)
    生成, 但奖励用【当前】动态阈值计算 —— 于是每个 case 额外记录
    'project_wind_levels'(用当前阈值对场景风速重新分档的结果), 阈值漂移一目了然。

    【相位】phase_feat > 0 时 H 观测用 Normalize_H(): 每个 case 的价格档映射到该档
    内的【代表相位】(300→210 = 03:30, 580→840 = 14:00, 980→540 = 09:00),
    避免"价格档 = LOW 却给 14:00 相位"这种训练中不会出现的 OOD 组合。

    Returns
    -------
    case_metrics : list of dict (length 9)  —— 每 case 的 5 步均值:
        r1 / c1(步级+块末gap) / gap_pen / cvm(c1_step>0 的步数) /
        pf_viol(P_F > 3 的步数, 规则下层下应≈0, 仅物理不可行时>0) /
        profit(5 步利润之和) / soc_track_err(块末 |target-SOC|) /
        project_wind_levels(该 case 内出现过的当前风速档) / phase_minute(代表相位)
    overall : dict —— 9 个 case 的均值 (cvm 为总计; profit 为 45 步总和),
        并附带当前 wind_low_max / wind_high_min
    """
    if scenario is None:
        wind_seq, price_seq, case_ids, blocks = build_9case_scenario(seed=seed)
    else:
        wind_seq, price_seq, case_ids, blocks = scenario

    obs = env_test.ScenarioReset(wind_seq, price_seq)
    phase_feat = int(phase_feat)
    norm_obs = Normalize_H(obs, representative_phase_minute(obs[0]), phase_feat)
    time = 1.0 / 60
    agent_H.is_training = False
    step = 0
    num_iterations = len(wind_seq)
    n_per_case = num_iterations // len(blocks)

    per_case = {i: {'r1': 0.0, 'c1': 0.0, 'cvm': 0, 'pf_viol': 0, 'profit': 0.0,
                    'gap_pen': 0.0, 'soc_track_err': 0.0, 'project_winds': set(),
                    'phase_minute': 0}
                for i in range(len(blocks))}
    df = pd.DataFrame()
    soc_target = SOC_TARGET_CENTER

    while step < num_iterations:
        if step % 5 == 0:
            action_H = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H)
            pw_tru = env_test.PowerOutput(obs[1], action_H)
            soc_target = soc_target_from_signal(action_H[-1])

        # ---- 下层: 规则控制器 (RBB 窗口内最大能力跟踪) ----
        _, rbb_info_obs = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=float(pw_tru),
            pg_prev=float(obs[-1]), E=float(obs[2]))
        pb_cmd = tracker.act(float(obs[2]), soc_target,
                             rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'])
        action_L = np.array([pb_cmd])

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
        # 相位: 由【该观测自身的价格档】映射到该档内的代表相位 (9-case 防 OOD)
        norm_obs2 = Normalize_H(obs2, representative_phase_minute(obs2[0]),
                                phase_feat)

        r1, c1_step, details = Cal_Rew_RLRB(
            obs, obs2, action_rbb, rbb_info,
            penalty_NN_H=penalty_NN_H, time=time,
            ptur=ptur, pb_cmd=pb_cmd,
        )

        # 块末: SOC 目标可行性罚 (豁免带 = 上层 --soc_tol, 与下层跟踪死区解耦)
        gap_pen = 0.0
        if step % 5 == 4:
            gap_pen = gap_penalty(soc_target, float(obs2[2]),
                                  tol=soc_tol_upper)
        c1 = c1_step + gap_pen

        bi = step // 5
        pc = per_case[bi]
        pc['r1'] += r1
        pc['c1'] += c1
        pc['gap_pen'] += gap_pen
        if c1_step > 0:
            pc['cvm'] += 1
        if details['P_F'] > 3.0 + 1e-4:
            pc['pf_viol'] += 1
        if step % 5 == 4:
            pc['soc_track_err'] = abs(soc_target - float(obs2[2]))
        # 当前（动态）阈值下该步实际落在哪一档 → 用于观察阈值漂移
        pc['project_winds'].add(details['wind_level'])
        pc['phase_minute'] = representative_phase_minute(obs[0])

        # profit: P_G * pr * dt - 490*max(-pb,0)*dt (与训练 episode_profit 同口径)
        profit_step = obs2[-1] * obs[0] * time - 490.0 * max(-float(pbrule), 0.0) * time
        pc['profit'] += profit_step

        if test_csv:
            df = pd.concat([df, pd.DataFrame([{
                'step': step,
                'case_wind': case_ids[step][0],
                'case_price': case_ids[step][1],
                'wind_level': details['wind_level'],
                'price_level': details['price_level'],
                'u': details['u'], 'pr': details['pr'],
                'r1': r1, 'c1_step': c1_step, 'gap_pen': gap_pen, 'c1': c1,
                'profit': profit_step,
                'soc_signal': details['soc_signal'],
                'soc_target': details['soc_target'],
                'soc_delta': details['soc_delta'],
                'SOC': details['SOC'],
                'pb_cmd': pb_cmd, 'pbrule': rbb_info['pbrule'],
                'clip_amount': rbb_info['clip_amount'],
                'P_F': details['P_F'],
                'action': action_rbb,
            }])], ignore_index=True)

        norm_obs = norm_obs2
        obs = obs2
        pw_tru = pw_tru2
        step += 1

    agent_H.is_training = True
    if test_csv and output:
        df.to_csv(f'{output}/test_s_rlrb_9case.csv', index=False)

    wind_low_max, wind_high_min = get_wind_thresholds()
    case_metrics = []
    for i, b in enumerate(blocks):
        pc = per_case[i]
        case_metrics.append({
            'wind_level': b['wind_level'],
            'price_level': b['price_level'],
            'r1': pc['r1'] / n_per_case,
            'c1': pc['c1'] / n_per_case,
            'gap_pen': pc['gap_pen'] / n_per_case,
            'cvm': pc['cvm'],
            'pf_viol': pc['pf_viol'],
            'profit': pc['profit'],
            'soc_track_err': pc['soc_track_err'],
            'project_wind_levels': sorted(pc['project_winds']),
            'phase_minute': int(pc['phase_minute']),
        })

    overall = {
        'r1': float(np.mean([m['r1'] for m in case_metrics])),
        'c1': float(np.mean([m['c1'] for m in case_metrics])),
        'gap_pen': float(np.mean([m['gap_pen'] for m in case_metrics])),
        'cvm': int(sum(m['cvm'] for m in case_metrics)),
        'pf_viol': int(sum(m['pf_viol'] for m in case_metrics)),
        'profit': float(sum(m['profit'] for m in case_metrics)),
        'soc_track_err': float(np.mean([m['soc_track_err'] for m in case_metrics])),
        # 当前动态阈值 (LLM 微调结果) 隨测试一并记录
        'wind_low_max': wind_low_max,
        'wind_high_min': wind_high_min,
        # 相位特征档位 (便于对照 4/6/9 维观测)
        'phase_feat': phase_feat,
    }
    return case_metrics, overall
