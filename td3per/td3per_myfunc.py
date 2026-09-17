"""
TD3PER — 上层 = TD3PERRB 口径 (非分段奖励 + SOC 目标信号), 下层 = RL BESS
=========================================================================
本文件是 td3per 的 "td3perrb 化" 改造:
  * 上层 (H) 的奖励/成本/信号语义 **完全移植自 td3perrb_myfunc.py**
    (非分段奖励: 收益权重恒 1.0, c1 小额波动系数恒 0.5);
  * 下层 (L) 仍是 td3per 自己的 RL BESS (TD3 + PER + EMA 对偶), 只把
    "信号 / 追踪目标" 由影子电价换成 **SOC 目标**。

上层 H (RL, 6 维动作 = 5 台风机桨距 + 1 个 SOC 目标信号)
--------------------------------------------------------
  a6 ∈ [-0.5, 0.5] → soc_target = 3 + 5·a6 ∈ [0.5, 5.5]   (块级(5 步)固定)
  r1 = 1.0·(pr·ptur·dt) + 1.0·(−pr·pbrule·dt) − 490·max(−pbrule,0)·dt − penalty_NN_H
     = pr·P_G·dt − 490·max(−pbrule,0)·dt − penalty_NN_H        (P_G = ptur − pbrule)
  c1 = 1.0·[10·(P_F−3)⁺ + 0.1·P_F] + W_GAP·unfeasible_gap        (W_GAP = 10)
       + [块末结算] 10·max(0, |soc_target − SOC_end| − SOC_TOL)  (SOC_TOL = 0.1)

下层 L (RL, 1 维动作 pb)
------------------------
  观测 (5 维蒸馏) = [pbrulemin/10, pbrulemax/10, soc_target/10, SOC/10, u/10]
      第 3 维由原 shadow_price(∈(−1,1)) 改为 **SOC 目标** (与 SOC 同刻度, 便于网络
      直接比较"目标 vs 自身"); 见 build_L_obs_distilled()。
  r2 = −100 · clip_amount            (clip_amount = |pb − pbrule|, RBB 硬裁剪量)
      —— 原影子套利项 (−pr·pbrule 形式) 随信号语义一起删除。
  c2 = 1000 · slack                  (能力内追踪失败量, 进拉格朗日约束口)
      slack = max(0, |target − SOC'| − |target − SOC_best|),
      SOC_best = SOC + g·clip(δ/g, 窗口)   ← 规则 SocTracker 的动作
      —— 只罚"本步能做到却没做"的部分: 块首够不着的跳变(结构性不可行)不计费,
         零成本动作恰好就是规则下层; 见 soc_track_slack()。

  即: 下层的职责被收敛成 "在 RBB 可行窗口内贴住上层给的 SOC 目标";
      收益/波动/可行性由上层 r1/c1 + RBB 硬裁剪负责, 下层不再需要套利激励。

Constants
---------
PRICE 分档: ≥980 HIGH, ≤300 LOW, 其余 MEDIUM  ← 仅诊断标签, 不进入奖励
WIND  分档: <7.0 LOW, ≥11.0 HIGH, 其余 MEDIUM ← 仅诊断标签, 不进入奖励
"""

import numpy as np
import pandas as pd

from util import *


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


WIND_LOW_MAX  = 7.0   # u < 7.0   → LOW wind
WIND_HIGH_MIN = 11.0  # u >= 11.0 → HIGH wind, else MEDIUM


def get_wind_level(u):
    """Classify wind speed into LOW / MEDIUM / HIGH."""
    if u < WIND_LOW_MAX:
        return 'LOW'
    elif u >= WIND_HIGH_MIN:
        return 'HIGH'
    else:
        return 'MEDIUM'


# =============================================================================
# 2. 非分段权重 (non-segmented weights) —— 与 td3perrb_myfunc.py 一致
# =============================================================================
# 机会侧 (利润 r1) 与风险侧 (c1) 分离, 但不做任何档位加权 —— 三个系数恒为 1.0。
# TD3PERRB 相对 TD3PRRLRB 的唯一差异就是这个; 本文件把该口径搬到 td3per 的上层。
# 风况/电价档位标签仍然计算, 但只作为日志与 CSV 的诊断列, 不进入任何系数。
W_TURBINE = 1.0      # wT: 非分段 ⇒ 恒为 1.0 (不做 风况×电价 加权)
W_BESS = 1.0         # wB: 非分段 ⇒ 恒为 1.0 (不做 电价加权 / SOC 贴边折减)

# c1 小额波动系数: 非分段 ⇒ 单一系数 0.5 (不按风况分档)
C1_SMALL_FLUC_COEF = 0.5   # 非分段 ⇒ c1 小额波动系数统一 0.5
# unfeasible_gap（平抑可行性缺口, MW）进入 c1 的系数
W_GAP = 10.0

# SOC 目标跟踪死区 (MWh): 0.1 MWh = 6 MW × 1 min
#   —— 该豁免**只属于上层 gap 罚** (记账约定): 上层只对超过 0.1 MWh 的块末残差付费。
#      下层 c2 没有死区 (任何"能力内该做没做"的 slack 都计费), 保证"下层能消掉的
#      残差不少于上层豁免的", 不会出现规则停下而上层照罚的幽灵税。
SOC_TOL = 0.1
# SOC 目标映射: a6 ∈ [-0.5, 0.5] → soc_target ∈ [0.5, 5.5]
SOC_TARGET_CENTER = 3.0
SOC_TARGET_SPAN = 5.0
SOC_TARGET_MIN, SOC_TARGET_MAX = 0.5, 5.5

# 退化系数 (元/MWh): 放电侧; 充电侧可以取更小 (k_c = 0 表示只在放电计退化)
DEG_COST_DISCHARGE = 490.0
DEG_COST_CHARGE = 0.0


def get_wT(wind_level, price_level):
    """Turbine revenue weight. 非分段 ⇒ 恒为 W_TURBINE (=1.0)."""
    return W_TURBINE


def get_wB(price_level, wind_level, SOC):
    """BESS revenue weight. 非分段 ⇒ 恒为 W_BESS (=1.0), 不含 SOC 贴边折减."""
    return W_BESS


# =============================================================================
# 3. SOC 安全带 (仅用于日志指标; 不参与任何成本)
# =============================================================================
# 非分段 ⇒ 单一安全带 [1.5, 4.5] (与 td3per 原 SOC 安全带常数一致)。
SOC_SAFE_MIN, SOC_SAFE_MAX = 1.5, 4.5


def Cal_soc_deviation(SOC, wind_level=None):
    """SOC 偏离安全带 (MWh) —— 仅日志指标, 单档 [SOC_SAFE_MIN, SOC_SAFE_MAX].

    注意: 改造后下层的 c2 不再以安全带为参照, 而以 **上层 SOC 目标** 为参照
    (见 lower_cost_soc_track); 本函数只剩诊断用途。
    """
    if SOC < SOC_SAFE_MIN:
        return SOC_SAFE_MIN - SOC
    if SOC > SOC_SAFE_MAX:
        return SOC - SOC_SAFE_MAX
    return 0.0


# =============================================================================
# 4. Upper-level r1
# =============================================================================

def Cal_r1_continuous(pr, ptur, pbrule, penalty_NN_H, wind_level, price_level,
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
# 5. SOC 目标映射 与 上层可行性(gap)惩罚
# =============================================================================

def soc_target_from_signal(a6):
    """上层信号 → SOC 目标: 3 + 5*a6, 并夹在 [0.5, 5.5] (合法目标带)."""
    return float(np.clip(SOC_TARGET_CENTER + SOC_TARGET_SPAN * float(a6),
                         SOC_TARGET_MIN, SOC_TARGET_MAX))


def gap_penalty(soc_target, SOC_end, tol=SOC_TOL):
    """上层可行性罚 (块末结算): 10 * max(0, |soc_target - SOC_end| - tol)."""
    return W_GAP * max(0.0, abs(float(soc_target) - float(SOC_end)) - tol)


# =============================================================================
# 6. Combined reward (H 层)
# =============================================================================

def Cal_Rew_RB(observation, observation2, action_rbb, rbb_info,
               penalty_NN_H=0.0, time=1.0 / 60, ptur=None,
               pb_cmd=None):
    """
    上层 (H) 奖励/成本 —— 非分段口径 (与 td3perrb_myfunc.Cal_Rew_RB 逐字一致).

    Parameters
    ----------
    observation  : [pr, u, E, P_G_last]
    observation2 : [pr', u', E', P_G]
    action_rbb   : [a1..a5, soc_signal, pbrule]
    rbb_info     : dict from RuleBasedBESS.compute_pbrule()
    penalty_NN_H : 上层动作越界罚
    ptur         : 风机功率 (MW)
    pb_cmd       : 下层给 RBB 之前的功率指令 (MW) —— 仅诊断列 (RL 下层 = pb_original)

    Returns
    -------
    r1, c1_step, details      # 注意: 不含块末 gap 罚 (由 main 在块末累加)
    """
    if ptur is None:
        raise ValueError("Cal_Rew_RB: ptur must be provided")

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
    r1 = Cal_r1_continuous(pr_old, ptur, pbrule, penalty_NN_H,
                           wind_level, price_level, SOC=SOC_decision, time=time)

    # ==== c1 (H 风险侧) ====
    # 非分段 ⇒ 小额波动系数取统一定值 0.5 (越限项系数恒 10)
    c1_small = C1_SMALL_FLUC_COEF
    if P_F > 3.0 + 1e-4:
        c1 = 10.0 * (P_F - 3.0) + c1_small * P_F
    else:
        c1 = c1_small * P_F
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
        'soc_deviation': soc_dev,
        'ptur': ptur,
        'pbrule': pbrule,
        'pb_cmd': float(pb_cmd) if pb_cmd is not None else np.nan,
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
# 6b. 下层 (L) 的观测 / 奖励 / 成本 —— SOC 目标追踪口径
# =============================================================================

# 下层 r2 的 RBB 硬裁剪惩罚系数 (td3per 原值 100)
L_CLIP_COEF = 100.0
# 下层 c2 的 SOC 目标追踪误差系数 (td3per 原 c2 系数 1000/MWh)
L_SOC_COEF = 1000.0


def build_L_obs_distilled(observation, pw_tru, applied_action_H, rbb_info):
    """下层蒸馏观测 (5 维): [pbrulemin/10, pbrulemax/10, soc_target/10, SOC/10, u/10].

    与改造前的唯一差异: 第 3 维由 `shadow_price = 2·a6 ∈ (−1,1)` 换成
    **SOC 目标** `soc_target/10` (∈[0.05, 0.55], 与 SOC/10 同刻度) —— 下层的
    "信号" 与 "追踪目标" 因此是同一个量, 网络可直接比较 "目标 vs 自身"。

    其余 4 维不变: 当前可行窗口 (RBB, 与 pb 无关) + 自身 SOC + 风速 u
    (u 保留在观测中, 供可行域/其它风速相关逻辑使用)。
    """
    soc_target = soc_target_from_signal(applied_action_H[-1])
    return np.array([
        rbb_info['pbrulemin'] / 10.0,
        rbb_info['pbrulemax'] / 10.0,
        soc_target / 10.0,
        float(observation[2]) / 10.0,
        float(observation[1]) / 10.0,  # u: 风速 (0~25 → 0~2.5)
    ])


def lower_reward(clip_amount, coef=L_CLIP_COEF):
    """下层 r2 = −coef · clip_amount (只保留 RBB 硬裁剪惩罚).

    改造前 r2 = 10·影子套利利润 − 100·clip_amount; 信号语义改为 SOC 目标后
    影子电价套利项失去依据 (且收益已由上层 r1 = pr·P_G·dt 全额记账),
    故只保留下层自己"别被窗口裁剪"的惩罚项。
    """
    return -float(coef) * float(clip_amount)


def soc_track_slack(soc_target, SOC_prev, SOC_next, pbrulemin, pbrulemax,
                    eta_c=0.98, eta_d=0.98, time=1.0 / 60):
    """下层"能力内追踪失败量"(slack, MWh) 及其诊断量。

    动机 (线性式 |target − SOC'| 的结构性缺陷)
    ------------------------------------------
    SOC 是纯积分器、目标块内固定, 且下层单步能力只有
        A = g·|pb|,  g_c = η_c·dt ≈ 0.0163 / g_d = dt/η_d ≈ 0.0170 MWh/MW,
    窗口 (平抑 ∩ 物理) 通常只放 ~3 MW → A ≈ 0.05 MWh/步 (额定 10 MW 也只有 0.17)。
    而目标每块(5 步)才变一次, 相邻块最多差 5 MWh ⇒ **块首的 |δ| 无论如何都消不掉**。
    直接以 |δ| 计费 ⇒ 每块固定吃一笔 1000~2000/步 的"不可控"惩罚:
      * sum_c2 变成"H 给的目标有多远"的代理, 而不是"L 追得多差";
      * d_block_L(20)/curriculum 恒被违反, lam2 迅速顶到 500 → 对偶退化为常数。

    本函数把 "不可控" 减掉, 只留 L 能控的部分 —— 即"本步能做到的最好"与
    "实际做到"之差:
        delta   = soc_target − SOC_prev
        g       = η_c·dt (delta>0 充电) / dt/η_d (delta<0 放电)
        pb_best = clip(delta/g, pbrulemin, pbrulemax)   # ← 规则 SocTracker 的动作
        d_best  = |delta − g·pb_best|                   # 该动作后的残差 (最优可达)
        d_next  = |soc_target − SOC_next|                # 实际执行后残差
        slack   = max(0, d_next − d_best)                # ∈ [0, 2A]

    性质
    ----
    * **块首 slack = 0**: 够不着的部分是结构性的 (由上层 gap 罚计费), 不罚下层;
    * **零成本动作恰好是规则下层 SocTracker 的 pb = clip(δ/g, 窗口)** ——
      规则版与 RL 版下层的**最优策略完全同一**, 两者对比变成纯粹的可学性对比;
    * 目标近时顶格充电/放电会被计费 → 天然学出"到位即停", 不产生自致过冲;
    * 反向动作全额计费 (d_next 增大), 梯度方向明确;
    * 取值紧界: |d_next − |δ|| ≤ A、|d_best − |δ|| ≤ A ⇒ slack ≤ 2A ≈ 0.34 MWh;
    * 窗口不可行 (pbrulemin > pbrulemax) 时 slack = 0 —— 被迫吞吐不可归责于 L。

    Returns
    -------
    slack : float  (MWh, ≥0)
    info  : dict   {delta, gain, pb_best, soc_best, d_best, d_next,
                    infeasible_window}
    """
    soc_target = float(soc_target)
    SOC_prev = float(SOC_prev)
    SOC_next = float(SOC_next)
    pbrulemin = float(pbrulemin)
    pbrulemax = float(pbrulemax)

    delta = soc_target - SOC_prev
    g = eta_c * time if delta > 0 else time / eta_d
    d_next = abs(soc_target - SOC_next)

    # 平坦窗口不可行 (平抑 ∩ 物理 = ∅): 本步的残差不可归责于 L
    if pbrulemin > pbrulemax:
        return 0.0, {
            'delta': delta, 'gain': g, 'pb_best': 0.0,
            'soc_best': SOC_prev, 'd_best': abs(delta), 'd_next': d_next,
            'infeasible_window': True,
        }

    pb_best = float(np.clip(delta / g, pbrulemin, pbrulemax))
    soc_best = SOC_prev + g * pb_best
    d_best = abs(soc_target - soc_best)
    slack = max(0.0, d_next - d_best)

    return slack, {
        'delta': delta, 'gain': g, 'pb_best': pb_best,
        'soc_best': soc_best, 'd_best': d_best, 'd_next': d_next,
        'infeasible_window': False,
    }


def lower_cost_soc_track(soc_target, SOC_prev, SOC_next, pbrulemin, pbrulemax,
                         coef=L_SOC_COEF, return_info=False, **kw):
    """下层 c2 = coef · slack (能力内追踪失败量, 进拉格朗日约束口).

    改造前 c2 = 1000·(相对固定安全带 [1.5,4.5] 的越界量) —— 那是一个**稀疏的
    安全约束** (SOC 在带内时恒为 0); 上一轮改成 1000·|soc_target − SOC'| 后变成
    **稠密的水平惩罚**, 其量级被块首不可控的目标跳变主导 (见 soc_track_slack)。
    本轮改为"能力内 slack"口径 —— 下层 lam2 对偶因此学的是
    "本可消掉却没消掉的残差" 的边际价格 (一个真正可满足的约束), 而不是目标距离。

    标定 (d_block_L)
    -----------------
    尽力追的 L: sum_c2 ≡ 0; 完全不动而 |δ|≥A: 每步 ~1000·A ≈ 50, 每块 ~250;
    最坏 (反向满功率): 每步 ≤ 1000·2A ≈ 340。
    因此 d_block_L = 20 现读作"允许每块浪费 ≈0.02 MWh 的可达量 (≈0.4 步能力)",
    地板 2.5 相应为"允许 ≈0.05 步能力"。想更松可取 20~50。
    """
    slack, info = soc_track_slack(soc_target, SOC_prev, SOC_next,
                                  pbrulemin, pbrulemax, **kw)
    cost = float(coef) * slack
    if return_info:
        return cost, info
    return cost


# =============================================================================
# 7. Action checking
# =============================================================================

def CheckAction(action, charge=10.):
    """Clip actions to feasible ranges. WT: [0,0.5], Signal: [-0.5,0.5], Bat: [-charge,charge].

    上层 (6 维): 与 td3perrb 一致 —— WT 越界与信号越界按 100/单位计罚, **不封顶**
    (保持越界区恒定的恢复力, 见 td3perrb/td3prrlrb 的 run160 诊断)。
    下层 (1 维): 保留 td3per 原口径 (根 myfunc.CheckAction 的 500/单位);
        该罚项在当前 r2 公式里不参与 (只作诊断)。
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
        # 不封顶: 保持恒定恢复力 (见 docstring)
        return applied, penalty

    elif len(action) == 5:
        for j in action[:5]:
            ma, mi = max(0.5, j), min(0, j)
            penalty += 100 * ((ma - 0.5) + (0. - mi))
        return np.clip(action[:5], 0., 0.5), penalty

    elif len(action) == 1:
        for j in action:
            ma, mi = max(charge, j), min(-charge, j)
            penalty += 500 * ((ma - charge) + (-charge - mi))
        return np.clip(action, -charge, charge), penalty

    return action, penalty


# =============================================================================
# 8. Normalization  (H 观测 = 4 维基础观测 + 可选 24h 相位特征)
# =============================================================================

def Normalize(obs, max_E=6.):
    return np.array([obs[0] / 1000, obs[1] / 10, obs[2] / 10, obs[3] / 10])


def Original(obs, max_E=6.):
    return np.array([obs[0] * 1000, obs[1] * 10, obs[2] * 10, obs[3] * 10])


# =============================================================================
# 8b. 24h 相位特征 (--phase_feat)
# =============================================================================
# 动机: H 原本看不到"一天中的位置"。而套利的关键决策之一 —— 在低价窗的哪一刻
# 把电充满 —— 必须要知道"本段还剩多久、下一段是涨还是跌"。
#
# 编码方式:
#   mode 0: 关 (4 维, 可复现旧 run)
#   mode 1: sin/cos(2π·minute/1440)                      → 6 维 (默认)
#   mode 2: mode 1 + [段内剩余比例, 剩余小时/8, 下一段电价/1000] → 9 维
#
# 相位来源: **不读 env 内部时钟**, 而是按"每个 episode 恰好 1440 步"这一事实
# 用 episode_steps % 1440 计算, 并与 env.CalPrice 的日程表对齐;
# 主循环里用 day_price_schedule() 做一致性自检。

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
    return REP_MINUTE.get(str(level).upper(), 210)


# =============================================================================
# 9. 9-case 测试 (下层为 RL BESS, 追踪 SOC 目标)
# =============================================================================

try:
    from test_scenario import build_9case_scenario
except ImportError:
    import sys as _sys, os as _os
    _HSRL_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _HSRL_ROOT not in _sys.path:
        _sys.path.insert(0, _HSRL_ROOT)
    from test_scenario import build_9case_scenario


def nine_case_test(agent_H, agent_L, env_test, rbb_system,
                   scenario=None, test_csv=False, output=None, seed=None,
                   phase_mode=0, phase_table=None):
    """
    TD3PER 45-step 9-case in-training test (上层 = 非分段 SOC 目标信号, 下层 = RL).

    Layout: wind LOW->MEDIUM->HIGH; within each wind segment price
    LOW->MEDIUM->HIGH; each case = 5 steps = one upper-level period.

    Returns
    -------
    case_metrics : list of dict (length 9)
        r1 / c1(步级 + 块末 gap) / gap_pen / r2(RL 下层) / c2(追踪误差) /
        cvm(c1_step>0 的步数) / pf_viol(P_F>3 的步数) /
        profit(5 步利润之和) / soc_track_err(块末 |target−SOC|)
    overall : dict —— 9 个 case 的均值 (cvm/pf_viol 为总计; profit 为 45 步总和)
    """
    if scenario is None:
        wind_seq, price_seq, case_ids, blocks = build_9case_scenario(seed=seed)
    else:
        wind_seq, price_seq, case_ids, blocks = scenario

    obs = env_test.ScenarioReset(wind_seq, price_seq)
    # 相位: 场景电价每 5 步就变一次, 直接用 wall-clock 相位会 OOD;
    # 这里按 case 的电价档映射到该档"代表性时刻"(段中点), 使测试与训练同分布,
    # 且跨 run 可比 (相同 case 相同相位)。
    _pminute = {i: rep_minute_for_price_level(price_seq[i]) for i in range(len(price_seq))}
    norm_obs = Normalize_H(obs, _pminute[0], phase_mode, phase_table)
    time = 1.0 / 60
    agent_H.is_training = False
    agent_L.is_training = False
    step = 0
    num_iterations = len(wind_seq)
    n_per_case = num_iterations // len(blocks)

    per_case = {i: {'r1': 0.0, 'c1': 0.0, 'r2': 0.0, 'c2': 0.0,
                    'cvm': 0, 'pf_viol': 0, 'profit': 0.0,
                    'gap_pen': 0.0, 'soc_track_err': 0.0}
                for i in range(len(blocks))}
    df = pd.DataFrame()
    soc_target = SOC_TARGET_CENTER

    while step < num_iterations:
        if step % 5 == 0:
            action_H = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H)
            pw_tru = env_test.PowerOutput(obs[1], action_H)
            soc_target = soc_target_from_signal(action_H[-1])

        ptur = float(pw_tru)
        pg_prev = float(obs[-1])
        E_current = float(obs[2])

        # ---- 下层 obs: 蒸馏 5 维 (第 3 维 = soc_target/10) ----
        _, rbb_info_obs = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=ptur, pg_prev=pg_prev, E=E_current)
        norm_obs_L = build_L_obs_distilled(obs, pw_tru, action_H, rbb_info_obs)
        if step % 5 == 0:
            agent_L.reset(norm_obs_L)
        original_action_L = agent_L.select_action(norm_obs_L)
        action_L, penalty_NN_L = CheckAction(original_action_L)

        action = np.concatenate((action_H, action_L))

        pb_original = float(action[-1])
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

        r1, c1_step, details = Cal_Rew_RB(
            obs, obs2, action_rbb, rbb_info,
            penalty_NN_H=penalty_NN_H, time=time,
            ptur=ptur, pb_cmd=pb_original,
        )

        # 块末: SOC 目标可行性罚 (上层)
        gap_pen = 0.0
        if step % 5 == 4:
            gap_pen = gap_penalty(soc_target, float(obs2[2]))
        c1 = c1_step + gap_pen

        # 下层 r2 / c2 (SOC 目标追踪口径: 能力内 slack)
        r2 = lower_reward(rbb_info['clip_amount'])
        c2, c2_info = lower_cost_soc_track(
            soc_target, E_current, float(obs2[2]),
            rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'],
            return_info=True)

        bi = step // 5
        pc = per_case[bi]
        pc['r1'] += r1
        pc['c1'] += c1
        pc['r2'] += r2
        pc['c2'] += c2
        pc['gap_pen'] += gap_pen
        if c1_step > 0:
            pc['cvm'] += 1
        if details['P_F'] > 3.0 + 1e-4:
            pc['pf_viol'] += 1
        if step % 5 == 4:
            pc['soc_track_err'] = abs(soc_target - float(obs2[2]))

        # profit: P_G * pr * dt - 490*max(-pb,0)*dt (与训练 episode_profit 同口径)
        profit_step = obs2[-1] * obs[0] * time - DEG_COST_DISCHARGE * max(-float(pbrule), 0.0) * time
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
                'r2': r2, 'c2': c2,
                'c2_slack': c2_info['d_next'] - c2_info['d_best'],
                'pb_best': c2_info['pb_best'],
                'd_best': c2_info['d_best'], 'd_next': c2_info['d_next'],
                'profit': profit_step,
                'soc_signal': details['soc_signal'],
                'soc_target': details['soc_target'],
                'soc_delta': details['soc_delta'],
                'SOC': details['SOC'],
                'pb_original': pb_original, 'pbrule': rbb_info['pbrule'],
                'clip_amount': rbb_info['clip_amount'],
                'P_F': details['P_F'],
                'action': action_rbb,
            }])], ignore_index=True)

        norm_obs = norm_obs2
        obs = obs2
        pw_tru = pw_tru2
        step += 1

    agent_H.is_training = True
    agent_L.is_training = True
    if test_csv and output:
        df.to_csv(f'{output}/test_s_9case.csv', index=False)

    case_metrics = []
    for i, b in enumerate(blocks):
        pc = per_case[i]
        case_metrics.append({
            'wind_level': b['wind_level'],
            'price_level': b['price_level'],
            'r1': pc['r1'] / n_per_case,
            'c1': pc['c1'] / n_per_case,
            'r2': pc['r2'] / n_per_case,
            'c2': pc['c2'] / n_per_case,
            'gap_pen': pc['gap_pen'] / n_per_case,
            'cvm': pc['cvm'],
            'pf_viol': pc['pf_viol'],
            'profit': pc['profit'],
            'soc_track_err': pc['soc_track_err'],
        })

    overall = {
        'r1': float(np.mean([m['r1'] for m in case_metrics])),
        'c1': float(np.mean([m['c1'] for m in case_metrics])),
        'r2': float(np.mean([m['r2'] for m in case_metrics])),
        'c2': float(np.mean([m['c2'] for m in case_metrics])),
        'gap_pen': float(np.mean([m['gap_pen'] for m in case_metrics])),
        'cvm': int(sum(m['cvm'] for m in case_metrics)),
        'pf_viol': int(sum(m['pf_viol'] for m in case_metrics)),
        'profit': float(sum(m['profit'] for m in case_metrics)),
        'soc_track_err': float(np.mean([m['soc_track_err'] for m in case_metrics])),
    }
    return case_metrics, overall
