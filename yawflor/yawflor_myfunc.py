"""
TD3PER — 上层 = TD3PERRB 口径 (非分段奖励 + SOC 目标信号), 下层 = RL BESS
=========================================================================
本文件是 td3per 的 "td3perrb 化" 改造:
  * 上层 (H) 的奖励/成本/信号语义 **完全移植自 td3perrb_myfunc.py**
    (非分段奖励: 收益权重与 c1 风险系数恒 1.0);
  * 下层 (L) 仍是 td3per 自己的 RL BESS (TD3 + PER + EMA 对偶), 只把
    "信号 / 追踪目标" 由影子电价换成 **SOC 目标**。

上层 H (RL, 6 维动作 = 5 台**偏航指令 n ∈ [−1,+1]** + 1 个 SOC 目标信号)
--------------------------------------------------------------------------------
  ⚠ yawflor 改造 (相对 td3per_myfunc 的**唯一**语义差异):
    动作维是**无量纲偏航指令** `n`, env 内 `γ = 25°·n`（动作带 [−1,+1]，带宽 2）；
    旧环境是 5 维轴向因子 `a ∈ [0,0.5]`（带宽 0.5）。奖励/成本/信号语义**完全不变**。
  a6 ∈ [-0.5, 0.5] → soc_target = 3 + 5·a6 ∈ [0.5, 5.5]   (块级(5 步)固定)
  r1 = 1.0·(pr·ptur·dt) + 1.0·(−pr·pbrule·dt) − 490·max(−pbrule,0)·dt − penalty_NN_H
     = pr·P_G·dt − 490·max(−pbrule,0)·dt − penalty_NN_H        (P_G = ptur − pbrule)
  c1 = 1.0·[10·(P_F−3)⁺ + 0.1·P_F] + W_GAP·unfeasible_gap        (W_GAP = 10)
       + [块末结算] 10·max(0, |soc_target − SOC_end| − SOC_TOL)  (SOC_TOL = 0.1)

下层 L (RL, 1 维动作 pb)
------------------------
  观测 (5 维蒸馏) = [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]
            规则的原配料: pg(t−1)/ptur 给窗口中心, SOC 与 soc_target 同刻度(第 2/4 维),
      n_rem/5 给"还剩几步" ⇒ δ 与摊还需量都能直接导出; 见 build_L_obs_distilled()。
  r2 = −1 · min(clip_amount, 20)              (clip_amount = |命令 − pbrule| = 两段 clip 合计)
      —— 原影子套利项 (−pr·pbrule 形式) 随信号语义一起删除。
   c2 = +1 · trackerr                 [MW] 两区间分治 (单一真源 hsrl_soctarget)
       参考集合 = Soctrack范围[δ/(g·n_rem), δ/g] ∩ 平抑窗口 [pbrulemin, pbrulemax]
       trackerr = dist(实际施加的 pb (由 SOC 变化反解), 参考集合)
       —— F ≠ ∅ ⇒ 零成本集合是该**区间** (L 可在「摊还端…一步填平端」间任选努力水平);
          F = ∅ ⇒ **不存在范围, 只有纯粹一个竭力值 r*** = 窗口里离该范围最近的端点,
                   顶到 r* 才不罚, 两侧偏离都按 |pbrule − r*| 线性计费
                   (够不着的缺口不可达 ⇒ 由 r1 块末 gap 罚承接)。

  即: 下层的职责被收敛成 "在 RBB 可行窗口内贴住上层给的 SOC 目标";
      收益/波动/可行性由上层 r1/c1 + RBB 硬裁剪负责, 下层不再需要套利激励。

Constants
---------
PRICE 分档: ≥980 HIGH, ≤300 LOW, 其余 MEDIUM  ← 仅诊断标签, 不进入奖励
WIND  分档: <7.0 LOW, ≥11.0 HIGH, 其余 MEDIUM ← 仅诊断标签, 不进入奖励
"""

import numpy as np
import hsrl_cost as hc
import hsrl_soctarget as hst
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

# c1 风险系数: 非分段 ⇒ 单一系数 (不按风况分档)
C1_RISK_COEF = 1.0
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
VIOL_DERATE_DIMS = 5       # 前 N 维 = **偏航维**; dim5(signal) 不计入 x
_ACT_HI, _ACT_LO = 1.0, -1.0  # 与 CheckAction 的前 5 维边界一致 (yawflor: [−1,+1])


def violation_amount(action, n_dims=None):
    """x = 前 n_dims 维 (**偏航维**, 边界 [−1,+1]) 的越界量之和。"""
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


def Cal_r1_continuous(pr, ptur, pbrule, penalty_NN_H, wind_level, price_level,
                      SOC=3.0, time=1.0 / 60, turbine_derate=1.0):
    """
    r1 = turbine_derate * wT * (pr * ptur * time)   ← 风机收益 (乘越界系数)
       + wB * (-pr * pbrule * time)          ← BESS 收益 (正=pbrule<0 放电)
       - 490 * max(-pbrule, 0) * time        ← 退化
       - penalty_NN_H

    turbine_derate=1.0 (默认) 时与旧口径逐位相同。
    """
    w_wind = get_wT(wind_level, price_level)
    w_bess = get_wB(price_level, wind_level, SOC)

    turbine_profit = w_wind * pr * ptur * time
    bess_profit    = w_bess * (-pr * pbrule * time)
    degradation    = (DEG_COST_DISCHARGE * max(-pbrule, 0.0)
                      + DEG_COST_CHARGE * max(pbrule, 0.0)) * time

    return (turbine_profit * turbine_derate + bess_profit - degradation
            - penalty_NN_H)


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
               pb_cmd=None, turbine_derate=1.0):
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
    turbine_derate : 越界乘性系数 f=exp(-c·x), 只乘风机收益项; 默认 1.0 (旧口径逐位相同)

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
                           wind_level, price_level, SOC=SOC_decision, time=time,
                           turbine_derate=turbine_derate)

    # ==== c1 (H 风险侧) ====
    c1_risk_coeff = C1_RISK_COEF   # 非分段 ⇒ 恒为 1.0 (不做风况分档)
    if P_F > 3.0 + 1e-4:
        c1 = c1_risk_coeff * (10.0 * (P_F - 3.0) + 0.1 * P_F)
    else:
        c1 = c1_risk_coeff * (0.1 * P_F)
    c1 = c1 + W_GAP * unfeasible_gap
    c1 = hc.c1_apply(c1, P_F)   # uniform: 波动当量替换 (legacy: 原样)

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

# 下层 r2 的 RBB 硬裁剪惩罚系数 (td3per 原值 100, 全库统一 ÷100 后为 1)
L_CLIP_COEF = hst.L_CLIP_COEF      # 单一真源 hsrl_soctarget
# 下层 c2 的 SOC 目标追踪误差系数 (td3per 原 c2 系数 1000/MWh, ÷100 后为 10)
L_SOC_COEF = hst.L_TRACK_COEF      # 单一真源 (新口径: MW)


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
    soc_target = soc_target_from_signal(applied_action_H[-1])
    return hst.build_L_obs(float(observation[-1]), float(pw_tru),
                           float(observation[2]), soc_target, n_rem)


def soc_track_slack(soc_target, SOC_prev, SOC_next, pbrulemin, pbrulemax, n_rem,
                    eta_c=0.98, eta_d=0.98, time=1.0 / 60):
    """兼容旧名：返回"能力内搁置量" (MWh) = trackerr·g 与诊断量。

    单一真源 = `hsrl_soctarget.soc_track_range`（两区间分治：Soctrack 范围 ∩ 平抑窗口）。
    """
    trackerr, info = hst.soc_track_range(soc_target, SOC_prev, SOC_next,
                                         pbrulemin, pbrulemax, n_rem,
                                         eta_c=eta_c, eta_d=eta_d, time=time)
    return info['slack'], info


def lower_reward(clip_amount, coef=None, cap=None):
    """下层 r2 = −coef·min(clip_amount, cap) [MW]（单一真源）。"""
    return hst.lower_reward(clip_amount, coef, cap)


def lower_cost_soc_track(soc_target, SOC_prev, SOC_next, pbrulemin, pbrulemax, n_rem,
                         coef=None, return_info=False, **kw):
    """下层 c2 = +coef · trackerr [MW]（两区间分治；单一真源 = hsrl_soctarget）。

    口径（本轮替换旧"能力内 slack"）：参考集合 = Soctrack 范围
    `[δ/(g·n_rem), δ/g]` ∩ 平抑窗口；空交集 ⇒ **不存在范围，只有纯粹的一个竭力值 r***
    = 窗口里离 Soctrack 范围最近的端点（零宽单点：只有顶到 r* 才不罚，两侧偏离都线性计费）。
    `trackerr` = 真正施加的 pb（由 SOC 变化反解）到参考集合的距离 [MW]。
    r2/c2 同为 MW 口径、系数同为 1 ⇒ 两者可直接相加/比较。
    """
    return hst.lower_cost_soc_track(soc_target, SOC_prev, SOC_next,
                                    pbrulemin, pbrulemax, n_rem,
                                    coef=coef, return_info=return_info, **kw)


# =============================================================================
# 7. Action checking
# =============================================================================

PENALTY_NN_H_COEF = 100.0  # [元/单位] H 层动作越界罚 (**偏航指令带 [−1,+1]**×5 + 信号 [±0.5])
                         #   ⚠ yawflor: 带宽从 0.5 (轴向因子 [0,0.5]) 变成 2 (偏航 [−1,+1])
                         #   ⇒ 系数必须 ÷4, 才能保持**"满带越界 = 200 元"这一经济口径不变**
                         #   (400 × 0.5 = 100 × 2 = 200 元; 与旧环境等值, 见 yawflor/README.md §4)。
                         #   单机物理不像旧环境那样有"Cp 悬崖", 但带外仍无物理含义 ⇒ 同样不封顶。
PENALTY_NN_L_COEF = 200.0  # [元/单位] 下层 1 维动作罚 —— **保持 200 不变** (仅诊断, 不进 r2 公式)


def CheckAction(action, charge=10.):
    """Clip actions to feasible ranges. YAW(n): [-1,+1], Signal: [-0.5,0.5], Bat: [-charge,charge].

    上层 (6 维): 与 td3per 一致 —— **偏航维**越界与信号越界按 PENALTY_NN_H_COEF 计罚,
    **不封顶** (保持越界区恒定的恢复力, 见 td3perrb/td3prrlrb 的 run160 诊断)。
    ⚠ yawflor: 前 5 维边界是 [−1,+1]（无量纲偏航指令），不是 [0,0.5]。
    下层 (1 维): 保留 td3per 原口径 (根 myfunc.CheckAction 的 500/单位);
        该罚项在当前 r2 公式里不参与 (只作诊断)。
    """
    penalty = 0
    SIGNAL_MIN, SIGNAL_MAX = -0.5, 0.5

    if len(action) == 6:
        for j in action[:5]:
            ma, mi = max(1.0, j), min(-1.0, j)
            penalty += PENALTY_NN_H_COEF * ((ma - 1.0) + (-1.0 - mi))
        sig = action[5]
        if sig > SIGNAL_MAX:
            penalty += PENALTY_NN_H_COEF * (sig - SIGNAL_MAX)
        elif sig < SIGNAL_MIN:
            penalty += PENALTY_NN_H_COEF * (SIGNAL_MIN - sig)
        applied = np.append(np.clip(action[:5], -1., 1.),
                            np.clip(action[5], SIGNAL_MIN, SIGNAL_MAX))
        # 不封顶: 保持恒定恢复力 (见 docstring)
        return applied, penalty

    elif len(action) == 5:
        for j in action[:5]:
            ma, mi = max(1.0, j), min(-1.0, j)
            penalty += PENALTY_NN_H_COEF * ((ma - 1.0) + (-1.0 - mi))
        return np.clip(action[:5], -1., 1.), penalty

    elif len(action) == 1:
        for j in action:
            ma, mi = max(charge, j), min(-charge, j)
            penalty += PENALTY_NN_L_COEF * ((ma - charge) + (-charge - mi))
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


# test9 存货结算参考价 —— **外生常数** (= 名义电价 hc.PR_REF, MED=580, 避免裸数字)。
# 为什么必须外生: 若取"当时电价", 每步的充放收益逐项抵消 (ΔSOC·(pr_ref − pr) ≡ 0)
#   ⇒ 电池对目标的贡献恒为 0 ⇒ 套利激励归零 (退化成没有电池)。
# 为什么取 580: 算上效率后充电闸门 = η_c·pr_ref = 568.4、放电闸门 = pr_ref/η_d = 591.8,
#   场景三价 LOW 300 / MED 580 / HIGH 980 ⇒ MED 恰落在中性带 [568.4, 591.8] 内。
SOC_SETTLE_PR = float(hc.PR_REF)
def nine_case_test(agent_H, agent_L, env_test, rbb_system,
                   scenario=None, test_csv=False, output=None, seed=None,
                   phase_mode=0, phase_table=None, viol_derate_coef=None):
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

    _soc0_of, _soc1_of = {}, {}     # bi -> 块首/块末储存电量 [MWh]
    per_case = {i: {'r1': 0.0, 'c1': 0.0, 'r2': 0.0, 'c2': 0.0,
                    'cvm': 0, 'pf_viol': 0, 'profit': 0.0,
                    'gap_pen': 0.0, 'soc_track_err': 0.0,
                    'turbine_profit': 0.0, 'derate_loss': 0.0, 'penalty_NN_H': 0.0}
                for i in range(len(blocks))}
    df = pd.DataFrame()
    soc_target = SOC_TARGET_CENTER

    while step < num_iterations:
        if step % 5 == 0:
            action_H_raw = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H_raw)
            # 越界乘性系数 (与训练同口径; 系数<=0 时恒 1.0 ⇒ 旧口径逐位不变)
            _derate = violation_derate(action_H_raw, viol_derate_coef)
            pw_tru = env_test.PowerOutput(obs[1], action_H)
            soc_target = soc_target_from_signal(action_H[-1])

        ptur = float(pw_tru)
        pg_prev = float(obs[-1])
        E_current = float(obs[2])

        # ---- 下层 obs: 5 维规则原配料 (单一真源 hst.build_L_obs) ----
        _, rbb_info_obs = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=ptur, pg_prev=pg_prev, E=E_current)
        norm_obs_L = build_L_obs_distilled(obs, pw_tru, action_H, rbb_info_obs, 5 - (step % 5))
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
            turbine_derate=_derate,
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
            5 - (step % 5),
            pbrule=float(rbb_info['pbrule']),   # 放电退化罚 (L_DEG_COEF)
            return_info=True)

        bi = step // 5
        if step % 5 == 0:
            _soc0_of[bi] = float(obs[2])        # 本块决策时刻的储存电量
        _soc1_of[bi] = float(obs2[2])             # 末次写入 = 本块末储存电量
        pc = per_case[bi]
        pc['r1'] += r1
        pc['c1'] += c1
        pc['r2'] += r2
        pc['c2'] += c2
        pc['gap_pen'] += gap_pen
        # r1 分解两项 (供审计: profit/45 - r1 == (罚 + 乘性削减)/45):
        #   turbine_profit = **未打折**的风机收益 (与 details 同一个量 ⇒ 记录口径不变)
        #   derate_loss    = turbine_profit * (1 - f), f = exp(-c·x) 的乘性削减量
        # 注意两者都按 5 步求和 (与 pc['profit'] 同口径, 不除 n_per_case)。
        pc['turbine_profit'] += details['turbine_profit']
        pc['derate_loss'] += details['turbine_profit'] * (1.0 - _derate)
        pc['penalty_NN_H'] += penalty_NN_H
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
                'c2_slack': c2_info['slack'],          # 与训练路径同口径 (trackerr·g [MWh])
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
            # r1 分解 (r1 = turbine_profit·f + bess_profit − degradation − penalty_NN_H):
            'turbine_profit': pc['turbine_profit'],   # 未打折的风机收益 (5 步之和)
            'derate_loss': pc['derate_loss'],         # 乘性削减量 (5 步之和; --viol_derate 0 时为 0)
            'penalty_NN_H': pc['penalty_NN_H'],       # 线性越界罚 (5 步之和)
        })

    # ---- 存货结算: 逐 case 追加 (profit 是现金流口径, 不含储存电量变动) ----
    for _i, _m in enumerate(case_metrics):
        _s0, _s1 = _soc0_of[_i], _soc1_of[_i]
        _m['soc_start'], _m['soc_end'] = _s0, _s1
        _m['dsoc'] = _s1 - _s0
        _m['profit_econ'] = float(_m['profit']) + (_s1 - _s0) * SOC_SETTLE_PR
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
        # r1 分解 (45 步总量; 供审计: profit/45 - r1 == (penalty_NN_H + derate_loss)/45)
        'turbine_profit': float(sum(m['turbine_profit'] for m in case_metrics)),
        'derate_loss': float(sum(m['derate_loss'] for m in case_metrics)),
        'penalty_NN_H': float(sum(m['penalty_NN_H'] for m in case_metrics)),
    }
    # ---- 存货结算: 45 步总量口径 (逐 case 的 dsoc 望远镜求和 = 首末之差) ----
    _b0, _bN = min(_soc0_of), max(_soc1_of)
    overall['soc_start'], overall['soc_end'] = _soc0_of[_b0], _soc1_of[_bN]
    overall['dsoc'] = _soc1_of[_bN] - _soc0_of[_b0]
    overall['profit_econ'] = float(overall['profit']) + overall['dsoc'] * SOC_SETTLE_PR
    return case_metrics, overall
