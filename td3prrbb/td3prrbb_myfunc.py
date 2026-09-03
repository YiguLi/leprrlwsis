"""
PRRBB — Wind × Price Segmented Reward (机会/风险分离版)
===================================================
Decomposes PG revenue into turbine (ptur) and BESS (pb) components.
机会侧（利润）用收益权重编码，风险侧（波动/平抑可行性/SOC）全部由 c1/c2 编码，
不再混入收益权重。

Upper (H) r1（机会侧，乘法分解）:
  r1 = wT * (pr * ptur * time)           ← 风机收益, wT = wT_u(u) × wT_p(pr)
     + wB * (-pr * pbrule * time)         ← BESS 收益, wB = wB_p(pr) × wB_soc(SOC,wind)
      - BESS_DEG_RMB_PER_MWH * max(-pbrule, 0) * time    ← 退化 (H 看 BESS 净收益)
      - penalty_NN_H

  Weight tables:
    wT_u:   LOW=1.2  MEDIUM=1.0  HIGH=0.9    (高风轻度降载倾向)
    wT_p:   LOW=1.0  MEDIUM=1.0  HIGH=1.05   (高价时段轻微加码发电)
    wB_p:   LOW=0.8  MEDIUM=1.0  HIGH=1.2    (低价时少罚充电→鼓励低谷充电)
    wB_soc: 带内(当前风速档安全带)=1.0, 带外=0.7 (SOC 贴边抑制套利)

  Wind boundaries: LOW < 7.0 m/s, MEDIUM [7.0, 11.0), HIGH >= 11.0 m/s

Upper (H) c1（风险侧）:
  c1 = r_u(u) × [10*(P_F-3)+ + 0.1*P_F] + W_GAP * unfeasible_gap
    r_u:   LOW=0.8  MEDIUM=1.0  HIGH=1.2
    W_GAP = 50.0  (平抑可行性缺口 unfeasible_gap 的直接代价, 单位 MW)

Lower (L):
  r2 = 10.0 * shadow_profit - 100.0 * clip_amount
    shadow_profit = -shadow_price * pbrule  (影子电价引导套利)
  c2 = 700.0 * soc_deviation   (SOC-safe 项进入拉格朗日约束, 保留原系数)
"""

import numpy as np
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
#   wB = wB_p(pr) × wB_soc(SOC, wind_level) — BESS 套利收益权重
# 风险全部进入 c1（波动+不可行缺口）与 c2（SOC 安全带），不再混入收益权重。

# 风机收益：低风鼓励多发（机会大、风险小），高风轻度降载倾向
WT_WIND = {'LOW': 1.2, 'MEDIUM': 1.0, 'HIGH': 0.9}
# 风机收益：高价时段轻微加码（降载的机会成本更大）
WT_PRICE = {'LOW': 1.0, 'MEDIUM': 1.0, 'HIGH': 1.05}
# BESS 收益：电价分档沿用原设计（低价时少罚充电→鼓励低谷充电）
WB_PRICE = {'LOW': 0.8, 'MEDIUM': 1.0, 'HIGH': 1.2}
# BESS 收益：SOC 贴边（超出当前风速档安全带）时压低套利权重
WB_SOC_OUTSIDE = 0.7

# c1 风险系数：三档（LOW≠MEDIUM≠HIGH）
C1_RISK = {'LOW': 0.8, 'MEDIUM': 1.0, 'HIGH': 1.2}
# unfeasible_gap（平抑可行性缺口, MW）进入 c1 的系数
W_GAP = 50.0


def get_wT(wind_level, price_level):
    """Turbine revenue weight: multiplicative wind × price decomposition."""
    return WT_WIND[wind_level] * WT_PRICE[price_level]


def get_wB(price_level, wind_level, SOC):
    """BESS revenue weight: price × SOC-margin decomposition.

    SOC-margin: 1.0 inside the wind-level safe band (SOC_SAFE_RANGES),
    WB_SOC_OUTSIDE otherwise (SOC 贴边时套利动作大概率被裁剪/不可行).
    """
    soc_min, soc_max = SOC_SAFE_RANGES[wind_level]
    w_soc = 1.0 if soc_min <= SOC <= soc_max else WB_SOC_OUTSIDE
    return WB_PRICE[price_level] * w_soc


# =============================================================================
# 4. Upper-Level (H): r1 with wind × price segmented profit decomposition
# =============================================================================

def Cal_r1_PRRBB(pr, ptur, pbrule, penalty_NN_H, wind_level, price_level,
                 SOC=3.0, time=1.0 / 60):
    """
    PRRBB upper-level r1 — 机会/风险分离的分段收益分解.

    r1 = wT * (pr * ptur * time)           ← 风机收益, wT = wT_u(u)×wT_p(pr)
       + wB * (-pr * pbrule * time)        ← BESS 收益, wB = wB_p(pr)×wB_soc(SOC,wind)
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
        Battery SOC at decision time (MWh), used by wB_soc margin check.
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

    return turbine_profit + bess_profit - degradation - penalty_NN_H


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
                   penalty_NN_H=0.0, time=1.0 / 60, ptur=None):
    """
    PRRBB reward: 机会/风险分离的分段奖励.
    Upper r1 = wT*pr*ptur*dt + wB*(-pr*pbrule*dt) - 退化 - penalty_NN_H
             （wT/wB 乘法分解: wT=wT_u×wT_p, wB=wB_p×wB_soc）
    Upper c1 = r_u(u)*[10*(P_F-3)+ + 0.1*P_F] + W_GAP*unfeasible_gap
    Lower r2 = 10*shadow_profit - 100*clip_amount
    Lower c2 = 700*soc_deviation

    Parameters
    ----------
    observation  : [pr, u, E, P_G_last]
    observation2 : [pr', u', E', P_G]
    action_rbb   : [a1..a5, shadow_price_signal, pbrule]
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

    shadow_price = float(action_rbb[5]) * 2.0  # (-0.5, 0.5) → (-1, 1)

    wind_level = get_wind_level(u)
    price_level = get_price_level(pr_old)
    SOC_decision = float(observation[2])  # 决策时刻 SOC，用于 wB_soc 贴边判断
    w_wind = get_wT(wind_level, price_level)
    w_bess = get_wB(price_level, wind_level, SOC_decision)

    # ==== Upper Level (H) ====
    r1 = Cal_r1_PRRBB(pr_old, ptur, pbrule, penalty_NN_H,
                       wind_level, price_level, SOC=SOC_decision, time=time)

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

    # ==== Lower Level (L) ====
    # r2: 影子套利利润 - 100*clip_amount (纯信号跟随, 无真实利润项)
    # c2: SOC 越界项进入拉格朗日约束（保留原系数 700/MWh）
    shadow_profit = -shadow_price * pbrule
    soc_dev = Cal_soc_deviation(SOC_new, wind_level)
    r2 = 10.0 * shadow_profit - 100.0 * clip_amount
    c2 = 700.0 * soc_dev

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
        'shadow_profit': shadow_profit,
        'shadow_price': shadow_price,
        'soc_deviation': soc_dev,
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

def CheckAction(action, charge=10.):
    """Clip actions to feasible ranges. WT: [0,0.5], Signal: [-0.5,0.5], Bat: [-charge,charge]."""
    penalty = 0
    SIGNAL_MIN, SIGNAL_MAX = -0.5, 0.5

    if len(action) == 7:
        for j in action[:5]:
            ma, mi = max(0.5, j), min(0, j)
            penalty += 200 *((ma - 0.5) + (0. - mi))
        sig = action[5]
        if sig > SIGNAL_MAX:    penalty += 200 *(sig - SIGNAL_MAX)
        elif sig < SIGNAL_MIN:  penalty += 200 *(SIGNAL_MIN - sig)
        for j in action[6:]:
            ma, mi = max(charge, j), min(-charge, j)
            penalty += 200 *((ma - charge) + (-charge - mi))
        a_H = np.append(np.clip(action[:5], 0., 0.5),
                        np.clip(action[5], SIGNAL_MIN, SIGNAL_MAX))
        a_L = np.clip(action[6:], -charge, charge)
        return np.concatenate((a_H, a_L)), penalty

    elif len(action) == 6:
        for j in action[:5]:
            ma, mi = max(0.5, j), min(0, j)
            penalty += 200 *((ma - 0.5) + (0. - mi))
        sig = action[5]
        if sig > SIGNAL_MAX:    penalty += 200 *(sig - SIGNAL_MAX)
        elif sig < SIGNAL_MIN:  penalty += 200 *(SIGNAL_MIN - sig)
        applied = np.append(np.clip(action[:5], 0., 0.5),
                            np.clip(action[5], SIGNAL_MIN, SIGNAL_MAX))
        return applied, penalty

    elif len(action) == 5:
        for j in action[:5]:
            ma, mi = max(0.5, j), min(0, j)
            penalty += 200 *((ma - 0.5) + (0. - mi))
        return np.clip(action[:5], 0., 0.5), penalty

    elif len(action) == 1:
        for j in action:
            ma, mi = max(charge, j), min(-charge, j)
            penalty += 200 *((ma - charge) + (-charge - mi))
        return np.clip(action, -charge, charge), penalty

    return action, penalty


# =============================================================================
# 8. Normalization (unchanged)
# =============================================================================

def Normalize(obs, max_E=6.):
    return np.array([obs[0] / 1000, obs[1] / 10, obs[2] / 10, obs[3] / 10])


def Original(obs, max_E=6.):
    return np.array([obs[0] * 1000, obs[1] * 10, obs[2] * 10, obs[3] * 10])


def build_L_obs_distilled(observation, pw_tru, applied_action_H, rbb_info):
    """TD3PRRBB 蒸馏 L 观测 (5维): [pbrulemin/10, pbrulemax/10, shadow_price, SOC/10, u/10].

    与 TD3PER 同步的五维蒸馏观测: 当前可行域 (RBB, 与 pb 无关) + H 影子价格
    信号 + 自身 SOC + 风速。u 使 PRRBB 风速分段 c2 SOC 安全带
    (SOC_SAFE_RANGES[wind_level]) 可观测。
    """
    shadow_price = float(applied_action_H[-1]) * 2.0  # (-0.5,0.5) → (-1,1)
    return np.array([
        rbb_info['pbrulemin'] / 10.0,
        rbb_info['pbrulemax'] / 10.0,
        shadow_price,
        float(observation[2]) / 10.0,
        float(observation[1]) / 10.0,  # u: 风速 (0~25 → 0~2.5)
    ])


# =============================================================================
# 9. Test Function
# =============================================================================

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

        # L 层观测蒸馏 (5维): [pbrulemin/10, pbrulemax/10, shadow_price, SOC/10, u/10]
        _, rbb_info_obs = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=float(pw_tru),
            pg_prev=float(obs[-1]), E=float(obs[2]))
        norm_obs_L = build_L_obs_distilled(obs, pw_tru, action_H, rbb_info_obs)
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
            ptur=ptur,
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
                'shadow_profit': details['shadow_profit'],
                'shadow_price': details['shadow_price'],
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


def nine_case_test(agent_H, agent_L, env_test, rbb_system,
                   scenario=None, test_csv=False, output=None, seed=None):
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
    norm_obs = Normalize(obs)
    time = 1.0 / 60
    agent_H.is_training = False
    agent_L.is_training = False
    step = 0
    num_iterations = len(wind_seq)
    n_per_case = num_iterations // len(blocks)

    per_case = {i: {'r1': 0.0, 'c1': 0.0, 'r2': 0.0, 'c2': 0.0,
                    'r': 0.0, 'c': 0.0, 'cvm': 0, 'profit': 0.0}
                for i in range(len(blocks))}
    df = pd.DataFrame()

    while step < num_iterations:
        if step % 5 == 0:
            action_H = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H)
            pw_tru = env_test.PowerOutput(obs[1], action_H)

        # L 层观测蒸馏 (5维): [pbrulemin/10, pbrulemax/10, shadow_price, SOC/10, u/10]
        _, rbb_info_obs = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=float(pw_tru),
            pg_prev=float(obs[-1]), E=float(obs[2]))
        norm_obs_L = build_L_obs_distilled(obs, pw_tru, action_H, rbb_info_obs)
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
        norm_obs2 = Normalize(obs2)

        r1, c1, r2, c2, details = Cal_Rew_PRRBB(
            obs, obs2, action_rbb, rbb_info,
            penalty_NN_H=penalty_NN_H, time=time,
            ptur=ptur,
        )

        bi = step // 5
        pc = per_case[bi]
        pc['r1'] += r1
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
            'profit': pc['profit'],  # 该 case 5 步 profit 之和
        })

    overall = {
        'r1': float(np.mean([m['r1'] for m in case_metrics])),
        'c1': float(np.mean([m['c1'] for m in case_metrics])),
        'r2': float(np.mean([m['r2'] for m in case_metrics])),
        'c2': float(np.mean([m['c2'] for m in case_metrics])),
        'r': float(np.mean([m['r'] for m in case_metrics])),
        'c': float(np.mean([m['c'] for m in case_metrics])),
        'cvm': int(sum(m['cvm'] for m in case_metrics)),
        'profit': float(sum(m['profit'] for m in case_metrics)),  # 45 步 profit 之和
    }
    return case_metrics, overall
