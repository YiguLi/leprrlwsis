"""
TD3PRLE — Dynamic Wind × Price Joint Segmentation (TD3 + LLM enhancement)
==========================================================================
Copy of prle_myfunc.py with the TD3PRRBB distilled L-observation addition.

Key difference from prrbb_myfunc.py: wind thresholds are dynamic
(mutable module-level state) rather than hardcoded constants.

Wind boundaries can be adjusted at runtime via set_wind_thresholds(),
and r1 rewards can be relabeled using relabel_r1_substep() with the
current threshold values.

TD3PRLE addition over prle_myfunc.py:
  - build_L_obs_distilled() — 5-dim distilled L observation
    [pbrulemin/10, pbrulemax/10, shadow_price, SOC/10, u/10] (from
    td3prrbb_myfunc.py); prrbb_S_test() and nine_case_test() use it.

Upper (H) r1:
  r1 = w_wind * (pr * ptur * time)     ← turbine profit, segmented
     + w_bess * (-pr * pbrule * time)   ← BESS profit, segmented
     - 490.0 * max(-pbrule, 0) * time   ← degradation, NO coefficient
     - penalty_NN_H

  Coefficients by (wind_level, price_level):
  ┌──────────┬───────────────┬───────────────┬───────────────┐
  │  Wind ↓  │  Price LOW    │  Price MEDIUM │  Price HIGH   │
  ├──────────┼───────────────┼───────────────┼───────────────┤
  │  LOW     │  WT:1.2 B:0.8 │  WT:1.2 B:1.0 │  WT:1.2 B:1.2 │
  │  MEDIUM  │  WT:1.0 B:0.8 │  WT:1.0 B:1.0 │  WT:1.2 B:1.2 │
  │  HIGH    │  WT:0.75 B:0.8│  WT:0.85 B:1.0│  WT:0.95 B:1.2│
  └──────────┴───────────────┴───────────────┴───────────────┘

  Wind boundaries: dynamic (default LOW < 7.0 m/s, HIGH >= 11.0 m/s)

Lower (L):
  r2 = 10.0 * shadow_profit - 100.0 * clip_amount
  c2 = 700.0 * soc_deviation   (SOC-safe 项进入拉格朗日约束, 保留原系数)
"""

import numpy as np
import pandas as pd
from copy import deepcopy

from util import *


# =============================================================================
# 0. Dynamic Wind Thresholds (mutable module-level state)
# =============================================================================

_wind_low_max  = 7.0   # u < _wind_low_max    → LOW wind
_wind_high_min = 11.0  # u >= _wind_high_min  → HIGH wind, else MEDIUM


def get_wind_thresholds():
    """Return current (wind_low_max, wind_high_min)."""
    return _wind_low_max, _wind_high_min


def set_wind_thresholds(low_max, high_min):
    """
    Set wind thresholds with validity clamping.

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


# =============================================================================
# 1. Price Classification (unchanged from PRRBB)
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
# 2. Wind Speed Classification (uses dynamic thresholds)
# =============================================================================

def get_wind_level(u):
    """Classify wind speed into LOW / MEDIUM / HIGH using current thresholds."""
    if u < _wind_low_max:
        return 'LOW'
    elif u >= _wind_high_min:
        return 'HIGH'
    else:
        return 'MEDIUM'


# =============================================================================
# 3. Joint Segmentation Coefficient Lookup
# =============================================================================
# HIGH wind (risk zone): ptur profit callback applied.
#   Low price → 0.75, Medium price → 0.85, High price → 0.95

# (wind_level, price_level) → (w_wind, w_bess)
SEG_COEFFS = {
    ('LOW',    'LOW'):    (1.2, 0.8),
    ('LOW',    'MEDIUM'): (1.2, 1.0),
    ('LOW',    'HIGH'):   (1.2, 1.2),
    ('MEDIUM', 'LOW'):    (1.0, 0.8),
    ('MEDIUM', 'MEDIUM'): (1.0, 1.0),
    ('MEDIUM', 'HIGH'):   (1.2, 1.2),
    ('HIGH',   'LOW'):    (0.75, 0.8),
    ('HIGH',   'MEDIUM'): (0.85, 1.0),
    ('HIGH',   'HIGH'):   (0.95, 1.2),
}


def get_seg_coeffs(wind_level, price_level):
    """Return (w_wind, w_bess) segmentation coefficients."""
    return SEG_COEFFS[(wind_level, price_level)]


# =============================================================================
# 4. Upper-Level (H): r1 with wind × price segmented profit decomposition
# =============================================================================

def Cal_r1_PRRBB(pr, ptur, pbrule, penalty_NN_H, wind_level, price_level,
                 time=1.0 / 60):
    """
    PRRBB upper-level r1 — decomposed PG profit with wind×price segmentation.

    r1 = w_wind * (pr * ptur * time)       ← turbine's grid contribution
       + w_bess * (-pr * pbrule * time)    ← BESS revenue (discharge → positive)
       - 490.0 * max(-pbrule, 0.0) * time  ← degradation (NO seg coefficient)
       - penalty_NN_H
    """
    w_wind, w_bess = get_seg_coeffs(wind_level, price_level)

    turbine_profit = w_wind * pr * ptur * time
    bess_profit    = w_bess * (-pr * pbrule * time)
    degradation    = 490.0 * max(-pbrule, 0.0) * time

    return turbine_profit + bess_profit - degradation - penalty_NN_H


# =============================================================================
# 4b. Reward Relabeling Helper (NEW for PRLE)
# =============================================================================

def relabel_r1_substep(u, pr, turbine_raw, bess_raw, degradation, penalty_NN_H):
    """
    Recompute r1 for a single sub-step using CURRENT wind thresholds.

    This is the core of reward relabeling: when wind thresholds change,
    this function reclassifies the wind level and recomputes the reward
    using the stored raw components (which are threshold-independent).

    Parameters
    ----------
    u : float
        Wind speed at this sub-step (m/s).
    pr : float
        Electricity price (RMB/MWh).
    turbine_raw : float
        pr * ptur * time  (threshold-independent turbine contribution).
    bess_raw : float
        -pr * pbrule * time  (threshold-independent BESS contribution).
    degradation : float
        490.0 * max(-pbrule, 0.0) * time  (threshold-independent).
    penalty_NN_H : float
        Action boundary penalty (threshold-independent).

    Returns
    -------
    r1 : float
        Recomputed reward under current wind thresholds.
    """
    wind_level = get_wind_level(u)
    price_level = get_price_level(pr)
    w_wind, w_bess = get_seg_coeffs(wind_level, price_level)
    return w_wind * turbine_raw + w_bess * bess_raw - degradation - penalty_NN_H


# c1 中 unfeasible_gap 的惩罚权重 (MW→成本系数, 与 PER/PRRBB 的 W_GAP 一致)
W_GAP = 50.0


def get_c1_risk_coeff(wind_level):
    """Risk-level coefficient on c1: HIGH wind ×1.2, LOW/MEDIUM wind ×0.8."""
    return 1.2 if wind_level == 'HIGH' else 0.8


def relabel_c1_substep(u, P_F, unfeasible_gap=0.0):
    """
    Recompute c1 for a single sub-step using CURRENT wind thresholds.

    c1 = c1_risk_coeff * (10.0 * max(0, P_F - 3.0) + 0.1 * P_F)
       + W_GAP * unfeasible_gap

    The base fluctuation penalty is threshold-independent (the 3.0 MW limit
    is hardcoded); only the risk coefficient depends on the wind threshold
    through get_wind_level(u). unfeasible_gap is also threshold-independent,
    so it is carried verbatim from the stored sub-step context. Relabeling
    keeps stored costs consistent with the cost signal under the current
    thresholds.

    Parameters
    ----------
    u : float
        Wind speed at this sub-step (m/s).
    P_F : float
        Grid power fluctuation |P_G(t) - P_G(t-1)| (MW).
    unfeasible_gap : float
        Infeasibility gap max(0, pbrulemin - pbrulemax) (MW), threshold-
        independent; read from the sub-step context (default 0.0 for
        contexts stored before the gap penalty was introduced).

    Returns
    -------
    c1 : float
        Recomputed fluctuation cost under current wind thresholds.
    """
    coeff = get_c1_risk_coeff(get_wind_level(u))
    if P_F > 3.0 + 1e-4:
        return coeff * (10.0 * (P_F - 3.0) + 0.1 * P_F) + W_GAP * unfeasible_gap
    return coeff * (0.1 * P_F) + W_GAP * unfeasible_gap


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
# 6. Combined Reward Function (forward computation during data collection)
# =============================================================================

def Cal_Rew_PRRBB(observation, observation2, action_rbb, rbb_info,
                   penalty_NN_H=0.0, time=1.0 / 60, ptur=None):
    """
    PRRBB reward: wind × price joint segmentation (upper);
    lower r2 = 10*shadow_profit - 100*clip_amount, c2 = 700*soc_deviation.

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
    w_wind, w_bess = get_seg_coeffs(wind_level, price_level)

    # ==== Upper Level (H) ====
    r1 = Cal_r1_PRRBB(pr_old, ptur, pbrule, penalty_NN_H,
                       wind_level, price_level, time)

    # c1: fluctuation penalty
    #   越限部分 10.0*(P_F-3.0) 仅在 P_F > 3.0 时生效；
    #   平滑项 0.1*P_F 对任意波动值 P_F（不仅限于越限值）都生效；
    #   W_GAP*unfeasible_gap: 波动∩物理约束不可行缺口 (阈值无关, 硬裁剪无法消除).
    # Risk-level coefficient on c1: HIGH wind ×1.2, LOW/MEDIUM wind ×0.8.
    c1_risk_coeff = get_c1_risk_coeff(wind_level)
    if P_F > 3.0 + 1e-4:
        c1 = c1_risk_coeff * (10.0 * (P_F - 3.0) + 0.1 * P_F) + W_GAP * unfeasible_gap
    else:
        c1 = c1_risk_coeff * (0.1 * P_F) + W_GAP * unfeasible_gap

    # ==== Lower Level (L) ====
    # r2: 影子套利利润 - 100*clip_amount（pbrule 裁剪惩罚进入 r2）
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
        'degradation': 490.0 * max(-pbrule, 0.0) * time,
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
    """TD3PRLE 蒸馏 L 观测 (5维): [pbrulemin/10, pbrulemax/10, shadow_price, SOC/10, u/10].

    与 TD3PER/TD3PRRBB 同步的五维蒸馏观测: 当前可行域 (RBB, 与 pb 无关) + H 影子价格
    信号 + 自身 SOC + 风速。u 使风速分段 c2 SOC 安全带 (SOC_SAFE_RANGES[wind_level])
    可观测 (u 同时决定 relabel 时的 wind_level 分类)。
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
    """TD3PRLE test with per-step detail recording (uses current dynamic thresholds)."""
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
                df.to_csv(f'{output}/test_s_td3prle.csv')
                np.save(f'{output}/test_s_td3prle.npy', data_list)

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
        df.to_csv(f'{output}/test_s_td3prle.csv')
        np.save(f'{output}/test_s_td3prle.npy', data_list)

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
    TD3PRLE 45-step 9-case in-training test (uses CURRENT dynamic thresholds).

    Designed case bands (fixed, 7/11): LOW < 7.0, MEDIUM [7.0, 11.0),
    HIGH >= 11.0. The reward is computed with the current dynamic wind
    thresholds; per-case 'project_wind_levels' records the actual
    project-side classification so threshold drift is visible.

    Returns
    -------
    case_metrics : list of dict (length 9)
        Per-case 5-step mean metrics + cvm + project_wind_levels;
        'profit' = the case's 5-step profit sum
        (profit_step = P_G*pr*dt - 4.9*max(-pbrule,0)*dt).
    overall : dict
        Mean over the 9 case means, plus current wind_low_max/wind_high_min;
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
                    'r': 0.0, 'c': 0.0, 'cvm': 0, 'profit': 0.0,
                    'project_winds': set()}
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
        pc['project_winds'].add(details['wind_level'])

        # profit: 上层利润 = P_G * pr * dt - 4.9*max(-pb,0)*dt（与训练 episode_profit 同口径）
        profit_step = obs2[-1] * obs[0] * time - 4.9 * max(-float(pbrule), 0.0) * time
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
        df.to_csv(f'{output}/test_s_td3prle_9case.csv', index=False)

    wind_low_max, wind_high_min = get_wind_thresholds()
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
            'project_wind_levels': sorted(pc['project_winds']),
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
        'wind_low_max': wind_low_max,
        'wind_high_min': wind_high_min,
        'profit': float(sum(m['profit'] for m in case_metrics)),  # 45 步 profit 之和
    }
    return case_metrics, overall
