"""
RBB (Rule-Based BESS) — Reward Functions, Action Checking, and Test Utilities
=============================================================================
Modified from perv0 myfunc.py. All changes in independent functions/files.

Key changes from PERv0:
  Upper-level (H):
    r1: revenue - penalty_NN_H (bessdeg removed)
    c1: max(0, pbrulemin - pbrulemax)  -- replaces H3/H4 fluctuation penalty

  Lower-level (L):
    r2: -(degradation cost) - max(0, |pb - pbrule|)  -- removes penalty_L, penalty_NN_L
    c2: -exp(-tracking_err) + 1  -- SOC-based tracking error (Option A)

  Signal: changed from power budget to SOC target.
  Tracking error formula reconstructed for SOC target semantics.
"""

import numpy as np
import pandas as pd
from copy import deepcopy

from util import *


# =============================================================================
# Reward Functions
# =============================================================================

def Cal_Rew_RBB(observation, observation2, action_all, rbb_info, time=1.0 / 60):
    """
    Compute RBB-modified rewards for upper and lower levels.

    Parameters
    ----------
    observation : np.ndarray
        Previous observation [pr, u, E, P_G_last].
    observation2 : np.ndarray
        Current observation [pr', u', E', P_G].
    action_all : np.ndarray
        Full action [a1..a5, signal, pb] (pb is the ORIGINAL before RBB clipping).
    rbb_info : dict
        Output from RuleBasedBESS.compute_pbrule().
    time : float
        Time step in hours (default 1/60).

    Returns
    -------
    r1 : float
        Upper-level profit (per step): revenue - penalty_NN_H.
        Note: penalty_NN_H must be subtracted separately by the caller.
    c1 : float
        Upper-level cost (per step): max(0, pbrulemin - pbrulemax).
    r2 : float
        Lower-level profit (per step): -(degradation) - |pb - pbrule|.
    c2 : float
        Lower-level cost (per step): exponential tracking deviation.
    r1_detail : dict
        Sub-components of r1.
    c1_detail : dict
        Sub-components of c1.
    r2_detail : dict
        Sub-components of r2.
    c2_detail : dict
        Sub-components of c2.
    """
    pr = observation[0]
    p_g = observation2[-1]

    pb_original = rbb_info['pb_original']
    pbrule = rbb_info['pbrule']
    clip_amount = rbb_info['clip_amount']
    unfeasible_gap = rbb_info['unfeasible_gap']

    # --- Upper Level (H) ---
    # r1: revenue from grid power (penalty_NN_H subtracted by caller)
    revenue_H = pr * p_g * time
    bessdeg = 490.0 * max(-pbrule, 0) * time  # BESS degradation cost, logged but no longer subtracted
    r1 = revenue_H  # caller subtracts penalty_NN_H

    # c1: unfeasible gap replaces H3/H4 fluctuation penalty
    c1 = unfeasible_gap

    # --- Lower Level (L) ---
    # r2: degradation cost - clip penalty (no arbitrage; arbitrage via SOC tracking)
    degradation_cost = 4.9 * max(-pbrule, 0)  # only discharge degradation
    r2 = -degradation_cost - clip_amount

    # r1_detail and r2_detail for test_detail recording
    r1_detail = {
        'revenue_H': revenue_H,
        # penalty_NN_H is added by caller
    }
    c1_detail = {
        'unfeasible_gap': unfeasible_gap,
    }
    r2_detail = {
        'degradation_cost': degradation_cost,
        'clip_amount': clip_amount,
    }
    c2_detail = {}  # c2 computed separately with tracking_err

    return r1, c1, r2, None, r1_detail, c1_detail, r2_detail, c2_detail


def Cal_TrackingErr_SOC(signal, E_current, tt, pbrule, time=1.0 / 60):
    """
    Compute SOC-based tracking error (RBB Option A).

    Measures tracking quality using the actual executed power (pbrule after
    RBB clipping), NOT the agent's intended power (pb_original). This ensures
    the tracking cost reflects physical reality rather than unexecuted intent.

    Formula:
        tracking_err = abs((soc_target - E_current) / ((5-tt) * time) - pbrule)

    Parameters
    ----------
    signal : float
        SOC target from upper-level agent (in MWh, physical units).
    E_current : float
        Current battery SOC (MWh).
    tt : int
        Step index within the 5-step block (0-indexed, 0..4).
    pbrule : float
        Actual BESS power after RBB clipping (MW). This is what the
        environment actually executed.
    time : float
        Time step in hours (default 1/60).

    Returns
    -------
    tracking_err : float
        Absolute tracking error (MW).
    c2 : float
        Exponential tracking cost: -exp(-tracking_err) + 1.
    """
    remaining_steps = 5 - tt
    if remaining_steps <= 0:
        remaining_steps = 1  # safety

    # Required average power to reach SOC target from current SOC
    required_power = (signal - E_current) / (remaining_steps * time)
    tracking_err = abs(required_power - pbrule)
    c2 = -np.exp(-tracking_err) + 1.0

    return tracking_err, c2


def Eval_Rew_RBB(observation, observation2, action_all, rbb_info, time=1.0 / 60):
    """
    Evaluation version of RBB reward (for test scripts).
    Uses same structure as training reward but without exploration noise.
    """
    return Cal_Rew_RBB(observation, observation2, action_all, rbb_info, time)


# =============================================================================
# Action Checking (unchanged from PERv0)
# =============================================================================

def CheckAction(action, charge=0.5):
    """
    Clip actions to feasible ranges and compute penalty for out-of-bound violations.

    WT actions are clipped to [0, 0.5]. The SOC target signal is clipped to
    [1.2, 4.8] (min_E to 0.8*max_E). Battery actions are clipped to [-charge, charge].

    Parameters
    ----------
    action : np.ndarray
        Raw action from the agent.
    charge : float
        Maximum charge/discharge rate for battery action (MW).

    Returns
    -------
    applied_action : np.ndarray
        Clipped action.
    penalty : float
        Action boundary violation penalty.
    """
    penalty = 0

    SIGNAL_MIN, SIGNAL_MAX = 1.2, 4.8  # SOC target bounds: min_E to 0.8*max_E

    if len(action) == 7:
        # Full action: 5 WT + 1 signal + 1 battery
        for j in action[:5]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 50 * ((ma - 0.5) + (0. - mi))
        # Signal clipping
        sig = action[5]
        if sig > SIGNAL_MAX:
            penalty += 50 * (sig - SIGNAL_MAX)
        elif sig < SIGNAL_MIN:
            penalty += 50 * (SIGNAL_MIN - sig)
        for j in action[6:]:
            ma = max(charge, j)
            mi = min(-charge, j)
            penalty += 50 * ((ma - charge) + (-charge - mi))
        a_H = np.clip(action[:5], 0., 0.5)
        a_H = np.append(a_H, np.clip(sig, SIGNAL_MIN, SIGNAL_MAX))
        a_L = np.clip(action[6:], -charge, charge)
        applied_action = np.concatenate((a_H, a_L))

    elif len(action) == 6:
        # Upper-level action: 5 WT + 1 signal
        for j in action[:5]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 50 * ((ma - 0.5) + (0. - mi))
        # Signal clipping
        sig = action[5]
        if sig > SIGNAL_MAX:
            penalty += 50 * (sig - SIGNAL_MAX)
        elif sig < SIGNAL_MIN:
            penalty += 50 * (SIGNAL_MIN - sig)
        applied_action = np.clip(action[:5], 0., 0.5)
        applied_action = np.append(applied_action, np.clip(sig, SIGNAL_MIN, SIGNAL_MAX))

    elif len(action) == 5:
        # WT only
        for j in action[:5]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 50 * ((ma - 0.5) + (0. - mi))
        applied_action = np.clip(action[:5], 0., 0.5)

    elif len(action) == 11:
        for j in action[:10]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 50 * ((ma - 0.5) + (0. - mi))
        applied_action = np.clip(action[:10], 0., 0.5)
        applied_action = np.append(applied_action, action[10])

    elif len(action) == 10:
        for j in action[:10]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 50 * ((ma - 0.5) + (0. - mi))
        applied_action = np.clip(action[:10], 0., 0.5)

    elif len(action) == 1:
        # Battery only
        for j in action:
            ma = max(charge, j)
            mi = min(-charge, j)
            penalty += 50 * ((ma - charge) + (-charge - mi))
        applied_action = np.clip(action, -charge, charge)

    return applied_action, penalty


# =============================================================================
# Normalization (unchanged)
# =============================================================================

def Normalize(obs, max_E=0.5):
    """Normalize observation to [0, ~1] range."""
    pr = obs[0] / 1000  # price: 300, 580, 980 → ~[0.3, 1.0]
    u = obs[1] / 10     # wind speed
    e = obs[2] / 10     # SOC
    p = obs[3] / 10     # grid power
    return np.array([pr, u, e, p])


def Original(obs, max_E=0.5):
    """Denormalize observation back to physical units."""
    pr = obs[0] * 1000
    u = obs[1] * 10
    e = obs[2] * 10
    p = obs[3] * 10
    return np.array([pr, u, e, p])


# =============================================================================
# Test Functions with test_detail Recording
# =============================================================================

def rbb_S_test(agent_H, agent_L, env_test, rbb_system,
               num_iterations=5, test_csv=False, output=None, day=0):
    """
    RBB version of S_test with full test_detail recording.

    Records per step:
      - pb_original, pbrule, pbfluctmin/max, pbphysicsmin/max, pbrulemin/max
      - clip_amount, unfeasible_gap, soc
      - All reward sub-component values

    Parameters
    ----------
    agent_H : SP_DDPG
        Upper-level (H) agent.
    agent_L : SP_DDPG
        Lower-level (L) agent.
    env_test : WindBattery
        Test environment instance.
    rbb_system : RuleBasedBESS
        Rule-based BESS system instance.
    num_iterations : int
        Number of test steps.
    test_csv : bool
        Whether to save CSV/npy output.
    output : str
        Output directory path.
    day : int
        Test day index for wind data.

    Returns
    -------
    episode_r1, episode_c1, episode_r2, episode_c2 : float
        Episode cumulative rewards.
    episode_reward, episode_cost : float
        Total episode reward and cost.
    action : np.ndarray
        Last action taken.
    cvm : int
        Count of steps with positive cost (volatility measure).
    """
    obs = env_test.TestReset(day)
    norm_obs = Normalize(obs)
    time = 1.0 / 60
    agent_H.is_training = False
    agent_L.is_training = False
    step = 0
    episode_reward = episode_cost = 0.0
    episode_r1 = episode_c1 = episode_r2 = episode_c2 = 0.0
    episode_action = list()
    df = pd.DataFrame()
    data_list = []
    cvm = 0

    while step < num_iterations:
        if step % 10 == 0:
            print(f"step: {step}")

        # ---- Upper-level action (every 5 steps) ----
        if step % 5 == 0:
            action_H = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H)
            pw_tru = env_test.PowerOutput(obs[1], action_H)
            pb_l = list()
            sum_r1 = 0.0
            sum_c1 = 0.0

        # ---- Lower-level action ----
        tt_test = step % 5
        norm_obs_L = np.append(norm_obs, [action_H[-1] / 10, tt_test])
        action_L = agent_L.select_action(norm_obs_L)
        action_L, penalty_NN_L = CheckAction(action_L)

        # Assemble full action
        action = np.concatenate((action_H, action_L))

        # ======== RBB: Rule-Based BESS Clipping ========
        pb_original = float(action[-1])
        ptur = float(pw_tru)
        pg_prev = float(obs[-1])       # P_G from previous step
        E_current = float(obs[2])       # Current SOC

        pbrule, rbb_info = rbb_system.compute_pbrule(
            pb_original=pb_original,
            ptur=ptur,
            pg_prev=pg_prev,
            E=E_current,
        )

        # Replace pb with pbrule for environment step
        action_rbb = action.copy()
        action_rbb[-1] = pbrule
        # ===============================================

        # Environment step
        pw_tru2, obs2, action_L_env, penalty_L, done, done_L, info = \
            env_test.TestStep(action_rbb[:6], action_rbb[6:], pw_tru)

        # Update action with actual battery power from env
        action_rbb[6:] = action_L_env
        episode_action.append(action_rbb)

        norm_obs2 = Normalize(obs2)

        # ---- RBB Rewards ----
        # Upper-level (H)
        bessdeg = 490.0 * max(-pbrule, 0) * time  # BESS degradation cost, logged but no longer subtracted
        r1 = obs[0] * obs2[-1] * time - penalty_NN_H
        c1 = rbb_info['unfeasible_gap']  # max(0, pbrulemin - pbrulemax)

        # r2: degradation cost - clip penalty - guidance penalty
        degradation_cost = 4.9 * max(-pbrule, 0)
        clip_amount = rbb_info['clip_amount']

        # Tracking deviation (SOC-based, Option A, uses pbrule)
        tt = step % 5
        signal = float(action[5])  # SOC target from upper agent

        # pbtargetreq: required avg power to hit SOC target in remaining steps
        remaining_steps = 5 - tt
        if remaining_steps > 0:
            pbtargetreq = (signal - E_current) / (remaining_steps * time)
        else:
            pbtargetreq = 0.0

        # Guidance penalty: one-sided, only penalize insufficient intent
        if rbb_info['pbrulemin'] <= pbtargetreq <= rbb_info['pbrulemax']:
            delta_E = signal - E_current
            if delta_E > 0:
                # Need charge: pb_original >= pbtargetreq means intent is sufficient
                intent_sufficient = (pb_original >= pbtargetreq)
            else:
                # Need discharge: pb_original <= pbtargetreq (more negative) is sufficient
                intent_sufficient = (pb_original <= pbtargetreq)
            if intent_sufficient:
                guidance_penalty = 0.0
            else:
                guidance_penalty = abs(pbrule - pbtargetreq)
        else:
            guidance_penalty = 0.0

        r2 = -degradation_cost - clip_amount - guidance_penalty
        tracking_err, c2 = Cal_TrackingErr_SOC(
            signal=signal,
            E_current=E_current,
            tt=tt,
            pbrule=pbrule,
            time=time,
        )

        # Accumulate
        sum_r1 += r1
        sum_c1 += c1

        if c1 > 0:
            cvm += 1

        # ---- Record test_detail ----
        if test_csv:
            df = pd.concat([df, pd.DataFrame([{
                'step': step,
                'obs': obs,
                'norm_obs_L': norm_obs_L,
                'action': action_rbb,
                'action_L': action_L_env,
                'r1': r1,
                'sum_r1': sum_r1,
                'c1': c1,
                'r2': r2,
                'c2': c2,
                'r': r1 + r2,
                'c': c1,
                # RBB info
                'pb_original': rbb_info['pb_original'],
                'pbrule': rbb_info['pbrule'],
                'pbfluctmin': rbb_info['pbfluctmin'],
                'pbfluctmax': rbb_info['pbfluctmax'],
                'pbphysicsmin': rbb_info['pbphysicsmin'],
                'pbphysicsmax': rbb_info['pbphysicsmax'],
                'pbrulemin': rbb_info['pbrulemin'],
                'pbrulemax': rbb_info['pbrulemax'],
                'clip_amount': rbb_info['clip_amount'],
                'unfeasible_gap': rbb_info['unfeasible_gap'],
                'soc': rbb_info['soc'],
                # Reward sub-components
                'revenue_H': obs[0] * obs2[-1] * time,
                'penalty_NN_H': penalty_NN_H,
                'degradation_cost': degradation_cost,
                'penalty_L': penalty_L,
                'tracking_err': tracking_err,
                'signal': signal,
            }])], ignore_index=True)

            # npy data
            data = {
                'step': step,
                'obs': obs,
                'obs2': obs2,
                'done': done,
                'r1': r1,
                'r2': r2,
                'c1': c1,
                'c2': c2,
                'r': r1 + r2,
                'c': c1,
                # RBB info
                'rbb_info': rbb_info,
                'tracking_err': tracking_err,
                'signal': signal,
                'degradation_cost': degradation_cost,
                'clip_amount': clip_amount,
            }
            data_list.append(data)
            if step % 50 == 0:
                df.to_csv(f'{output}/test_s_rbb.csv')
                np.save(f'{output}/test_s_rbb.npy', data_list)

        episode_reward += (r1 + r2)
        episode_cost += c1
        episode_r1 += r1
        episode_r2 += r2
        episode_c1 += c1
        episode_c2 += c2

        # Update state for next step
        norm_obs = norm_obs2
        obs = obs2
        pw_tru = pw_tru2
        step += 1

    agent_H.is_training = True
    agent_L.is_training = True
    if test_csv:
        df.to_csv(f'{output}/test_s_rbb.csv')
        np.save(f'{output}/test_s_rbb.npy', data_list)

    return (episode_r1, episode_c1, episode_r2, episode_c2,
            episode_reward, episode_cost, action_rbb, cvm)
