#!/usr/bin/env python3
"""
RBB (Rule-Based BESS) — Test Script with Full test_detail Recording
====================================================================
Loads trained models and runs evaluation with complete RBB metric logging.

Records per step:
  - pb_original, pbrule (clipping before/after)
  - pbfluctmin, pbfluctmax (fluctuation smoothing range)
  - pbphysicsmin, pbphysicsmax (physical feasible range)
  - pbrulemin, pbrulemax (final feasible range)
  - clip_amount = |pb - pbrule| (total clipping amount)
  - unfeasible_gap = max(0, pbrulemin-pbrulemax) (no-intersection gap)
  - soc (per-step SOC)
  - All reward function sub-component values
"""

import os
import sys
import numpy as np
import argparse
import torch
import pandas as pd

# Ensure both rbb/ and parent (HSRL) directories are on path
_RBB_DIR = os.path.dirname(os.path.abspath(__file__))
_HSRL_DIR = os.path.dirname(_RBB_DIR)
sys.path.insert(0, _RBB_DIR)
sys.path.insert(0, _HSRL_DIR)

from env import WindBattery
from PER_ddpg import SP_DDPG
from util import *
from rbb_rule_bess import RuleBasedBESS
from rbb_myfunc import (
    CheckAction, Normalize, Original,
    Cal_TrackingErr_SOC, rbb_S_test,
)

import warnings
warnings.filterwarnings("ignore")

import matlab
import matlab.engine


def rbb_test_detailed(test_folder, env_test, rbb_system, args,
                      num_iterations=140, day=0):
    """
    Run RBB test with full test_detail recording.

    Parameters
    ----------
    test_folder : str
        Path to the trained model folder.
    env_test : WindBattery
        Test environment.
    rbb_system : RuleBasedBESS
        Rule-Based BESS system.
    args : argparse.Namespace
        Configuration arguments.
    num_iterations : int
        Number of test steps.
    day : int
        Test day index (1-4 for wind scenarios).

    Returns
    -------
    episode_reward : float
    episode_cost : float
    cvm : int
        Count of volatility minutes.
    df_detail : pd.DataFrame
        Full per-step detail DataFrame.
    """
    # Load agents
    agent_H = SP_DDPG(nb_states=4, nb_actions=6, args=args)
    agent_L = SP_DDPG(nb_states=5, nb_actions=1, args=args)
    agent_H.load_weights(test_folder)
    agent_L.load_weights(test_folder)

    agent_H.is_training = False
    agent_L.is_training = False

    # Initialize environment
    obs = env_test.TestReset(day)
    norm_obs = Normalize(obs)
    time = 1.0 / 60
    step = 0

    episode_reward = episode_cost = 0.0
    episode_r1 = episode_c1 = episode_r2 = episode_c2 = 0.0
    cvm = 0
    episode_action = list()

    # Detail recording
    detail_records = []
    df_detail = pd.DataFrame()

    print(f"=== RBB Test: {test_folder}, day={day}, steps={num_iterations} ===")

    while step < num_iterations:
        if step % 50 == 0:
            print(f"  step: {step}/{num_iterations}")

        # ---- Upper-level action (every 5 steps) ----
        if step % 5 == 0:
            action_H = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H)
            pw_tru = env_test.PowerOutput(obs[1], action_H)
            pb_l = list()
            sum_r1 = 0.0
            sum_c1 = 0.0

        # ---- Lower-level action ----
        norm_obs_L = np.append(norm_obs, action_H[-1] / 10)
        action_L = agent_L.select_action(norm_obs_L)
        action_L, penalty_NN_L = CheckAction(action_L)

        action = np.concatenate((action_H, action_L))

        # ======== RBB: Rule-Based BESS Clipping ========
        pb_original = float(action[-1])
        ptur = float(pw_tru)
        pg_prev = float(obs[-1])
        E_current = float(obs[2])

        pbrule, rbb_info = rbb_system.compute_pbrule(
            pb_original=pb_original,
            ptur=ptur,
            pg_prev=pg_prev,
            E=E_current,
        )

        action_rbb = action.copy()
        action_rbb[-1] = pbrule
        # ===============================================

        # Environment step
        pw_tru2, obs2, action_L_env, penalty_L, done, done_L, info = \
            env_test.TestStep(action_rbb[:6], action_rbb[6:], pw_tru)

        action_rbb[6:] = action_L_env
        episode_action.append(action_rbb)
        norm_obs2 = Normalize(obs2)

        # ---- RBB Rewards ----
        # Upper level (H)
        bessdeg = 490.0 * max(-pbrule, 0) * time  # BESS degradation cost, logged but no longer subtracted
        r1 = obs[0] * obs2[-1] * time - penalty_NN_H
        c1 = rbb_info['unfeasible_gap']
        sum_r1 += r1
        sum_c1 += c1

        if c1 > 0:
            cvm += 1

        # Lower level (L)
        tt = step % 5
        signal = float(action[5])
        degradation_cost = 4.9 * max(-pbrule, 0)
        clip_amount = rbb_info['clip_amount']
        r2 = -degradation_cost - clip_amount

        tracking_err, c2 = Cal_TrackingErr_SOC(
            signal=signal,
            E_current=E_current,
            tt=tt,
            pbrule=pbrule,
            time=time,
        )

        r = r1 + r2
        c = c1

        # ---- Record detail ----
        detail_records.append({
            'step': step,
            # Observations
            'pr': obs[0],
            'u': obs[1],
            'E': obs[2],
            'P_G_prev': obs[3],
            'pr_next': obs2[0],
            'u_next': obs2[1],
            'E_next': obs2[2],
            'P_G': obs2[3],
            # Actions
            'signal_soc_target': signal,
            'pb_original': pb_original,
            'pbrule': pbrule,
            'action_H_0': action_H[0],
            'action_H_1': action_H[1],
            'action_H_2': action_H[2],
            'action_H_3': action_H[3],
            'action_H_4': action_H[4],
            # RBB ranges
            'pbfluctmin': rbb_info['pbfluctmin'],
            'pbfluctmax': rbb_info['pbfluctmax'],
            'pbphysicsmin': rbb_info['pbphysicsmin'],
            'pbphysicsmax': rbb_info['pbphysicsmax'],
            'pbrulemin': rbb_info['pbrulemin'],
            'pbrulemax': rbb_info['pbrulemax'],
            # RBB metrics
            'clip_amount': clip_amount,
            'unfeasible_gap': rbb_info['unfeasible_gap'],
            'soc': rbb_info['soc'],
            # Rewards
            'r1': r1,
            'c1': c1,
            'r2': r2,
            'c2': c2,
            'r': r,
            'c': c,
            # Reward sub-components
            'revenue_H': obs[0] * obs2[-1] * time,
            'penalty_NN_H': penalty_NN_H,
            'degradation_cost': degradation_cost,
            'penalty_L': penalty_L,
            'tracking_err': tracking_err,
            # Turbine power
            'ptur': ptur,
            'pw_tru2': pw_tru2,
        })

        # Accumulate
        episode_reward += r
        episode_cost += c
        episode_r1 += r1
        episode_r2 += r2
        episode_c1 += c1
        episode_c2 += c2

        # State transition
        norm_obs = norm_obs2
        obs = obs2
        pw_tru = pw_tru2
        step += 1

    # Build DataFrame
    df_detail = pd.DataFrame(detail_records)

    # Print summary
    print(f"\n=== RBB Test Summary ===")
    print(f"  Total steps: {num_iterations}")
    print(f"  Episode r1 (upper profit):     {episode_r1:.4f}")
    print(f"  Episode c1 (unfeasible gap):   {episode_c1:.4f}")
    print(f"  Episode r2 (lower profit):     {episode_r2:.4f}")
    print(f"  Episode c2 (tracking dev):     {episode_c2:.4f}")
    print(f"  Episode reward (total):        {episode_reward:.4f}")
    print(f"  Episode cost (total):          {episode_cost:.4f}")
    print(f"  CVM (volatility minutes):      {cvm}")
    print(f"  Avg clip_amount:               {df_detail['clip_amount'].mean():.4f} MW")
    print(f"  Avg unfeasible_gap:            {df_detail['unfeasible_gap'].mean():.4f} MW")
    print(f"  Max unfeasible_gap:            {df_detail['unfeasible_gap'].max():.4f} MW")
    print(f"  Avg tracking_err:              {df_detail['tracking_err'].mean():.4f}")
    print(f"  Avg |pb_original|:             {df_detail['pb_original'].abs().mean():.4f} MW")
    print(f"  Avg |pbrule|:                  {df_detail['pbrule'].abs().mean():.4f} MW")

    agent_H.is_training = True
    agent_L.is_training = True

    return episode_reward, episode_cost, cvm, df_detail


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RBB Test with Detail Recording')

    parser.add_argument('--mode', default='test', type=str,
                        help='support option: train/test')
    parser.add_argument('--env', default='RBB-TEST', type=str,
                        help='environment name')
    parser.add_argument('--hidden1', default=16, type=int,
                        help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=100, type=int,
                        help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float,
                        help='policy net learning rate')
    parser.add_argument('--warmup', default=200, type=int, help='warmup steps')
    parser.add_argument('--discount', default=0.98, type=float, help='')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=20000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float,
                        help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int, help='')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter_H', default=200000, type=int, help='')
    parser.add_argument('--epsilon', default=50000, type=int, help='')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str,
                        help='Resuming model path for testing')
    parser.add_argument('--cuda', default=True, type=bool, help='Whether to use CUDA')
    parser.add_argument('--lam', default=0.1, type=float, help='safe constraint')
    parser.add_argument('--lr_lam', default=10e-3, type=float,
                        help='learning rate for lambda')
    parser.add_argument('--cost_single_limit', default=3, type=float,
                        help='cost_bound for single step')
    parser.add_argument('--PER_alpha', default=0.6, type=float, help='')
    parser.add_argument('--PER_beta', default=0.4, type=float, help='')
    parser.add_argument('--PER_epsilon', default=0.01, type=float, help='')
    parser.add_argument('--beta_increment_per_sampling', default=0., type=float, help='')

    # RBB-specific parameters
    parser.add_argument('--rbb_fluctuation_limit', default=3.0, type=float,
                        help='Max grid power fluctuation between steps (MW)')
    parser.add_argument('--rbb_rated_power', default=0.5, type=float,
                        help='BESS rated charge/discharge power (MW)')

    args = parser.parse_args()

    max_E = 0.50
    nb_turbine = 5
    upload = True
    args.max_episode_length = 500
    args.train_iter_H = 1000 * args.max_episode_length
    args.seed = np.random.randint(0, 10000)

    # Set device and seeds
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        print("Using GPU:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    # Start MATLAB engine
    engine = matlab.engine.start_matlab()

    # Test config
    day = 4
    test_folder = 'output/PER-run394'  # FIXME: update to RBB model path
    alg = 'DDPG-S-RBB'

    args.env = 'TEST-RBB-' + test_folder.split('-')[-1]
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    num_iterations = 140

    # Create environment
    env_test = WindBattery(max_E, nb_turbine, upload, engine)

    if args.seed > 0:
        np.random.seed(args.seed)

    # Create RBB system
    rbb_system = RuleBasedBESS(
        max_E=max_E,
        min_E_factor=0.2,
        rated_power=args.rbb_rated_power,
        eta_c=0.98,
        eta_d=0.98,
        time=1.0 / 60,
        fluctuation_limit=args.rbb_fluctuation_limit,
    )

    # Run detailed test
    episode_reward, episode_cost, cvm, df_detail = rbb_test_detailed(
        test_folder=test_folder,
        env_test=env_test,
        rbb_system=rbb_system,
        args=args,
        num_iterations=num_iterations,
        day=day,
    )

    # Save detailed results
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    df_detail.to_csv(f'{output_dir}/test_detail_rbb_day{day}.csv', index=False)
    print(f"\nTest detail saved to: {output_dir}/test_detail_rbb_day{day}.csv")
    print(f"Columns: {list(df_detail.columns)}")
