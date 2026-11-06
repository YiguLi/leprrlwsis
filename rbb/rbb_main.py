#!/usr/bin/env python3
"""
RBB (Rule-Based BESS) — Main Training Loop
==========================================
Modified from PERv0 (PER_main0.py). All new RBB code in independent files.

Key changes from PERv0:
  1. RuleBasedBESS system clips pb → pbrule before env.step().
  2. Upper-level signal is now SOC target (was power budget).
  3. Upper-level r1: revenue - penalty_NN_H; c1: max(0, pbrulemin-pbrulemax).
  4. Lower-level r2: -(degradation) - clip_amount (no arbitrage; arbitrage via SOC tracking).
  5. Tracking deviation uses SOC-based formula (Option A).
  6. test_detail records all RBB metrics per step.
"""

import os
import numpy as np
import argparse
from copy import deepcopy
import torch

import sys
# Ensure both rbb/ and parent (HSRL) directories are on path
_RBB_DIR = os.path.dirname(os.path.abspath(__file__))
_HSRL_DIR = os.path.dirname(_RBB_DIR)
sys.path.insert(0, _RBB_DIR)
sys.path.insert(0, _HSRL_DIR)

from env import WindBattery
from PER_ddpg import SP_DDPG
from util import *
from rbb_myfunc import (
    CheckAction, Normalize, Original,
    Cal_TrackingErr_SOC, rbb_S_test,
)
from rbb_rule_bess import RuleBasedBESS

import pandas as pd
from memory import RingBuffer

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

import matlab
import matlab.engine


def train_rbb(num_iterations_H,
              agent_H,
              agent_L,
              env,
              env_test,
              rbb_system,
              output,
              max_episode_length=None,
              debug=False,
              upload=False,
              use_csv=False,
              engine=None):
    """
    RBB training loop with Rule-Based BESS integration.

    The RBB system clips the BESS agent's pb to pbrule before env.step(),
    ensuring that the actual BESS power satisfies both:
      - Grid fluctuation constraints (|pg(t) - pg(t-1)| <= 3 MW)
      - Physical SOC and rated power constraints

    Reward changes:
      Upper (H): r1 unchanged, c1 = max(0, pbrulemin-pbrulemax)
      Lower (L): r2 = -(degradation) - max(0, |pb-pbrule|), c2 = SOC tracking
    """
    agent_H.is_training = True
    agent_L.is_training = True
    step = episode = episode_steps = 0
    episode_reward = episode_cost = episode_profit = 0.0
    observation = None
    time = 1.0 / 60
    lamo_H = deepcopy(agent_H.lam)
    lamo_L = deepcopy(agent_L.lam)

    if use_csv:
        df = pd.DataFrame()
        df_episode = pd.DataFrame()
        df_test = pd.DataFrame()

    # ---- 1440-step window counters for env metrics ----
    window_violation_count = 0   # steps where |P_F| > fluctuation_limit
    window_power_sum = 0.0        # cumulative P_G over the window

    while step < num_iterations_H:
        # ---- Start of episode ----
        prYellow('########### step:{} ###########'.format(step))
        if observation is None:
            observation = deepcopy(env.reset())  # [Pr, U, E, P_last]
            norm_observation = Normalize(observation)
            agent_H.reset(norm_observation)

        # ================================================================
        # Upper Level (H): WT agent + SOC target signal
        # ================================================================
        if episode_steps % 5 == 0:
            sum_r1 = 0.0
            sum_c1 = 0.0
            if step <= args.warmup:
                applied_action_H = original_action_H = agent_H.random_action()
                penalty_NN_H = 0
            else:
                original_action_H = agent_H.select_action(norm_observation)
                applied_action_H, penalty_NN_H = CheckAction(original_action_H)
            norm_observation_H = norm_observation
            pw_tru = env.PowerOutput(observation[1], applied_action_H)

        prGreen('--H-- ori a_t = {}'.format(round_list(original_action_H, 3)))

        # ================================================================
        # Lower Level (L): BESS agent
        # ================================================================
        tt = episode_steps % 5
        norm_observation_L = np.append(norm_observation, [applied_action_H[-1] / 10, tt])
        prPurple('--L-- obs = {}.'.format(round_list(norm_observation_L, 3)))

        if episode_steps % 5 == 0:
            agent_L.reset(norm_observation_L)
            pb_l = list()

        if step <= args.warmup:
            applied_action_L = original_action_L = agent_L.random_action()
            penalty_NN_L = 0
        else:
            original_action_L = agent_L.select_action(norm_observation_L)
            applied_action_L, penalty_NN_L = CheckAction(original_action_L)
        prGreen('--L-- ori a_t = {}'.format(round_list(original_action_L, 3)))

        # Assemble full action
        action = np.concatenate((applied_action_H, applied_action_L))
        original_action = np.concatenate((original_action_H, original_action_L))

        # ================================================================
        # RBB: Rule-Based BESS Clipping
        # ================================================================
        pb_original = float(action[-1])
        ptur = float(pw_tru)
        pg_prev = float(observation[-1])    # P_G from end of previous step
        E_current = float(observation[2])    # Current SOC

        pbrule, rbb_info = rbb_system.compute_pbrule(
            pb_original=pb_original,
            ptur=ptur,
            pg_prev=pg_prev,
            E=E_current,
        )

        # Replace pb with pbrule for environment
        action_rbb = action.copy()
        action_rbb[-1] = pbrule
        # ================================================================

        # ---- Environment Step ----
        pw_tru2, observation2, action_L_env, penalty_L, done, done_L, info = \
            env.step(action_rbb[:6], action_rbb[6:], pw_tru)

        action_rbb[6:] = action_L_env
        prGreen('Applied action = {}.'.format(round_list(action_rbb, 3)))

        norm_observation2 = Normalize(observation2)
        tt2 = (episode_steps + 1) % 5
        norm_observation2_L = np.append(norm_observation2, [applied_action_H[-1] / 10, tt2])

        # ================================================================
        # RBB Rewards
        # ================================================================

        # --- Upper Level (H) ---
        # r1: revenue - penalty_NN_H
        bessdeg = 490.0 * max(-pbrule, 0) * time  # BESS degradation cost, logged but no longer subtracted
        r1 = observation[0] * observation2[-1] * time - penalty_NN_H

        # c1: unfeasible gap replaces H3/H4 fluctuation penalty
        c1 = rbb_info['unfeasible_gap']

        sum_r1 += r1
        sum_c1 += c1
        prGreen('r1 = {}'.format(round(r1, 3)))
        prGreen('c1 (unfeasible_gap) = {}'.format(round(c1, 3)))

        # ---- 1440-step window: track actual grid fluctuation ----
        P_F = abs(observation2[-1] - observation[-1])  # |pg(t) - pg(t-1)|
        if P_F > rbb_system.fluctuation_limit + 1e-4:
            window_violation_count += 1
        window_power_sum += observation2[-1]  # cumulative P_G

        # --- Lower Level (L) ---
        tt = episode_steps % 5
        signal = float(action[5])  # SOC target from upper agent
        pb_l.append(pb_original)

        # pbtargetreq: required avg power to hit SOC target in remaining steps
        remaining_steps = 5 - tt
        if remaining_steps > 0:
            pbtargetreq = (signal - E_current) / (remaining_steps * time)
        else:
            pbtargetreq = 0.0

        # Guidance penalty: one-sided, only penalize insufficient intent
        pbrulemin = rbb_info['pbrulemin']
        pbrulemax = rbb_info['pbrulemax']
        if pbrulemin <= pbtargetreq <= pbrulemax:
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

        # r2: degradation cost - clip penalty - guidance penalty
        degradation_cost = 4.9 * max(-pbrule, 0)       # 仅放电折旧
        clip_amount = rbb_info['clip_amount']
        r2 = -degradation_cost - clip_amount - guidance_penalty

        # c2: SOC-based tracking deviation (Option A, uses pbrule)
        tracking_err, c2 = Cal_TrackingErr_SOC(
            signal=signal,
            E_current=E_current,
            tt=tt,
            pbrule=pbrule,
            time=time,
        )

        # Overall reward and cost
        r = r1 + r2
        c = c1

        # ================================================================
        # Episode termination
        # ================================================================
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        if episode_steps % 5 == 4:
            # Upper agent observes cumulative r1, c1 over the 5-step block
            agent_H.observe_jump(
                norm_observation_H, original_action_H,
                sum_r1, sum_c1, norm_observation2, done
            )
            done_L = True

        # Lower agent observes per-step r2, c2
        agent_L.observe_jump(
            norm_observation_L, original_action_L,
            r2, c2, norm_observation2_L, done_L
        )

        # ================================================================
        # Update Policies
        # ================================================================
        if step > args.warmup:
            if episode_steps % 5 == 4:
                update_num = 0
                while update_num < 5:
                    agent_H.update_policy()
                    agent_L.update_policy()
                    update_num += 1

                # Lambda adjustment
                if abs(lamo_H - agent_H.lam) <= 0.1:
                    agent_H.cost_limit = agent_H.cost_limit / 2
                if abs(lamo_L - agent_L.lam) <= 0.1:
                    agent_L.cost_limit = agent_L.cost_limit / 2

                lamo_H = deepcopy(agent_H.lam)
                lamo_L = deepcopy(agent_L.lam)

            # Test evaluation
            if episode_steps % 50 == 0:
                r1_test, c1_test, r2_test, c2_test, r_test, c_test, action_test, cvm_test = \
                    rbb_S_test(agent_H, agent_L, env_test, rbb_system,
                               num_iterations=5, test_csv=False, output=output, day=0)

                if use_csv:
                    df_test = pd.concat([df_test, pd.DataFrame([{
                        'r1': r1_test, 'c1': c1_test,
                        'r2': r2_test, 'c2': c2_test,
                        'reward': r_test, 'cost': c_test,
                        'action': action_test,
                    }])], ignore_index=True)
                    if step % 50 == 0:
                        df_test.to_csv('{}/df_test_rbb.csv'.format(output))

                if upload:
                    writer.add_scalar('test/r1', r1_test, step)
                    writer.add_scalar('test/c1', c1_test, step)
                    writer.add_scalar('test/r2', r2_test, step)
                    writer.add_scalar('test/c2', c2_test, step)
                    writer.add_scalar('test/r', r_test, step)
                    writer.add_scalar('test/c', c_test, step)
                    writer.add_scalar('test/cvm', cvm_test, step)

        # ================================================================
        # Logging
        # ================================================================
        if use_csv:
            df = pd.concat([df, pd.DataFrame([{
                'step': step,
                'observation': observation,
                'observation2': observation2,
                'norm_observation_L': norm_observation_L,
                'original_action_H': original_action_H,
                'original_action_L': original_action_L,
                'action_original': original_action,
                'action_rbb': action_rbb,
                'r1': r1,
                'c1': c1,
                'r2': r2,
                'c2': c2,
                'reward': r,
                'cost': c,
                'lam1': agent_H.lam,
                'lam2': agent_L.lam,
                # RBB metrics
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
                'degradation_cost': degradation_cost,
                'bessdeg': bessdeg,
                'tracking_err': tracking_err,
                'signal': signal,
                'soctarget': signal,
                'penalty_NN_H': penalty_NN_H,
                'penalty_L': penalty_L,
                # Guidance metrics
                'pbtargetreq': pbtargetreq,
                'guidance_penalty': guidance_penalty,
            }])], ignore_index=True)
            if step % 50 == 0:
                df.to_csv('{}/df_rbb.csv'.format(output))

        # ---- Periodic checkpoint (every 2000 steps) ----
        if step > 0 and step % 2000 == 0:
            agent_H.save_checkpoint(output, step)
            agent_L.save_checkpoint(output, step)
            prCyan('Checkpoint saved at step {}'.format(step))

        if upload:
            writer.add_scalar('r1', r1, step)
            writer.add_scalar('c1', c1, step)
            writer.add_scalar('r2', r2, step)
            writer.add_scalar('c2', c2, step)
            writer.add_scalar('wind_speed', env.u, step)
            writer.add_scalar('reward', r, step)
            writer.add_scalar('cost', c, step)
            writer.add_scalar('lam1', agent_H.lam, step)
            writer.add_scalar('lam2', agent_L.lam, step)
            writer.add_scalar('d1', agent_H.cost_limit, step)
            writer.add_scalar('d2', agent_L.cost_limit, step)
            # RBB-specific metrics
            writer.add_scalar('rbb/clip_amount', rbb_info['clip_amount'], step)
            writer.add_scalar('rbb/unfeasible_gap', rbb_info['unfeasible_gap'], step)
            writer.add_scalar('rbb/pbrule', rbb_info['pbrule'], step)
            writer.add_scalar('rbb/pb_original', rbb_info['pb_original'], step)
            writer.add_scalar('rbb/soc', rbb_info['soc'], step)
            writer.add_scalar('rbb/bessdeg', bessdeg, step)
            if episode_steps % 5 == 4:
                writer.add_scalar('sum_r1', sum_r1, step)
                writer.add_scalar('sum_c1', sum_c1, step)

            # ---- Action logging (every 5 steps) ----
            if episode_steps % 5 == 4:
                # Upper action: 5 WT + 1 signal (SOC target)
                for i in range(5):
                    writer.add_scalar('action/a_H_original_{}'.format(i),
                                      original_action_H[i], step)
                    writer.add_scalar('action/a_H_clipped_{}'.format(i),
                                      applied_action_H[i], step)
                writer.add_scalar('action/signal_soc_target',
                                  applied_action_H[5], step)
                # PB: original + rule-clipped
                writer.add_scalar('action/pb_original', pb_original, step)
                writer.add_scalar('action/pbrule', pbrule, step)

            # ---- Env 1440-step window metrics ----
            if step > 0 and step % 1440 == 0:
                writer.add_scalar('env/window_violation_count',
                                  window_violation_count, step)
                writer.add_scalar('env/window_power_sum',
                                  window_power_sum, step)
                # Reset window counters
                window_violation_count = 0
                window_power_sum = 0.0

        # ================================================================
        # State transition
        # ================================================================
        step += 1
        episode_steps += 1
        episode_reward += r
        episode_cost += c
        episode_profit += r1 + r2

        norm_observation = deepcopy(norm_observation2)
        observation = deepcopy(observation2)
        pw_tru = pw_tru2

        # ================================================================
        # End of Episode
        # ================================================================
        if done:
            agent_H.save_model(output)
            agent_L.save_model(output)
            prGreen('##########{}: episode_profit:{} episode_reward:{} step:{}'.format(
                episode, round(episode_profit, 2), round(episode_reward, 2), step))

            if upload:
                writer.add_scalar('episode_reward', episode_reward, episode)
                writer.add_scalar('episode_profit', episode_profit, episode)
                writer.add_scalar('episode_cost', episode_cost, episode)
                writer.add_scalar('wind_day', env.j, episode)

            if use_csv:
                df_episode = pd.concat([df_episode, pd.DataFrame([{
                    'episode': episode,
                    'episode_reward': episode_reward,
                    'episode_profit': episode_profit,
                    'episode_cost': episode_cost,
                    'wind_day': env.j,
                }])], ignore_index=True)
                df_episode.to_csv('{}/df_episode_rbb.csv'.format(output))

            # Reset for next episode
            observation = None
            episode_steps = 0
            episode_reward = episode_cost = episode_profit = 0.0
            episode += 1

    if use_csv:
        df.to_csv('{}/df_rbb.csv'.format(output))
        df_episode.to_csv('{}/df_episode_rbb.csv'.format(output))
        df_test.to_csv('{}/df_test_rbb.csv'.format(output))


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RBB: Rule-Based BESS Training')

    parser.add_argument('--mode', default='train', type=str,
                        help='support option: train/test')
    parser.add_argument('--env', default='RBB', type=str,
                        help='environment name')
    parser.add_argument('--hidden1', default=16, type=int,
                        help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=100, type=int,
                        help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float,
                        help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=200, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.98, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=40000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float,
                        help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int,
                        help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int,
                        help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter_H', default=200000, type=int,
                        help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int,
                        help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str,
                        help='Resuming model path for testing')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Whether to use CUDA')
    parser.add_argument('--lam', default=3, type=float, help='safe constraint')
    parser.add_argument('--lr_lam', default=0.01, type=float,
                        help='learning rate for lambda')
    parser.add_argument('--cost_single_limit', default=0.5, type=float,
                        help='cost_bound for single step')
    parser.add_argument('--PER_alpha', default=0.6, type=float,
                        help='[0~1] convert the importance of TD error to priority')
    parser.add_argument('--PER_beta', default=0.4, type=float,
                        help='importance-sampling, from initial value increasing to 1')
    parser.add_argument('--PER_epsilon', default=0.01, type=float,
                        help='small amount to avoid zero priority')
    parser.add_argument('--beta_increment_per_sampling', default=0., type=float, help='')
    parser.add_argument('--nb_turbine', default=5, type=int,
                        help='number of turbines')

    # RBB-specific parameters
    parser.add_argument('--rbb_fluctuation_limit', default=3.0, type=float,
                        help='Max grid power fluctuation between steps (MW)')
    parser.add_argument('--rbb_rated_power', default=0.5, type=float,
                        help='BESS rated charge/discharge power (MW)')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    max_E = 0.50
    upload = True
    use_csv = True
    args.max_episode_length = 1400
    args.train_iter_H = 1000 * args.max_episode_length
    # 只在**未显式指定**（--seed 默认 -1）时随机化。显式 --seed >= 0 必须被尊重，
    # 否则"同 seed 配对 A/B"会被随机种子混淆（历史 bug：本行曾无条件覆盖用户传入的 --seed）。
    if args.seed is None or int(args.seed) < 0:
        args.seed = np.random.randint(0, 10000)
    args.cost_single_limit = 1000.0

    # Set device and random seeds
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
            torch.cuda.manual_seed_all(args.seed)

    if upload:
        writer = SummaryWriter(args.output)
        writer.add_scalar('seed', args.seed)

    # Start MATLAB engine
    engine = matlab.engine.start_matlab()

    # Create environments
    env = WindBattery(max_E, args.nb_turbine, upload, engine)
    env_test = WindBattery(max_E, args.nb_turbine, upload, engine)

    if args.seed > 0:
        np.random.seed(args.seed)

    # Create RBB system (BESS agent side)
    rbb_system = RuleBasedBESS(
        max_E=max_E,
        min_E_factor=0.2,
        rated_power=args.rbb_rated_power,
        eta_c=0.98,
        eta_d=0.98,
        time=1.0 / 60,
        fluctuation_limit=args.rbb_fluctuation_limit,
    )

    # Create agents
    agent_H = SP_DDPG(nb_states=4, nb_actions=6, args=args, lam_max=100.0)
    agent_L = SP_DDPG(nb_states=6, nb_actions=1, args=args, lam_max=500.0)

    if args.mode == 'train':
        train_rbb(args.train_iter_H,
                  agent_H,
                  agent_L,
                  env,
                  env_test,
                  rbb_system,
                  args.output,
                  max_episode_length=args.max_episode_length,
                  debug=args.debug,
                  upload=upload,
                  use_csv=use_csv,
                  engine=engine)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))

    if upload:
        writer.close()
