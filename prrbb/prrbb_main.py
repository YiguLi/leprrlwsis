#!/usr/bin/env python3
"""
PRRBB (PER + Segmented Reward) — Main Training Loop
=====================================================
Modified from PER_main0.py. Uses state-dependent segmented reward functions
from prrbb_myfunc.py instead of flat reward functions.

Key changes from PERv0:
  1. Reward computation uses Cal_Rew_PRRBB() with regime-dependent weights
  2. Upper-level r1: augmented revenue - SOC-scaled degradation - penalty_NN_H
  3. Upper-level c1: unfeasible_gap + SOC danger zone penalty
  4. Lower-level r2: dynamic w_soc * R_soc_safety + w_shadow * R_shadow_follow
  5. Lower-level c2: Tier 1 hard constraints (fluctuation + feasibility)
  6. All regime classifications and dynamic weights are logged

Preserved from PERv0:
  - Shadow price signal convention: range (-0.5, 0.5)
  - RBB hard clipping before env.step (via RuleBasedBESS)
  - SP-DDPG agent architecture
  - 5-step block structure for upper/lower level interaction
"""

import os
import numpy as np
import argparse
from copy import deepcopy
import torch

import sys
# Ensure both prrbb/ and parent (HSRL) directories are on path
_PRRBB_DIR = os.path.dirname(os.path.abspath(__file__))
_HSRL_DIR = os.path.dirname(_PRRBB_DIR)
sys.path.insert(0, _PRRBB_DIR)
sys.path.insert(0, _HSRL_DIR)

from env import WindBattery
from PER_ddpg import SP_DDPG
from util import *
from prrbb_myfunc import (
    CheckAction, Normalize, Original,
    Cal_Rew_PRRBB, nine_case_test,
    get_price_level,
)
from prrbb_rule_bess import RuleBasedBESS

import pandas as pd
from memory import RingBuffer

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

import matlab
import matlab.engine


def train_prrbb(num_iterations_H,
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
    PRRBB training loop with segmented reward functions.

    Reward structure (机会/风险分离版, prrbb_myfunc.py):
      Upper (H): r1 = wT*(pr*ptur*time) + wB*(-pr*pbrule*time)
                      - 490*max(-pbrule,0)*time - penalty_NN_H
                      wT=wT_u(u)*wT_p(pr), wB=wB_p(pr)*wB_soc(SOC,wind)
                 c1 = r_u(u)*(10*max(0,P_F-3) + 0.1*P_F) + 50*unfeasible_gap
      Lower (L): r2 = 10*shadow_profit - 100*clip_amount + 0.5*real_profit
                 c2 = 700*soc_deviation  (SOC-safe 项, 拉格朗日约束)

    Lagrange multiplier updates:
      Upper (H): EMA dual-ascent on ACTUAL block cost (EMA 方案)
                 EMA   ← (1-0.02)*EMA + 0.02*sum_c1          (每 5 步块)
                 lam1  ← clip(lam1 + 0.01*(EMA - d_block), 3, 20), 初始 5
                 d_block 初始 1.2, curriculum 连续满足则减半, 地板 0.3
                 (每MW量纲对齐: λ≥2.55 时 c1 惩罚≥r1 收益)
      Lower (L): critic-based (unchanged): lam2 ∈ [1, 10],
                 cost_limit 减半地板 0.0
    """
    agent_H.is_training = True
    agent_L.is_training = True
    step = episode = episode_steps = 0
    episode_reward = episode_cost = episode_cost2 = episode_profit = 0.0
    episode_bess_profit = 0.0
    observation = None
    time = 1.0 / 60
    lamo_L = deepcopy(agent_L.lam)

    # Curriculum learning trackers
    consecutive_satisfied_H = 0
    consecutive_satisfied_L = 0

    # ---- H-layer EMA dual-ascent (actual-cost based lam update) ----
    d_block_H = 1.2          # 每块 c1 均值的约束目标（初始）
    d_block_H_min = 0.3      # curriculum 收紧地板

    if use_csv:
        df = pd.DataFrame()
        df_episode = pd.DataFrame()
        df_test = pd.DataFrame()

    # ---- 1440-step window counters for env metrics ----
    window_violation_count = 0
    window_power_sum = 0.0

    while step < num_iterations_H:
        # ---- Start of episode ----
        prYellow('########### step:{} ###########'.format(step))
        if observation is None:
            observation = deepcopy(env.reset())  # [Pr, U, E, P_last]
            norm_observation = Normalize(observation)
            agent_H.reset(norm_observation)

        # ================================================================
        # Upper Level (H): WT agent + shadow price signal
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
        norm_observation_L = np.append(norm_observation, applied_action_H[-1])
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
        original_action = action.copy()

        # ================================================================
        # RBB: Rule-Based BESS Clipping (hard clipping before env.step)
        # ================================================================
        pb_original = float(action[-1])
        ptur = float(pw_tru)
        pg_prev = float(observation[-1])
        E_current = float(observation[2])

        pbrule, rbb_info = rbb_system.compute_pbrule(
            pb_original=pb_original,
            ptur=ptur,
            pg_prev=pg_prev,
            E=E_current,
        )

        # Clip pb to pbrule before passing to environment
        action_rbb = action.copy()
        action_rbb[-1] = pbrule

        # ---- Environment Step ----
        pw_tru2, observation2, action_L_env, penalty_L, done, done_L, info = \
            env.step(action_rbb[:6], action_rbb[6:], pw_tru)

        action_rbb[6:] = action_L_env
        prGreen('Applied action = {}.'.format(round_list(action_rbb, 3)))

        norm_observation2 = Normalize(observation2)
        norm_observation2_L = np.append(norm_observation2, applied_action_H[-1])

        # ================================================================
        # PRRBB Segmented Reward Computation
        # ================================================================
        r1, c1, r2, c2, reward_details = Cal_Rew_PRRBB(
            observation, observation2, action_rbb, rbb_info,
            penalty_NN_H=penalty_NN_H, time=time,
            ptur=ptur,
        )

        # Total reward and cost (used for H's r1 and c1; L uses r2, c2)
        r = r1 + r2
        c = c1

        sum_r1 += r1
        sum_c1 += c1

        # Track shadow price for logging
        shadow_price = reward_details['shadow_price']

        # ---- Track BESS original intent vs actual ----
        pb_l.append(pb_original)

        prGreen('r1 = {} (turb={}, bess={}, deg={})'.format(
            round(r1, 3), round(reward_details['turbine_profit'], 3),
            round(reward_details['bess_profit'], 3),
            round(reward_details['degradation'], 3)))
        prGreen('c1 = {} (fluct={})'.format(
            round(c1, 3), round(reward_details['c1_fluct'], 3)))
        prGreen('r2 = {} (shadow_profit={}, clip={})'.format(
            round(r2, 3), round(reward_details['shadow_profit'], 3),
            round(reward_details['clip_amount'], 3)))
        prGreen('c2 = {} (soc_dev={})'.format(
            round(c2, 3), round(reward_details['soc_deviation'], 3)))
        prGreen('price={} p_level={} wind={:.1f} w_level={} (wT={}, wB={})'.format(
            reward_details['pr'], reward_details['price_level'],
            reward_details['u'], reward_details['wind_level'],
            reward_details['w_wind'], reward_details['w_bess']))

        # ---- 1440-step window: track actual grid fluctuation ----
        if reward_details['P_F'] > 3.0 + 1e-4:
            window_violation_count += 1
        window_power_sum += observation2[-1]

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
                while update_num < 10:
                    agent_H.update_policy()
                    agent_L.update_policy()
                    update_num += 1

                    # Track consecutive satisfaction for curriculum learning
                    # H: EMA 方案 — 判据为实际成本 EMA ≤ d_block
                    if agent_H.ema_cost is not None and agent_H.ema_cost <= d_block_H:
                        consecutive_satisfied_H += 1
                    else:
                        consecutive_satisfied_H = 0
                    if agent_L.lam_loss < 0:
                        consecutive_satisfied_L += 1
                    else:
                        consecutive_satisfied_L = 0

                # Curriculum convergence: tighten constraint targets when
                # satisfied for 100 consecutive policy updates.
                #   H (EMA): halve d_block, floor d_block_H_min
                #   L (critic): halve cost_limit, floor 0.0
                if consecutive_satisfied_H >= 100:
                    d_block_H = max(d_block_H / 2, d_block_H_min)
                    consecutive_satisfied_H = 0
                    prGreen('H: d_block halved to {}'.format(d_block_H))
                if consecutive_satisfied_L >= 100:
                    agent_L.cost_limit = max(agent_L.cost_limit / 2, 0.0)
                    consecutive_satisfied_L = 0
                    prGreen('L: cost_limit halved to {}'.format(agent_L.cost_limit))

                # ---- H-layer: EMA dual ascent on actual block cost ----
                # 每 5 步块一次：EMA ← (1-α)EMA + α·sum_c1，随后
                # lam1 ← clip(lam1 + η·(EMA − d_block), 0.1, 10)
                agent_H.update_ema_cost(sum_c1)
                agent_H.update_lam_ema(d_block_H)

                # Lambda adjustment (backup mechanism, L only)
                if abs(lamo_L - agent_L.lam) <= 0.1:
                    agent_L.cost_limit = agent_L.cost_limit / 2

                lamo_L = deepcopy(agent_L.lam)

            # Test evaluation: 9-case segmented test every 480 episode steps
            if episode_steps % 480 == 0:
                case_metrics, overall = nine_case_test(
                    agent_H, agent_L, env_test, rbb_system)

                if use_csv:
                    df_test = pd.concat([df_test, pd.DataFrame([{
                        'step': step,
                        'wind_level': m['wind_level'],
                        'price_level': m['price_level'],
                        'r1': m['r1'], 'c1': m['c1'],
                        'r2': m['r2'], 'c2': m['c2'],
                        'reward': m['r'], 'cost': m['c'],
                        'cvm': m['cvm'],
                        'profit': m['profit'],
                    } for m in case_metrics])], ignore_index=True)
                    # 附加一行 OVERALL：45 步 profit 之和等整体指标
                    df_test = pd.concat([df_test, pd.DataFrame([{
                        'step': step,
                        'wind_level': 'OVERALL',
                        'price_level': 'OVERALL',
                        'r1': overall['r1'], 'c1': overall['c1'],
                        'r2': overall['r2'], 'c2': overall['c2'],
                        'reward': overall['r'], 'cost': overall['c'],
                        'cvm': overall['cvm'],
                        'profit': overall['profit'],
                    }])], ignore_index=True)
                    df_test.to_csv('{}/df_test9.csv'.format(output))

                if upload:
                    writer.add_scalar('test9/overall/r1', overall['r1'], step)
                    writer.add_scalar('test9/overall/c1', overall['c1'], step)
                    writer.add_scalar('test9/overall/r2', overall['r2'], step)
                    writer.add_scalar('test9/overall/c2', overall['c2'], step)
                    writer.add_scalar('test9/overall/r', overall['r'], step)
                    writer.add_scalar('test9/overall/c', overall['c'], step)
                    writer.add_scalar('test9/overall/cvm', overall['cvm'], step)
                    writer.add_scalar('test9/overall/profit', overall['profit'], step)
                    for m in case_metrics:
                        tag = 'test9/{}_{}'.format(m['wind_level'], m['price_level'])
                        writer.add_scalar(tag + '/r1', m['r1'], step)
                        writer.add_scalar(tag + '/c1', m['c1'], step)
                        writer.add_scalar(tag + '/r2', m['r2'], step)
                        writer.add_scalar(tag + '/c2', m['c2'], step)
                        writer.add_scalar(tag + '/r', m['r'], step)
                        writer.add_scalar(tag + '/c', m['c'], step)
                        writer.add_scalar(tag + '/cvm', m['cvm'], step)
                        writer.add_scalar(tag + '/profit', m['profit'], step)

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
                'r1': r1, 'c1': c1, 'r2': r2, 'c2': c2,
                'reward': r, 'cost': c,
                'lam1': agent_H.lam, 'lam2': agent_L.lam,
                'd_block_H': d_block_H,
                'ema_c1_H': agent_H.ema_cost,
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
                'is_infeasible': float(rbb_info['is_infeasible']),
                'inevitable_violation': rbb_info['inevitable_violation'],
                # PRRBB price segmentation info
                'wind_level': reward_details['wind_level'],
                'price_level': reward_details['price_level'],
                'w_wind': reward_details['w_wind'],
                'w_bess': reward_details['w_bess'],
                'pr': reward_details['pr'],
                'u': reward_details['u'],
                # PRRBB reward sub-components
                'turbine_profit': reward_details['turbine_profit'],
                'bess_profit': reward_details['bess_profit'],
                'revenue': reward_details['revenue'],
                'degradation': reward_details['degradation'],
                'soc_deviation': reward_details['soc_deviation'],
                'shadow_profit': reward_details['shadow_profit'],
                'shadow_price': reward_details['shadow_price'],
                'c1_fluct': reward_details['c1_fluct'],
                'clip_amount': reward_details['clip_amount'],
                'unfeasible_gap': reward_details['unfeasible_gap'],
                'penalty_NN_H': penalty_NN_H,
                'SOC': reward_details['SOC'],
                'P_F': reward_details['P_F'],
                'penalty_L': penalty_L,
                'penalty_NN_L': penalty_NN_L,
                # Convenience
                'bess_profit': reward_details['shadow_profit'] - reward_details['degradation'],
            }])], ignore_index=True)
            if step % 50 == 0:
                df.to_csv('{}/df_prrbb.csv'.format(output))

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
            writer.add_scalar('d1', d_block_H, step)
            writer.add_scalar('d2', agent_L.cost_limit, step)
            if agent_H.ema_cost is not None:
                writer.add_scalar('ema_c1_H', agent_H.ema_cost, step)
            writer.add_scalar('soc', float(observation[2]), step)
            writer.add_scalar('P_F', reward_details['P_F'], step)

            # PRRBB wind×price segmentation
            writer.add_scalar('seg/price_level',
                              {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(reward_details['price_level'], 0),
                              step)
            writer.add_scalar('seg/wind_level',
                              {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(reward_details['wind_level'], 0),
                              step)
            writer.add_scalar('seg/pr', reward_details['pr'], step)
            writer.add_scalar('seg/u', reward_details['u'], step)
            writer.add_scalar('seg/w_wind', reward_details['w_wind'], step)
            writer.add_scalar('seg/w_bess', reward_details['w_bess'], step)

            # PRRBB reward sub-components (upper): decomposed profit
            writer.add_scalar('reward_upper/turbine_profit', reward_details['turbine_profit'], step)
            writer.add_scalar('reward_upper/bess_profit', reward_details['bess_profit'], step)
            writer.add_scalar('reward_upper/degradation', reward_details['degradation'], step)

            # PRRBB reward sub-components (lower)
            writer.add_scalar('reward_lower/shadow_profit', reward_details['shadow_profit'], step)
            writer.add_scalar('reward_lower/soc_deviation', reward_details['soc_deviation'], step)
            writer.add_scalar('reward_lower/shadow_price', shadow_price, step)
            writer.add_scalar('reward_lower/clip_amount', reward_details['clip_amount'], step)

            # RBB-specific metrics
            writer.add_scalar('rbb/clip_amount', rbb_info['clip_amount'], step)
            writer.add_scalar('rbb/unfeasible_gap', rbb_info['unfeasible_gap'], step)
            writer.add_scalar('rbb/pbrule', float(pbrule), step)
            writer.add_scalar('rbb/pb_original', pb_original, step)
            writer.add_scalar('rbb/is_infeasible', float(rbb_info['is_infeasible']), step)
            writer.add_scalar('rbb/inevitable_violation', rbb_info['inevitable_violation'], step)

            if episode_steps % 5 == 4:
                writer.add_scalar('sum_r1', sum_r1, step)
                writer.add_scalar('sum_c1', sum_c1, step)

            # ---- Action logging (every 5 steps) ----
            if episode_steps % 5 == 4:
                for i in range(5):
                    writer.add_scalar('action/a_H_original_{}'.format(i),
                                      original_action_H[i], step)
                    writer.add_scalar('action/a_H_clipped_{}'.format(i),
                                      applied_action_H[i], step)
                writer.add_scalar('action/shadow_price_signal',
                                  applied_action_H[5], step)
                writer.add_scalar('action/pb_original', pb_original, step)
                writer.add_scalar('action/pbrule', float(pbrule), step)

            # ---- Env 1440-step window metrics ----
            if step > 0 and step % 1440 == 0:
                writer.add_scalar('env/window_violation_count',
                                  window_violation_count, step)
                writer.add_scalar('env/window_power_sum',
                                  window_power_sum, step)
                window_violation_count = 0
                window_power_sum = 0.0

        # ================================================================
        # State transition
        # ================================================================
        step += 1
        episode_steps += 1
        episode_reward += r
        episode_cost += c
        episode_cost2 += c2
        episode_profit += observation2[-1] * observation[0] * time \
            - 4.9 * max(-float(pbrule), 0) * time
        episode_bess_profit += reward_details['shadow_profit'] \
            - 4.9 * max(-float(pbrule), 0) * time

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
                writer.add_scalar('episode_cost', episode_cost, episode)
                writer.add_scalar('episode_cost2', episode_cost2, episode)
                writer.add_scalar('episode_profit', episode_profit, episode)
                writer.add_scalar('episode_bess_profit', episode_bess_profit, episode)
                writer.add_scalar('wind_day', env.j, episode)

            if use_csv:
                df_episode = pd.concat([df_episode, pd.DataFrame([{
                    'episode': episode,
                    'episode_reward': episode_reward,
                    'episode_cost': episode_cost,
                    'episode_cost2': episode_cost2,
                    'episode_profit': episode_profit,
                    'bess_profit': episode_bess_profit,
                    'wind_day': env.j,
                }])], ignore_index=True)
                df_episode.to_csv('{}/df_episode_prrbb.csv'.format(output))

            # Reset for next episode
            observation = None
            episode_steps = 0
            episode_reward = episode_cost = episode_cost2 = episode_profit = 0.0
            episode_bess_profit = 0.0
            episode += 1

    if use_csv:
        df.to_csv('{}/df_prrbb.csv'.format(output))
        df_episode.to_csv('{}/df_episode_prrbb.csv'.format(output))
        df_test.to_csv('{}/df_test9.csv'.format(output))


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PRRBB: PER + Segmented Reward Training')

    parser.add_argument('--mode', default='train', type=str,
                        help='support option: train/test')
    parser.add_argument('--env', default='PRRBB', type=str,
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
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=20000, type=int, help='memory size')
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
    parser.add_argument('--lr_lam', default=0.1, type=float,
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

    # RBB parameters
    parser.add_argument('--rbb_fluctuation_limit', default=3.0, type=float,
                        help='Max grid power fluctuation between steps (MW)')
    parser.add_argument('--rbb_rated_power', default=10.0, type=float,
                        help='BESS rated charge/discharge power (MW)')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    max_E = 6.0
    upload = True
    use_csv = True
    args.max_episode_length = 1440  # 1 day = 24h * 60m = 1440 steps
    args.train_iter_H = 1000 * args.max_episode_length
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

    # Create RBB system (hard clipping before env.step)
    rbb_system = RuleBasedBESS(
        max_E=max_E,
        min_E_factor=0.0,
        rated_power=args.rbb_rated_power,
        eta_c=0.98,
        eta_d=0.98,
        time=1.0 / 60,
        fluctuation_limit=args.rbb_fluctuation_limit,
    )

    # Create agents (PER convention: H=6 dims, L=5 dims)
    # H-layer: EMA dual-ascent on actual c1 (α=0.02, η=0.01)
    #          lam1 ∈ [3, 20], 初始 5.0
    #          (每MW量纲对齐核算: 需 λ ≥ 2.55 保证任何风速×电价下惩罚≥利润)
    agent_H = SP_DDPG(nb_states=4, nb_actions=6, args=args,
                      lam_max=20.0, lam_min=3.0,
                      ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01)
    agent_H.lam = 5.0  # lam1 初始值
    agent_L = SP_DDPG(nb_states=5, nb_actions=1, args=args, lam_max=10.0, lam_min=1.0)
    agent_L.cost_limit = 10.0  # L-layer c2 dedicated initial value

    if args.mode == 'train':
        train_prrbb(args.train_iter_H,
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
