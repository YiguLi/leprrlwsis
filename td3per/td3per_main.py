#!/usr/bin/env python3
"""
TD3PER (TD3 + PER) — Main Training Loop — real-profit variant
=============================================================
Modified from PER_main0.py. Key changes:

  1. Uses SP_TD3 agent (TD3: twin critics with min aggregation,
     delayed policy update, target policy smoothing) combined with
     PER and the Lagrangian cost constraint.
  2. New hyper-parameters: --policy_delay, --policy_noise, --noise_clip,
     --cost_critic_mode.
  3. Lower level (BESS) uses REAL electricity price instead of the
     learned shadow price: L observation dim 3 = price/100, and
     r2 = real_profit - bessdeg - 100*clip_amount with
     real_profit = -pbrule*price/20 (低谷充电/高峰放电为正),
     bessdeg = 490*放电/20 (退化成本, 与利润项同分母 /20, 盈亏平衡≈490).
  4. Lower-level episode length for Q-value bootstrapping is the full
     day (1440 steps); done_L is True only at the episode end.

Preserved from PER:
  - 5-step block structure for upper/lower level interaction
  - PER buffer with importance sampling
  - RBB rule clipping before env.step
  - EMA dual ascent for lambda (actual collected cost)
  - Curriculum learning with d_block halving
  - Checkpointing, CSV logging, TensorBoard, 9-case test
"""
import os
import sys
import numpy as np
import argparse
from copy import deepcopy
import torch

# Ensure both td3per/ and parent (HSRL) directories are on path
_TD3PER_DIR = os.path.dirname(os.path.abspath(__file__))
_HSRL_DIR = os.path.dirname(_TD3PER_DIR)
sys.path.insert(0, _TD3PER_DIR)
sys.path.insert(0, _HSRL_DIR)

from env import WindBattery
from td3per.td3per_ddpg import SP_TD3
from util import *
from td3per_myfunc import *
from rbb.rbb_rule_bess import RuleBasedBESS

import pandas as pd
from memory import RingBuffer

from torch.utils.tensorboard import SummaryWriter

# 不显示warning
import warnings
warnings.filterwarnings("ignore")

import matlab
import matlab.engine
engine = matlab.engine.start_matlab()

# H 层 c1 中 unfeasible_gap 的惩罚权重 (MW→成本系数, 与 PRRBB W_GAP 一致)
W_GAP = 50.0

# TD3PER: TD3 + PER + CL
def train(num_iterations_H,
          agent_H,
          agent_L,
          env,
          env_test,
          rbb_system,
          output,
          max_episode_length=None,
          debug=False,
          upload=False,
          use_csv=False):

    agent_H.is_training = True
    agent_L.is_training = True
    step = episode = episode_steps = 0
    episode_reward = episode_cost = episode_cost2 = episode_profit = 0.
    episode_bess_profit = 0.
    observation = None
    prev_soc = None   # 上一 episode 末 SOC，跨 episode 继承（None = 首个 episode，用默认重置值）
    time = 1/60
    # Curriculum learning: consecutive steps where critic_c < cost_limit
    consecutive_satisfied_H = 0
    consecutive_satisfied_L = 0
    # H 层 EMA 对偶 —— 每 5 步块实际成本预算 d_block_H (与 sum_c1 同口径)
    # 按波动设计: c1=0.1*P_F (P_F<=3 时 c=0), 预算 = 0.5*P_F目标;
    # 初值 2.5 = 允许 5 MW/步; 地板 1.25 = 允许 2.5 MW/步 (不再收得更紧)
    d_block_H = 2.5         # 初值: 每块 sum_c1 预算 (≈允许 5 MW/步)
    d_block_H_min = 1.25    # curriculum 收紧地板 (≈允许 2.5 MW/步)
    # B1: L 层 EMA 对偶 —— 每 5 步块实际成本预算 d_block_L (与 sum_c2 同口径)
    d_block_L = 20.0        # 初值: 每块 sum_c2 预算 (≈每步 4, 按 c2 系数 1000 标定)
    d_block_L_min = 2.5     # curriculum 收紧地板 (≈每步 0.5)
    if use_csv:
        df = pd.DataFrame()
        df_episode = pd.DataFrame()
        df_test = pd.DataFrame()
        df_update = pd.DataFrame()  # H 层更新数据（TD/Q/batch 统计）

    # ---- 1440-step window counters for env metrics ----
    window_violation_count = 0   # steps where |P_F| > 3 MW
    window_power_sum = 0.0        # cumulative P_G over the window

    while step < num_iterations_H:
        # start of episode 重置环境
        prYellow('########### step:{} ###########'.format(step))
        if observation is None:
            observation = deepcopy(env.reset())  # [Pr,U,E,P_last]
            # SOC 继承前一 episode：env.reset() 默认把 E 重置为 3.0，
            # 此处恢复为上一 episode 末的 SOC（首个 episode 无前值，保持默认）。
            if prev_soc is not None:
                env.E = prev_soc
                observation[2] = prev_soc
            norm_observation = Normalize(observation)
            agent_H.reset(norm_observation)
        
        # agent pick action 上层选择动作 
        if episode_steps % 5 == 0:   
            sum_r1 = 0 
            sum_c1 = 0 
            sum_c2 = 0     # B1: L 层 5 步块累计成本 (喂 EMA 对偶)     
            if step <= args.warmup:
                applied_action_H = original_action_H = agent_H.random_action() 
                penalty_NN_H = 0
            else:
                original_action_H = agent_H.select_action(norm_observation)
                applied_action_H, penalty_NN_H = CheckAction(original_action_H)
            norm_observation_H = norm_observation
            pw_tru = env.PowerOutput(observation[1], applied_action_H)
        prGreen('--H-- ori a_t = {}'.format(round_list(original_action_H, 3)))
        
        
        # L 层观测蒸馏 (5维): [pbrulemin/10, pbrulemax/10, price/100, SOC/10, u/10]
        # 可行域 (RBB, 与 pb 无关, 预计算) + 真实电价 (300/580/980 → 3.0/5.8/9.8)
        # + 自身 SOC + 风速; 不再直接看 pr/P_G/ptur —— 分层: L 在约束内按真实电价套利.
        ptur = float(pw_tru)
        pg_prev = float(observation[-1])
        E_current = float(observation[2])
        _, rbb_info_obs = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=ptur, pg_prev=pg_prev, E=E_current)
        norm_observation_L = build_L_obs_distilled(
            observation, pw_tru, applied_action_H, rbb_info_obs)
        prPurple('--L-- obs = {}.'.format(round_list(norm_observation_L,3)))
        # 下层每一个5-step都是一个episode
        if episode_steps % 5 == 0:    
            agent_L.reset(norm_observation_L)
            pb_l = list()
        
        # agent pick action 下层选择动作
        if step <= args.warmup:    
            applied_action_L = original_action_L = agent_L.random_action()
            penalty_NN_L = 0
        else:
            original_action_L = agent_L.select_action(norm_observation_L) 
            applied_action_L, penalty_NN_L = CheckAction(original_action_L)
        prGreen('--L-- ori a_t = {}'.format(round_list(original_action_L, 3)))
        
        action = np.concatenate((applied_action_H, applied_action_L))
        original_action = action.copy()

        # ================================================================
        # RBB: Rule-Based BESS Clipping (hard clipping before env.step)
        # ================================================================
        pb_original = float(action[-1])

        pbrule, rbb_info = rbb_system.compute_pbrule(
            pb_original=pb_original,
            ptur=ptur,
            pg_prev=pg_prev,
            E=E_current,
        )

        # Clip pb to pbrule before passing to environment
        action_rbb = action.copy()
        action_rbb[-1] = pbrule

        # env response with next_observation, reward, terminate_info
        pw_tru2, observation2, action_L, penalty_L, done, done_L, info = \
            env.step(action_rbb[:6], action_rbb[6:], pw_tru)
        P_F = info['P_F']  # 电网功率波动 |P_G(t) - P_G(t-1)|
        action_rbb[6:] = action_L
        prGreen('Applied action = {}.'.format(round_list(action_rbb, 3)))
        norm_observation2 = Normalize(observation2)
        # 下一状态 L 观测: 用下一状态重算可行域 (pb 无关)
        _, rbb_info_obs2 = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=float(pw_tru2),
            pg_prev=float(observation2[-1]), E=float(observation2[2]))
        norm_observation2_L = build_L_obs_distilled(
            observation2, pw_tru2, applied_action_H, rbb_info_obs2)
        
        r, c = Cal_Rew(observation, observation2, action_rbb)
        r1 = observation[0] * observation2[-1] * time - penalty_NN_H - 490.0 * max(-float(pbrule), 0) * time

        # SOC safe: [1.5, 4.5] (conservative, equivalent to HIGH wind in segmented schemes)
        # —— c2 拉格朗日约束项的 SOC 越界计算
        soc_safe_min = 1.5
        soc_safe_max = 4.5
        soc = float(observation2[2])
        soc_deviation = 0.0
        if soc < soc_safe_min:
            soc_deviation = soc_safe_min - soc
        elif soc > soc_safe_max:
            soc_deviation = soc - soc_safe_max

        # c1: 拉格朗日惩罚 = 越限惩罚(c) + 平滑项 0.1*P_F + W_GAP*unfeasible_gap
        #     （unfeasible_gap: 波动约束∩物理约束为空时的不可行缺口, 硬裁剪无法消除）
        c1 = c + 0.1 * P_F + W_GAP * rbb_info['unfeasible_gap']
        sum_r1 += r1
        sum_c1 += c1
        prGreen('r1 = {}'.format(round(r1, 3)))
        prGreen('c1 (fluct={}, no_soc_safe) = {}'.format(round(c, 3), round(c1, 3)))

        # ---- 1440-step window: track actual grid fluctuation ----
        if P_F > 3.0 + 1e-4:
            window_violation_count += 1
        window_power_sum += observation2[-1]  # cumulative P_G
        
        pb_l.append(pb_original)  # track original intent

        clip_amount = rbb_info['clip_amount']
        is_infeasible = rbb_info['is_infeasible']
        inevitable_violation = rbb_info['inevitable_violation']

        # r2: 真实套利利润 - bess 退化成本 - 100*clip_amount（pbrule 裁剪惩罚进入 r2）
        # real_profit = -pbrule*price/20 (price=observation[0] 真实电价 RMB/MWh;
        # 低谷 300 充电 / 高峰 980 放电 → real_profit > 0; 不再乘 10 放大)
        price = float(observation[0])
        real_profit = -float(pbrule) * price / 20.0
        # bessdeg: BESS 退化成本 = 490*放电/20 (放电 D → 24.5D;
        # 与利润项 real_profit 同分母 /20: 盈亏平衡电价 = 20*24.5 = 490,
        # 与 H 层 r1 的退化系数 490 口径一致 → 低谷亏、平峰微利、高峰赚)
        bessdeg = 490.0 * max(-float(pbrule), 0.0) / 20.0

        # clip_amount 惩罚系数恢复 100 (real_profit 去掉 *10 后, 500 相对利润项过强)
        r2 = real_profit - bessdeg - 100.0 * clip_amount

        # c2: SOC 越界项进入拉格朗日约束（保留原系数 1000/MWh）
        c2 = 1000.0 * soc_deviation
        sum_c2 += c2  # B1: 累计块成本 (每 5 步归零)

        # prGreen('r2 = {}'.format(round(r2, 3)))
        # prGreen('c2 = {}'.format(round(c2, 3)))
        
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True 
        if episode_steps % 5 == 4:
            agent_H.observe_jump(norm_observation_H, original_action_H, sum_r1, sum_c1, norm_observation2, done)
            # H 层 EMA 对偶上升 —— 用 5 步块实际成本 sum_c1 更新 lam1 (与 d_block_H 同口径)
            if step > args.warmup:
                agent_H.update_ema_cost(sum_c1)
                agent_H.update_lam_ema(d_block_H)
            # B1: L 层 EMA 对偶上升 —— 用 5 步块实际成本 sum_c2 更新 lam2 (与 d_block_L 同口径)
            if step > args.warmup:
                agent_L.update_ema_cost(sum_c2)
                agent_L.update_lam_ema(d_block_L)
            # c2 = 0.01 * abs(sum(pb_l)-action[5]) + c2 # todo 是否要*0.1， 有关 signal 的 terminal reward 
        # L 层 episode 长度 = 1440 步（整日）：仅当日结束时 done_L=True，Q 值跨 5 步块引导
        done_L = done
        agent_L.observe_jump(norm_observation_L, original_action_L, r2, c2, norm_observation2_L, done_L)
        
        if step > args.warmup:
            #--------------------Update----------------------
            if episode_steps % 5 == 4:
                # Curriculum convergence: halve cost_limit when
                # critic_c < cost_limit for 100 consecutive policy updates
                # UTD 2:1 -> 1:1: one policy update per new transition
                # (5 env steps per block -> 5 updates), TD3 paper default.
                update_num = 0
                while update_num < 5:
                    agent_H.update_policy()
                    agent_L.update_policy()
                    update_num += 1

                    # Track consecutive satisfaction for H (EMA 实际成本判据,
                    # 替换原 critic-based lam_loss —— EMA 模式下 lam_loss 恒为 0)
                    if agent_H.ema_cost is not None and agent_H.ema_cost <= d_block_H:
                        consecutive_satisfied_H += 1
                    else:
                        consecutive_satisfied_H = 0
                    # Track consecutive satisfaction for L (B1: EMA 实际成本判据,
                    # 替换原 critic-based lam_loss —— EMA 模式下 lam_loss 恒为 0)
                    if agent_L.ema_cost is not None and agent_L.ema_cost <= d_block_L:
                        consecutive_satisfied_L += 1
                    else:
                        consecutive_satisfied_L = 0

                if consecutive_satisfied_H >= 100:
                    d_block_H = max(d_block_H / 2, d_block_H_min)  # H: 收紧块预算
                    consecutive_satisfied_H = 0
                    prGreen('H: d_block_H halved to {}'.format(d_block_H))
                if consecutive_satisfied_L >= 100:
                    d_block_L = max(d_block_L / 2, d_block_L_min)  # B1: 收紧块预算
                    consecutive_satisfied_L = 0
                    prGreen('L: d_block_L halved to {}'.format(d_block_L))

                # ---- H 层更新数据记录（本块 10 批均值 → TB/csv） ----
                # 记录"神经网络更新时实际采用的数据"：TD/Q/target/batch 奖励成本/ISW。
                if agent_H.record_update_stats and len(agent_H.update_stats) > 0:
                    _stats = agent_H.update_stats
                    _mean = {k: float(np.mean([s[k] for s in _stats]))
                             for k in _stats[0]}
                    agent_H.update_stats = []  # 清空，下一块重收
                    if upload:
                        for _k, _v in _mean.items():
                            writer.add_scalar('update/' + _k, _v, step)
                    if use_csv:
                        df_update = pd.concat([df_update, pd.DataFrame(
                            [{'step': step, **_mean}])], ignore_index=True)
                        df_update.to_csv('{}/update_stats_H.csv'.format(output))

            # Test: 9-case segmented evaluation every 480 episode steps
            # (fixed scenario seed 42 -> same wind trajectory each test, so
            # consecutive test9 events are point-comparable and smooth)
            if episode_steps % 480 == 0:
                case_metrics, overall = nine_case_test(
                    agent_H, agent_L, env_test, rbb_system, seed=42)
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

        

        # 上层保存到data
        if use_csv:
            df = pd.concat([df, pd.DataFrame([{
                'step': step,
                'observation': observation,
                'norm_observation_L': norm_observation_L,
                'original_action_H': original_action_H,
                'original_action_L': original_action_L,
                'action': action_rbb,
                'r1': r1,
                'c1': c1,
                'r2': r2,
                'c2': c2,
                'soc_deviation': soc_deviation,
                'reward': r,
                'cost': c,
                'lam1': agent_H.lam,
                'lam2': agent_L.lam,
                'price': price,
                'real_profit': real_profit,
                'bessdeg': bessdeg,
                'clip_amount': clip_amount,
                'pb_original': pb_original,
                'pbrule': float(pbrule),
                'soc': float(observation[2]),
                'soc_next': float(observation2[2]),
                'pbrulemin': rbb_info['pbrulemin'],
                'pbrulemax': rbb_info['pbrulemax'],
                'unfeasible_gap': rbb_info['unfeasible_gap'],
                'is_infeasible': float(is_infeasible),
                'inevitable_violation': inevitable_violation,
                'P_F': P_F,
            }])], ignore_index=True)
            if step % 50 == 0:
                df.to_csv('{}/df.csv'.format(output))

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
            writer.add_scalar('soc', float(observation[2]), step)
            writer.add_scalar('soc_deviation', soc_deviation, step)
            writer.add_scalar('P_F', P_F, step)  # 电网功率波动
            # Real price (observation 栏)
            writer.add_scalar('action/price', price, step)
            # Lower reward components
            writer.add_scalar('reward_lower/real_profit', real_profit, step)
            writer.add_scalar('reward_lower/bessdeg', bessdeg, step)
            # RBB-specific metrics
            writer.add_scalar('rbb/clip_amount', clip_amount, step)
            writer.add_scalar('rbb/pbrule', float(pbrule), step)
            writer.add_scalar('rbb/pb_original', pb_original, step)
            writer.add_scalar('rbb/unfeasible_gap', rbb_info['unfeasible_gap'], step)
            writer.add_scalar('rbb/is_infeasible', float(is_infeasible), step)
            writer.add_scalar('rbb/inevitable_violation', inevitable_violation, step)

            if episode_steps % 5 == 4:
                writer.add_scalar('sum_r1', sum_r1, step)
                writer.add_scalar('sum_c1', sum_c1, step)

            # ---- Action logging (every 5 steps) ----
            if episode_steps % 5 == 4:
                # Upper action: 5 WT + 1 signal
                for i in range(5):
                    writer.add_scalar('action/a_H_original_{}'.format(i),
                                      original_action_H[i], step)
                    writer.add_scalar('action/a_H_clipped_{}'.format(i),
                                      applied_action_H[i], step)
                # PB: original + rule-clipped
                writer.add_scalar('action/pb_original', pb_original, step)
                writer.add_scalar('action/pbrule', float(pbrule), step)

            # ---- Env 1440-step window metrics ----
            if step > 0 and step % 1440 == 0:
                writer.add_scalar('env/window_violation_count',
                                  window_violation_count, step)
                writer.add_scalar('env/window_power_sum',
                                  window_power_sum, step)
                # Reset window counters
                window_violation_count = 0
                window_power_sum = 0.0

        # update
        step += 1
        episode_steps += 1
        episode_reward += r
        episode_cost += c
        episode_cost2 += c2
        # 上层利润: pg * 实际电价 * dt - 4.9 * pb * dt
        episode_profit += observation2[-1] * observation[0] * time - 4.9 * max(-float(pbrule), 0) * time
        # BESS盈亏: 真实套利 - 退化成本（与 r2 中 bessdeg 同口径）
        episode_bess_profit += real_profit - bessdeg
        # update env
        # Oh_observation = deepcopy(Oh_observation2)
        norm_observation = deepcopy(norm_observation2)
        observation = deepcopy(observation2)
        pw_tru = pw_tru2

        #-------------------------------------episode-----------------------------
        if done:
            # save intermideate model
            agent_H.save_model(output)
            agent_L.save_model(output)
            prGreen('##########{}: episode_reward:{} step:{}'.format(episode, round(episode_reward,2), step))

            if upload:
                writer.add_scalar('episode_reward', episode_reward, episode)
                writer.add_scalar('episode_cost', episode_cost, episode)
                writer.add_scalar('episode_cost2', episode_cost2, episode)
                writer.add_scalar('episode_profit', episode_profit, episode)
                writer.add_scalar('env/bessprofit', episode_bess_profit, episode)
                writer.add_scalar('wind_day', env.j, episode)

            if use_csv:
                df_episode = pd.concat([df_episode, pd.DataFrame([{
                    'episode': episode,
                    'episode_reward': episode_reward,
                    'episode_cost': episode_cost,
                    'episode_cost2': episode_cost2,
                    'episode_profit': episode_profit,
                    'bessprofit': episode_bess_profit,
                    'wind_day': env.j,
                }])], ignore_index=True)
                df_episode.to_csv('{}/df_episode.csv'.format(output))

            # end of episode
            # reset
            prev_soc = float(observation[2])  # 保存本 episode 末 SOC，供下一 episode 继承
            observation = None
            episode_steps = 0
            episode_reward = episode_cost = episode_profit = 0.
            episode_bess_profit = 0.
            episode += 1
    if use_csv:
        df.to_csv('{}/df.csv'.format(output))
        df_episode.to_csv('{}/df_episode.csv'.format(output))
        df_test.to_csv('{}/df_test9.csv'.format(output))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode',
                        default='train',
                        type=str,
                        help='support option: train/test')
    parser.add_argument('--env',
                        default='TD3PER',
                        type=str,
                        help='open-ai gym environment')
    parser.add_argument('--hidden1',
                        default=16,
                        type=int,
                        help='hidden num of first fully connect layer')  # 400
    parser.add_argument('--hidden2',
                        default=100,
                        type=int,
                        help='hidden num of second fully connect layer')  # 300
    parser.add_argument('--rate',
                        default=0.01,
                        type=float,
                        help='learning rate')
    parser.add_argument('--prate',
                        default=0.0001,
                        type=float,
                        help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup',
                        default=200,
                        type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.98, type=float, help='')
    # 0.99
    parser.add_argument('--bsize', default=32, type=int,
                        help='minibatch size')  #64
    parser.add_argument('--rmsize',
                        default=20000,
                        type=int,
                        help='memory size') # 6000000
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau',
                        default=0.001,
                        type=float,
                        help='moving average for target network')
    parser.add_argument('--ou_theta',
                        default=0.15,
                        type=float,
                        help='noise theta')
    parser.add_argument('--ou_sigma',
                        default=0.2,
                        type=float,
                        help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes',
                        default=20,
                        type=int,
                        help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps',
                        default=2000,
                        type=int,
                        help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter_H',
                        default=200000,
                        type=int,
                        help='train iters each timestep')
    parser.add_argument('--epsilon',
                        default=50000,
                        type=int,
                        help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume',
                        default='default',
                        type=str,
                        help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay')
    parser.add_argument('--cuda', default=True, type=bool, help='Whether to use CUDA')
    parser.add_argument('--lam', default=3, type=float, help='safe constraint')
    parser.add_argument('--lr_lam', default=0.1, type=float, help='learning rate for lambda')
    parser.add_argument('--cost_single_limit', default=0.5, type=float, help='cost_bound for single step')

    parser.add_argument('--PER_alpha', default=0.6, type=float, 
                        help='[0~1] convert the importance of TD error to priority. 0 - no prioritization, 1 - full prioritization')
    parser.add_argument('--PER_beta', default=0.4, type=float, 
                        help='importance-sampling, from initial value increasing to 1. 0 - no corrections, 1 - full correction')
    parser.add_argument('--PER_epsilon', default=0.01, type=float, 
                        help='small amount to avoid zero priority')
    parser.add_argument('--beta_increment_per_sampling', default=3e-5, type=float, 
                        help='β退火增量/次更新: 0.6/20000次更新 → 10k env步内 0.4→1 (100k预算)')
    parser.add_argument('--nb_turbine', default=5, type=int,
                        help='number of turbines')

    # ---- TD3 specific ----
    parser.add_argument('--policy_delay', default=2, type=int,
                        help='TD3 delayed policy update interval (critic updates per actor update)')
    parser.add_argument('--policy_noise', default=0.1, type=float,
                        help='target policy smoothing noise std, as fraction of action amplitude')
    parser.add_argument('--noise_clip', default=0.3, type=float,
                        help='target policy smoothing noise clip, as fraction of action amplitude')
    parser.add_argument('--cost_critic_mode', default='max', type=str,
                        choices=['min', 'max'],
                        help='cost twin-critic aggregation: max (safe, recommended) or min (TD3 paper; risks cost underestimation -> Q_c collapse)')

    # RBB parameters (for soft penalty computation only, no hard clipping)
    parser.add_argument('--rbb_fluctuation_limit', default=3.0, type=float,
                        help='Max grid power fluctuation between steps (MW)')
    parser.add_argument('--rbb_rated_power', default=10.0, type=float,
                        help='BESS rated charge/discharge power (MW)')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    #TODO 修改的命令ppyp
    max_E = 6.
    upload = True
    use_csv = True
    args.max_episode_length = 1440  # 1d = 24h*60m = 1440 steps
    args.train_iter_H = 1000 * args.max_episode_length
    args.seed = np.random.randint(0, 10000)
    args.cost_single_limit = 1000. #TODO

    # args.warmup = 5

    # Set device and random seeds
    if (args.cuda and torch.cuda.is_available()):
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        print("Using GPU:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Set random seeds
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子

    if upload:
        writer = SummaryWriter(args.output)
        writer.add_scalar('seed', args.seed)
    
    env = WindBattery(max_E, args.nb_turbine, upload, engine)
    env_test = WindBattery(max_E, args.nb_turbine, upload, engine)

    if args.seed > 0:
        np.random.seed(args.seed)

    # RBB system for soft penalty computation only (no hard clipping in PER)
    rbb_system = RuleBasedBESS(
        max_E=max_E,
        min_E_factor=0.0,
        rated_power=args.rbb_rated_power,
        eta_c=0.98,
        eta_d=0.98,
        time=1.0 / 60,
        fluctuation_limit=args.rbb_fluctuation_limit,
    )


    # H 层对偶改 EMA 实际成本更新 —— lam1 不再读 critic (Q_c)；
    #     lam1 ∈ [3, 50] 初始 5.0 (与 PRRBB H 层同配置, c1 公式同量级)
    agent_H = SP_TD3(nb_states = 4, nb_actions = 6, args = args,
                     lam_max=50.0, lam_min=3.0,
                     ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                     record_update_stats=True,
                     policy_delay=args.policy_delay,
                     policy_noise=args.policy_noise,
                     noise_clip=args.noise_clip,
                     cost_critic_mode=args.cost_critic_mode)
    agent_H.lam = 5.0
    # B1: L 层对偶改 EMA 实际成本更新 —— lam2 不再读 critic (Q_c)，断高估→顶死传导；
    #     lam2 ∈ [10, 100]，ema_alpha=0.02 (时间常数~50块=250步), ema_lam_lr=0.01
    #     蒸馏观测 5 维 = [pbrulemin/10, pbrulemax/10, price/100, SOC/10, u/10]
    agent_L = SP_TD3(nb_states = 5, nb_actions = 1, args = args,
                     lam_max=100.0, lam_min=10.0,
                     ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                     policy_delay=args.policy_delay,
                     policy_noise=args.policy_noise,
                     noise_clip=args.noise_clip,
                     cost_critic_mode=args.cost_critic_mode)
    agent_L.cost_limit = 10.0  # L层c2专用初始值（覆盖全局cost_single_limit=1000）
    #TODO 是否预训练
    # agent_H.load_weights('output/1_2310S-run18')
    # agent_L.load_weights('output/1_2310S-run18')

    if args.mode == 'train':
        train(args.train_iter_H,
              agent_H,
              agent_L,
              env,
              env_test,
              rbb_system,
              args.output,
              max_episode_length=args.max_episode_length,
              debug=args.debug,
              upload=upload,
              use_csv=use_csv)
    if upload:
        writer.close()

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
