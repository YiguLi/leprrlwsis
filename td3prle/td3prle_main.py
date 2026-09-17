#!/usr/bin/env python3
"""
TD3PRLE (TD3 + PRLE) — Main Training Loop
==========================================
Modified from td3prrbb_main.py (TD3 + PRRBB). Adds the PRLE layer:
dynamic wind thresholds adjusted by an LLM meta-controller and batch
reward relabeling for non-stationary rewards.

Parent packages:
  - Algorithm layer (td3prrbb/td3prrbb_main.py):
      SP_TD3_PRLE agents — twin critics (profit channel: min aggregation,
      cost channel: max/safe aggregation with clamp>=0), delayed policy
      update, target policy smoothing, PRLEBuffer (PER + context store)
      with importance sampling, EMA dual ascent on actual per-block cost,
      H-layer update-stats recording, distilled 5-dim L observation,
      9-case test with fixed seed 42, per-episode SOC random reset [0.5, 5.5].
  - Reward layer (prle/prle_main.py):
      Cal_Rew_PRRBB 机会/风险分离的分段奖励 with DYNAMIC wind thresholds,
      RBB hard clipping before env.step, sub-step context accumulation,
      WindowStats + LLM threshold adjustment at 7200-step (5-day) boundaries,
      buffer relabeling when thresholds change.

Reward structure (td3prle_myfunc.Cal_Rew_PRRBB, 严格对照 td3prrbb):
  Upper (H): r1 = wT*(pr*ptur*time) + wB*(-pr*pbrule*time)
                  - BESS_DEG_RMB_PER_MWH*max(-pbrule,0)*time - penalty_NN_H
                  (wT = wT_u(u)×wT_p(pr), wB ≡ 1.0;
                   【本轮变更】取消"按 SOC 位置折减 BESS 盈利 ×0.7"的机制,
                   WB_SOC_OUTSIDE 0.7 → 1.0, SOC 位置不再影响 H 的收益权重;
                   与 td3prrbb 的差异仅在于此)
             c1 = 10*max(P_F-3,0) + C1_SMALL_FLUC[档]*P_F + W_GAP*unfeasible_gap
                  + 块末结算项 c1_soc_track = 10*|soc_target − SOC_块末|
  Lower (L): r2 = -min(100*(clip_amount + clip_over), 1000)
             (clip_amount = 送进 RBB 的动作被可行窗口夹掉的量;
              clip_over = 原始意图被 CheckAction 夹到 ±10 MW 之外的部分,
              两项同系数合成同一个裁剪惩罚, 并受【总上限】封顶 1000 RMB/step
              ⇔ 合计裁剪量 ≤10 MW; 见 myfunc.L_CLIP_PEN_COEF / _TOTAL_CAP)
             c2 = 10*(SOC_new − soc_target)²   (目标中心二次势, ×10 惩罚系数)

【信号语义 (本轮变更)】H 第 6 维输出 signal 不再是影子价格, 而是 SOC 目标的归一化指令;
  H 输出范围不变 ((-0.5, 0.5)), 由下层线性映射 (myfunc.signal_to_soc_target):
      soc_target = 3.0 + 5.0·signal ∈ [0.5, 5.5] MWh
  L 层把 soc_target 作为 c2 中心 (以 ×10 计费跟踪), 同时进 L 观测量 (soc_target/10);
  H 层在 5 步块末结算时对 |soc_target − SOC_块末| 计 10 倍惩罚, 计入 c1 约束口
  (与波动/不可行缺口共用 d_block_H, 由 lam1 对偶驱动 H 给出下层可达的目标)。

Lagrange multiplier updates (EMA dual ascent on ACTUAL block cost):
  H: EMA   ← (1-0.02)*EMA + 0.02*sum_c1          (每 5 步块, sum_c1 含块末 SOC 跟踪项)
     lam1  ← clip(lam1 + 0.01*(EMA − d_block_H), 50, 500), 初始 100.0
     d_block_H 初始 12.0 (c1 平滑系数已回调 0.1, 预算未随动), curriculum 连续 100 次满足则减半, 地板 3.0
  L: lam2  ← clip(lam2 + 0.01*(EMA − d_block_L), 2, 200), 初始 100.0
     d_block_L 初始 40.0 (= 10·2²: 目标中心二次势下单步贴 ±2 的成本值), curriculum 减半地板 = 40 (不收紧)

折扣 (γ):
  H 层 γ = --discount (默认 0.98, 块级语义: 1/(1−γ)=50 块视野);
  L 层 γ = --l_discount (默认 0.98, 步级语义; L 层每 5 步块终止 → bootstrap 视野 ≤5 步,
  γ 只影响块内 5 步的加权, 故取与 H 相同的 0.98)。

LLM meta-controller (from prle):
  - 每 7200 步（5 天）窗口边界, episode >= 3 后调用 LLM 分析窗口统计
    并给出 (adjust_low_max, adjust_high_min) 阈值调整建议;
  - 调整范围硬约束: 阈值只能在 原设定 ±1 m/s 内变化 (整个训练过程);
  - 阈值实际变化后, 对 H 层回放缓冲 relabel_all() 重算奖励/成本;
  - LLM 调用失败时回退占位模式 (返回 (0.0, 0.0)), 训练继续。
"""
import os
import sys
import numpy as np
import argparse
from copy import deepcopy
import torch

# Ensure both td3prle/ and parent (HSRL) directories are on path
_TD3PRLE_DIR = os.path.dirname(os.path.abspath(__file__))
_HSRL_DIR = os.path.dirname(_TD3PRLE_DIR)
sys.path.insert(0, _TD3PRLE_DIR)
sys.path.insert(0, _HSRL_DIR)

from env import WindBattery
from td3prle.td3prle_ddpg import SP_TD3_PRLE
from util import *
from td3prle.td3prle_myfunc import (
    CheckAction, Normalize, Original,
    Cal_Rew_PRRBB, nine_case_test,
    build_L_obs_distilled,
    get_price_level,
    get_wind_thresholds, set_wind_thresholds,
    relabel_r1_substep, relabel_c1_substep,
    relabel_c2_substep,
    signal_to_soc_target, soc_track_penalty,
    WB_PRICE, WB_SOC_OUTSIDE, WT_PRICE, WT_WIND,
)
from td3prle.td3prle_llm import (
    WindowStats, llm_analyze_thresholds, create_llm_call_fn,
)
from td3prle.td3prle_rule_bess import RuleBasedBESS

import pandas as pd
from memory import RingBuffer

from torch.utils.tensorboard import SummaryWriter

# 不显示warning
import warnings
warnings.filterwarnings("ignore")

import matlab
import matlab.engine
engine = matlab.engine.start_matlab()


# TD3PRLE: TD3 + PRLE 动态阈值分段奖励 + CL + LLM 元控制
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
          use_csv=False,
          adj_bounds=None):

    agent_H.is_training = True
    agent_L.is_training = True
    step = episode = episode_steps = 0
    episode_reward = episode_cost = episode_cost2 = episode_profit = 0.
    episode_bess_profit = 0.
    observation = None
    # 每 episode (1440 步) 起点 SOC 均匀随机重置 [0.5, 5.5]，
    # 替代原跨 episode 继承 (prev_soc) — 保证日初 SOC 覆盖安全带，避免长期锁死窄带
    soc_reset_low, soc_reset_high = 0.5, 5.5
    time = 1/60
    # Curriculum learning: consecutive steps where EMA cost <= d_block
    consecutive_satisfied_H = 0
    consecutive_satisfied_L = 0
    # H 层 EMA 对偶 —— 每 5 步块实际成本预算 d_block_H (与 sum_c1 同口径)
    # td3prle 标定: c1 = 10*max(P_F-3,0) + C1_SMALL_FLUC[档]*P_F + W_GAP*unfeasible_gap
    #            (HIGH 风速 ×1.2, LOW/MEDIUM ×0.8; W_GAP=10, td3prle 调参: 原 50 → 10)
    d_block_H = 12.0        # 初值: 每块 sum_c1 预算 (平滑系数曾 0.1→1.0 ×10 重标; 现已回调 0.1, 预算未随动)
    d_block_H_min = 3.0     # curriculum 收紧地板
    # L 层 EMA 对偶 —— 每 5 步块实际成本预算 d_block_L (与 sum_c2 同口径)
    # c2 = C2_SOC_COEFF(10)*(SOC_new−soc_target)² 目标中心二次势;
    # 预算 d_block_L = 40 = 10·2² (单步贴 ±2 MWh 处的成本值; ×10 系数下与旧
    #     100·(SOC−3)² / 400 同刻度), 地板 = 初值 → 预算永不收紧。
    d_block_L = 40.0        # 初值: 每块 sum_c2 预算 (目标中心二次势标定)
    d_block_L_min = 40.0    # 收紧地板 = 初值 → max(d/2, 40)≡40, 预算永不收紧:
                            # SOC 保持软裕度, 不随 curriculum 渐渐被钉向目标

    # ---- TD3PRLE: Window statistics for LLM analysis ----
    window_stats = WindowStats()
    threshold_history = []         # records of past adjustments
    min_episodes_before_llm = 3    # cold-start protection: wait for initial convergence
    block_ctx_H = []               # accumulates sub-step context for current 5-step block

    # ---- TD3PRLE: Create LLM call function (reads from config file by default) ----
    llm_call_fn = create_llm_call_fn(provider=args.llm_provider,
                                     config_path=args.llm_config)
    prCyan('TD3PRLE: LLM ready (provider={})'.format(
        args.llm_provider or 'from config'))

    # ---- TD3PRLE: Startup API connectivity test ----
    prCyan('TD3PRLE: Testing LLM API connectivity...')
    test_prompt = '请回复"OK"，不要包含任何其他内容。'
    try:
        test_response = llm_call_fn(test_prompt)
        prGreen('TD3PRLE: LLM API test passed. Response: "{}"'.format(
            test_response.strip()[:120]))
    except Exception as e:
        prRed('TD3PRLE: LLM API test FAILED — {}'.format(e))
        prRed('TD3PRLE: Training will continue but threshold adjustments will be skipped.')
        llm_call_fn = None  # fall back to placeholder mode
    if use_csv:
        df = pd.DataFrame()
        df_episode = pd.DataFrame()
        df_test = pd.DataFrame()
        df_update = pd.DataFrame()  # H 层更新数据（TD/Q/batch 统计）
        df_flushed_rows = 0          # 已增量落盘的 df 行数 (df 跨 episode 只增不减)
        df_update_flushed_rows = 0   # 已增量落盘的 df_update 行数

    # ---- 1440-step window counters for env metrics ----
    window_violation_count = 0   # steps where |P_F| > 3 MW
    window_power_sum = 0.0        # cumulative P_G over the window

    while step < num_iterations_H:
        # start of episode 重置环境
        prYellow('########### step:{} ###########'.format(step))
        if observation is None:
            observation = deepcopy(env.reset())  # [Pr,U,E,P_last]
            # SOC 每 episode 随机重置: env.reset() 默认把 E 重置为 3.0,
            # 此处覆盖为 [0.5, 5.5] 均匀随机起点 (每个 episode 一致)
            env.E = np.random.uniform(soc_reset_low, soc_reset_high)
            observation[2] = env.E
            norm_observation = Normalize(observation)
            agent_H.reset(norm_observation)

        # agent pick action 上层选择动作
        if episode_steps % 5 == 0:
            sum_r1 = 0
            sum_c1 = 0
            sum_c2 = 0     # L 层 5 步块累计成本 (喂 EMA 对偶)
            c1_soc_track = 0.0   # 本块末结算的 SOC 跟踪惩罚 (块末赋值, 记账用)
            block_ctx_H = []  # reset sub-step context for new 5-step block
            if step <= args.warmup:
                applied_action_H = original_action_H = agent_H.random_action()
                penalty_NN_H = 0
            else:
                original_action_H = agent_H.select_action(norm_observation)
                applied_action_H, penalty_NN_H = CheckAction(original_action_H)
            norm_observation_H = norm_observation
            pw_tru = env.PowerOutput(observation[1], applied_action_H)
            # 上层信号 → SOC 目标 (下层映射; 块内保持不变)
            soc_target_H = signal_to_soc_target(applied_action_H[5])
        prGreen('--H-- ori a_t = {}'.format(round_list(original_action_H, 3)))
        prPurple('--H-- soc_target = {:.3f} MWh (signal a6 = {:.3f})'.format(
            soc_target_H, applied_action_H[5]))

        # L 层观测蒸馏 (5维): [pbrulemin/10, pbrulemax/10, soc_target/10, SOC/10, u/10]
        # 可行域 (RBB, 与 pb 无关, 预计算) + H 给定 SOC 目标 + 自身 SOC + 风速;
        # 不再直接看 pr/P_G/ptur —— 分层: H 给目标, L 在约束内执行.
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
        # L 层原始意图被【执行器上限 ±charge(±10 MW)】夹掉的量 (MW, ≥0):
        #   |original_action_L| − |applied_action_L|; 计入 r2 的裁剪惩罚
        #   (与 clip_amount 同系数 100, 见 myfunc.L_CLIP_PEN_COEF)
        l_clip_over = float(np.sum(np.maximum(
            np.abs(np.asarray(original_action_L, dtype=float)) -
            np.abs(np.asarray(applied_action_L, dtype=float)), 0.0)))
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
        action_rbb[6:] = action_L
        prGreen('Applied action = {}.'.format(round_list(action_rbb, 3)))
        norm_observation2 = Normalize(observation2)
        # 下一状态 L 观测: 用下一状态重算可行域 (pb 无关)
        _, rbb_info_obs2 = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=float(pw_tru2),
            pg_prev=float(observation2[-1]), E=float(observation2[2]))
        norm_observation2_L = build_L_obs_distilled(
            observation2, pw_tru2, applied_action_H, rbb_info_obs2)

        # ================================================================
        # PRRBB Segmented Reward Computation (机会/风险分离版)
        # ================================================================
        r1, c1, r2, c2, reward_details = Cal_Rew_PRRBB(
            observation, observation2, action_rbb, rbb_info,
            penalty_NN_H=penalty_NN_H, time=time,
            ptur=ptur, l_clip_over=l_clip_over,
        )

        # Total reward and cost (H uses r1/c1; L uses r2/c2)
        r = r1 + r2
        c = c1

        sum_r1 += r1
        sum_c1 += c1
        sum_c2 += c2  # 累计块成本 (每 5 步归零)

        pb_l.append(pb_original)  # track original intent

        clip_amount = rbb_info['clip_amount']
        is_infeasible = rbb_info['is_infeasible']
        inevitable_violation = rbb_info['inevitable_violation']

        prGreen('r1 = {} (turb={}, bess={}, deg={})'.format(
            round(r1, 3), round(reward_details['turbine_profit'], 3),
            round(reward_details['bess_profit'], 3),
            round(reward_details['degradation'], 3)))
        prGreen('c1 = {} (fluct={})'.format(
            round(c1, 3), round(reward_details['c1_fluct'], 3)))
        prGreen('r2 = {} (clip={}, clip_over={}, capped={})'.format(
            round(r2, 3), round(reward_details['clip_amount'], 3),
            round(reward_details['clip_over'], 3),
            int(reward_details['clip_pen_capped'])))
        prGreen('c2 = {} (soc_dev={}, soc_track_err={})'.format(
            round(c2, 3), round(reward_details['soc_deviation'], 3),
            round(reward_details['soc_track_err'], 3)))
        prGreen('price={} p_level={} wind={:.1f} w_level={} (wT={}, wB={})'.format(
            reward_details['pr'], reward_details['price_level'],
            reward_details['u'], reward_details['wind_level'],
            reward_details['w_wind'], reward_details['w_bess']))

        # ---- 1440-step window: track actual grid fluctuation ----
        if reward_details['P_F'] > 3.0 + 1e-4:
            window_violation_count += 1
        window_power_sum += observation2[-1]  # cumulative P_G

        # ---- TD3PRLE: Accumulate sub-step context for H-agent reward relabeling ----
        sub_step_ctx = {
            'u': float(observation[1]),
            'pr': float(observation[0]),
            'turbine_raw': float(reward_details['pr'] * ptur * time),
            'bess_raw': float(-reward_details['pr'] * pbrule * time),
            'degradation': float(BESS_DEG_RMB_PER_MWH * max(-pbrule, 0.0) * time),
            'penalty_NN_H': float(penalty_NN_H),
            'P_F': float(reward_details['P_F']),
            'unfeasible_gap': float(rbb_info['unfeasible_gap']),  # relabel 原样带入
        }
        block_ctx_H.append(sub_step_ctx)

        # ---- TD3PRLE: Window statistics tracking ----
        window_stats.add(r1, reward_details['wind_level'],
                         reward_details['P_F'], rbb_info['is_infeasible'])

        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True
        if episode_steps % 5 == 4:
            # 上层【块末结算】: 目标 SOC 与实际 SOC 的差值惩罚
            #   c1_soc_track = C1_SOC_TRACK * |soc_target − SOC_块末|
            # 计入 c1 约束口 (与波动/不可行缺口同一预算 d_block_H), 由 lam1 对偶驱动
            # H 给出下层实际可达的目标; 块内逐步 c1 不入账本项 (一次性结算)。
            c1_soc_track = soc_track_penalty(soc_target_H, float(observation2[2]))
            sum_c1 = sum_c1 + c1_soc_track
            # episode 结算成本同口径累计 (df_episode.episode_cost = 逐步波动成本 + 块末跟踪项)
            episode_cost += c1_soc_track
            # TD3PRLE: pass accumulated sub-step context for reward relabeling
            agent_H.observe_jump(
                norm_observation_H, original_action_H,
                sum_r1, sum_c1, norm_observation2, done,
                context=list(block_ctx_H)  # pass a copy
            )
            # H 层 EMA 对偶上升 —— 用 5 步块实际成本 sum_c1 更新 lam1 (与 d_block_H 同口径)
            if step > args.warmup:
                agent_H.update_ema_cost(sum_c1)
                agent_H.update_lam_ema(d_block_H)
            # L 层 EMA 对偶上升 —— 用 5 步块实际成本 sum_c2 更新 lam2 (与 d_block_L 同口径)
            if step > args.warmup:
                agent_L.update_ema_cost(sum_c2)
                agent_L.update_lam_ema(d_block_L)
            done_L = True
        # L 层每步过渡也存上下文 {u_t, SOC_new, soc_target}: c2 = 10·(SOC_new−target)²
        # 目标由 H 信号给定 (与风速档无关), 故 relabel 时把存档目标原样传回即为恒等
        agent_L.observe_jump(norm_observation_L, original_action_L, r2, c2,
                             norm_observation2_L, done_L,
                             context=[{'u': float(observation[1]),
                                       'SOC_new': float(observation2[2]),
                                       'soc_target': float(soc_target_H)}])

        if step > args.warmup:
            #--------------------Update----------------------
            if episode_steps % 5 == 4:
                # Curriculum convergence: halve d_block when EMA cost
                # <= d_block for 100 consecutive policy updates
                # H 层本轮实验: 更新频率 5 → 2 次/块 (块内前 2 次与 L 交替);
                # L 层维持每块 10 次 (UTD 2:1)。
                update_num = 0
                while update_num < 2:
                    agent_H.update_policy()
                    agent_L.update_policy()
                    update_num += 1

                # L 层补足至每块 10 次 (while 内已 2 次, 此处再 8 次; UTD 2:1)
                for _ in range(8):
                    agent_L.update_policy()

                # curriculum 计数保持原节奏 (每块 5 次 × 两 agent):
                # EMA 成本每块只在块末更新一次, 块内多次检查结果一致、仅推进计数器,
                # 故判定阈值 100 不变时, d_block 减半所需"满足块数"与改动前相同
                # (不随 H 更新次数下调而变化)。
                for _ in range(5):
                    # Track consecutive satisfaction for H (EMA 实际成本判据)
                    if agent_H.ema_cost is not None and agent_H.ema_cost <= d_block_H:
                        consecutive_satisfied_H += 1
                    else:
                        consecutive_satisfied_H = 0
                    # Track consecutive satisfaction for L (EMA 实际成本判据)
                    if agent_L.ema_cost is not None and agent_L.ema_cost <= d_block_L:
                        consecutive_satisfied_L += 1
                    else:
                        consecutive_satisfied_L = 0

                if consecutive_satisfied_H >= 100:
                    d_block_H = max(d_block_H / 2, d_block_H_min)  # H: 收紧块预算
                    consecutive_satisfied_H = 0
                    prGreen('H: d_block_H halved to {}'.format(d_block_H))
                if consecutive_satisfied_L >= 100:
                    d_block_L = max(d_block_L / 2, d_block_L_min)  # L: 收紧块预算
                    consecutive_satisfied_L = 0
                    prGreen('L: d_block_L halved to {}'.format(d_block_L))

                # ---- H 层更新数据记录（本块 H 更新批均值 → TB/csv） ----
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
                        # 增量落盘: 首段覆盖写, 其后追加新行 (避免每 5 步块全量重写增长表)
                        _up_csv = '{}/update_stats_H.csv'.format(output)
                        if df_update_flushed_rows == 0:
                            df_update.to_csv(_up_csv, index=False)
                        else:
                            df_update.iloc[df_update_flushed_rows:].to_csv(
                                _up_csv, mode='a', header=False, index=False)
                        df_update_flushed_rows = len(df_update)

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
                        # c1 = 结算总成本 (含块末 SOC 跟踪项); c1_soc_track 为其分量
                        'c1_soc_track': m['c1_soc_track'],
                        'profit': m['profit'],
                        'project_wind_levels': ','.join(m['project_wind_levels']),
                        'wind_low_max': overall['wind_low_max'],
                        'wind_high_min': overall['wind_high_min'],
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
                        'c1_soc_track': overall['c1_soc_track'],
                        'profit': overall['profit'],
                        'project_wind_levels': '',
                        'wind_low_max': overall['wind_low_max'],
                        'wind_high_min': overall['wind_high_min'],
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
                    writer.add_scalar('test9/overall/c1_soc_track', overall['c1_soc_track'], step)
                    writer.add_scalar('test9/overall/profit', overall['profit'], step)
                    writer.add_scalar('test9/wind_low_max', overall['wind_low_max'], step)
                    writer.add_scalar('test9/wind_high_min', overall['wind_high_min'], step)
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
                'clip_over': reward_details['clip_over'],   # ±10 夹掉的原始意图量 (MW)
                'clip_pen_capped': reward_details['clip_pen_capped'],  # 1=触到总上限
                'unfeasible_gap': rbb_info['unfeasible_gap'],
                'soc': rbb_info['soc'],
                'is_infeasible': float(rbb_info['is_infeasible']),
                'inevitable_violation': rbb_info['inevitable_violation'],
                # PRRBB wind×price segmentation info
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
                'soc_target': reward_details['soc_target'],
                'soc_track_err': reward_details['soc_track_err'],
                'c1_soc_track': c1_soc_track,   # 仅块末步非零 (结算项)
                'c1_fluct': reward_details['c1_fluct'],
                'penalty_NN_H': penalty_NN_H,
                'SOC': reward_details['SOC'],
                'P_F': reward_details['P_F'],
                'penalty_L': penalty_L,
                'penalty_NN_L': penalty_NN_L,
                # Convenience
                'soc_next': float(observation2[2]),
                # BESS 真实盈亏 (真实电价, 未加 wB 权重): -pr*pbrule*dt - 退化
                'bess_profit_net': (-reward_details['pr'] * reward_details['pbrule'] * time
                                    - reward_details['degradation']),
                # TD3PRLE: current wind thresholds
                'wind_low_max': get_wind_thresholds()[0],
                'wind_high_min': get_wind_thresholds()[1],
            }])], ignore_index=True)
            if step % 50 == 0:
                # 增量落盘: 首段覆盖写(截断旧文件), 其后只追加自上次落盘以来的新行。
                # 之前每 50 步对跨 episode 只增不减的整表 df 做全量 to_csv, 是 50
                # 整数倍 step 卡顿的根源; 崩溃时最多丢 <50 行, 与原先语义一致。
                _df_csv = '{}/df_td3prle.csv'.format(output)
                if df_flushed_rows == 0:
                    df.to_csv(_df_csv, index=False)
                else:
                    df.iloc[df_flushed_rows:].to_csv(
                        _df_csv, mode='a', header=False, index=False)
                df_flushed_rows = len(df)

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
            writer.add_scalar('soc_deviation', reward_details['soc_deviation'], step)
            writer.add_scalar('P_F', reward_details['P_F'], step)  # 电网功率波动

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
            writer.add_scalar('reward_lower/soc_deviation', reward_details['soc_deviation'], step)
            writer.add_scalar('reward_lower/soc_track_err', reward_details['soc_track_err'], step)
            writer.add_scalar('reward_lower/clip_amount', reward_details['clip_amount'], step)
            writer.add_scalar('reward_lower/clip_over', reward_details['clip_over'], step)
            writer.add_scalar('reward_lower/clip_pen_capped', reward_details['clip_pen_capped'], step)

            # SOC 目标 (H 信号映射后的物理目标) 与块末结算跟踪惩罚
            writer.add_scalar('action/soc_target', soc_target_H, step)
            writer.add_scalar('action/signal_soc', applied_action_H[5], step)
            writer.add_scalar('c1_soc_track', c1_soc_track, step)
            # RBB-specific metrics
            writer.add_scalar('rbb/clip_amount', clip_amount, step)
            writer.add_scalar('rbb/pbrule', float(pbrule), step)
            writer.add_scalar('rbb/pb_original', pb_original, step)
            writer.add_scalar('rbb/unfeasible_gap', rbb_info['unfeasible_gap'], step)
            writer.add_scalar('rbb/is_infeasible', float(is_infeasible), step)
            writer.add_scalar('rbb/inevitable_violation', inevitable_violation, step)

            # TD3PRLE: dynamic wind thresholds
            wind_low, wind_high = get_wind_thresholds()
            writer.add_scalar('prle/wind_low_max', wind_low, step)
            writer.add_scalar('prle/wind_high_min', wind_high, step)

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
                writer.add_scalar('action/soc_signal_raw',
                                  applied_action_H[5], step)
                writer.add_scalar('action/soc_target_mapped',
                                  soc_target_H, step)
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

        # ================================================================
        # TD3PRLE: LLM Threshold Adjustment at 7200-step (5-day) boundaries
        # ================================================================
        if step > 0 and step % 7200 == 0:
            prCyan('--- TD3PRLE: 7200-step window boundary (step {}) ---'.format(step))

            if episode >= min_episodes_before_llm:
                current_low_max, current_high_min = get_wind_thresholds()
                summary = window_stats.summary()

                prCyan('Window summary: r1_avg={}, viol_rate={}, HIGH_pct={}%'.format(
                    summary['r1_avg'], summary['violation_rate'],
                    summary['high_wind']['pct']))

                # Call LLM for threshold adjustment suggestions
                # 传入原设定阈值, 使提示词说明 ±1 m/s 调整带 (由 adj_bounds 中点还原)
                if adj_bounds is not None:
                    orig_low_max = (adj_bounds[0][0] + adj_bounds[0][1]) / 2.0
                    orig_high_min = (adj_bounds[1][0] + adj_bounds[1][1]) / 2.0
                else:
                    orig_low_max = orig_high_min = None
                adjust_low, adjust_high = llm_analyze_thresholds(
                    summary, threshold_history,
                    current_low_max, current_high_min,
                    llm_call_fn=llm_call_fn,
                    orig_low_max=orig_low_max,
                    orig_high_min=orig_high_min
                )

                # 阈值只能在 原设定 ±1 m/s 范围内调整 (整个训练过程硬约束)
                if adj_bounds is not None:
                    (low_min, low_max), (high_min, high_max) = adj_bounds
                    new_low_max = float(np.clip(current_low_max + adjust_low,
                                                low_min, low_max))
                    new_high_min = float(np.clip(current_high_min + adjust_high,
                                                 high_min, high_max))
                else:
                    new_low_max = current_low_max + adjust_low
                    new_high_min = current_high_min + adjust_high
                # set_wind_thresholds 内部仍有物理钳制兜底 (low ∈ [3, high-1] 等)
                actual_low, actual_high = set_wind_thresholds(new_low_max, new_high_min)

                if actual_low != current_low_max or actual_high != current_high_min:
                    prYellow('TD3PRLE: Thresholds changed: '
                             'LOW_MAX {:.1f}→{:.1f}, HIGH_MIN {:.1f}→{:.1f}'.format(
                                 current_low_max, actual_low,
                                 current_high_min, actual_high))

                    # ---- Relabel replay buffer rewards with new thresholds ----
                    n_relabeled = agent_H.memory.relabel_all(
                        relabel_r1_substep, relabel_c1_substep)
                    prGreen('TD3PRLE: Relabeled {} H-agent transitions'.format(n_relabeled))
                    # L-agent buffer: r2 与阈值无关; c2 = 10·(SOC_new−soc_target)² 的目标中心
                    # 二次势也只依赖存档的 soc_target (与风速无关) → relabel 为恒等,
                    # 保留调用仅为与 relabel_c2_substep 接口一致。
                    n_relabeled_L = agent_L.memory.relabel_c2_all(relabel_c2_substep)
                    prGreen('TD3PRLE: Relabeled {} L-agent costs'.format(n_relabeled_L))

                    # Record this adjustment in history
                    threshold_history.append({
                        'step': step,
                        'episode': episode,
                        'old_low_max': current_low_max,
                        'new_low_max': actual_low,
                        'old_high_min': current_high_min,
                        'new_high_min': actual_high,
                        'summary': summary,
                    })
                else:
                    prCyan('TD3PRLE: No threshold change needed')
            else:
                prCyan('TD3PRLE: Skipping LLM analysis (cold start: episode {} < {})'.format(
                    episode, min_episodes_before_llm))

            # Reset window statistics for the next window
            window_stats.reset()

        # update
        step += 1
        episode_steps += 1
        episode_reward += r
        episode_cost += c
        episode_cost2 += c2
        # 上层利润: pg * 实际电价 * dt - BESS_DEG_RMB_PER_MWH * pb * dt
        episode_profit += observation2[-1] * observation[0] * time - BESS_DEG_RMB_PER_MWH * max(-float(pbrule), 0) * time
        # BESS盈亏: 真实电价下的 BESS 净收益 - 退化成本
        # (信号已改为 SOC 目标, r2 不再含影子套利项 → 此处改用真实电价口径)
        episode_bess_profit += -observation[0] * float(pbrule) * time - BESS_DEG_RMB_PER_MWH * max(-float(pbrule), 0) * time
        # update env
        norm_observation = deepcopy(norm_observation2)
        observation = deepcopy(observation2)
        pw_tru = pw_tru2

        #-------------------------------------episode-----------------------------
        if done:
            # save intermideate model
            agent_H.save_model(output)
            agent_L.save_model(output)
            prGreen('##########{}: episode_profit:{} episode_reward:{} step:{}'.format(
                episode, round(episode_profit, 2), round(episode_reward, 2), step))

            if upload:
                writer.add_scalar('episode_reward', episode_reward, episode)
                writer.add_scalar('episode_cost', episode_cost, episode)
                writer.add_scalar('episode_cost2', episode_cost2, episode)
                writer.add_scalar('episode_profit', episode_profit, episode)
                writer.add_scalar('env/bessprofit', episode_bess_profit, episode)
                writer.add_scalar('wind_day', env.j, episode)
                # TD3PRLE: log thresholds per episode
                wind_low, wind_high = get_wind_thresholds()
                writer.add_scalar('prle/episode_wind_low_max', wind_low, episode)
                writer.add_scalar('prle/episode_wind_high_min', wind_high, episode)

            if use_csv:
                df_episode = pd.concat([df_episode, pd.DataFrame([{
                    'episode': episode,
                    'episode_reward': episode_reward,
                    'episode_cost': episode_cost,
                    'episode_cost2': episode_cost2,
                    'episode_profit': episode_profit,
                    'bess_profit': episode_bess_profit,
                    'wind_day': env.j,
                    'wind_low_max': get_wind_thresholds()[0],
                    'wind_high_min': get_wind_thresholds()[1],
                }])], ignore_index=True)
                df_episode.to_csv('{}/df_episode_td3prle.csv'.format(output))

            # end of episode
            # reset
            # 下个 episode 起点 SOC 由循环顶部在 [0.5, 5.5] 内随机重置，不再继承
            observation = None
            episode_steps = 0
            episode_reward = episode_cost = episode_profit = 0.
            episode_bess_profit = 0.
            episode += 1
    if use_csv:
        df.to_csv('{}/df_td3prle.csv'.format(output))
        df_episode.to_csv('{}/df_episode_td3prle.csv'.format(output))
        df_test.to_csv('{}/df_test9.csv'.format(output))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode',
                        default='train',
                        type=str,
                        help='support option: train/test')
    parser.add_argument('--env',
                        default='TD3PRLE',
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
    # L 层(BESS)专用策略更新超参 (步骤A: 验证 TD3 smoothing/delay 拖慢下层跟随 pbrule 的假设)
    # H 层保持 TD3 默认 (--policy_delay 2, --policy_noise 0.1); L 层放松为接近 DDPG 节奏:
    #   --l_policy_delay 1 → 每批 critic 更新后都更新 actor; --l_policy_noise 0.01 ≈ 关闭 target smoothing
    parser.add_argument('--l_policy_delay', default=2, type=int,
                        help='L 层策略更新间隔: 1=每批更新(≈DDPG节奏, 步骤A) / 2=TD3延迟(默认, 与H一致)')
    parser.add_argument('--l_policy_noise', default=0.1, type=float,
                        help='L 层 target smoothing noise std (动作幅度比例): 0.01≈关闭(步骤A) / 0.1=TD3默认')
    # L 层折扣 (步级语义) 与 H 层解耦: L 层每 5 步块即终止 (done_L=True),
    # bootstrap 视野 ≤5 步, γ 只影响块内 5 步加权 → 取 0.98 (与 H 的块级 γ 同值)。
    parser.add_argument('--l_discount', default=0.98, type=float,
                        help='L 层(BESS)折扣: 步级语义; L episode=5 步块, 视野≤5 步, 默认 0.98')
    parser.add_argument('--cost_critic_mode', default='max', type=str,
                        choices=['min', 'max'],
                        help='cost twin-critic aggregation: max (safe, recommended) or min (TD3 paper; risks cost underestimation -> Q_c collapse)')

    # RBB parameters (hard clipping + feasible-range observation inputs)
    parser.add_argument('--rbb_fluctuation_limit', default=3.0, type=float,
                        help='Max grid power fluctuation between steps (MW)')
    parser.add_argument('--rbb_rated_power', default=10.0, type=float,
                        help='BESS rated charge/discharge power (MW)')

    # TD3PRLE: LLM configuration
    parser.add_argument('--llm_provider', default=None, type=str,
                        choices=['anthropic', 'openai', 'deepseek'],
                        help='LLM API provider (overrides config file). '
                             'Default: read from td3prle/llm_config.json')
    parser.add_argument('--llm_config', default=None, type=str,
                        help='Path to LLM config JSON file. '
                             'Default: td3prle/llm_config.json')

    # 风速分段起点分岔: percentile(33/66分位) / kmeans(簇中心中点) / expert(专家锚点 8.5/12.0)
    parser.add_argument('--seg', default='expert',
                        choices=['percentile', 'kmeans', 'expert'],
                        help='风速分段起点方法: percentile | kmeans | expert (默认 expert); '
                             '启动后由 LLM 元控制器在此基础上在线调整')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    # ---- 风速分段起点: 按 --seg 应用（percentile / kmeans / expert），LLM 在此基础上微调 ----
    from data_piecewise.boundaries import apply_segmentation
    apply_segmentation(args.seg, set_wind_thresholds)

    # ---- 阈值调整带: 整个训练过程中, LLM 调整不得越出 原设定 ±1 m/s ----
    orig_low_max, orig_high_min = get_wind_thresholds()
    adj_bounds = ((orig_low_max - 1.0, orig_low_max + 1.0),
                  (orig_high_min - 1.0, orig_high_min + 1.0))
    prCyan('TD3PRLE: 阈值原设定 LOW_MAX={:.1f} HIGH_MIN={:.1f} m/s; '
           'LLM 调整带 = 原设定 ±1 m/s → LOW∈[{:.1f},{:.1f}] HIGH∈[{:.1f},{:.1f}]'.format(
               orig_low_max, orig_high_min,
               orig_low_max - 1.0, orig_low_max + 1.0,
               orig_high_min - 1.0, orig_high_min + 1.0))

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

    # ---- H 层收益权重留痕 (【本轮变更】wB 与 c1 波动项的分段系数已去掉) ----
    prCyan('H 层收益权重: wT=wT_u(u)×wT_p(pr), wB≡1.0, c1 波动项系数≡1.0 '
           '(WB_SOC_OUTSIDE={}, WB_PRICE={}), WT_WIND={}, WT_PRICE={}'.format(
               WB_SOC_OUTSIDE, WB_PRICE, WT_WIND, WT_PRICE))

    env = WindBattery(max_E, args.nb_turbine, upload, engine)
    env_test = WindBattery(max_E, args.nb_turbine, upload, engine)

    if args.seed > 0:
        np.random.seed(args.seed)

    # RBB system (hard clipping before env.step + L 层可行域观测输入)
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
    #     lam1 ∈ [50, 500] 初始 100.0 (高初值+高下限: 对偶恒处强惩罚区, 上限放宽留裕度)
    agent_H = SP_TD3_PRLE(nb_states = 4, nb_actions = 6, args = args,
                          lam_max=500.0, lam_min=50.0,
                          ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                          record_update_stats=True,
                          policy_delay=args.policy_delay,
                          policy_noise=args.policy_noise,
                          noise_clip=args.noise_clip,
                          cost_critic_mode=args.cost_critic_mode)
    agent_H.lam = 100.0
    # L 层对偶改 EMA 实际成本更新 —— lam2 不再读 critic (Q_c), 与 td3prrbb 同配置;
    #     lam2 ∈ [2, 200] 初始 100.0 (min=2: 带内拖拽 < 套利边际, 保留 ±2 软裕度;
    #     max=200 顶配), ema_alpha=0.02, ema_lam_lr=0.01
    #     蒸馏观测 5 维 = [pbrulemin/10, pbrulemax/10, soc_target/10, SOC/10, u/10]
    # L 层(步骤A): policy_delay/--policy_noise 独立于 H —— 放松到 delay=1+noise≈0 时节奏接近 DDPG
    # (twin critic 与 EMA 对偶仍保留, 仅去掉延迟更新与 target smoothing 对窄可行窗的抹平)
    agent_L = SP_TD3_PRLE(nb_states = 5, nb_actions = 1, args = args,
                          lam_max=200.0, lam_min=2.0,
                          ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                          policy_delay=args.l_policy_delay,
                          policy_noise=args.l_policy_noise,
                          noise_clip=args.noise_clip,
                          cost_critic_mode=args.cost_critic_mode)
    agent_L.lam = 100.0
    agent_L.cost_limit = 10.0  # L层c2专用初始值（覆盖全局cost_single_limit=1000）
    # L 层折扣与 H 解耦: 步级语义, L episode = 5 步块 → bootstrap 视野 ≤5 步
    agent_L.discount = args.l_discount
    prYellow('L 层(BESS)策略超参: policy_delay={}, policy_noise={} (H 层 TD3: delay={}, noise={}); '
             '折扣: L γ={} (步级, 块内 ≤5 步) / H γ={} (块级)'.format(
                 args.l_policy_delay, args.l_policy_noise, args.policy_delay, args.policy_noise,
                 agent_L.discount, agent_H.discount))
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
              use_csv=use_csv,
              adj_bounds=adj_bounds)
    if upload:
        writer.close()

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
