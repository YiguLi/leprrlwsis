#!/usr/bin/env python3
"""
TD3LEPRRLRB (TD3 + LLM-Enhanced Piecewise Reward + Rule-Based BESS)
==================================================================
= TD3PRRLRB + PRLE 的 LLM 元控制器层 (关系类比 td3prrbb → td3prle)。

架构 (与 td3prrlrb 完全一致):
H (RL, 6 维动作 = 5 台风机桨距 + 1 个 SOC 目标信号):
    * 唯一决策者: 追求收益, 并为一个"不可行的 SOC 目标"付费。
    * a6 ∈ [-0.5, 0.5] → soc_target = 3 + 5*a6 ∈ [0.5, 5.5], 块级(5 步)固定。
L (规则, SocTracker, 1 维动作 pb):
    * 在 RBB 可行窗口 [pbrulemin, pbrulemax] 内以【最大能力】跟踪 soc_target,
      死区 0.1 MWh (=6 MW·min, 与控制波动裕度同源)。
    * 物理可行 + 平抑波动由窗口硬保证 → clip_amount ≡ 0;
      因此块末残差 |soc_target - SOC_end| ≈ 不可行量。

奖励 / 成本
-----------
H: r1 = wT*(pr*ptur*dt) + wB*(-pr*pbrule*dt) - 490*max(-pbrule,0)*dt - penalty_NN_H
    c1 = C1_RISK[wind]*[10*(P_F-3)+ + 0.1*P_F] + 10*unfeasible_gap
         + [块末结算] 10*max(0, |soc_target - SOC_end| - SOC_TOL)

对偶上升 (仅 H, EMA 实际块成本)
    EMA  ← (1-0.02)*EMA + 0.02*sum_c1
    lam1 ← clip(lam1 + 0.01*(EMA - d_block_H), 3, 100), 初始 5.0
    d_block_H 初值 50.0, 地板 5.0; 连续 100 次满足则减半

【本项目新增: LLM 分段微调层】
    1. 风速分段阈值不再是常量: 启动时按 --seg (percentile/kmeans/expert) 定起点
       (模块默认 7.0/11.0, 即 td3prrlrb 的写死值);
    2. 每 --llm_window 步 (默认 7200 = 5 天) 且 episode >= --llm_min_episodes 时,
       把窗口统计交给 LLM, 由它给出 (ΔLOW_MAX, ΔHIGH_MIN);
    3. 硬约束: 整个训练过程阈值只能在【原设定 ±1 m/s】内移动;
    4. 阈值实际变化后, 立刻对 H 层回放池 relabel_all() 用新阈值重算历史 (r1, c1),
       避免非平稳奖励把 critic 拟合目标搞乱;
    5. LLM 不可用 (未装 SDK / 未配 key / 连通性失败) 时退化为占位模式
       (返回 0 调整), 训练照常进行 —— 此时行为等价于 td3prrlrb。
"""
import os
import sys
import numpy as np
import argparse
from copy import deepcopy
import torch

# Ensure both td3leprrlrb/ and parent (HSRL) directories are on path
_TD3LEPRRLRB_DIR = os.path.dirname(os.path.abspath(__file__))
_HSRL_DIR = os.path.dirname(_TD3LEPRRLRB_DIR)
sys.path.insert(0, _TD3LEPRRLRB_DIR)
sys.path.insert(0, _HSRL_DIR)

from env import WindBattery
from td3leprrlrb.td3leprrlrb_ddpg import SP_TD3_LEPRRLRB
from util import *
from td3leprrlrb.td3leprrlrb_myfunc import (
    CheckAction, Normalize,
    Cal_Rew_RLRB, gap_penalty, soc_target_from_signal, nine_case_test,
    SOC_TOL,
    get_wind_thresholds, set_wind_thresholds,
    relabel_r1_substep, relabel_c1_substep,
    WT_WIND, WT_PRICE, WB_PRICE, WB_SOC_OUTSIDE,
)
from td3leprrlrb.td3leprrlrb_llm import (
    WindowStats, llm_analyze_thresholds, create_llm_call_fn,
)
from td3leprrlrb.td3leprrlrb_rule_bess import (
    RuleBasedBESS, SocTracker, block_cap_charge, block_cap_discharge,
)

import pandas as pd

from torch.utils.tensorboard import SummaryWriter

# 不显示warning
import warnings
warnings.filterwarnings("ignore")

import matlab
import matlab.engine
engine = matlab.engine.start_matlab()


RUN_TAG = 'td3leprrlrb'


def _flush_csv(path, rows, written):
    """把累积的 rows 以【追加】方式写入 CSV, 只写新增行 (首次写表头)。

    避免"每 N 步整表重写"在高迭代数时产生的卡顿 (行数线性增长)。
    """
    if not rows:
        return written
    chunk = pd.DataFrame(rows)
    chunk.index = range(written, written + len(rows))
    if written == 0:
        if os.path.exists(path):
            os.remove(path)
        chunk.to_csv(path, mode='w', index=True)
    else:
        chunk.to_csv(path, mode='a', header=False, index=True)
    return written + len(rows)


# TD3LEPRRLRB: TD3 + 分段奖励 + 规则下层 + LLM 分段微调
def train(num_iterations_H,
          agent_H,
          tracker,
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
    step = episode = episode_steps = 0
    episode_reward = episode_cost = episode_profit = 0.
    episode_bess_profit = 0.
    episode_gap_pen = 0.
    episode_deg_charge = episode_deg_discharge = 0.
    observation = None
    time = 1/60
    # Curriculum learning: consecutive blocks where EMA cost <= d_block_H
    consecutive_satisfied_H = 0
    # H 层 EMA 对偶 —— 每 5 步块实际成本预算 d_block_H (与 sum_c1 同口径)
    #   sum_c1 的构成: 波动成本 + 10*unfeasible_gap + 块末 SOC 跟踪罚
    d_block_H = 50.0        # 初值: 每块 sum_c1 预算 (先松后紧)
    d_block_H_min = 5.0     # 地板(min): 10*max(0,|δ|-0.1) ≤ 5 → 含死区总 |δ| ≤ 0.6

    # ---- LEPRRLRB: LLM 元控制器状态 ----
    window_stats = WindowStats()          # 一个 LLM 调整窗口内的运行统计
    threshold_history = []                # 历次阈值调整记录 (喂回提示词)
    block_ctx_H = []                      # 当前 5 步块的逐子步 relabel 上下文
    llm_window = int(getattr(args, 'llm_window', 7200))
    min_episodes_before_llm = int(getattr(args, 'llm_min_episodes', 3))
    llm_relabel_count = 0                 # 阈值变更次数 (落盘用)

    # ---- LEPRRLRB: 建立 LLM 调用函数 (默认读 td3leprrlrb/llm_config.json) ----
    llm_call_fn = None
    try:
        llm_call_fn = create_llm_call_fn(provider=args.llm_provider,
                                        config_path=args.llm_config)
        prCyan('LEPRRLRB: LLM ready (provider={})'.format(
            args.llm_provider or 'from config'))
    except Exception as e:
        prRed('LEPRRLRB: LLM 初始化失败 ({}) — 退化为占位模式 (阈值不调整)'.format(e))

    # ---- LEPRRLRB: 启动连通性自检 (失败则退化为占位模式) ----
    if llm_call_fn is not None:
        prCyan('LEPRRLRB: Testing LLM API connectivity...')
        try:
            test_response = llm_call_fn('请回复"OK"，不要包含任何其他内容。')
            prGreen('LEPRRLRB: LLM API test passed. Response: "{}"'.format(
                str(test_response).strip()[:120]))
        except Exception as e:
            prRed('LEPRRLRB: LLM API test FAILED — {}'.format(e))
            prRed('LEPRRLRB: 训练继续, 但阈值调整将被跳过。')
            llm_call_fn = None

    if use_csv:
        # 增量落盘: 只把新增行追加进 CSV
        df_rows = []
        df_written = 0
        df_episode_rows = []
        df_episode_written = 0
        df_test_rows = []
        df_test_written = 0
        df_update_rows = []   # H 层更新数据（TD/Q/batch 统计）
        df_update_written = 0
        df_llm_rows = []      # LLM 阈值调整记录
        df_llm_written = 0

    # ---- 1440-step window counters for env metrics ----
    window_violation_count = 0   # steps where |P_F| > 3 MW
    window_power_sum = 0.0        # cumulative P_G over the window

    while step < num_iterations_H:
        # start of episode 重置环境
        prYellow('########### step:{} ###########'.format(step))
        if observation is None:
            observation = deepcopy(env.reset())  # [Pr,U,E,P_last]
            # SOC 随机化: 每个 episode(1440步) 从 U[0.5,5.5] 重新采样
            reset_E = float(np.random.uniform(0.5, 5.5))
            env.E = reset_E
            observation[2] = reset_E
            norm_observation = Normalize(observation)
            agent_H.reset(norm_observation)

        # ================================================================
        # 上层 H: 每 5 步决策一次 (块级)
        # ================================================================
        if episode_steps % 5 == 0:
            sum_r1 = 0
            sum_c1 = 0
            sum_gap_pen = 0.0
            block_ctx_H = []          # 新块: 重置 relabel 上下文
            if step <= args.warmup:
                applied_action_H = original_action_H = agent_H.random_action()
                penalty_NN_H = 0
            else:
                original_action_H = agent_H.select_action(norm_observation)
                applied_action_H, penalty_NN_H = CheckAction(original_action_H)
            norm_observation_H = norm_observation
            pw_tru = env.PowerOutput(observation[1], applied_action_H)
            # SOC 目标 (块级固定)
            soc_signal = float(applied_action_H[5])
            soc_target = soc_target_from_signal(soc_signal)

        prGreen('--H-- ori a_t = {}'.format(round_list(original_action_H, 3)))

        ptur = float(pw_tru)
        pg_prev = float(observation[-1])
        E_current = float(observation[2])

        # 可行窗口 (与 pb 无关, 预计算) —— 供规则控制器与可行性诊断使用
        _, rbb_info_obs = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=ptur, pg_prev=pg_prev, E=E_current)

        # ================================================================
        # 下层 L: 规则控制器 (每一步执行, 窗口内最大能力跟踪 soc_target)
        # ================================================================
        pb_cmd = tracker.act(E_current, soc_target,
                             rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'])
        applied_action_L = np.array([pb_cmd])

        action = np.concatenate((applied_action_H, applied_action_L))
        original_action = action.copy()

        # ================================================================
        # RBB: Rule-Based BESS 硬限幅 (安全性兜底)
        # ================================================================
        pb_original = float(action[-1])
        pbrule, rbb_info = rbb_system.compute_pbrule(
            pb_original=pb_original,
            ptur=ptur,
            pg_prev=pg_prev,
            E=E_current,
        )

        action_rbb = action.copy()
        action_rbb[-1] = pbrule

        # env response with next_observation, reward, terminate_info
        pw_tru2, observation2, action_L_env, penalty_L, done, done_L, info = \
            env.step(action_rbb[:6], action_rbb[6:], pw_tru)
        action_rbb[6:] = action_L_env
        prGreen('Applied action = {}.'.format(round_list(action_rbb, 3)))
        norm_observation2 = Normalize(observation2)

        # ================================================================
        # PRRBB 分段奖励 (机会/风险分离; 仅 H 层)
        # ================================================================
        r1, c1_step, reward_details = Cal_Rew_RLRB(
            observation, observation2, action_rbb, rbb_info,
            penalty_NN_H=penalty_NN_H, time=time,
            ptur=ptur, pb_cmd=pb_cmd,
        )

        sum_r1 += r1
        sum_c1 += c1_step

        # ---- 块末结算: SOC 目标可行性罚 ----
        gap_raw = abs(soc_target - float(observation2[2]))
        gap_pen = 0.0
        target_feasible = True
        if episode_steps % 5 == 4:
            # tol 必须与下层规则死区同源 (tracker.soc_tol), 否则 --soc_tol 改动后
            # 规则停在死区边缘时上层会被收一笔永远消不掉的"幽灵税"。
            gap_pen = gap_penalty(soc_target, float(observation2[2]),
                                  tol=tracker.soc_tol)
            sum_c1 += gap_pen
            sum_gap_pen += gap_pen

        # ---- 可达性诊断: 规则控制在本块的最大能力 ----
        cap_c = block_cap_charge(rbb_info_obs['pbrulemax'], eta_c=0.98, time=time)
        cap_d = block_cap_discharge(rbb_info_obs['pbrulemin'], eta_d=0.98, time=time)
        target_feasible = (soc_target <= E_current + cap_c + tracker.soc_tol) and \
                          (soc_target >= E_current - cap_d - tracker.soc_tol)

        clip_amount = rbb_info['clip_amount']
        is_infeasible = rbb_info['is_infeasible']
        inevitable_violation = rbb_info['inevitable_violation']

        prGreen('r1 = {} (turb={}, bess={}, deg={})'.format(
            round(r1, 3), round(reward_details['turbine_profit'], 3),
            round(reward_details['bess_profit'], 3),
            round(reward_details['degradation'], 3)))
        prGreen('c1 = {} (fluct={}, gap_pen={})'.format(
            round(sum_c1 if episode_steps % 5 == 4 else c1_step, 3),
            round(reward_details['c1_fluct'], 3), round(gap_pen, 4)))
        prGreen('soc_signal={:.4f} soc_target={:.3f} SOC={:.3f} gap_raw={:.3f} '
                '(tol={}, feasible={})'.format(
                    soc_signal, soc_target, reward_details['SOC'],
                    gap_raw, tracker.soc_tol, target_feasible))
        prGreen('price={} p_level={} wind={:.1f} w_level={} (wT={}, wB={})'.format(
            reward_details['pr'], reward_details['price_level'],
            reward_details['u'], reward_details['wind_level'],
            reward_details['w_wind'], reward_details['w_bess']))

        # ---- 1440-step window: track actual grid fluctuation ----
        if reward_details['P_F'] > 3.0 + 1e-4:
            window_violation_count += 1
        window_power_sum += observation2[-1]  # cumulative P_G

        # ================================================================
        # LEPRRLRB: 逐子步 relabel 上下文 (阈值变动后据此精确重算历史 r1/c1)
        #   与阈值无关的原料全部缓存: pr·ptur·dt / −pr·pbrule·dt / 退化 /
        #   越界罚; 阈值相关的只有分档权重与风险系数。
        #   SOC 也一并缓存: 当前 WB_SOC_OUTSIDE=1.0 下 wB 已不依赖它, 但保留该
        #   字段才能让"改回 0.7 折减"的回退路径 relabel 依然精确。
        #   gap_pen 在块末回填到本块最后一个子步 (见下方 episode_steps%5==4)。
        # ================================================================
        step_profit = (observation2[-1] * observation[0] * time
                       - 490.0 * max(-pbrule, 0) * time)
        block_ctx_H.append({
            'u': float(reward_details['u']),
            'pr': float(reward_details['pr']),
            'turbine_raw': float(reward_details['pr'] * ptur * time),
            'bess_raw': float(-reward_details['pr'] * pbrule * time),
            'degradation': float(reward_details['degradation']),
            'penalty_NN_H': float(penalty_NN_H),
            'SOC': float(reward_details['SOC_decision']),
            'P_F': float(reward_details['P_F']),
            'unfeasible_gap': float(rbb_info['unfeasible_gap']),
            'gap_pen': 0.0,
        })

        # ---- LEPRRLRB: LLM 窗口统计 (逐步指标) ----
        window_stats.add(r1, reward_details['wind_level'],
                         reward_details['P_F'], rbb_info['is_infeasible'],
                         profit=step_profit)

        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        if episode_steps % 5 == 4:
            # H 层 5 步结算: 已把 gap_pen 计入 sum_c1
            # LEPRRLRB: 块末 SOC 目标罚回填进本块最后一个子步的上下文,
            # 使 relabel_all() 重算成本时不会丢掉这一项 (它与风速阈值无关)。
            block_ctx_H[-1]['gap_pen'] = float(gap_pen)
            window_stats.add_block_end(gap_pen, gap_raw)
            agent_H.observe_jump(norm_observation_H, original_action_H,
                                 sum_r1, sum_c1, norm_observation2, done,
                                 context=list(block_ctx_H))
            # H 层 EMA 对偶上升 —— 用块实际成本 sum_c1 更新 lam1 (与 d_block_H 同口径)
            if step > args.warmup:
                agent_H.update_ema_cost(sum_c1)
                agent_H.update_lam_ema(d_block_H)

        if step > args.warmup:
            #--------------------Update----------------------
            if episode_steps % 5 == 4:
                # H 层更新频率: 每块 2 次
                for _ in range(2):
                    agent_H.update_policy()
                    if agent_H.ema_cost is not None and agent_H.ema_cost <= d_block_H:
                        consecutive_satisfied_H += 1
                    else:
                        consecutive_satisfied_H = 0

                if consecutive_satisfied_H >= 100:
                    d_block_H = max(d_block_H / 2, d_block_H_min)
                    consecutive_satisfied_H = 0
                    prGreen('H: d_block_H halved to {}'.format(d_block_H))

                # ---- H 层更新数据记录（本块 H 2 批均值 → TB/csv） ----
                if agent_H.record_update_stats and len(agent_H.update_stats) > 0:
                    _stats = agent_H.update_stats
                    _mean = {k: float(np.mean([s[k] for s in _stats]))
                             for k in _stats[0]}
                    agent_H.update_stats = []
                    if upload:
                        for _k, _v in _mean.items():
                            writer.add_scalar('update/' + _k, _v, step)
                    if use_csv:
                        df_update_rows.append({'step': step, **_mean})
                        df_update_written = _flush_csv(
                            '{}/update_stats_H.csv'.format(output),
                            df_update_rows, df_update_written)
                        df_update_rows = []

            # Test: 9-case segmented evaluation every 480 episode steps
            if episode_steps % 480 == 0:
                case_metrics, overall = nine_case_test(
                    agent_H, tracker, env_test, rbb_system, seed=42)
                if use_csv:
                    for m in case_metrics:
                        df_test_rows.append({
                            'step': step,
                            'wind_level': m['wind_level'],
                            'price_level': m['price_level'],
                            'r1': m['r1'], 'c1': m['c1'],
                            'gap_pen': m['gap_pen'],
                            'cvm': m['cvm'],
                            'pf_viol': m['pf_viol'],
                            'profit': m['profit'],
                            'soc_track_err': m['soc_track_err'],
                            # LEPRRLRB: 该 case 在【当前阈值】下实际落入的档
                            'project_wind_levels':
                                ','.join(m['project_wind_levels']),
                            'wind_low_max': overall['wind_low_max'],
                            'wind_high_min': overall['wind_high_min'],
                        })
                    df_test_rows.append({
                        'step': step,
                        'wind_level': 'OVERALL',
                        'price_level': 'OVERALL',
                        'r1': overall['r1'], 'c1': overall['c1'],
                        'gap_pen': overall['gap_pen'],
                        'cvm': overall['cvm'],
                        'pf_viol': overall['pf_viol'],
                        'profit': overall['profit'],
                        'soc_track_err': overall['soc_track_err'],
                        'project_wind_levels': '',
                        'wind_low_max': overall['wind_low_max'],
                        'wind_high_min': overall['wind_high_min'],
                    })
                    df_test_written = _flush_csv(
                        '{}/df_test9.csv'.format(output),
                        df_test_rows, df_test_written)
                    df_test_rows = []
                if upload:
                    writer.add_scalar('test9/overall/r1', overall['r1'], step)
                    writer.add_scalar('test9/overall/c1', overall['c1'], step)
                    writer.add_scalar('test9/overall/gap_pen', overall['gap_pen'], step)
                    writer.add_scalar('test9/overall/cvm', overall['cvm'], step)
                    writer.add_scalar('test9/overall/pf_viol', overall['pf_viol'], step)
                    writer.add_scalar('test9/overall/profit', overall['profit'], step)
                    writer.add_scalar('test9/overall/soc_track_err',
                                      overall['soc_track_err'], step)
                    writer.add_scalar('test9/wind_low_max',
                                      overall['wind_low_max'], step)
                    writer.add_scalar('test9/wind_high_min',
                                      overall['wind_high_min'], step)
                    for m in case_metrics:
                        tag = 'test9/{}_{}'.format(m['wind_level'], m['price_level'])
                        writer.add_scalar(tag + '/r1', m['r1'], step)
                        writer.add_scalar(tag + '/c1', m['c1'], step)
                        writer.add_scalar(tag + '/gap_pen', m['gap_pen'], step)
                        writer.add_scalar(tag + '/cvm', m['cvm'], step)
                        writer.add_scalar(tag + '/pf_viol', m['pf_viol'], step)
                        writer.add_scalar(tag + '/profit', m['profit'], step)
                        writer.add_scalar(tag + '/soc_track_err',
                                          m['soc_track_err'], step)

        # 上层保存到data —— 只累积行, 每 50 步增量 flush
        if use_csv:
            df_rows.append({
                'step': step,
                'observation': observation,
                'observation2': observation2,
                'original_action_H': original_action_H,
                'applied_action_H': applied_action_H,
                'action_pb_cmd': pb_cmd,
                'action_rbb': action_rbb,
                'r1': r1, 'c1_step': c1_step,
                'c1': c1_step + gap_pen,
                'gap_raw': gap_raw, 'gap_pen': gap_pen,
                'target_feasible': float(target_feasible),
                'lam1': agent_H.lam,
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
                # 规则下层能力诊断
                'd_cap_c': cap_c,
                'd_cap_d': cap_d,
                # PRRBB wind×price segmentation info
                'wind_level': reward_details['wind_level'],
                'price_level': reward_details['price_level'],
                # LEPRRLRB: 本步所用阈值 (LLM 微调后可回溯每一步的分档依据)
                'wind_low_max': get_wind_thresholds()[0],
                'wind_high_min': get_wind_thresholds()[1],
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
                'soc_signal': reward_details['soc_signal'],
                'soc_target': reward_details['soc_target'],
                'soc_delta': reward_details['soc_delta'],
                'c1_fluct': reward_details['c1_fluct'],
                'penalty_NN_H': penalty_NN_H,
                'SOC': reward_details['SOC'],
                'P_F': reward_details['P_F'],
                'soc_next': float(observation2[2]),
                'bess_profit_net': reward_details['bess_profit'] - reward_details['degradation'],
            })
            if step % 50 == 0:
                df_written = _flush_csv(
                    '{}/df_{}.csv'.format(output, RUN_TAG), df_rows, df_written)
                df_rows = []

        # ---- Periodic checkpoint (every 2000 steps) ----
        if step > 0 and step % 2000 == 0:
            agent_H.save_checkpoint(output, step)
            prCyan('Checkpoint saved at step {}'.format(step))

        if upload:
            writer.add_scalar('r1', r1, step)
            writer.add_scalar('c1', c1_step, step)
            writer.add_scalar('gap_pen', gap_pen, step)
            writer.add_scalar('gap_raw', gap_raw, step)
            writer.add_scalar('wind_speed', env.u, step)
            writer.add_scalar('reward', r1, step)
            writer.add_scalar('cost', c1_step, step)
            writer.add_scalar('lam1', agent_H.lam, step)
            writer.add_scalar('d1', d_block_H, step)
            if agent_H.ema_cost is not None:
                writer.add_scalar('ema_c1_H', agent_H.ema_cost, step)
            writer.add_scalar('soc', float(observation[2]), step)
            writer.add_scalar('soc_deviation', reward_details['soc_deviation'], step)
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
            # LEPRRLRB: 当前动态阈值 (LLM 分段微调轨迹)
            writer.add_scalar('seg/wind_low_max', get_wind_thresholds()[0], step)
            writer.add_scalar('seg/wind_high_min', get_wind_thresholds()[1], step)

            # 收益分解
            writer.add_scalar('reward_upper/turbine_profit', reward_details['turbine_profit'], step)
            writer.add_scalar('reward_upper/bess_profit', reward_details['bess_profit'], step)
            writer.add_scalar('reward_upper/degradation', reward_details['degradation'], step)

            # SOC 目标跟踪
            writer.add_scalar('target/soc_target', soc_target, step)
            writer.add_scalar('target/soc_signal', soc_signal, step)
            writer.add_scalar('target/gap_raw', gap_raw, step)
            writer.add_scalar('target/gap_pen', gap_pen, step)
            writer.add_scalar('target/target_feasible', float(target_feasible), step)
            writer.add_scalar('target/d_cap_c', cap_c, step)
            writer.add_scalar('target/d_cap_d', cap_d, step)
            writer.add_scalar('target/pb_cmd', pb_cmd, step)

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
                writer.add_scalar('sum_gap_pen', sum_gap_pen, step)
                for i in range(5):
                    writer.add_scalar('action/a_H_original_{}'.format(i),
                                      original_action_H[i], step)
                    writer.add_scalar('action/a_H_clipped_{}'.format(i),
                                      applied_action_H[i], step)
                writer.add_scalar('action/soc_signal',
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
        # LEPRRLRB: LLM 阈值微调 (每 llm_window 步 = 默认 5 天窗口边界)
        # ================================================================
        if step > 0 and step % llm_window == 0:
            prCyan('--- LEPRRLRB: {} 步窗口边界 (step {}) ---'.format(
                llm_window, step))

            if episode >= min_episodes_before_llm:
                current_low_max, current_high_min = get_wind_thresholds()
                summary = window_stats.summary()

                prCyan('窗口统计: r1_avg={}, profit_avg={}, 越限率={}, HIGH 占比={}%, '
                       'gap罚/块={}'.format(
                           summary['r1_avg'], summary['profit_avg'],
                           summary['violation_rate'],
                           summary['high_wind']['pct'],
                           summary['gap_pen_per_block']))

                # 由 ±1 m/s 调整带还原"原设定"给提示词用
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
                    orig_high_min=orig_high_min)

                # 硬约束: 整个训练过程只能在 原设定 ±1 m/s 内移动
                if adj_bounds is not None:
                    (low_min, low_max), (high_min, high_max) = adj_bounds
                    new_low_max = float(np.clip(current_low_max + adjust_low,
                                                low_min, low_max))
                    new_high_min = float(np.clip(current_high_min + adjust_high,
                                                 high_min, high_max))
                else:
                    new_low_max = current_low_max + adjust_low
                    new_high_min = current_high_min + adjust_high

                # set_wind_thresholds 内部还有物理钳制兜底 (low∈[3, high-1] 等)
                actual_low, actual_high = set_wind_thresholds(new_low_max,
                                                             new_high_min)
                changed = (actual_low != current_low_max
                           or actual_high != current_high_min)

                if changed:
                    prYellow('LEPRRLRB: 阈值变更: LOW_MAX {:.3f}→{:.3f}, '
                             'HIGH_MIN {:.3f}→{:.3f}'.format(
                                 current_low_max, actual_low,
                                 current_high_min, actual_high))

                    # ---- 用新阈值重算回放池里历史 H 层转移的 (r1, c1) ----
                    n_relabeled = agent_H.memory.relabel_all(
                        relabel_r1_substep, relabel_c1_substep)
                    llm_relabel_count += 1
                    prGreen('LEPRRLRB: relabel 了 {} 条 H 层转移'.format(n_relabeled))

                    threshold_history.append({
                        'step': step,
                        'episode': episode,
                        'old_low_max': current_low_max,
                        'new_low_max': actual_low,
                        'old_high_min': current_high_min,
                        'new_high_min': actual_high,
                        'n_relabeled': n_relabeled,
                        'summary': summary,
                    })
                    if use_csv:
                        df_llm_rows.append({
                            'step': step,
                            'episode': episode,
                            'old_low_max': current_low_max,
                            'new_low_max': actual_low,
                            'old_high_min': current_high_min,
                            'new_high_min': actual_high,
                            'adjust_low_max': adjust_low,
                            'adjust_high_min': adjust_high,
                            'n_relabeled': n_relabeled,
                            'r1_avg': summary['r1_avg'],
                            'profit_avg': summary['profit_avg'],
                            'violation_rate': summary['violation_rate'],
                            'high_wind_pct': summary['high_wind']['pct'],
                            'high_wind_violation_rate':
                                summary['high_wind']['violation_rate'],
                            'infeasible_count': summary['infeasible_count'],
                            'gap_pen_per_block': summary['gap_pen_per_block'],
                            'soc_track_err_avg': summary['soc_track_err_avg'],
                        })
                        df_llm_written = _flush_csv(
                            '{}/df_llm_adjust.csv'.format(output),
                            df_llm_rows, df_llm_written)
                        df_llm_rows = []
                else:
                    prCyan('LEPRRLRB: 阈值无需调整 (LLM 增量 = ({:.3f}, {:.3f}))'.format(
                        adjust_low, adjust_high))

                if upload:
                    writer.add_scalar('llm/wind_low_max', actual_low, step)
                    writer.add_scalar('llm/wind_high_min', actual_high, step)
                    writer.add_scalar('llm/adjust_low_max', adjust_low, step)
                    writer.add_scalar('llm/adjust_high_min', adjust_high, step)
                    writer.add_scalar('llm/changed', float(changed), step)
                    writer.add_scalar('llm/window_r1_avg',
                                      summary['r1_avg'], step)
                    writer.add_scalar('llm/window_profit_avg',
                                      summary['profit_avg'], step)
                    writer.add_scalar('llm/window_violation_rate',
                                      summary['violation_rate'], step)
                    writer.add_scalar('llm/window_gap_pen_per_block',
                                      summary['gap_pen_per_block'], step)
            else:
                prCyan('LEPRRLRB: 冷启动保护, 跳过 LLM 分析 '
                       '(episode {} < {})'.format(episode,
                                                  min_episodes_before_llm))

            # 下一窗口重新统计
            window_stats.reset()

        # update
        step += 1
        episode_steps += 1
        episode_reward += r1
        episode_cost += c1_step + gap_pen
        episode_gap_pen += gap_pen
        # 上层利润: pg * 实际电价 * dt - 490.0 * max(-pb,0) * dt（退化 490 元/MWh）
        episode_profit += observation2[-1] * observation[0] * time - 490.0 * max(-float(pbrule), 0) * time
        episode_bess_profit += reward_details['bess_profit'] - 490.0 * max(-float(pbrule), 0) * time
        episode_deg_charge += max(float(pbrule), 0.0) * time
        episode_deg_discharge += max(-float(pbrule), 0.0) * time
        # update env
        norm_observation = deepcopy(norm_observation2)
        observation = deepcopy(observation2)
        pw_tru = pw_tru2

        #-------------------------------------episode-----------------------------
        if done:
            agent_H.save_model(output)
            prGreen('##########{}: episode_profit:{} episode_reward:{} step:{}'.format(
                episode, round(episode_profit, 2), round(episode_reward, 2), step))

            if upload:
                writer.add_scalar('episode_reward', episode_reward, episode)
                writer.add_scalar('episode_cost', episode_cost, episode)
                writer.add_scalar('episode_gap_pen', episode_gap_pen, episode)
                writer.add_scalar('episode_profit', episode_profit, episode)
                writer.add_scalar('env/bessprofit', episode_bess_profit, episode)
                writer.add_scalar('env/deg_charge', episode_deg_charge, episode)
                writer.add_scalar('env/deg_discharge', episode_deg_discharge, episode)
                writer.add_scalar('wind_day', env.j, episode)
                # LEPRRLRB: 每 episode 记录当前动态阈值
                _wl, _wh = get_wind_thresholds()
                writer.add_scalar('llm/episode_wind_low_max', _wl, episode)
                writer.add_scalar('llm/episode_wind_high_min', _wh, episode)

            if use_csv:
                df_episode_rows.append({
                    'episode': episode,
                    'episode_reward': episode_reward,
                    'episode_cost': episode_cost,
                    'episode_gap_pen': episode_gap_pen,
                    'episode_profit': episode_profit,
                    'bess_profit': episode_bess_profit,
                    'deg_charge': episode_deg_charge,
                    'deg_discharge': episode_deg_discharge,
                    'wind_day': env.j,
                    'wind_low_max': get_wind_thresholds()[0],
                    'wind_high_min': get_wind_thresholds()[1],
                })
                df_episode_written = _flush_csv(
                    '{}/df_episode_{}.csv'.format(output, RUN_TAG),
                    df_episode_rows, df_episode_written)
                df_episode_rows = []

            # end of episode
            observation = None
            episode_steps = 0
            episode_reward = episode_cost = episode_profit = 0.
            episode_bess_profit = 0.
            episode_gap_pen = 0.
            episode_deg_charge = episode_deg_discharge = 0.
            episode += 1

    if use_csv:
        # 收尾 flush: 最后一次 flush 后新增的行在此补齐 (追加模式)
        _flush_csv('{}/df_{}.csv'.format(output, RUN_TAG), df_rows, df_written)
        _flush_csv('{}/df_episode_{}.csv'.format(output, RUN_TAG),
                   df_episode_rows, df_episode_written)
        _flush_csv('{}/df_test9.csv'.format(output), df_test_rows, df_test_written)
        _flush_csv('{}/update_stats_H.csv'.format(output),
                   df_update_rows, df_update_written)
        _flush_csv('{}/df_llm_adjust.csv'.format(output),
                   df_llm_rows, df_llm_written)
    if llm_relabel_count:
        prCyan('LEPRRLRB: 本 run 共发生 {} 次阈值调整与 relabel; '
               '最终阈值 LOW_MAX={:.3f} HIGH_MIN={:.3f}'.format(
                   llm_relabel_count, *get_wind_thresholds()))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode',
                        default='train',
                        type=str,
                        help='support option: train/test')
    parser.add_argument('--env',
                        default='TD3LEPRRLRB',
                        type=str,
                        help='open-ai gym environment (决定 output/<env>-runN 目录名)')
    parser.add_argument('--hidden1',
                        default=16,
                        type=int,
                        help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2',
                        default=100,
                        type=int,
                        help='hidden num of second fully connect layer')
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
    parser.add_argument('--bsize', default=32, type=int,
                        help='minibatch size')
    parser.add_argument('--rmsize',
                        default=20000,
                        type=int,
                        help='memory size')
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
    parser.add_argument('--cuda', default=True, type=bool, help='Whether to use CUDA')
    parser.add_argument('--lam', default=3, type=float, help='safe constraint')
    parser.add_argument('--lr_lam', default=0.1, type=float, help='learning rate for lambda')
    parser.add_argument('--cost_single_limit', default=0.5, type=float, help='cost_bound for single step')

    parser.add_argument('--PER_alpha', default=0.6, type=float,
                        help='[0~1] convert the importance of TD error to priority')
    parser.add_argument('--PER_beta', default=0.4, type=float,
                        help='importance-sampling, from initial value increasing to 1')
    parser.add_argument('--PER_epsilon', default=0.01, type=float,
                        help='small amount to avoid zero priority')
    parser.add_argument('--beta_increment_per_sampling', default=3e-5, type=float,
                        help='β退火增量/次更新')
    parser.add_argument('--nb_turbine', default=5, type=int,
                        help='number of turbines')

    # ---- TD3 specific ----
    parser.add_argument('--policy_delay', default=2, type=int,
                        help='TD3 delayed policy update interval')
    parser.add_argument('--policy_noise', default=0.1, type=float,
                        help='target policy smoothing noise std')
    parser.add_argument('--noise_clip', default=0.3, type=float,
                        help='target policy smoothing noise clip')
    parser.add_argument('--cost_critic_mode', default='max', type=str,
                        choices=['min', 'max'],
                        help='cost twin-critic aggregation: max (safe) or min')

    # RBB parameters
    parser.add_argument('--rbb_fluctuation_limit', default=3.0, type=float,
                        help='Max grid power fluctuation between steps (MW)')
    parser.add_argument('--rbb_rated_power', default=10.0, type=float,
                        help='BESS rated charge/discharge power (MW)')
    parser.add_argument('--soc_tol', default=SOC_TOL, type=float,
                        help='SOC 目标死区 (MWh), 与规则下层死区同源')

    # ---- LEPRRLRB: LLM 分段微调 ----
    parser.add_argument('--llm_provider', default=None, type=str,
                        choices=['anthropic', 'openai', 'deepseek'],
                        help='LLM API provider (覆盖配置文件). '
                             '默认读 td3leprrlrb/llm_config.json')
    parser.add_argument('--llm_config', default=None, type=str,
                        help='LLM 配置 JSON 路径. 默认 td3leprrlrb/llm_config.json')
    parser.add_argument('--llm_window', default=7200, type=int,
                        help='LLM 调整窗口长度 (步). 默认 7200 = 5 天; '
                             '每个窗口边界调用一次 LLM 微调风速分段阈值')
    parser.add_argument('--llm_min_episodes', default=3, type=int,
                        help='冷启动保护: episode 数达到该值后才允许 LLM 调整阈值')
    parser.add_argument('--seg', default='expert',
                        choices=['percentile', 'kmeans', 'expert'],
                        help='风速分段【起点】方法: percentile | kmeans | expert '
                             '(默认 expert); 起点确定后由 LLM 在 原设定 ±1 m/s 内在线微调')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    # ---- LEPRRLRB: 风速分段起点 (percentile / kmeans / expert) ----
    # LLM 的微调在此基础上进行; 模块默认 7.0/11.0 即 td3prrlrb 的写死值。
    from data_piecewise.boundaries import apply_segmentation
    apply_segmentation(args.seg, set_wind_thresholds)

    # ---- LEPRRLRB: 阈值调整带 = 原设定 ±1 m/s (整个训练过程硬约束) ----
    orig_low_max, orig_high_min = get_wind_thresholds()
    adj_bounds = ((orig_low_max - 1.0, orig_low_max + 1.0),
                  (orig_high_min - 1.0, orig_high_min + 1.0))
    prCyan('LEPRRLRB: 阈值原设定 LOW_MAX={:.2f} HIGH_MIN={:.2f} m/s; '
           'LLM 调整带 = 原设定 ±1 m/s → LOW∈[{:.2f},{:.2f}] HIGH∈[{:.2f},{:.2f}]'.format(
               orig_low_max, orig_high_min,
               orig_low_max - 1.0, orig_low_max + 1.0,
               orig_high_min - 1.0, orig_high_min + 1.0))
    prCyan('LEPRRLRB: 收益权重表 WT_WIND={}, WT_PRICE={}, WB_PRICE={}, '
           'WB_SOC_OUTSIDE={} (SOC 贴边折减已取消: wB 只由电价档决定, 与 td3prle 对齐)'
           .format(WT_WIND, WT_PRICE, WB_PRICE, WB_SOC_OUTSIDE))

    max_E = 6.
    upload = True
    use_csv = True
    args.max_episode_length = 1440  # 1d = 24h*60m = 1440 steps
    args.train_iter_H = 1000 * args.max_episode_length
    args.seed = np.random.randint(0, 10000)
    args.cost_single_limit = 1000.

    # Set device and random seeds
    if (args.cuda and torch.cuda.is_available()):
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

    env = WindBattery(max_E, args.nb_turbine, upload, engine)
    env_test = WindBattery(max_E, args.nb_turbine, upload, engine)

    if args.seed > 0:
        np.random.seed(args.seed)

    # RBB system: 可行性/波动窗口 (下层规则依赖它, 同时作为硬限幅兜底)
    rbb_system = RuleBasedBESS(
        max_E=max_E,
        min_E_factor=0.0,
        rated_power=args.rbb_rated_power,
        eta_c=0.98,
        eta_d=0.98,
        time=1.0 / 60,
        fluctuation_limit=args.rbb_fluctuation_limit,
    )

    # 下层: 规则控制器 (无 RL, 无 lam2)
    tracker = SocTracker(soc_tol=args.soc_tol, eta_c=0.98, eta_d=0.98,
                         time=1.0 / 60, rated_power=args.rbb_rated_power)

    # 上层: TD3 + EMA 实际成本对偶 (lam1 ∈ [3, 100], 初始 5.0)
    agent_H = SP_TD3_LEPRRLRB(nb_states=4, nb_actions=6, args=args,
                              lam_max=100.0, lam_min=3.0,
                              ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                              record_update_stats=True,
                              policy_delay=args.policy_delay,
                              policy_noise=args.policy_noise,
                              noise_clip=args.noise_clip,
                              cost_critic_mode=args.cost_critic_mode)
    agent_H.lam = 5.0

    if args.mode == 'train':
        train(args.train_iter_H,
              agent_H,
              tracker,
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
