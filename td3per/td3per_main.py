#!/usr/bin/env python3
"""
TD3PER — 上层 = TD3PERRB 口径 (非分段奖励 + SOC 目标信号), 下层 = RL BESS 追 SOC 目标
====================================================================================
本文件是 td3per 的 "td3perrb 化" 改造 (原地修改):

  * 上层 H: **与 td3perrb 完全一致**
      - 动作 6 维 = 5 台风机桨距 [0,0.5] + 1 个 SOC 目标信号 a6 ∈ [-0.5, 0.5];
      - a6 → soc_target = 3 + 5·a6 ∈ [0.5, 5.5] (块级(5 步)固定);
      - 非分段奖励 r1 / c1 (收益权重与风险系数恒 1.0, W_GAP = 10);
      - 块末 SOC 目标可行性罚 gap_pen = 10·max(0, |target − SOC_末| − 0.1);
      - 观测 = Normalize(obs) 4 维 + 24h 相位特征 (--phase_feat 1 → 6 维);
      - 对偶: lam1 ∈ [3, 100] 初值 5.0, d_block_H = 50 → 地板 5 (EMA 实际块成本);
      - 每块 2 次更新; 逐维探索噪声 (--wt_noise / --sig_noise_max / --sig_noise_min);
      - --epsilon 4000 (H 决策次数 ⇒ 20k 环境步); actor 输出层 WT bias 初始化到 Cp 高效区。
  * 下层 L: **td3per 自己的 RL BESS 操作保留** (TD3 + PER + EMA 对偶 + RBB 硬裁剪),
      只把"信号 / 追踪目标"换成 SOC 目标:
      - 观测 (5 维蒸馏) = [pbrulemin/10, pbrulemax/10, **soc_target/10**, SOC/10, u/10]
        (第 3 维原为 shadow_price);
      - r2 = −100 · clip_amount            (原影子套利项随信号语义一起删除);
      - c2 = 1000 · slack,  slack = max(0, |target−SOC'| − |target−SOC_best|),
        SOC_best = SOC + g·clip(δ/g, RBB 窗口)  ← 规则 SocTracker 的动作;
        只罚"本步能做到却没做"的部分 —— 块首够不着的目标跳变不计费, 零成本动作
        恰好是规则下层 (见 td3per_myfunc.soc_track_slack);
      - lam2 ∈ [10, 500], d_block_L = 20 → 地板 2.5, 每块 10 次更新;
      - L 层折扣/探索按 td3per 原口径: --l_discount 0.98 (步级), --l_epsilon 50000。

奖励 / 成本口径
----------------
H: r1 = pr·P_G·dt − 490·max(−pbrule,0)·dt − penalty_NN_H          (P_G = ptur − pbrule)
    c1 = 1.0·[10·(P_F−3)⁺ + 0.1·P_F] + 10·unfeasible_gap
         + [块末] 10·max(0, |soc_target − SOC_末| − 0.1)
L: r2 = −100·clip_amount
    c2 = 1000·|soc_target − SOC'|

对偶上升 (两层都是 EMA 实际块成本)
----------------------------------
    H: EMA ← 0.98·EMA + 0.02·sum_c1;  lam1 ← clip(lam1 + 0.01·(EMA − d_block_H), 3, 100)
    L: EMA ← 0.98·EMA + 0.02·sum_c2;  lam2 ← clip(lam2 + 0.01·(EMA − d_block_L), 10, 500)
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
from td3per.td3per_myfunc import (
    CheckAction, Normalize, Normalize_H,
    Cal_Rew_RB, gap_penalty, soc_target_from_signal, nine_case_test,
    build_L_obs_distilled, lower_reward, lower_cost_soc_track,
    SOC_TOL,
    PHASE_DIM, day_price_schedule, phase_features, phase_daily_table,
    phase_minute_of_day,
)
from rbb.rbb_rule_bess import RuleBasedBESS

import pandas as pd

from torch.utils.tensorboard import SummaryWriter

# 不显示warning
import warnings
warnings.filterwarnings("ignore")

import matlab
import matlab.engine
engine = matlab.engine.start_matlab()


RUN_TAG = 'td3per'


def init_wt_action_bias(agent, lo=0.25, hi=0.40, sig_value=0.0, verbose=True):
    """把 actor 输出层前 5 维 (WT 轴向诱导因子 a) 的 bias 初始化到 Cp 高效区。

    为什么必须做 (run160/初始策略诊断):
      `model.py` 的 Actor 只把 fc4.weight 初始化成 ±init_w(0.003),
      fc4.bias 仍是 PyTorch 默认 U(−1/√fan_in)=U(±0.25) (fan_in=hidden1=16);
      而 a 与功率系数的关系是 Cp = 4a(1−a)² (见 WFSim_port.m):
        a=0 → Cp=0 (该机零出力/停机),  a=1/3 → Cp=0.593 (Betz 最大),  a=0.5 → Cp=0.5。
      于是初始策略 ≈ bias: 实测 5 个种子 → 死机 2~4 台、全场 ΣCp 仅 0.47~0.72
      (效率 16%~24%)。策略从"半停机"出发, 而 a<0 区被 CheckAction 裁掉后梯度为 0
      → 若没有恢复力就会长期停在停机态。
    bias∈[lo,hi]=[0.25,0.40] 对应 Cp≈0.56~0.59, 即从"正常工作点"出发。
    SOC 目标信号维 (第 6 维) 置 sig_value=0 → 初始 soc_target = 3.0 (SOC 带中点)。

    `model.py` 为多项目共享文件, 不修改; 这里构造后覆写 actor 与 actor_target 的 bias。
    """
    a = agent.actor
    if not hasattr(a, 'fc4'):
        return False
    nb = a.fc4.bias.data.shape[0]
    with torch.no_grad():
        b = a.fc4.bias.data
        n_wt = min(5, nb)
        b[:n_wt] = torch.tensor(np.random.uniform(lo, hi, n_wt), dtype=b.dtype)
        if nb > n_wt:
            b[n_wt:] = float(sig_value)
        tgt = agent.actor_target.fc4.bias.data
        tgt.copy_(b)
    if verbose:
        cp = 4 * np.array([b[:n_wt].mean()]) * (1 - np.array([b[:n_wt].mean()])) ** 2
        prGreen('init_wt_action_bias: WT bias = {} (mean a = {:.3f} -> Cp = {:.3f}); '
                'signal bias = {}'.format(
                    np.round(b[:n_wt].numpy(), 3), float(b[:n_wt].mean()),
                    float(cp[0]), float(b[n_wt]) if nb > n_wt else None))
    return True


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


# TD3PER: 上层 = TD3PERRB 口径 (非分段 + SOC 目标信号), 下层 = RL BESS 追 SOC 目标
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
    episode_gap_pen = 0.
    episode_deg_charge = episode_deg_discharge = 0.
    observation = None
    time = 1/60
    # 上层 gap 罚的豁免裕度 (默认 0.1 MWh = SOC_TOL); 下层 c2 无死区 —— 两者分开。
    soc_tol_upper = float(getattr(args, 'soc_tol', SOC_TOL))
    # Curriculum: consecutive blocks where EMA actual cost <= d_block
    consecutive_satisfied_H = 0
    consecutive_satisfied_L = 0
    # H 层 EMA 对偶 —— 每 5 步块实际成本预算 d_block_H (与 sum_c1 同口径)
    #   sum_c1 = 波动成本 + 10·unfeasible_gap + 块末 SOC 目标 gap 罚
    d_block_H = 50.0        # 初值: 每块 sum_c1 预算 (先松后紧; 与 td3perrb 一致)
    d_block_H_min = 5.0     # 地板: 10·max(0,|δ|−0.1) ≤ 5 → 含死区总 |δ| ≤ 0.6
    # L 层 EMA 对偶 —— 每 5 步块实际成本预算 d_block_L (与 sum_c2 同口径)
    #   sum_c2 = 1000·slack 的块内累计, slack = "能力内追踪失败量" (见 myfunc)
    #   * 尽力追的 L: sum_c2 ≡ 0;  完全不动而 |δ|≥A: 每步 ~50, 每块 ~250;
    #     最坏 (反向满功率): 每步 ≤ 340。
    #   * 于是 d_block_L = 20 现读作"允许每块浪费 ≈0.02 MWh 的可达量 (≈0.4 步能力)",
    #     地板 2.5 为"≈0.05 步能力" —— 是一个**可满足**的约束, lam2 不再必然贴顶。
    d_block_L = 20.0        # 初值: 每块 sum_c2 预算
    d_block_L_min = 2.5     # curriculum 收紧地板
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

    # ---- 1440-step window counters for env metrics ----
    window_violation_count = 0   # steps where |P_F| > 3 MW
    window_power_sum = 0.0        # cumulative P_G over the window

    # ---- 24h 相位特征 (--phase_feat) ----
    phase_mode = int(getattr(args, 'phase_feat', 0))
    phase_table = phase_daily_table() if phase_mode >= 2 else None
    phase_sched = day_price_schedule()
    phase_mismatch = 0
    phase_checked = 0

    while step < num_iterations_H:
        # start of episode 重置环境
        prYellow('########### step:{} ###########'.format(step))
        if observation is None:
            observation = deepcopy(env.reset())  # [Pr,U,E,P_last]
            # SOC 随机化: 每个 episode(1440步) 从 U[0.5,5.5] 重新采样
            reset_E = float(np.random.uniform(0.5, 5.5))
            env.E = reset_E
            observation[2] = reset_E
            norm_observation = Normalize_H(observation, phase_minute_of_day(0),
                                           phase_mode, phase_table)
            agent_H.reset(norm_observation)

        # 相位自检: env 的电价必须与 day_price_schedule() 在 episode_steps%1440 处一致
        pm_now = phase_minute_of_day(episode_steps)
        if phase_mode > 0:
            if abs(float(observation[0]) - phase_sched[pm_now]) > 1e-6:
                phase_mismatch += 1
            phase_checked += 1

        # ================================================================
        # 上层 H: 每 5 步决策一次 (块级) —— 与 td3perrb 完全一致
        # ================================================================
        if episode_steps % 5 == 0:
            sum_r1 = 0
            sum_c1 = 0
            sum_c2 = 0     # L 层 5 步块累计成本 (喂 EMA 对偶)
            sum_gap_pen = 0.0
            if step <= args.warmup:
                applied_action_H = original_action_H = agent_H.random_action()
                penalty_NN_H = 0
            else:
                original_action_H = agent_H.select_action(norm_observation)
                applied_action_H, penalty_NN_H = CheckAction(original_action_H)
            norm_observation_H = norm_observation
            pw_tru = env.PowerOutput(observation[1], applied_action_H)
            # SOC 目标 (块级固定) —— 上层信号语义
            soc_signal = float(applied_action_H[5])
            soc_target = soc_target_from_signal(soc_signal)
            pb_l = list()

        prGreen('--H-- ori a_t = {}'.format(round_list(original_action_H, 3)))

        ptur = float(pw_tru)
        pg_prev = float(observation[-1])
        E_current = float(observation[2])

        # 可行窗口 (与 pb 无关, 预计算) —— 供下层观测与可行性诊断使用
        _, rbb_info_obs = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=ptur, pg_prev=pg_prev, E=E_current)

        # ================================================================
        # 下层 L 观测 (5 维蒸馏): 第 3 维 = soc_target/10 (原 shadow_price)
        # ================================================================
        norm_observation_L = build_L_obs_distilled(
            observation, pw_tru, applied_action_H, rbb_info_obs)
        prPurple('--L-- obs = {}.'.format(round_list(norm_observation_L, 3)))
        # 下层每一个 5-step 块都是一个 episode
        if episode_steps % 5 == 0:
            agent_L.reset(norm_observation_L)

        # ================================================================
        # 下层 L: RL BESS (td3per 自己的操作, 每步决策)
        # ================================================================
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
        # RBB: 规则可行窗口 + 硬裁剪 (安全性兜底, 不变)
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
        norm_observation2 = Normalize_H(observation2,
                                       phase_minute_of_day(episode_steps + 1),
                                       phase_mode, phase_table)

        # ================================================================
        # 上层 H 奖励/成本 (非分段, 与 td3perrb 同口径)
        # ================================================================
        r1, c1_step, reward_details = Cal_Rew_RB(
            observation, observation2, action_rbb, rbb_info,
            penalty_NN_H=penalty_NN_H, time=time,
            ptur=ptur, pb_cmd=pb_original,
        )
        P_F = reward_details['P_F']

        sum_r1 += r1
        sum_c1 += c1_step

        # ---- 块末结算: SOC 目标可行性罚 (上层豁免 tol = --soc_tol, 默认 0.1) ----
        gap_raw = abs(soc_target - float(observation2[2]))
        gap_pen = 0.0
        if episode_steps % 5 == 4:
            gap_pen = gap_penalty(soc_target, float(observation2[2]), tol=soc_tol_upper)
            sum_c1 += gap_pen
            sum_gap_pen += gap_pen

        # ================================================================
        # 下层 L 奖励/成本 (SOC 目标追踪口径)
        #   r2 = −100·clip_amount                 (RBB 硬裁剪量)
        #   c2 = 1000·slack, slack = max(0, |target−SOC'| − |target−SOC_best|)
        #        SOC_best = SOC + g·clip(δ/g, 窗口) —— 规则 SocTracker 的动作;
        #        只罚"本步能做到却没做"的部分 (块首够不着的跳变不计费)
        # ================================================================
        clip_amount = rbb_info['clip_amount']
        is_infeasible = rbb_info['is_infeasible']
        inevitable_violation = rbb_info['inevitable_violation']

        r2 = lower_reward(clip_amount)
        c2, c2_info = lower_cost_soc_track(
            soc_target, E_current, float(observation2[2]),
            rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'],
            return_info=True)
        c2_slack = c2_info['d_next'] - c2_info['d_best']
        sum_c2 += c2

        # 下一状态 L 观测: 用下一状态重算可行域 (pb 无关; soc_target 块内固定)
        _, rbb_info_obs2 = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=float(pw_tru2),
            pg_prev=float(observation2[-1]), E=float(observation2[2]))
        norm_observation2_L = build_L_obs_distilled(
            observation2, pw_tru2, applied_action_H, rbb_info_obs2)

        prGreen('r1 = {} (turb={}, bess={}, deg={})'.format(
            round(r1, 3), round(reward_details['turbine_profit'], 3),
            round(reward_details['bess_profit'], 3),
            round(reward_details['degradation'], 3)))
        prGreen('c1 = {} (fluct={}, gap_pen={})'.format(
            round(sum_c1 if episode_steps % 5 == 4 else c1_step, 3),
            round(reward_details['c1_fluct'], 3), round(gap_pen, 4)))
        prGreen('soc_signal={:.4f} soc_target={:.3f} SOC={:.3f} gap_raw={:.3f} '
                '(tol_upper={})'.format(
                    soc_signal, soc_target, reward_details['SOC'],
                    gap_raw, soc_tol_upper))
        prGreen('r2 = {} (clip_amount={}) | c2 = {} (slack={:.4f} MWh, '
                'd_next={:.4f} vs d_best={:.4f}, pb_best={:.3f})'.format(
                    round(r2, 3), round(clip_amount, 3), round(c2, 3),
                    c2_slack, c2_info['d_next'], c2_info['d_best'],
                    c2_info['pb_best']))
        prGreen('price={} p_level={} wind={:.1f} w_level={} (wT={}, wB={})'.format(
            reward_details['pr'], reward_details['price_level'],
            reward_details['u'], reward_details['wind_level'],
            reward_details['w_wind'], reward_details['w_bess']))

        pb_l.append(pb_original)

        # ---- 1440-step window: track actual grid fluctuation ----
        if P_F > 3.0 + 1e-4:
            window_violation_count += 1
        window_power_sum += observation2[-1]  # cumulative P_G

        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        # ---- 块末: H/L 两层各自结算 (H 用块级 sum, L 用步级 transition) ----
        if episode_steps % 5 == 4:
            agent_H.observe_jump(norm_observation_H, original_action_H,
                                 sum_r1, sum_c1, norm_observation2, done)
            # H 层 EMA 对偶上升 —— 用块实际成本 sum_c1 更新 lam1 (与 d_block_H 同口径)
            if step > args.warmup:
                agent_H.update_ema_cost(sum_c1)
                agent_H.update_lam_ema(d_block_H)
            # L 层 EMA 对偶上升 —— 用块实际成本 sum_c2 更新 lam2 (与 d_block_L 同口径)
            if step > args.warmup:
                agent_L.update_ema_cost(sum_c2)
                agent_L.update_lam_ema(d_block_L)
            done_L = True
        agent_L.observe_jump(norm_observation_L, original_action_L,
                             r2, c2, norm_observation2_L, done_L)

        if step > args.warmup:
            #--------------------Update----------------------
            if episode_steps % 5 == 4:
                # H 层更新频率: 每块 2 次 (td3perrb 口径)
                for _ in range(2):
                    agent_H.update_policy()
                    # Track consecutive satisfaction for H (EMA 实际成本判据)
                    if agent_H.ema_cost is not None and agent_H.ema_cost <= d_block_H:
                        consecutive_satisfied_H += 1
                    else:
                        consecutive_satisfied_H = 0

                # L 层更新频率: 每块 10 次 (td3per 口径, L 只有 1 维动作)
                for _ in range(10):
                    agent_L.update_policy()
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

                # ---- H 层更新数据记录（本块 2 批均值 → TB/csv） ----
                if agent_H.record_update_stats and len(agent_H.update_stats) > 0:
                    _stats = agent_H.update_stats
                    _mean = {k: float(np.mean([s[k] for s in _stats]))
                             for k in _stats[0]}
                    agent_H.update_stats = []  # 清空，下一块重收
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
            # (fixed scenario seed 42 -> same wind trajectory each test)
            if episode_steps % 480 == 0:
                case_metrics, overall = nine_case_test(
                    agent_H, agent_L, env_test, rbb_system, seed=42,
                    phase_mode=phase_mode, phase_table=phase_table)
                if use_csv:
                    for m in case_metrics:
                        df_test_rows.append({
                            'step': step,
                            'wind_level': m['wind_level'],
                            'price_level': m['price_level'],
                            'r1': m['r1'], 'c1': m['c1'],
                            'r2': m['r2'], 'c2': m['c2'],
                            'gap_pen': m['gap_pen'],
                            'cvm': m['cvm'],
                            'pf_viol': m['pf_viol'],
                            'profit': m['profit'],
                            'soc_track_err': m['soc_track_err'],
                        })
                    df_test_rows.append({
                        'step': step,
                        'wind_level': 'OVERALL',
                        'price_level': 'OVERALL',
                        'r1': overall['r1'], 'c1': overall['c1'],
                        'r2': overall['r2'], 'c2': overall['c2'],
                        'gap_pen': overall['gap_pen'],
                        'cvm': overall['cvm'],
                        'pf_viol': overall['pf_viol'],
                        'profit': overall['profit'],
                        'soc_track_err': overall['soc_track_err'],
                    })
                    df_test_written = _flush_csv(
                        '{}/df_test9.csv'.format(output),
                        df_test_rows, df_test_written)
                    df_test_rows = []
                if upload:
                    writer.add_scalar('test9/overall/r1', overall['r1'], step)
                    writer.add_scalar('test9/overall/c1', overall['c1'], step)
                    writer.add_scalar('test9/overall/r2', overall['r2'], step)
                    writer.add_scalar('test9/overall/c2', overall['c2'], step)
                    writer.add_scalar('test9/overall/gap_pen', overall['gap_pen'], step)
                    writer.add_scalar('test9/overall/cvm', overall['cvm'], step)
                    writer.add_scalar('test9/overall/pf_viol', overall['pf_viol'], step)
                    writer.add_scalar('test9/overall/profit', overall['profit'], step)
                    writer.add_scalar('test9/overall/soc_track_err',
                                      overall['soc_track_err'], step)
                    for m in case_metrics:
                        tag = 'test9/{}_{}'.format(m['wind_level'], m['price_level'])
                        writer.add_scalar(tag + '/r1', m['r1'], step)
                        writer.add_scalar(tag + '/c1', m['c1'], step)
                        writer.add_scalar(tag + '/r2', m['r2'], step)
                        writer.add_scalar(tag + '/c2', m['c2'], step)
                        writer.add_scalar(tag + '/gap_pen', m['gap_pen'], step)
                        writer.add_scalar(tag + '/cvm', m['cvm'], step)
                        writer.add_scalar(tag + '/pf_viol', m['pf_viol'], step)
                        writer.add_scalar(tag + '/profit', m['profit'], step)
                        writer.add_scalar(tag + '/soc_track_err',
                                          m['soc_track_err'], step)

        # 上层/下层逐步数据 —— 只累积行, 每 50 步增量 flush
        if use_csv:
            df_rows.append({
                'step': step,
                'observation': observation,
                'observation2': observation2,
                # 24h 相位 (phase_feat>0): 分钟数 + 可选段内特征
                'phase_minute': pm_now,
                'phase_sin': float(np.sin(2 * np.pi * pm_now / 1440)),
                'phase_cos': float(np.cos(2 * np.pi * pm_now / 1440)),
                'phase_rest_frac': (float(phase_table[pm_now, 0]) if phase_table is not None else np.nan),
                'phase_rest_h': (float(phase_table[pm_now, 1]) * 8 if phase_table is not None else np.nan),
                'phase_next_pr': (float(phase_table[pm_now, 2]) * 1000 if phase_table is not None else np.nan),
                # 上层
                'original_action_H': original_action_H,
                'applied_action_H': applied_action_H,
                'soc_signal': soc_signal,
                'soc_target': soc_target,
                # 下层 (RL BESS)
                'norm_observation_L': norm_observation_L,
                'original_action_L': original_action_L,
                'applied_action_L': applied_action_L,
                'action_rbb': action_rbb,
                'r1': r1, 'c1_step': c1_step, 'c1': c1_step + gap_pen,
                'r2': r2, 'c2': c2, 'c2_slack': c2_slack,
                'pb_best': c2_info['pb_best'],
                'd_best': c2_info['d_best'], 'd_next': c2_info['d_next'],
                'window_infeasible': float(c2_info['infeasible_window']),
                'gap_raw': gap_raw, 'gap_pen': gap_pen,
                'soc_track_err': abs(soc_target - float(observation2[2])),
                'lam1': agent_H.lam,
                'lam2': agent_L.lam,
                'd_block_H': d_block_H,
                'd_block_L': d_block_L,
                'ema_c1_H': agent_H.ema_cost,
                'ema_c2_L': agent_L.ema_cost,
                # RBB metrics
                'pb_original': rbb_info['pb_original'],
                'pbrule': rbb_info['pbrule'],
                'pbfluctmin': rbb_info['pbfluctmin'],
                'pbfluctmax': rbb_info['pbfluctmax'],
                'pbphysicsmin': rbb_info['pbphysicsmin'],
                'pbphysicsmax': rbb_info['pbphysicsmax'],
                'pbrulemin': rbb_info['pbrulemin'],
                'pbrulemax': rbb_info['pbrulemax'],
                'clip_amount': clip_amount,
                'unfeasible_gap': rbb_info['unfeasible_gap'],
                'is_infeasible': float(is_infeasible),
                'inevitable_violation': inevitable_violation,
                # 风况/电价档位标签 —— 非分段奖励下仅作诊断, 不进入任何系数
                'wind_level': reward_details['wind_level'],
                'price_level': reward_details['price_level'],
                'w_wind': reward_details['w_wind'],
                'w_bess': reward_details['w_bess'],
                'pr': reward_details['pr'],
                'u': reward_details['u'],
                # 非分段奖励的收益分解 (wT=wB=1.0 ⇒ 等于 ptur/(-pbrule) 项本身)
                'turbine_profit': reward_details['turbine_profit'],
                'bess_profit': reward_details['bess_profit'],
                'revenue': reward_details['revenue'],
                'degradation': reward_details['degradation'],
                'soc_deviation': reward_details['soc_deviation'],
                'soc_delta': reward_details['soc_delta'],
                'c1_fluct': reward_details['c1_fluct'],
                'penalty_NN_H': penalty_NN_H,
                'penalty_NN_L': penalty_NN_L,
                'SOC': reward_details['SOC'],
                'P_F': P_F,
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
            agent_L.save_checkpoint(output, step)
            prCyan('Checkpoint saved at step {}'.format(step))

        if upload:
            writer.add_scalar('r1', r1, step)
            writer.add_scalar('c1', c1_step, step)
            writer.add_scalar('c2', c2, step)
            writer.add_scalar('r2', r2, step)
            writer.add_scalar('gap_pen', gap_pen, step)
            writer.add_scalar('gap_raw', gap_raw, step)
            writer.add_scalar('soc_track_err', abs(soc_target - float(observation2[2])), step)
            writer.add_scalar('wind_speed', env.u, step)
            writer.add_scalar('reward', r1, step)
            writer.add_scalar('cost', c1_step, step)
            writer.add_scalar('lam1', agent_H.lam, step)
            writer.add_scalar('lam2', agent_L.lam, step)
            writer.add_scalar('d1', d_block_H, step)
            writer.add_scalar('d2', d_block_L, step)
            if agent_H.ema_cost is not None:
                writer.add_scalar('ema_c1_H', agent_H.ema_cost, step)
            if agent_L.ema_cost is not None:
                writer.add_scalar('ema_c2_L', agent_L.ema_cost, step)
            writer.add_scalar('soc', float(observation[2]), step)
            writer.add_scalar('soc_deviation', reward_details['soc_deviation'], step)
            writer.add_scalar('P_F', P_F, step)

            # 24h 相位 (与 --phase_feat 对应; 小时数即为可读的时间轴)
            writer.add_scalar('phase/hour', pm_now / 60.0, step)
            if phase_table is not None:
                writer.add_scalar('phase/rest_frac', float(phase_table[pm_now, 0]), step)
                writer.add_scalar('phase/rest_hours', float(phase_table[pm_now, 1]) * 8, step)
                writer.add_scalar('phase/next_price', float(phase_table[pm_now, 2]) * 1000, step)
            if phase_checked > 0:
                writer.add_scalar('phase/check_mismatch_frac',
                                  phase_mismatch / float(phase_checked), step)

            # 风况/电价档位 (非分段奖励下仅作诊断; w_wind/w_bess 恒为 1.0)
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

            # 收益分解
            writer.add_scalar('reward_upper/turbine_profit', reward_details['turbine_profit'], step)
            writer.add_scalar('reward_upper/bess_profit', reward_details['bess_profit'], step)
            writer.add_scalar('reward_upper/degradation', reward_details['degradation'], step)
            writer.add_scalar('reward_lower/r2', r2, step)
            writer.add_scalar('reward_lower/c2', c2, step)
            writer.add_scalar('reward_lower/c2_slack', c2_slack, step)

            # SOC 目标跟踪 / 下层可达性诊断
            writer.add_scalar('target/soc_target', soc_target, step)
            writer.add_scalar('target/soc_signal', soc_signal, step)
            writer.add_scalar('target/gap_raw', gap_raw, step)
            writer.add_scalar('target/gap_pen', gap_pen, step)
            writer.add_scalar('target/pb_best', c2_info['pb_best'], step)
            writer.add_scalar('target/d_best', c2_info['d_best'], step)
            writer.add_scalar('target/d_next', c2_info['d_next'], step)

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
                writer.add_scalar('sum_c2', sum_c2, step)
                writer.add_scalar('sum_gap_pen', sum_gap_pen, step)
                # Upper action: 5 WT + 1 SOC 目标信号
                for i in range(5):
                    writer.add_scalar('action/a_H_original_{}'.format(i),
                                      original_action_H[i], step)
                    writer.add_scalar('action/a_H_clipped_{}'.format(i),
                                      applied_action_H[i], step)
                writer.add_scalar('action/soc_signal', applied_action_H[5], step)
                writer.add_scalar('action/soc_target', soc_target, step)
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

        # update
        step += 1
        episode_steps += 1
        episode_reward += r1
        episode_cost += c1_step + gap_pen
        episode_cost2 += c2
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
            agent_L.save_model(output)
            prGreen('##########{}: episode_profit:{} episode_reward:{} step:{}'.format(
                episode, round(episode_profit, 2), round(episode_reward, 2), step))

            if upload:
                writer.add_scalar('episode_reward', episode_reward, episode)
                writer.add_scalar('episode_cost', episode_cost, episode)
                writer.add_scalar('episode_cost2', episode_cost2, episode)
                writer.add_scalar('episode_gap_pen', episode_gap_pen, episode)
                writer.add_scalar('episode_profit', episode_profit, episode)
                writer.add_scalar('env/bessprofit', episode_bess_profit, episode)
                writer.add_scalar('env/deg_charge', episode_deg_charge, episode)
                writer.add_scalar('env/deg_discharge', episode_deg_discharge, episode)
                writer.add_scalar('wind_day', env.j, episode)

            if use_csv:
                df_episode_rows.append({
                    'episode': episode,
                    'episode_reward': episode_reward,
                    'episode_cost': episode_cost,
                    'episode_cost2': episode_cost2,
                    'episode_gap_pen': episode_gap_pen,
                    'episode_profit': episode_profit,
                    'bess_profit': episode_bess_profit,
                    'deg_charge': episode_deg_charge,
                    'deg_discharge': episode_deg_discharge,
                    'wind_day': env.j,
                })
                df_episode_written = _flush_csv(
                    '{}/df_episode_{}.csv'.format(output, RUN_TAG),
                    df_episode_rows, df_episode_written)
                df_episode_rows = []

            # end of episode
            # 下个 episode 起点 SOC 由循环顶部在 [0.5, 5.5] 内随机重置，不再继承
            observation = None
            episode_steps = 0
            episode_reward = episode_cost = episode_cost2 = episode_profit = 0.
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

    # 相位自检结论 (应恒为 0; >0 说明 episode_steps 与 env 电价时钟失配)
    if phase_mode > 0:
        prGreen('phase check: mode={} dims={} checked={} mismatch={} ({:.4f}%)'.format(
            phase_mode, PHASE_DIM[phase_mode], phase_checked, phase_mismatch,
            100.0 * phase_mismatch / max(phase_checked, 1)))
        if phase_mismatch > 0:
            prRed('WARNING: ep 电价时钟与 day_price_schedule() 不一致 '
                  '({} 步) → 相位特征失效, 请检查 env.reset()/max_episode_length'.format(
                      phase_mismatch))


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
    # 0.99: H 层 γ 以【块(5 min)】为单位生效 (observe_jump 聚合 5 步后 discount 只用一次)
    #       ⇒ 半衰期 69 块 ≈ 5.8 h, 覆盖 4~7 h 的低价充/高价放周期 (td3perrb 口径)
    parser.add_argument('--discount', default=0.99, type=float, help='')
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
                        default=4000,
                        type=int,
                        help='【H 层】探索噪声 ε 由 1 线性衰减到 0 所需的【决策次数】。'
                             'H 每 5 个环境步决策一次 ⇒ 4000 次 = 20k 环境步'
                             '(信号维幅值此后由地板 sig_noise_min 保底)')
    parser.add_argument('--l_epsilon',
                        default=50000,
                        type=int,
                        help='【L 层】探索噪声 ε 衰减所需的【决策次数】(L 每环境步决策一次)。'
                             '默认 50000 = td3per 下层原口径 (与 H 解耦: H 用 --epsilon)')
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
    # L 层(BESS)专用策略更新超参 (与 H 解耦; td3per 原口径)
    parser.add_argument('--l_policy_delay', default=2, type=int,
                        help='L 层策略更新间隔: 1=每批更新(≈DDPG节奏) / 2=TD3延迟(默认, 与H一致)')
    parser.add_argument('--l_policy_noise', default=0.1, type=float,
                        help='L 层 target smoothing noise std (动作幅度比例): 0.01≈关闭 / 0.1=TD3默认')
    # L 层折扣 (步级语义) 与 H 层解耦: L episode = 5 步块 (done_L=True), 视野 ≤5 步
    parser.add_argument('--l_discount', default=0.98, type=float,
                        help='L 层(BESS)折扣: 步级语义; 默认 0.98 (td3per 原口径; H 为块级 0.99)')
    parser.add_argument('--cost_critic_mode', default='max', type=str,
                        choices=['min', 'max'],
                        help='cost twin-critic aggregation: max (safe, recommended) or min')

    # RBB parameters (硬裁剪 + L 层可行域观测输入)
    parser.add_argument('--rbb_fluctuation_limit', default=3.0, type=float,
                        help='Max grid power fluctuation between steps (MW)')
    parser.add_argument('--rbb_rated_power', default=10.0, type=float,
                        help='BESS rated charge/discharge power (MW)')

    # 上层 gap 罚的豁免裕度 (下层 c2 无死区)
    parser.add_argument('--soc_tol', default=SOC_TOL, type=float,
                        help='【上层】块末 gap 罚的豁免裕度 (MWh): H 只对超过该值的残差付费 '
                             '(默认 0.1)')

    # ---- 探索噪声幅度 (逐维; 与 td3perrb 上层同口径) ----
    parser.add_argument('--wt_noise', default=0.05, type=float,
                        help='WT 5 维探索噪声幅度 (±wt_noise*eps)')
    parser.add_argument('--sig_noise_max', default=0.2, type=float,
                        help='SOC 目标信号维探索噪声幅度上限 (±sig_noise_max*eps)')
    parser.add_argument('--sig_noise_min', default=0.05, type=float,
                        help='SOC 目标信号维探索噪声幅度地板 (eps 衰减完后保底)')

    # ---- 24h 相位特征 (--phase_feat) ----
    parser.add_argument('--phase_feat', default=1, type=int, choices=[0, 1, 2],
                        help='H 层观测的 24h 相位特征: 0=关(4维); 1=sin/cos(6维, 默认); '
                             '2=1+[段内剩余比例, 剩余小时/8, 下一段电价/1000](9维)')

    # ---- 输出层 bias 初始化 (与 td3perrb 一致) ----
    parser.add_argument('--wt_bias_init', default=1, type=int, choices=[0, 1],
                        help='1: 把 actor 输出层 WT 5 维 bias 初始化到 Cp 高效区 '
                             '[0.25,0.40]; 0: 用 PyTorch 默认')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    max_E = 6.
    upload = True
    use_csv = True
    args.max_episode_length = 1440  # 1d = 24h*60m = 1440 steps
    args.train_iter_H = 1000 * args.max_episode_length
    args.seed = np.random.randint(0, 10000)
    args.cost_single_limit = 1000.

    # args.warmup = 5

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

    # RBB system: 可行窗口 (供 L 层观测) + 硬裁剪兜底
    rbb_system = RuleBasedBESS(
        max_E=max_E,
        min_E_factor=0.0,
        rated_power=args.rbb_rated_power,
        eta_c=0.98,
        eta_d=0.98,
        time=1.0 / 60,
        fluctuation_limit=args.rbb_fluctuation_limit,
    )

    # 上层 H: TD3 + EMA 实际成本对偶 (lam1 ∈ [3, 100], 初始 5.0) —— 与 td3perrb 一致
    nb_states_H = 4 + PHASE_DIM[int(getattr(args, 'phase_feat', 0))]
    print('H nb_states = {} (4 base + {} phase dims, --phase_feat={})'.format(
        nb_states_H, PHASE_DIM[int(getattr(args, 'phase_feat', 0))], args.phase_feat))
    agent_H = SP_TD3(nb_states=nb_states_H, nb_actions=6, args=args,
                     lam_max=100.0, lam_min=3.0,
                     ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                     record_update_stats=True,
                     policy_delay=args.policy_delay,
                     policy_noise=args.policy_noise,
                     noise_clip=args.noise_clip,
                     cost_critic_mode=args.cost_critic_mode,
                     wt_noise=args.wt_noise,
                     sig_noise_max=args.sig_noise_max,
                     sig_noise_min=args.sig_noise_min)
    agent_H.lam = 5.0
    if args.wt_bias_init:
        init_wt_action_bias(agent_H, lo=0.25, hi=0.40, sig_value=0.0)

    # 下层 L: 保留 td3per 自己的 RL BESS 操作
    #   * 观测 5 维蒸馏 = [pbrulemin/10, pbrulemax/10, soc_target/10, SOC/10, u/10]
    #   * r2 = −100·clip_amount; c2 = 1000·|soc_target − SOC'|
    #   * lam2 ∈ [10, 500] 初始 10.0; 每块 10 次更新; 步级折扣 --l_discount
    agent_L = SP_TD3(nb_states=5, nb_actions=1, args=args,
                     lam_max=500.0, lam_min=10.0,
                     ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                     policy_delay=args.l_policy_delay,
                     policy_noise=args.l_policy_noise,
                     noise_clip=args.noise_clip,
                     cost_critic_mode=args.cost_critic_mode)
    agent_L.lam = 10.0
    agent_L.cost_limit = 10.0  # L层c2专用初始值（覆盖全局cost_single_limit=1000）
    agent_L.discount = args.l_discount          # L 步级折扣 (与 H 的块级折扣解耦)
    agent_L.depsilon = 1.0 / max(args.l_epsilon, 1)  # L 探索衰减 (与 H 的 --epsilon 解耦)
    prYellow('L 层(BESS)策略超参: policy_delay={}, policy_noise={} (H 层 TD3: delay={}, noise={}); '
             '折扣: L γ={} (步级, 块内 ≤5 步) / H γ={} (块级); '
             '探索衰减: L 每 {} 次决策 / H 每 {} 次决策'.format(
                 args.l_policy_delay, args.l_policy_noise, args.policy_delay, args.policy_noise,
                 agent_L.discount, agent_H.discount, args.l_epsilon, args.epsilon))
    prCyan('下层奖励/成本口径: r2 = -100·clip_amount; '
           'c2 = 1000·slack, slack = max(0, |target-SOC\'| - |target-SOC_best|), '
           'SOC_best = SOC + g·clip(δ/g, RBB窗口) (能力内追踪失败量); '
           'L 观测第 3 维 = soc_target/10 (原 shadow_price)')
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
