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
    * 规则只有一条: pb = clip(ΔSOC/gain, pbrulemin, pbrulemax) —— "一步填平 SOC
      残差"所需功率投影到 RBB 可行窗口, **无跟踪死区、无附加限幅机制**:
      残差 0.01 MWh 也追 (pb 仅 0.61 MW); 窗口不含 0 时只做窗口允许的最小被迫吞吐。
    * 物理可行 + 平抑波动由窗口硬保证 → 可行窗口步 clip_amount ≡ 0;
      因此块末残差 |soc_target - SOC_end| ≈ 不可行量 (窗口行走造成的过冲除外, 见 README §9)。

【死区归属】0.1 MWh 的豁免只给上层: 上层 gap 罚按 --soc_tol 豁免 0.1 MWh
  (H 不必为零头付费), 下层则尽力追到 0。不对称方向安全, 无幽灵税;
  只有把 --tracker_dead_zone 调成 > --soc_tol 时幽灵税才会回归。

奖励 / 成本
-----------
H: r1 = wT*(pr*ptur*dt) + wB*(-pr*pbrule*dt) - 490*max(-pbrule,0)*dt - penalty_NN_H
    c1 = 1.0*[10*(P_F-3)+ + 0.1*P_F] + 10*unfeasible_gap  (【本轮变更】分段系数已去掉)
         + [块末结算] 10*max(0, |soc_target - SOC_end| - SOC_TOL)

对偶上升 (仅 H, EMA 实际块成本)
    EMA  ← (1-0.02)*EMA + 0.02*sum_c1
    lam1 ← clip(lam1 + 0.01*(EMA - d_block_H), 3, 100), 初始 5.0
    d_block_H 初值 50.0, 地板 5.0; 连续 100 次满足则减半

【本项目新增: LLM 分段微调层】
    0. 启动自检: __main__ 在解析参数后立刻打印 LLM 配置 (provider/model/掩码 key)
       并做一次真实连通性测试 (PASSED/FAILED + 耗时), 结果醒目地留在控制台最前面;
       自检不通过不中断训练, 只降级为占位模式。
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
import time
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
    CheckAction, Normalize, Normalize_H,
    Cal_Rew_RLRB, gap_penalty, soc_target_from_signal, nine_case_test,
    SOC_TOL,
    get_wind_thresholds, set_wind_thresholds,
    relabel_r1_substep, relabel_c1_substep,
    WT_WIND, WT_PRICE, WB_PRICE, WB_SOC_OUTSIDE,
    REP_MINUTE, PHASE_DIM, phase_dim, phase_features, phase_log_fields,
    phase_daily_table, day_price_schedule, representative_phase_minute,
    PHASE_REP_MINUTE,
)
from td3leprrlrb.td3leprrlrb_llm import (
    WindowStats, llm_analyze_thresholds, create_llm_call_fn, load_llm_config,
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
import hsrl_cost as hc   # H 层 c1/λ/d_block 统一口径
import hsrl_agent_init as hinit   # critic 输出层 bias(价值基线)
import hsrl_warmup as hwarm       # warmup 调度 + 规则下层 + transition 填入
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


# =============================================================================
# P0.1: 训练起始点 —— 把 H 层 actor 输出层 bias 覆写到 Cp 高效区
# =============================================================================
# 病因: model.Actor.init_weights() 只初始化 fc1/fc2/fc3 的 weight 与 fc4.weight
#   (±init_w=0.003), fc4.bias 仍是 PyTorch 默认 U(±1/√fan_in), fan_in = hidden1 = 16
#   ⇒ U(±0.25)。而 WFSim_port.m 的 Cp = 4a(1−a)² 在 a=0 时 = 0 ⇒ a≈0 就是停机。
#   初始策略的"基准工作点"≈ bias 本身, 实测初始 ΣCp 只有 ~0.4 (≈14% 效率),
#   5 个种子里 2~4 台机被推到 a<0 (CheckAction 夹到 0) = 永久停机, 策略得先从
#   "停机"这个病态起点自己爬出来。
# 处方: 前 5 维 (风机桨距) bias 采在 U(0.25, 0.40) —— Cp 的高效区 (a=1/3 时
#   Cp=16/27≈0.5926 最优, 该区间正好夹住它); 第 6 维 (SOC 目标信号) 置 0
#   ⇒ 初始目标 SOC = 3.0, 中性且不泄露策略。
# 不动共享的 model.py; fc4.weight 仍是 ±0.003, 所以"策略强度"依然随机, 只是
#   "基准工作点"落在高效区。

WT_BIAS_LOW = 0.25
WT_BIAS_HIGH = 0.40
CP_MAX_PER_TURBINE = 16.0 / 27.0        # a=1/3 处的 Cp 上限 4a(1−a)²


def wt_cp_summary(bias, n_wt=5, a_lo=0.0, a_hi=0.5):
    """由 fc4.bias 的前 n_wt 维评估【初始工作点】.

    a 先按 env.PowerOutput 的口径 clip 到 [a_lo, a_hi] (a<0 被夹到 0 ⇒ Cp=0 = 停机),
    再算 Cp = 4a(1−a)²。返回 dict:
        sum_cp    : Σ Cp (物理口径, 与风机实际出力对应)
        eff_pct   : Σ Cp 占 5×(a=1/3 最优 Cp=16/27) 的百分比
        cp        : 逐台 Cp
        n_stopped : a<0 的台数 (被夹到 0 = 停机)
        sum_cp_raw: 不做 clip 的原始 Σ 4a(1−a)² (负 bias 会给出负 Cp, 仅作对照)
    """
    a_raw = np.asarray(bias[:n_wt], dtype=float)
    a = np.clip(a_raw, a_lo, a_hi)
    cp = 4.0 * a * (1.0 - a) ** 2
    total = float(cp.sum())
    return {
        'sum_cp': total,
        'eff_pct': 100.0 * total / (n_wt * CP_MAX_PER_TURBINE),
        'cp': cp,
        'n_stopped': int(np.sum(a_raw < 0.0)),
        'sum_cp_raw': float(np.sum(4.0 * a_raw * (1.0 - a_raw) ** 2)),
    }


def init_wt_action_bias(agent, low=WT_BIAS_LOW, high=WT_BIAS_HIGH, n_wt=5,
                        verbose=True):
    """覆写 H 层 actor / actor_target 的 fc4.bias (前 n_wt 维 U(low,high), 末维 0).

    Returns
    -------
    bias : np.ndarray  实际写入的 bias 向量 (便于打印/留档)
    """
    if agent is None:
        return None
    if not hasattr(agent, "actor") or not hasattr(agent.actor, "fc4"):
        raise ValueError("init_wt_action_bias: agent.actor.fc4 不存在")
    with torch.no_grad():
        for net in (agent.actor, agent.actor_target):
            b = net.fc4.bias
            if b.numel() != n_wt + 1:
                raise ValueError(
                    "init_wt_action_bias: fc4.bias 长度 {} != 预期 {} "
                    "(5 桨距 + 1 信号)".format(b.numel(), n_wt + 1))
            b[:n_wt].uniform_(float(low), float(high))
            b[n_wt] = 0.0
        # target 必须与 actor 完全一致 (构造函数里的 hard_update 之后又被改过)
        agent.actor_target.fc4.bias.copy_(agent.actor.fc4.bias)
    bias = agent.actor.fc4.bias.detach().cpu().numpy().copy()
    if verbose:
        s = wt_cp_summary(bias, n_wt)
        prCyan('LEPRRLRB[P0.1]: WT bias <- U({:.2f},{:.2f}) x{} + 信号维 0; '
               '初始 sum_Cp = {:.3f} (效率 {:.1f}%, 最优 {:.3f}), 死机 {}/{} 台; '
               '逐台 Cp = {}'.format(
                   low, high, n_wt, s['sum_cp'], s['eff_pct'],
                   n_wt * CP_MAX_PER_TURBINE, s['n_stopped'], n_wt,
                   np.round(s['cp'], 4).tolist()))
    return bias


# =============================================================================
# P0.3 自检: 电价时钟一致性 (相位特征依赖"第几步 = 当日第几分钟")
# =============================================================================

def price_clock_check(env, verbose=True):
    """用 env.CalPrice 走满一天, 与 day_price_schedule() 对比.

    env.CalPrice 是"先自增再判档"(count_price 从 0 起), 其第 k 次调用对应
    当日第 k 分钟 (k=1..1440, 表索引 k%1440)。训练第 j 步 (0-based) 用的正是
    第 j+1 分钟的电价, 即相位特征相对 env 时钟"领先 1 分钟"(1/480 的段内比例
    偏差, 对择时无实质影响) —— 该偏差在此一并报出。

    Returns
    -------
    mismatches : list of (k, env_price, schedule_price)
    """
    sched = day_price_schedule()
    saved = (env.count_price, env.pr)
    mismatches = []
    env.count_price = 0
    try:
        for k in range(1, REP_MINUTE + 1):
            p = env.CalPrice()
            exp = sched[k % REP_MINUTE]
            if p != exp:
                mismatches.append((k, p, exp))
    finally:
        env.count_price, env.pr = saved      # 复原 (episode 开始时 reset() 会重设)
    if verbose:
        known = [m for m in mismatches if m[0] == REP_MINUTE]
        unknown = [m for m in mismatches if m[0] != REP_MINUTE]
        if not unknown:
            prGreen('LEPRRLRB[P0.3]: 电价时钟自检 {}/{} 分钟与日程表一致; '
                    '唯一例外在第 {} 分钟 (env.CalPrice 在 count_price==1440 时落入 '
                    'else 档 → 980, 属 env 已知怪癖, 落在 episode 最后一步)'.format(
                        REP_MINUTE - len(mismatches), REP_MINUTE, REP_MINUTE))
        else:
            prRed('LEPRRLRB[P0.3]: 电价时钟自检发现 {} 处【非预期】不一致 (共 {} 处): '
                  '{}'.format(len(unknown), len(mismatches), unknown[:6]))
        prCyan('LEPRRLRB[P0.3]: 相位特征用人类时钟 (index = 当日第几分钟); '
               'env 时钟领先 1 分钟 (训练第 j 步 = 日程表第 j+1 分钟)')
    return mismatches


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
          adj_bounds=None,
          llm_call_fn=None,
          phase_feat=0,
          soc_tol_upper=SOC_TOL):

    agent_H.is_training = True
    step = episode = episode_steps = 0
    episode_reward = episode_cost = episode_profit = 0.
    episode_bess_profit = 0.
    episode_gap_pen = 0.
    episode_deg_charge = episode_deg_discharge = 0.
    observation = None
    time = 1/60
    phase_feat = int(phase_feat)
    # Curriculum learning: consecutive blocks where EMA cost <= d_block_H
    consecutive_satisfied_H = 0
    # H 层 EMA 对偶 —— 每 5 步块实际成本预算 d_block_H (与 sum_c1 同口径)
    #   sum_c1 的构成: 波动成本 + 10*unfeasible_gap + 块末 SOC 跟踪罚
    _h = hc.h_settings()   # uniform: 统一 d=15 / legacy: 原值
    d_block_H = _h['d_block'] if _h else 50.0        # 初值: 每块 sum_c1 预算 (先松后紧)
    d_block_H_min = _h['d_block_min'] if _h else 5.0     # 地板(min): 10*max(0,|δ|-0.1) ≤ 5 → 含死区总 |δ| ≤ 0.6

    # ---- LEPRRLRB: LLM 元控制器状态 ----
    window_stats = WindowStats()          # 一个 LLM 调整窗口内的运行统计
    threshold_history = []                # 历次阈值调整记录 (喂回提示词)
    block_ctx_H = []                      # 当前 5 步块的逐子步 relabel 上下文
    llm_window = int(getattr(args, 'llm_window', 7200))
    min_episodes_before_llm = int(getattr(args, 'llm_min_episodes', 3))
    llm_relabel_count = 0                 # 阈值变更次数 (落盘用)

    # ---- LEPRRLRB: LLM 调用函数由 __main__ 的【启动自检】构建并传入 ----
    # (放在 __main__ 而非此处, 是为了让 LLM 连通性验证在运行一开始就打印出来,
    #  不必等 MATLAB 引擎启动 + 分段计算 + env/agent 构造全部完成。)
    # llm_call_fn is None ⇒ 占位模式: 阈值永不调整, 行为等价于 td3prrlrb。
    if llm_call_fn is None:
        prYellow('LEPRRLRB: LLM 不可用 → 占位模式: 阈值保持起始值不变, '
                 '行为等价于 td3prrlrb')

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

    # ---- warmup 期调度 + critic bias 触发状态 ----
    #   warmup_sched: 上层确定性动作调度 (wt 0.33 起每块 ±0.05 clip[0.20,0.45]; 信号维恒 0
    #                 ⇒ SOC 目标语义的项目 soc_target ≡ 3, 影子电价语义的项目影子价 ≡ 0)
    #   bias_init   : warmup 块样本收集 + bias 一次性应用
    #                 (修 step == args.warmup 与块末 episode_steps%5==4 不同余导致永不触发的 bug;
    #                  触发点 = warmup 之后第一个块末, 仍在首次策略更新之前)
    warmup_sched, warmup_rule, bess_fill_stats = hwarm.build_from_args(args, agent_H)
    bias_init = hinit.BiasInitTracker.from_args(args)
    while step < num_iterations_H:
        # start of episode 重置环境
        prYellow('########### step:{} ###########'.format(step))
        # P0.3 相位: episode 恰好 1440 步 (env.reset 里 count_price=0, step 恒 done=False)
        #   ⇒ 当日分钟 = episode_steps % 1440; 下一步的相位 = (episode_steps+1) % 1440
        phase_minute = episode_steps % REP_MINUTE
        phase_minute2 = (episode_steps + 1) % REP_MINUTE
        if observation is None:
            observation = deepcopy(env.reset())  # [Pr,U,E,P_last]
            # SOC 随机化: 每个 episode(1440步) 从 U[0.5,5.5] 重新采样
            reset_E = float(np.random.uniform(0.5, 5.5))
            env.E = reset_E
            observation[2] = reset_E
            norm_observation = Normalize_H(observation, phase_minute, phase_feat)
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
                _wa = warmup_sched.action()          # None ⇒ 对照臂 (--warmup_sched random)
                applied_action_H = original_action_H = (
                    agent_H.random_action() if _wa is None else _wa)
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
        norm_observation2 = Normalize_H(observation2, phase_minute2, phase_feat)

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
            # 【死区归属】0.1 MWh 豁免只给上层: 下层 SocTracker 默认死区 0
            # (尽力追随), 这里按 --soc_tol 只对"超过豁免带"的残差收费。
            # 不对称方向是安全的 (L 能消的残差 ⊇ H 豁免的), 不会产生幽灵税;
            # 但若把 --tracker_dead_zone 调成 > --soc_tol, 幽灵税会回归。
            gap_pen = gap_penalty(soc_target, float(observation2[2]),
                                  tol=soc_tol_upper)
            if hc.mode() == 'uniform':
                sum_r1 -= gap_pen   # uniform: 追踪偏好进 r (固定价 W_GAP=10 元/MWh)
            else:
                sum_c1 += gap_pen
            sum_gap_pen += gap_pen

        # ---- 可达性诊断: 规则控制在本块的最大能力 ----
        # 上限就是窗口本身 (tracker 的动作 = clip(need, 窗口), 无附加限幅)
        cap_c = block_cap_charge(rbb_info_obs['pbrulemax'], eta_c=0.98, time=time)
        cap_d = block_cap_discharge(rbb_info_obs['pbrulemin'], eta_d=0.98, time=time)
        target_feasible = (soc_target <= E_current + cap_c + soc_tol_upper) and \
                          (soc_target >= E_current - cap_d - soc_tol_upper)

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
                    gap_raw, soc_tol_upper, target_feasible))
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
            # ---- critic 输出层 bias(价值基线): warmup 后首个块末一次性应用 ----
            bias_init.on_block_end(step, sum_r1, sum_c1, agent_H)

            agent_H.observe_jump(norm_observation_H, original_action_H,
                                 sum_r1, sum_c1, norm_observation2, done,
                                 context=list(block_ctx_H))
            # H 层 EMA 对偶上升 —— 用块实际成本 sum_c1 更新 lam1 (与 d_block_H 同口径)
            if step > args.warmup:
                agent_H.update_ema_cost(sum_c1)
                hc.dual_update(agent_H, d_block_H)   # uniform=乘性 / legacy=加性

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
                    agent_H, tracker, env_test, rbb_system, seed=42,
                    phase_feat=phase_feat, soc_tol_upper=soc_tol_upper)
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
                            # P0.3: 该 case 使用的代表相位
                            'phase_minute': m['phase_minute'],
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
                        'phase_minute': -1,
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
                # ---- P0.3 相位记录 (+6 列): 供事后核对"择时"是否真的发生 ----
                **phase_log_fields(phase_minute),
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

            # P0.3 相位 (24h 电价时钟)
            _pf = phase_log_fields(phase_minute)
            writer.add_scalar('phase/minute', _pf['phase_minute'], step)
            writer.add_scalar('phase/sin', _pf['phase_sin'], step)
            writer.add_scalar('phase/cos', _pf['phase_cos'], step)
            writer.add_scalar('phase/seg_remain_frac', _pf['seg_remain_frac'], step)
            writer.add_scalar('phase/seg_remain_hours', _pf['seg_remain_hours'], step)
            writer.add_scalar('phase/next_price', _pf['next_price'], step)
            writer.add_scalar('phase/n_states', agent_H.nb_states, step)

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
                        help='【上层】gap 罚的豁免带 (MWh): H 只对超过该带的目标残差付费。'
                             '默认 0.1 (=6 MW·min)。与下层跟踪死区已解耦 —— '
                             '下层用什么死区由 --tracker_dead_zone 决定')
    parser.add_argument('--tracker_dead_zone', default=0.0, type=float,
                        help='【下层】SocTracker 的跟踪死区 (MWh)。默认 0.0 = 尽力追随'
                             '(pb = clip(ΔSOC/gain, 可行窗口), 残差 0.01 MWh 也追; '
                             'δ=0 时只做窗口允许的最小被迫吞吐); '
                             '设为 >0 复现旧的"死区内不动作"行为, 但必须 ≤ --soc_tol, '
                             '否则会出现规则停住、上层照罚的幽灵税')

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

    # ---- P0.1 / P0.3: 起始点与 24h 相位 ----
    parser.add_argument('--wt_bias_init', default=1, type=int, choices=[0, 1],
                        help='P0.1 起始点: 1(默认)=把 H 层 actor 输出层 bias 覆写为 '
                             'WT∈U(0.25,0.40) + 信号维 0 (Cp 高效区, 消除"初始停机"病态); '
                             '0=保留 PyTorch 默认 bias U(±0.25) (复现旧 run)')
    parser.add_argument('--phase_feat', default=1, type=int, choices=[0, 1, 2],
                        help='P0.3 24h 相位特征: 0=无(4 维) / 1(默认)=sin,cos(6 维) / '
                             '2=+[段内剩余比例, 剩余小时/8, 下一段电价/1000] (9 维)')

    parser.add_argument('--value_scale', default=hc.VALUE_SCALE, type=float,
                        help='H 层 r1/c1 进入值函数回归空间的统一缩放 s '
                             '(默认 1000: 把回归目标从 O(1e4) 压到 O(10); '
                             '1 = 关闭归一化, 只影响 critic 回归空间)')
    parser.add_argument('--c1-mode', default='uniform',
                        choices=['uniform', 'legacy'],
                        help='H 层 c1 口径: uniform=波动当量(默认, hsrl_cost 单一真源); '
                             'legacy=各项目原公式(对照臂)')
    parser.add_argument('--critic_bias_init', default='estimate',
                        choices=['estimate', 'none'],
                        help='critic 输出层 bias 的价值基线初始化: estimate=用 warmup 观测的'
                             '块收益估计 r̄/(1-γ) (默认); none=不初始化(对照臂)')
    parser.add_argument('--critic_bias_p', default=-1.0, type=float,
                        help='显式指定利润 critic 基线 (>0 生效, 覆盖 estimate)')
    parser.add_argument('--critic_bias_c', default=-1.0, type=float,
                        help='显式指定成本 critic 基线 (>0 生效, 覆盖 estimate)')
    parser.add_argument('--max_steps', default=0, type=int,
                        help='>0 时覆盖 train_iter_H, 用于短程 A/B')
    hwarm.add_warmup_args(parser)
    args = parser.parse_args()
    hc.set_mode(args.c1_mode)
    hc.set_value_scale(args.value_scale)   # 值函数回归空间归一化 (单一真源)
    _HCFG = hc.h_settings()
    prCyan(hc.summary())
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    # ================================================================
    # LEPRRLRB: LLM 启动自检 —— 运行一开始就验证 LLM 分段微调层是否可用
    #   自检不通过不中断训练: llm_call_fn=None ⇒ 占位模式 (阈值保持起始值,
    #   行为等价于 td3prrlrb)。自检通过则把所有 LLM 输出留在控制台最前面,
    #   便于确认"这次 run 到底有没有在调阈值"。
    # ================================================================
    _SEP = '=' * 78
    prCyan(_SEP)
    prCyan('LEPRRLRB: LLM 分段微调层 · 启动自检')
    prCyan(_SEP)
    llm_call_fn = None
    try:
        _cfg = load_llm_config(args.llm_config)
        _prov = args.llm_provider or _cfg['provider']
        _key = _cfg['api_key'] or ''
        _key_show = ('{}...{}'.format(_key[:6], _key[-4:])
                     if len(_key) > 12 else ('(未设置)' if not _key else _key))
        prCyan('LEPRRLRB: 配置 provider={} model={} api_key={} (来源: {})'.format(
            _prov, _cfg['model'] or '(provider 默认)', _key_show,
            'llm_config.json' if _cfg['api_key'] else '环境变量/缺省'))
        llm_call_fn = create_llm_call_fn(provider=args.llm_provider,
                                         config_path=args.llm_config)
        prCyan('LEPRRLRB: LLM ready (provider={})'.format(
            args.llm_provider or 'from config'))
        prCyan('LEPRRLRB: Testing LLM API connectivity ...')
        _t0 = time.time()
        _resp = llm_call_fn('请回复"OK"，不要包含任何其他内容。')
        prGreen('LEPRRLRB: LLM API test PASSED ({:.1f}s). Response: "{}"'.format(
            time.time() - _t0, str(_resp).strip()[:120]))
    except Exception as e:
        prRed('LEPRRLRB: LLM API test FAILED — {}: {}'.format(type(e).__name__, e))
        prRed('LEPRRLRB: 训练将继续, 但【阈值调整会被跳过】(占位模式, 等价于 td3prrlrb)。')
        prRed('LEPRRLRB: 排查顺序: --llm_provider / --llm_config 路径 → '
              'llm_config.json 的 api_key → provider SDK 是否安装 → 网络。')
        llm_call_fn = None
    prCyan(_SEP)

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
           'WB_SOC_OUTSIDE={} (【本轮变更】BESS 收益项与 c1 波动项的分段系数都已去掉: wB ≡ 1.0, c1 系数 ≡ 1.0)'
           .format(WT_WIND, WT_PRICE, WB_PRICE, WB_SOC_OUTSIDE))

    max_E = 6.
    upload = True
    use_csv = True
    args.max_episode_length = 1440  # 1d = 24h*60m = 1440 steps
    args.train_iter_H = 1000 * args.max_episode_length
    if getattr(args, 'max_steps', 0) and args.max_steps > 0:
        args.train_iter_H = int(args.max_steps)
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

    # ---- P0.3: 电价时钟一致性自检 (相位特征的前提) ----
    price_clock_check(env)

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
    #   跟踪死区默认 0 (尽力追随): 0.1 MWh 的豁免只给上层的 gap 罚 (--soc_tol)。
    #   动作规则: pb = clip(ΔSOC/gain, pbrulemin, pbrulemax) —— 无附加限幅机制。
    tracker = SocTracker(soc_tol=args.tracker_dead_zone, eta_c=0.98, eta_d=0.98,
                         time=1.0 / 60, rated_power=args.rbb_rated_power)
    prCyan('LEPRRLRB: SOC 死区归属 — 下层 tracker_dead_zone={} MWh '
           '(0=尽力追随) / 上层 gap 豁免 --soc_tol={} MWh'.format(
               args.tracker_dead_zone, args.soc_tol))
    if args.tracker_dead_zone > args.soc_tol:
        prRed('LEPRRLRB: ⚠ tracker_dead_zone ({}) > --soc_tol ({}) → '
              '规则会停在上层仍在收费的区间里 (幽灵税)'.format(
                  args.tracker_dead_zone, args.soc_tol))

    # 上层: TD3 + EMA 实际成本对偶 (lam1 ∈ [3, 100], 初始 5.0)
    # P0.3: 观测维数 = 4 (pr,u,E,P_G) + 相位特征 (0/2/5)
    _phase_dim = phase_dim(args.phase_feat)
    prCyan('LEPRRLRB[P0.3]: --phase_feat={} ⇒ 相位特征 {} 维, H 观测 {} 维 '
           '(代表相位: {}; 相位周期 {} 步)'.format(
               args.phase_feat, _phase_dim, 4 + _phase_dim,
               PHASE_REP_MINUTE, REP_MINUTE))
    agent_H = SP_TD3_LEPRRLRB(nb_states=4 + _phase_dim, nb_actions=6, args=args,
                              lam_max=(_HCFG['lam_max'] if _HCFG else 100.0), lam_min=(_HCFG['lam_min'] if _HCFG else 3.0),
                              ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                              record_update_stats=True,
                              policy_delay=args.policy_delay,
                              policy_noise=args.policy_noise,
                              noise_clip=args.noise_clip,
                              cost_critic_mode=args.cost_critic_mode)
    agent_H.lam = _HCFG['lam_init'] if _HCFG else 5.0
    # 值函数回归空间归一化 (s = hc.value_scale()): 只作用于 H 层 r1/c1 进入
    # critic 之前; buffer/对偶/日志/CSV 仍是原始口径 (见 hsrl_cost 末尾小节)。
    agent_H.value_scale = hc.value_scale()

    if _HCFG:
        agent_H.ema_lam_lr = _HCFG['ema_lam_lr']   # uniform: 乘性对偶步长

    # ---- P0.1: 起始点 (覆写 actor/actor_target 输出层 bias) ----
    if args.wt_bias_init:
        init_wt_action_bias(agent_H)
    else:
        _b = agent_H.actor.fc4.bias.detach().cpu().numpy()
        _s = wt_cp_summary(_b)
        prYellow('LEPRRLRB[P0.1]: --wt_bias_init 0 -> 保留 PyTorch 默认起点; '
                 '初始 sum_Cp = {:.3f} (效率 {:.1f}%), 死机 {}/5 台; bias = {}'.format(
                     _s['sum_cp'], _s['eff_pct'], _s['n_stopped'],
                     np.round(_b, 4).tolist()))

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
              adj_bounds=adj_bounds,
              llm_call_fn=llm_call_fn,
              phase_feat=args.phase_feat,
              soc_tol_upper=args.soc_tol)
        if upload:
            writer.close()

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
