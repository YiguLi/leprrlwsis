#!/usr/bin/env python3
"""
TD3PRRBB (TD3 + PRRBB) — Main Training Loop
===========================================
TD3 (Twin Delayed DDPG) agents combined with PRRBB's Wind × Price
Segmented Reward structure.

Parent packages:
  - Algorithm layer (td3per/td3per_main.py):
      SP_TD3 agents — twin critics (profit channel: min aggregation,
      cost channel: max/safe aggregation with clamp>=0), delayed policy
      update, target policy smoothing, PER buffer with importance
      sampling, EMA dual ascent on actual per-block cost, H-layer
      update-stats recording, 9-case test with fixed seed 42.
  - Reward layer (prrbb/prrbb_main.py):
      Cal_Rew_PRRBB 机会/风险分离的分段奖励, RBB hard clipping before
      env.step, wind×price segmentation logging, d_block calibration.

Reward structure (td3prrbb_myfunc.Cal_Rew_PRRBB):
  Upper (H): r1 = wT*(pr*ptur*time) + wB*(-pr*pbrule*time)
                  - BESS_DEG_RMB_PER_MWH*max(-pbrule,0)*time - penalty_NN_H
                  wT = wT_u(u)×wT_p(pr), wB ≡ 1.0 (【本轮变更】BESS 分段系数已去掉)
             c1: uniform(默认 --c1-mode) = P_F + 20*(P_F-3)+   [MW·step]  (hsrl_cost.fluct_cost)
                 legacy  = r_u(u)*[10*(P_F-3)+ + 0.1*P_F] + W_GAP(50)*unfeasible_gap
  Lower (L) —— 两区间分治 (hsrl_soctarget):
             r2 = −1·min(clip_amount, 20)      [MW] 命令与实际之差 (两段 clip 合计)
             c2 = +1·trackerr         [MW] 实际施加 pb 到参考集合的距离
                  参考集合 = Soctrack范围[δ/(g·n_rem), δ/g] ∩ 平抑窗口
                    F ≠ ∅ ⇒ **区间**（L 可在「摊还端…一步填平端」间任选努力水平）
                    F = ∅ ⇒ **不存在范围, 只有纯粹一个竭力值 r*** = W 里离 B 最近的端点:
                            只有顶到 r* 才不罚, 两侧偏离都按 |pbrule − r*| 线性计费
                            (够不着的缺口不可达 ⇒ 由上层块末 gap 罚承接)

Lagrange multiplier updates (EMA dual ascent on ACTUAL block cost):
  H: EMA   ← (1-0.02)*EMA + 0.02*sum_c1          (每 5 步块)
     uniform: 乘性 lam1 ← clip(lam1*exp(0.02*(EMA/d_block_H − 1))),
              lam1 ∈ [0.967, 4.833], 初值 0.967, d_block_H = 15.0 (= 3 MW × 5 步)
     legacy : 加性 lam1（本项目原式, lam1 ∈ [3, 50]）
  L: EMA ← 0.98*EMA + 0.02*sum_c2
     lam2 ← clip(lam2*exp(0.02*(EMA/d_block_L − 1)), 0.5, 2), 初值 1.0  (无量纲兑换率: 每 MW c2 至多值 2 MW r2 / 至少 0.5)
     d_block_L 初始 1.2 (⇔ 允许搁置 0.02 MWh/块), curriculum 减半地板 0.15
"""
import os
import sys
import numpy as np
import argparse
from copy import deepcopy
import torch

# Ensure both td3prrbb/ and parent (HSRL) directories are on path
_TD3PRRBB_DIR = os.path.dirname(os.path.abspath(__file__))
_HSRL_DIR = os.path.dirname(_TD3PRRBB_DIR)
sys.path.insert(0, _TD3PRRBB_DIR)
sys.path.insert(0, _HSRL_DIR)

from env import WindBattery
from td3prrbb.td3prrbb_ddpg import SP_TD3
from util import *
from td3prrbb.td3prrbb_myfunc import (
    CheckAction, Normalize, Normalize_H, Original,
    Cal_Rew_PRRBB, nine_case_test,
    violation_derate, VIOL_DERATE_COEF,
    build_L_obs_distilled,
    get_price_level,
    get_wind_thresholds, set_wind_thresholds,
    PHASE_DIM, day_price_schedule, phase_daily_table, phase_minute_of_day,
)
from td3prrbb.td3prrbb_rule_bess import RuleBasedBESS

import pandas as pd
from memory import RingBuffer

from torch.utils.tensorboard import SummaryWriter

# 不显示warning
import warnings
warnings.filterwarnings("ignore")

import matlab
import matlab.engine
import hsrl_cost as hc   # H 层 c1/λ/d_block 统一口径
import hsrl_soctarget as hst   # 下层 SOC 目标单一真源
import hsrl_agent_init as hinit   # critic 输出层 bias(价值基线)
import hsrl_warmup as hwarm       # warmup 调度 + 规则下层 + transition 填入
engine = matlab.engine.start_matlab()


# TD3PRRBB: TD3 + PRRBB 分段奖励 + CL
def init_wt_action_bias(agent, lo=0.05, hi=0.45, sig_value=0.0, verbose=True):
    """把 H 层 actor / actor_target 输出层 `fc4.bias` 的前 5 维(风机轴向诱导因子)初始化到
    中性宽带 **U(0.05, 0.45)**(5 维各自独立随机), 第 6 维(信号)置 sig_value。

    动机(run160 诊断, 与 td3perrb/td3prrlrb 同): `model.py` 只把 `fc4.weight` 初始化到
    ±3e-3, `fc4.bias` 仍是 PyTorch 默认 U(±1/√fan_in); 而 `Cp = 4a(1−a)²`(见 WFSim_port.m):
    a=0 → Cp=0(该机停机), a=1/3 → 0.593(Betz 最优), a=0.5 → 0.5。
    初始策略 ≈ bias ⇒ 不设的话从"半停机"起步(实测死机 2~4 台、全场 ΣCp 仅 0.47~0.72),
    且 a<0 区被 `CheckAction` 裁掉后梯度为 0 ⇒ 没有恢复力就会长期停在停机态。
    U(0.05,0.45) 覆盖"上游降载 → 末台单机最优"的整个区间, **故意不预设**风场最优的形状
    (它随尾流状态变化, 是**策略要学的函数**);
    **5 维各自随机**另有打破机组间初始对称的作用。

    不改共享的 model.py; 构造后覆写 actor 与 actor_target。
    """
    if agent is None or not hasattr(agent, 'actor') or not hasattr(agent.actor, 'fc4'):
        return False
    nb = agent.actor.fc4.bias.data.shape[0]
    with torch.no_grad():
        b = agent.actor.fc4.bias.data
        n_wt = min(5, nb)
        b[:n_wt] = torch.tensor(np.random.uniform(lo, hi, n_wt), dtype=b.dtype)
        if nb > n_wt:
            b[n_wt:] = float(sig_value)
        tgt = getattr(agent, 'actor_target', None)
        if tgt is not None and hasattr(tgt, 'fc4'):
            tgt.fc4.bias.data.copy_(b)
    if verbose:
        a_mean = float(b[:n_wt].mean())
        prGreen('init_wt_action_bias: WT bias = {} (mean a = {:.3f} -> Cp = {:.3f}); '
                'signal = {}'.format(np.round(b[:n_wt].numpy(), 3), a_mean,
                                     4 * a_mean * (1 - a_mean) ** 2,
                                     float(b[n_wt]) if nb > n_wt else None))
    return b[:n_wt].numpy().copy()


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
    # 每 episode (1440 步) 起点 SOC 均匀随机重置 [0.5, 5.5]，
    # 替代原跨 episode 继承 (prev_soc) — 保证日初 SOC 覆盖安全带，避免长期锁死窄带
    soc_reset_low, soc_reset_high = 0.5, 5.5
    time = 1/60
    # Curriculum learning: consecutive steps where EMA cost <= d_block
    consecutive_satisfied_H = 0
    consecutive_satisfied_L = 0
    # H 层 EMA 对偶 —— 每 5 步块实际成本预算 d_block_H (与 sum_c1 同口径)
    # PRRBB 标定: c1 = r_u*[10*(P_F-3)+ + 0.1*P_F] + 50*unfeasible_gap
    _h = hc.h_settings()   # uniform: 统一 d=15 / legacy: 原值
    d_block_H = _h['d_block'] if _h else 1.2         # 初值: 每块 sum_c1 预算
    d_block_H_min = _h['d_block_min'] if _h else 0.3     # curriculum 收紧地板
    # L 层 EMA 对偶 —— 每 5 步块实际成本预算 d_block_L (与 sum_c2 同口径)
    # 下层 SOC 目标口径 (单一真源 hsrl_soctarget): r2 = −1·min(clip_amount, 20), c2 = +1·trackerr [MW]
    d_block_L = hst.L_D_BLOCK
    d_block_L_min = hst.L_D_BLOCK_MIN
    soc_tol_upper = float(getattr(args, "soc_tol", hc.SOC_TOL))   # 块末残差豁免
    if use_csv:
        df = pd.DataFrame()
        df_written = 0
        df_episode = pd.DataFrame()
        df_episode_written = 0
        df_test = pd.DataFrame()
        df_test_written = 0
        df_update = pd.DataFrame()  # H 层更新数据（TD/Q/batch 统计）
        df_update_written = 0

    # ---- 1440-step window counters for env metrics ----
    window_violation_count = 0   # steps where |P_F| > 3 MW
    window_power_sum = 0.0        # cumulative P_G over the window

    # ---- warmup 期调度 + critic bias 触发状态 ----
    #   warmup_sched: 上层确定性动作调度 (wt 0.33 起每块 ±0.05 clip[0.20,0.45]; 信号维恒 0
    #                 ⇒ SOC 目标语义的项目 soc_target ≡ 3, 影子电价语义的项目影子价 ≡ 0)
    #   bias_init   : warmup 块样本收集 + bias 一次性应用
    #                 (修 step == args.warmup 与块末 episode_steps%5==4 不同余导致永不触发的 bug;
    #                  触发点 = warmup 之后第一个块末, 仍在首次策略更新之前)
    warmup_sched, warmup_rule, bess_fill_stats = hwarm.build_from_args(args, agent_H, agent_L)
    # 残余动作自检: warmup 规则臂写进 buffer 的是 Δ≡0 ⇒ 它的"命令"必须与策略期基线
    # (同一个 hst.rule_action + 同一个 rbb_system 参数) **逐位相同**, 否则 warmup 行自相矛盾
    hwarm.check_residual_baseline(
        warmup_rule, warmup_sched, args,
        lambda _tgt, _E, _lo, _hi: hst.rule_action(
            _tgt, _E, _lo, _hi, 5,
            eta_c=rbb_system.eta_c, eta_d=rbb_system.eta_d,
            time=rbb_system.time, rated_power=rbb_system.rated_power),
        soc_target_from_signal=hst.soc_target_from_signal)
    bias_init = hinit.BiasInitTracker.from_args(args)
    # 与 H agent 绑定同一 (γ, s): 估计口径 = critic 回归口径, 结构上不会分叉
    bias_init.value_scale = float(getattr(agent_H, 'value_scale', hc.value_scale()))
    bias_init.discount = float(agent_H.discount)

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
            # SOC 每 episode 随机重置: env.reset() 默认把 E 重置为 3.0,
            # 此处覆盖为 [0.5, 5.5] 均匀随机起点 (每个 episode 一致)
            env.E = np.random.uniform(soc_reset_low, soc_reset_high)
            observation[2] = env.E
            norm_observation = Normalize_H(observation, phase_minute_of_day(0),
                                           phase_mode, phase_table)
            agent_H.reset(norm_observation)

        # 相位自检: env 的电价必须与 day_price_schedule() 在 episode_steps%1440 处一致
        pm_now = phase_minute_of_day(episode_steps)
        if phase_mode > 0:
            if abs(float(observation[0]) - phase_sched[pm_now]) > 1e-6:
                phase_mismatch += 1
            phase_checked += 1

        # agent pick action 上层选择动作
        if episode_steps % 5 == 0:
            sum_r1 = 0
            sum_c1 = 0
            sum_c2 = 0     # L 层 5 步块累计成本 (喂 EMA 对偶)
            sum_gap_pen = 0.0
            if step <= args.warmup:
                _wa = warmup_sched.action()          # None ⇒ 对照臂 (--warmup_sched random)
                applied_action_H = original_action_H = (
                    agent_H.random_action() if _wa is None else _wa)
                penalty_NN_H = 0
            else:
                original_action_H = agent_H.select_action(norm_observation)
                applied_action_H, penalty_NN_H = CheckAction(original_action_H)
            viol_derate = violation_derate(original_action_H, args.viol_derate)
            norm_observation_H = norm_observation
            pw_tru = env.PowerOutput(observation[1], applied_action_H)
        prGreen('--H-- ori a_t = {}'.format(round_list(original_action_H, 3)))

        # L 层观测蒸馏 (5维): [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]
        # 可行域 (RBB, 与 pb 无关, 预计算) + H 影子价格信号 + 自身 SOC + 风速;
        # 不再直接看 pr/P_G/ptur —— 分层: H 给信号, L 在约束内执行.
        ptur = float(pw_tru)
        pg_prev = float(observation[-1])
        E_current = float(observation[2])
        _, rbb_info_obs = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=ptur, pg_prev=pg_prev, E=E_current)
        norm_observation_L = build_L_obs_distilled(
            observation, pw_tru, applied_action_H, rbb_info_obs,
            5 - (episode_steps % 5))
        prPurple('--L-- obs = {}.'.format(round_list(norm_observation_L,3)))
        # 下层每一个5-step都是一个episode
        if episode_steps % 5 == 0:
            agent_L.reset(norm_observation_L)
            pb_l = list()

        # agent pick action 下层选择动作
        # ---- 残余动作: 命令 = 规则基线 + Δ (Δ 由 actor 输出, 初值 0) ----
        # 基线 = 规则动作 (单一真源 hst.rule_action, 与 SocTracker.act 逐位一致):
        #   Δ≡0 ⇒ 命令逐位等于规则下层 ⇒ r2≡0、c2≡0 —— "先稳定到规则"是**结构保证**。
        #   基线负责 clip/物理限幅/被迫吞吐等非线性, 网络只学"相对规则偏多少 [MW]"。
        _n_rem_L = 5 - (episode_steps % 5)
        _soc_tgt_L = hst.soc_target_from_signal(float(applied_action_H[5]))
        pb_rule = hst.rule_action(
            _soc_tgt_L, E_current, rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'],
            _n_rem_L, eta_c=rbb_system.eta_c, eta_d=rbb_system.eta_d,
            time=rbb_system.time, rated_power=rbb_system.rated_power)
        if step <= args.warmup:
            # warmup 期下层 = 规则行为体 (rule 臂: Δ=0, 命令 = 基线)
            if warmup_rule is not None:
                delta_L = np.zeros(1, dtype=np.float64)
                pb_cmd_L = float(pb_rule)
            else:               # 对照臂 (--warmup_bess random): 仍是"绝对随机功率"命令
                delta_L = np.asarray(agent_L.random_action(), dtype=float).reshape(-1)
                pb_cmd_L = float(delta_L[0])
                delta_L = np.array([pb_cmd_L - float(pb_rule)])
            penalty_NN_L = 0
        else:
            delta_L = np.asarray(agent_L.select_action(norm_observation_L),
                                 dtype=float).reshape(-1)
            pb_cmd_L = float(pb_rule) + float(delta_L[0])
        original_action_L = delta_L          # 入 buffer 的动作 = Δ (actor 的动作空间)
        applied_action_L, penalty_NN_L = CheckAction(np.array([pb_cmd_L]))
        prGreen('--L-- rule = {:.3f} | Δ = {:.3f} | cmd = {:.3f}'.format(
            float(pb_rule), float(delta_L[0]), float(pb_cmd_L)))
        prGreen('--L-- ori a_t = {}'.format(round_list(original_action_L, 3)))

        action = np.concatenate((applied_action_H, applied_action_L))
        original_action = action.copy()

        # ================================================================
        # RBB: Rule-Based BESS Clipping (hard clipping before env.step)
        # ================================================================
        # 交给 RBB 的是**命令值**（CheckAction 之前）—— 这样 clip_amount = |cmd − pbrule|
        # 自动含两段 clip: ① CheckAction 的 ±10 夹裁 + ② 平抑窗口夹裁。
        # 因窗口 ⊆ ±10, 传命令值与传夹后值给出的 pbrule 逐位相同 (见 test_l_lower_unit);
        # 若改回传夹后值, |cmd|>10 时该方向梯度恒 0 (死区)。见 hsrl_soctarget.lower_reward。
        pb_original = float(pb_cmd_L)   # 命令值 (CheckAction 前) = 规则基线 + Δ

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
        norm_observation2 = Normalize_H(observation2,
                                        phase_minute_of_day(episode_steps + 1),
                                        phase_mode, phase_table)
        # 下一状态 L 观测: 用下一状态重算可行域 (pb 无关)
        _, rbb_info_obs2 = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=float(pw_tru2),
            pg_prev=float(observation2[-1]), E=float(observation2[2]))
        norm_observation2_L = build_L_obs_distilled(
            observation2, pw_tru2, applied_action_H, rbb_info_obs2,
            5 - ((episode_steps + 1) % 5))

        # ================================================================
        # PRRBB Segmented Reward Computation (机会/风险分离版)
        # ================================================================
        r1, c1, r2, c2, reward_details = Cal_Rew_PRRBB(
            observation, observation2, action_rbb, rbb_info,
            penalty_NN_H=penalty_NN_H, time=time,
            ptur=ptur, n_rem=5 - (episode_steps % 5),
            turbine_derate=viol_derate,
        )

        # Total reward and cost (H uses r1/c1; L uses r2/c2)
        r = r1 + r2
        c = c1

        sum_r1 += r1
        sum_c1 += c1
        sum_c2 += c2  # 累计块成本 (每 5 步归零)

        # SOC 目标/信号 (块内固定; 供 L 观测、块末罚与日志用) —— 必须在下面的
        # 块末 gap 罚之前取出, 否则 UnboundLocalError
        soc_target = reward_details['soc_target']
        soc_signal = reward_details['soc_signal']

        # ---- 块末结算: SOC 目标可行性罚 (豁免 tol = --soc_tol, 默认 0.1) ----
        #   偏好用固定价 W_GAP=10 元/MWh 进 r1 (uniform); legacy 仍进 c1
        gap_raw = abs(soc_target - float(observation2[2]))
        gap_pen = 0.0
        if episode_steps % 5 == 4:
            gap_pen = hst.gap_penalty(soc_target, float(observation2[2]),
                                      tol=soc_tol_upper)
            if hc.mode() == 'uniform':
                sum_r1 -= gap_pen
            else:
                sum_c1 += gap_pen
            sum_gap_pen += gap_pen

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
        prGreen('r2 = {} (clip={})'.format(
            round(r2, 3), round(reward_details['clip_amount'], 3)))
        prGreen('c2 = {} (slack={:.4f}, soc_dev={} | target={:.2f}, SOC={:.2f})'.format(
            round(c2, 3), reward_details['soc_slack'],
            round(reward_details['soc_deviation'], 3),
            soc_target, float(observation2[2])))
        prGreen('price={} p_level={} wind={:.1f} w_level={} (wT={}, wB={})'.format(
            reward_details['pr'], reward_details['price_level'],
            reward_details['u'], reward_details['wind_level'],
            reward_details['w_wind'], reward_details['w_bess']))

        # ---- 1440-step window: track actual grid fluctuation ----
        if reward_details['P_F'] > 3.0 + 1e-4:
            window_violation_count += 1
        window_power_sum += observation2[-1]  # cumulative P_G

        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True
        if episode_steps % 5 == 4:
            # ---- critic 输出层 bias(价值基线): warmup 后首个块末一次性应用 ----
            bias_init.on_block_end(step, sum_r1, sum_c1, agent_H)

            agent_H.observe_jump(norm_observation_H, original_action_H, sum_r1, sum_c1, norm_observation2, done)
            # H 层 EMA 对偶上升 —— 用 5 步块实际成本 sum_c1 更新 lam1 (与 d_block_H 同口径)
            if step > args.warmup:
                agent_H.update_ema_cost(sum_c1)
                hc.dual_update(agent_H, d_block_H)   # uniform=乘性 / legacy=加性
            # L 层 EMA 对偶上升 —— 用 5 步块实际成本 sum_c2 更新 lam2 (与 d_block_L 同口径)
            if step > args.warmup:
                agent_L.update_ema_cost(sum_c2)
                hst.dual_update_L(agent_L, d_block_L)
            # ---- L 层 curriculum: EMA 成本连续 L_CURRICULUM_BLOCKS 块满足预算 ⇒ 预算减半 ----
            #   计数改为块末每块 1 次 (此前嵌在更新循环里 = 每块 10 次, "100" 的含义随 UTD 漂移)
            if agent_L.ema_cost is not None and agent_L.ema_cost <= d_block_L:
                consecutive_satisfied_L += 1
            else:
                consecutive_satisfied_L = 0
            if consecutive_satisfied_L >= hst.L_CURRICULUM_BLOCKS:
                d_block_L = max(d_block_L / 2, d_block_L_min)
                consecutive_satisfied_L = 0
                prGreen('L: d_block_L halved to {}'.format(d_block_L))
            done_L = True
        hwarm.fill_bess_transition(
            agent_L, norm_observation_L, original_action_L, r2, c2,
            norm_observation2_L, done_L,
            applied_pb=float(pbrule), warmup=bool(step <= args.warmup),
            cmd_pb=float(pb_cmd_L),
            rule_pb=float(pb_rule),
            stats=bess_fill_stats)

        if step > args.warmup:
            #--------------------Update----------------------
            if episode_steps % 5 == 4:
                # H/L 层更新频率(UTD): H 每块 args.utd_h 次、L 每块 args.utd_l 次
                #   (默认 2 / 10, 与其余 5 个 TD3 main 的 td3per 口径一致)。
                #   两者交织在**同一个**循环里 ⇒ 用 hc.utd_plan 拆成"交替 n_both + 各自补足"。
                _n_both, _n_h, _n_l = hc.utd_plan(args.utd_h, args.utd_l)
                for _ in range(_n_both):
                    agent_H.update_policy()
                    agent_L.update_policy()
                for _ in range(_n_h):
                    agent_H.update_policy()
                for _ in range(_n_l):
                    agent_L.update_policy()
                    # (L 的 curriculum 计数已移到块末: 每块 1 次, 阈值 hst.L_CURRICULUM_BLOCKS)

                # H 层 curriculum 计数: **每块 1 次**(与 UTD 解耦);
                #   EMA 成本每块只在块末更新一次 ⇒ 块内重复检查结论相同。
                #   阈值 = 块数 hc.H_CURRICULUM_BLOCKS(=50, 等价于历史的"2 次/块 × 100")。
                if agent_H.ema_cost is not None and agent_H.ema_cost <= d_block_H:
                    consecutive_satisfied_H += 1
                else:
                    consecutive_satisfied_H = 0

                if consecutive_satisfied_H >= hc.H_CURRICULUM_BLOCKS:
                    d_block_H = max(d_block_H / 2, d_block_H_min)  # H: 收紧块预算
                    consecutive_satisfied_H = 0
                    prGreen('H: d_block_H halved to {}'.format(d_block_H))
                # (L 的减半已移到块末: 每块 1 次)

                # ---- H 层更新数据记录（本块 H 2 批均值 → TB/csv） ----
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
                        df_update_written = flush_csv_append(
                            '{}/update_stats_H.csv'.format(output),
                            df_update, df_update_written)

            # Test: 9-case segmented evaluation every 480 episode steps
            # (fixed scenario seed 42 -> same wind trajectory each test, so
            # consecutive test9 events are point-comparable and smooth)
            if episode_steps % 480 == 0:
                case_metrics, overall = nine_case_test(
                    agent_H, agent_L, env_test, rbb_system, seed=42,
                    phase_mode=phase_mode, phase_table=phase_table,
                    viol_derate_coef=args.viol_derate)
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
                        'turbine_profit': m['turbine_profit'],
                        'derate_loss': m['derate_loss'],
                        'penalty_NN_H': m['penalty_NN_H'],
                        # 存货结算 (口径修正, 不改训练路径): profit 只记现金流, 不含储存电量
                        # 的变动; 按外生账面价 580 (myfunc.SOC_SETTLE_PR = hc.PR_REF) 结算 ΔSOC。
                        'soc_start': m['soc_start'], 'soc_end': m['soc_end'],
                        'dsoc': m['dsoc'], 'profit_econ': m['profit_econ'],
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
                        'turbine_profit': overall['turbine_profit'],
                        'derate_loss': overall['derate_loss'],
                        'penalty_NN_H': overall['penalty_NN_H'],
                        'soc_start': overall['soc_start'], 'soc_end': overall['soc_end'],
                        'dsoc': overall['dsoc'], 'profit_econ': overall['profit_econ'],
                    }])], ignore_index=True)
                    df_test_written = flush_csv_append(
                        '{}/df_test9.csv'.format(output), df_test, df_test_written)
                if upload:
                    writer.add_scalar('test9/overall/r1', overall['r1'], step)
                    writer.add_scalar('test9/overall/c1', overall['c1'], step)
                    writer.add_scalar('test9/overall/r2', overall['r2'], step)
                    writer.add_scalar('test9/overall/c2', overall['c2'], step)
                    writer.add_scalar('test9/overall/r', overall['r'], step)
                    writer.add_scalar('test9/overall/c', overall['c'], step)
                    writer.add_scalar('test9/overall/cvm', overall['cvm'], step)
                    writer.add_scalar('test9/overall/profit', overall['profit'], step)
                    writer.add_scalar('test9/overall/profit_econ', overall['profit_econ'], step)
                    writer.add_scalar('test9/overall/dsoc', overall['dsoc'], step)
                    writer.add_scalar('test9/overall/soc_end', overall['soc_end'], step)
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
                'original_action_L': delta_L,              # = Δ (actor 输出; 残余动作)
                'delta_L': float(delta_L[0]),
                'pb_rule': float(pb_rule),
                'pb_cmd': float(pb_cmd_L),
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
                'soc_signal': reward_details['soc_signal'],
                'soc_target': reward_details['soc_target'],
                'soc_slack': reward_details['soc_slack'],
                'soc_track_err': abs(soc_target - float(observation2[2])),
                'c1_fluct': reward_details['c1_fluct'],
                'penalty_NN_H': penalty_NN_H,
                'SOC': reward_details['SOC'],
                'P_F': reward_details['P_F'],
                'penalty_L': penalty_L,
                'penalty_NN_L': penalty_NN_L,
                # Convenience
                'soc_next': float(observation2[2]),
                'bess_profit_net': reward_details['bess_profit'] - reward_details['degradation'],
            }])], ignore_index=True)
            if step % 50 == 0:
                df_written = flush_csv_append(
                    '{}/df_td3prrbb.csv'.format(output), df, df_written)

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
            writer.add_scalar('reward_lower/soc_slack', reward_details['soc_slack'], step)
            writer.add_scalar('target/soc_target', soc_target, step)
            writer.add_scalar('target/soc_signal', soc_signal, step)
            writer.add_scalar('soc_track_err', abs(soc_target - float(observation2[2])), step)
            writer.add_scalar('gap_penalty', gap_pen, step)
            writer.add_scalar('sum_gap_pen', sum_gap_pen, step)
            writer.add_scalar('reward_lower/clip_amount', reward_details['clip_amount'], step)

            # Shadow price (action栏)
            writer.add_scalar('action/soc_target', soc_target, step)
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
                writer.add_scalar('action/soc_signal_raw',
                                  applied_action_H[5], step)
                writer.add_scalar('action/pb_original', pb_original, step)
                writer.add_scalar('action/pb_rule', float(pb_rule), step)
                writer.add_scalar('action/delta_L', float(delta_L[0]), step)
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
        # 上层利润: pg * 实际电价 * dt - BESS_DEG_RMB_PER_MWH * pb * dt
        episode_profit += observation2[-1] * observation[0] * time - BESS_DEG_RMB_PER_MWH * max(-float(pbrule), 0) * time
        # BESS盈亏: 影子套利 - 退化成本
        episode_bess_profit += reward_details['bess_profit'] - BESS_DEG_RMB_PER_MWH * max(-float(pbrule), 0) * time
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
                df_episode_written = flush_csv_append(
                    '{}/df_episode_td3prrbb.csv'.format(output),
                    df_episode, df_episode_written)

            # end of episode
            # reset
            # 下个 episode 起点 SOC 由循环顶部在 [0.5, 5.5] 内随机重置，不再继承
            observation = None
            episode_steps = 0
            episode_reward = episode_cost = episode_profit = 0.
            episode_bess_profit = 0.
            episode += 1
    if use_csv:
        df.to_csv('{}/df_td3prrbb.csv'.format(output))
        df_episode.to_csv('{}/df_episode_td3prrbb.csv'.format(output))
        df_test.to_csv('{}/df_test9.csv'.format(output))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode',
                        default='train',
                        type=str,
                        help='support option: train/test')
    parser.add_argument('--env',
                        default='TD3PRRBB',
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
    # 0.95: H 层 γ 块级语义 ⇒ 半衰期 13.5 块 ≈ 68 min; 1/(1−γ) = 20 块 ≈ 100 min
    parser.add_argument('--discount', default=0.95, type=float, help='')
    # ---- 24h 相位特征 (--phase_feat) ----
    parser.add_argument('--phase_feat', default=0, type=int, choices=[0, 1, 2],
                        help='H 层观测的 24h 相位特征: 0=关(4维, 默认, 时钟过拟合治理见 README); 1=sin/cos(6维); '
                             '2=1+[段内剩余比例, 剩余小时/8, 下一段电价/1000](9维)')
    # 0.99
    # ---- H 层 n 步回报 / twin-target 聚合 ----
    #   冻结批离线验证 (agentutil/test_nstep_h.py 的动机, 见 2026-09 记录):
    #   1-step + min-target 在 40k 次更新后只到 MC 真值的 34 %;
    #   n=10 + mean-target 在 20k 次就到 90 % (n=10 + min 为 76 %)。
    #   实现: H 层把连续 n 个块折算进一条转移 (reward=Σγ^k r_k, 滑窗 stride-1:
    #   每个块都是某条转移的起点 ⇒ 状态覆盖 = 块数, 条目数 ≈ 块数), 每条转移自带
    #   bootstrap 系数 γ^n ⇒ 需要自助传播的深度从 1/(1−γ)=50 降到 50/n。
    #   ---- 默认值 = 10 (生产口径) ----
    #   依据: run12/run13 的 **2v2 观测** (非受控 A/B; 尚无同代码的 --nstep 1 对照臂):
    #     |dQp/da| @20k  n=5: 7.9 / 5.4  vs  n=1: 1.02 / 1.29 (5~8x, 且 n=5 不衰减)
    #     0-20k 块奖励   n=5: 181 / 160  vs  n=1: 55.5 / 90.7 (元)
    #     自洽比 @30k    n=5: 0.59/0.60  vs  n=1: 0.34 / 0.47
    #   **复现历史 run1~run10 请显式传 --nstep 1** (那条路径逐位不变)。
    parser.add_argument('--nstep', default=10, type=int,
                        help='H 层 n 步回报, 滑窗 stride-1 (默认 10 = 生产口径; '
                             '1 = 复现 run1~run10 的 1-step 历史口径; 建议 10)')
    parser.add_argument('--target_agg', default='min', type=str,
                        choices=['min', 'mean'],
                        help='H 层 twin profit-critic 的 target 聚合 '
                             '(min = TD3 原口径; mean = 去掉悲观偏差)')
    parser.add_argument('--bsize', default=256, type=int,
                        help='minibatch size')  #64
    parser.add_argument('--rmsize',
                        default=40000,
                        type=int,
                        help='memory size') # 6000000
    # ---- 越界乘性惩罚 ----
    #   动机 (agentutil/probe_q_profile_boundary.py 一维 Q(a_j) 剖面实测, run13 ckpt40000):
    #     线性罚 200/单位在 critic 空间只有 200/1000 = 0.2, 而框内经济斜率 0.27~4.5
    #     ⇒ 跨界几乎不花钱, Q 峰贴在约束边界上 (dim4 峰 a=0.05, dim5 峰甚至在界外 -0.60)
    #     ⇒ 平滑 actor 逼近角点解时一半状态落界外 (dim4 25.7%, dim5 47.1%)。
    #     且典型越界(深度~0.09)的罚贡献 = 0.018, 比 critic 自身 TD 误差 0.64 小 30+ 倍,
    #     边界位置落在 critic 噪声里。
    #   做法: r1 的风机收益项乘 f = exp(-c·x), x = Σ 前 5 维越界量 (dim5=signal 不计)。
    #     * 乘 turbine_profit (>=0 恒成立) 而非净利润: 排除亏损步上 f<1 鼓励越界的符号陷阱;
    #       且界面斜率 = c·(该步风机收益), 与收益同尺度、随风速/电价自动缩放。
    #     * r1 单调不增 (f<=1 且 turbine_profit>=0) ⇒ 越界永不改善奖励。
    #     * **只动 r1**: env 物理量、revenue/turbine_profit/degradation/profit 等记录列不变。
    #   ⚠ **默认 0 = 关闭** (run14 实测 c=5.0 有害: 奖励电平被策略自身越界率绑架
    #     ⇒ critic 高估约 2x ⇒ 正反馈; 详见 README 的 --viol_derate 条目)。
    #   传 0 = 逐位复现历史 run1~run14。
    parser.add_argument('--viol_derate', default=VIOL_DERATE_COEF, type=float,
                        help='越界乘性惩罚系数 c: 风机收益 × exp(-c·Σ前5维越界量) '
                             '(默认 %.1f = **关闭**; 5.0 = 已证伪的旧口径, 见 README run14 分析)'
                             % VIOL_DERATE_COEF)
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
    parser.add_argument('--soc_tol', default=hc.SOC_TOL, type=float,
                        help='块末 SOC 目标残差豁免 (MWh); 只属于上层 gap 罚')
    parser.add_argument('--epsilon',
                        default=50000,
                        type=int,
                        help='linear decay of exploration policy')
    # ---- 探索噪声（**与 td3per 逐字同格式**：逐维独立抽样 + 地板）----
    parser.add_argument('--wt_noise', default=0.05, type=float,
                        help='WT 维探索噪声幅度 (±wt_noise*eps)')
    parser.add_argument('--sig_noise_max', default=0.2, type=float,
                        help='SOC 目标信号维探索噪声幅度上限 (±sig_noise_max*eps)')
    parser.add_argument('--sig_noise_min', default=0.01, type=float,
                        help='SOC 目标信号维探索噪声**地板** (eps 收敛到它, 不再降到 0)')
    parser.add_argument('--wt_noise_floor', default=0.01, type=float,
                        help='WT 维探索噪声**地板** (0 = 探索窗口结束后完全不再探索)')
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

    # ---- UTD（每块 update_policy 次数）: 启动指令可调; 默认 = 历史口径, 逐位不变 ----
    parser.add_argument('--utd_h', default=4, type=int,
                        help='H 层每块 update_policy 次数 (默认 4; 复现历史传 --utd_h 2; 调整它不再改变 d_block_H 的减半节奏)')
    parser.add_argument('--utd_l', default=10, type=int,
                        help='L 层每块 update_policy 次数 (默认 10 = 历史口径; 与 H 交替 + 补足)')

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
    parser.add_argument('--l_discount', default=0.98, type=float,
                        help='L 层(BESS)折扣: **步级**语义 (H 为块级), 默认 0.98')
    parser.add_argument('--l_epsilon', default=50000,
                        help='L 层探索衰减步数 (与 H 的 --epsilon 解耦), 默认 50000')
    parser.add_argument('--l_noise_amp', default=1.0, type=float,
                        help='L 层探索噪声半宽 [MW] (残余动作量级; 旧式 ±0.05 比所需小 ~50 倍)')
    parser.add_argument('--cost_critic_mode', default='max', type=str,
                        choices=['min', 'max'],
                        help='cost twin-critic aggregation: max (safe, recommended) or min (TD3 paper; risks cost underestimation -> Q_c collapse)')

    # RBB parameters (for soft penalty computation only, no hard clipping)
    parser.add_argument('--rbb_fluctuation_limit', default=3.0, type=float,
                        help='Max grid power fluctuation between steps (MW)')
    parser.add_argument('--rbb_rated_power', default=10.0, type=float,
                        help='BESS rated charge/discharge power (MW)')

    # 风速分段方法分岔: percentile(33/66分位) / kmeans(簇中心中点) / expert(专家锚点 8.5/12.0)
    parser.add_argument('--seg', default='expert',
                        choices=['percentile', 'kmeans', 'expert'],
                        help='风速分段方法: percentile | kmeans | expert (默认 expert)')

    parser.add_argument('--wt_bias_init', default=1, type=int, choices=[0, 1],
                        help='风机输出层 bias 初始值 = U(0.05,0.45) x5 (中性宽带): 1=开(默认) 0=关')
    parser.add_argument('--value_scale', default=hc.VALUE_SCALE, type=float,
                        help='H 层 r1/c1 进入值函数回归空间的统一缩放 s '
                             '(默认 1000: 把回归目标从 O(1e4) 压到 O(10); '
                             '1 = 关闭归一化, 只影响 critic 回归空间)')
    parser.add_argument('--cost_scale', default=hc.COST_SCALE_TD3, type=float,
                        help='H 层 c1 进入值函数回归空间的缩放 s_c（与 s_p 组成双尺度; '
                             'λ 由 hsrl_cost.lam_scale()=s_c/s_p 自动重标 ⇒ 物理价不变; '
                             '<=0 ⇒ 与 s_p 相同 = 旧行为）')
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
    hc.set_cost_scale(args.cost_scale)   # cost 通道缩放 (λ 自动跟随)
    _HCFG = hc.h_settings()
    prCyan(hc.summary())
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    # ---- 风速分段边界: 按 --seg 应用（percentile / kmeans / expert）----
    from data_piecewise.boundaries import apply_segmentation
    apply_segmentation(args.seg, set_wind_thresholds)

    #TODO 修改的命令ppyp
    max_E = 6.
    upload = True
    use_csv = True
    args.max_episode_length = 1440  # 1d = 24h*60m = 1440 steps
    args.train_iter_H = 1000 * args.max_episode_length
    if getattr(args, 'max_steps', 0) and args.max_steps > 0:
        args.train_iter_H = int(args.max_steps)
    # 只在**未显式指定**（--seed 默认 -1）时随机化。显式 --seed >= 0 必须被尊重，
    # 否则"同 seed 配对 A/B"会被随机种子混淆（历史 bug：本行曾无条件覆盖用户传入的 --seed）。
    if args.seed is None or int(args.seed) < 0:
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
    # H 观测维 = 4 (基础) + 相位维 (--phase_feat; 与其余五个 TD3 方法统一口径)
    nb_states_H = 4 + PHASE_DIM[int(getattr(args, 'phase_feat', 0))]
    print('H nb_states = {} (4 base + {} phase dims, --phase_feat={})'.format(
        nb_states_H, PHASE_DIM[int(getattr(args, 'phase_feat', 0))], args.phase_feat))
    agent_H = SP_TD3(nb_states = nb_states_H, nb_actions = 6, args = args,
                     lam_max=(_HCFG['lam_max'] if _HCFG else 50.0), lam_min=(_HCFG['lam_min'] if _HCFG else 3.0),
                     ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                     record_update_stats=True,
                     policy_delay=args.policy_delay,
                     policy_noise=args.policy_noise,
                     noise_clip=args.noise_clip,
                              wt_noise=args.wt_noise,
                              sig_noise_max=args.sig_noise_max,
                              sig_noise_min=args.sig_noise_min,
                              wt_noise_floor=args.wt_noise_floor,
                     cost_critic_mode=args.cost_critic_mode,
nstep=args.nstep, target_agg=args.target_agg)
    agent_H.lam = _HCFG['lam_init'] if _HCFG else 5.0
    # 值函数回归空间归一化 (s = hc.value_scale()): 只作用于 H 层 r1/c1 进入
    # critic 之前; buffer/对偶/日志/CSV 仍是原始口径 (见 hsrl_cost 末尾小节)。
    agent_H.value_scale = hc.value_scale()
    agent_H.cost_scale = hc.cost_scale()
    # ---- 风机起始点: actor 输出层 bias = U(0.05,0.45) x5 (中性宽带; 策略接管时的起点) ----
    if args.wt_bias_init:
        init_wt_action_bias(agent_H, lo=0.05, hi=0.45, sig_value=0.0)

    if _HCFG:
        agent_H.ema_lam_lr = _HCFG['ema_lam_lr']   # uniform: 乘性对偶步长
    # L 层对偶改 EMA 实际成本更新 —— lam2 不再读 critic (Q_c)，断高估→顶死传导；
    #     lam2 ∈ [L_LAM_MIN, L_LAM_MAX] = [0.5, 2] —— r2/c2 同为 MW ⇒ λ2 是**无量纲兑换率**
    #       ("每 MW c2 至多值 2 MW r2 / 至少值 0.5 MW r2"), 关于 1.0 对数对称; 初值 1.0
    #     ema_alpha = 0.02 (时间常数 ~50 块), **乘性**对偶 λ2 ← λ2·exp(0.02·(EMA/d_block_L − 1))
    #     d_block_L = 1.2 ⇔ 0.02 MWh/块; curriculum 连续 50 块满足 ⇒ 减半 (地板 0.15)
    #     蒸馏观测 5 维 = [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]
    # L 层(步骤A): policy_delay/--policy_noise 独立于 H —— 放松到 delay=1+noise≈0 时节奏接近 DDPG
    # (twin critic 与 EMA 对偶仍保留, 仅去掉延迟更新与 target smoothing 对窄可行窗的抹平)
    agent_L = SP_TD3(nb_states = 5, nb_actions = 1, args = args,
                     lam_max=hst.L_LAM_MAX, lam_min=hst.L_LAM_MIN,
                     ema_lam=True, ema_alpha=0.02, ema_lam_lr=hst.L_EMA_LAM_LR,
                     policy_delay=args.l_policy_delay,
                     policy_noise=args.l_policy_noise,
                     noise_clip=args.noise_clip,
                              wt_noise=args.wt_noise,
                              sig_noise_max=args.sig_noise_max,
                              sig_noise_min=args.sig_noise_min,
                              wt_noise_floor=args.wt_noise_floor,
                     cost_critic_mode=args.cost_critic_mode)
    agent_L.cost_limit = hst.L_COST_LIMIT  # L层c2专用初始值（覆盖全局cost_single_limit=1000）
    agent_L.lam = hst.L_LAM_INIT           # λ2 初值 = 中性兑换率 1.0 (单一真源; 此前漏设, 继承 --lam 默认 3)
    agent_L.value_scale = 1.0   # L 层 r2/c2 不归一化 (值函数尺度本就 O(1))
    # ---- 残余动作: L actor 输出层零初始化 ⇒ t=0 时 Δ≡0 ⇒ 起点逐位等于规则下层 ----
    hinit.zero_actor_output(agent_L, verbose=True)
    # L 层探索半宽 [MW] (残余动作量级): 旧式 ±0.05 MW 比所需动作小 ~50 倍
    agent_L.noise_halfwidth = float(getattr(args, 'l_noise_amp', 1.0))
    agent_L.discount = args.l_discount      # L 步级折扣 (与 H 的块级折扣解耦)
    agent_L.depsilon = 1.0 / max(args.l_epsilon, 1)   # L 探索衰减 (与 H 的 --epsilon 解耦)

    prYellow('L 层(BESS)策略超参: policy_delay={}, policy_noise={} (H 层 TD3: delay={}, noise={})'.format(
        args.l_policy_delay, args.l_policy_noise, args.policy_delay, args.policy_noise))
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
