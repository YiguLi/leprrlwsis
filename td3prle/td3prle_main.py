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
             c1 = 1.0*[10*(P_F-3)+ + 0.1*P_F] + W_GAP*unfeasible_gap
                  + 块末结算项 c1_soc_track = 10*|soc_target − SOC_块末|
  Lower (L) —— 两区间分治, 单一真源 hsrl_soctarget (与 prle/td3per/td3prrbb 逐位同口径):
             r2 = −1·min(clip_amount, 20)      [MW] 命令与实际之差 (两段 clip 合计)
             c2 = +1·trackerr         [MW] 实际施加 pb 到"参考集合"的距离
                  参考集合 = Soctrack范围[δ/(g·n_rem), δ/g] ∩ 平抑窗口
                    F ≠ ∅ ⇒ **区间**（L 可在「摊还端…一步填平端」间任选努力水平）
                    F = ∅ ⇒ **不存在范围, 只有纯粹一个竭力值 r*** = 平抑窗口里离目标 B 最近的端点:
                            只有顶到 r* 才不罚, 两侧偏离都按 |pbrule − r*| 线性计费

【信号语义 (本轮变更)】H 第 6 维输出 signal 不再是影子价格, 而是 SOC 目标的归一化指令;
  H 输出范围不变 ((-0.5, 0.5)), 由下层线性映射 (myfunc.signal_to_soc_target):
      soc_target = 3.0 + 5.0·signal ∈ [0.5, 5.5] MWh
  L 层把 soc_target 作为 c2 的跟踪参照 (trackerr, MW 口径), 同时进 L 观测量 (soc_target/10);
  H 层在 5 步块末结算时对 |soc_target − SOC_块末| 计 10 倍惩罚, 计入 c1 约束口
  (与波动/不可行缺口共用 d_block_H, 由 lam1 对偶驱动 H 给出下层可达的目标)。

Lagrange multiplier updates (EMA dual ascent on ACTUAL block cost):
  H: EMA   ← (1-0.02)*EMA + 0.02*sum_c1          (每 5 步块, sum_c1 含块末 SOC 跟踪项)
     lam1  ← clip(lam1 + 0.01*(EMA − d_block_H), 50, 500), 初始 100.0
     d_block_H 初始 12.0 (c1 平滑系数已回调 0.1, 预算未随动), curriculum 连续 100 次满足则减半, 地板 3.0
  L: lam2 ← clip(lam2·exp(hst.L_EMA_LAM_LR·(EMA_c2/d_block_L − 1)), hst.L_LAM_MIN, hst.L_LAM_MAX), 初值 hst.L_LAM_INIT = 1.0
     d_block_L 初始 hst.L_D_BLOCK = 1.2 (⇔ 0.02 MWh/块), curriculum 减半地板 hst.L_D_BLOCK_MIN = 0.15

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
    CheckAction, Normalize, Normalize_H, Original,
    Cal_Rew_PRRBB, nine_case_test,
    violation_derate, VIOL_DERATE_COEF,
    build_L_obs_distilled,
    get_price_level,
    get_wind_thresholds, set_wind_thresholds,
    relabel_r1_substep, relabel_c1_substep,
    signal_to_soc_target,
    WB_PRICE, WB_SOC_OUTSIDE, WT_PRICE, WT_WIND,
    PHASE_DIM, day_price_schedule, phase_daily_table, phase_minute_of_day,
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
import hsrl_cost as hc   # H 层 c1/λ/d_block 统一口径
import hsrl_soctarget as hst   # 下层 SOC 目标单一真源
import hsrl_agent_init as hinit   # critic 输出层 bias(价值基线)
import hsrl_warmup as hwarm       # warmup 调度 + 规则下层 + transition 填入
engine = matlab.engine.start_matlab()


# TD3PRLE: TD3 + PRLE 动态阈值分段奖励 + CL + LLM 元控制
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
    # PRLE 标定: c1 = c1_risk_coeff*[10*(P_F-3)+ + 0.1*P_F] + W_GAP*unfeasible_gap
    #            (HIGH 风速 ×1.2, LOW/MEDIUM ×0.8; W_GAP=10, td3prle 调参: 原 50 → 10)
    _h = hc.h_settings()   # uniform: 统一 d=15 / legacy: 原值
    d_block_H = _h['d_block'] if _h else 12.0        # 初值: 每块 sum_c1 预算 (平滑系数曾 0.1→1.0 ×10 重标; 现已回调 0.1, 预算未随动)
    d_block_H_min = _h['d_block_min'] if _h else 3.0     # curriculum 收紧地板
    # L 层 EMA 对偶 —— 每 5 步块实际成本预算 d_block_L (与 sum_c2 同口径)
    # 下层 SOC 目标口径 (单一真源 hsrl_soctarget, 与 prle/td3per/td3prrbb 同):
    #     r2 = −1·min(clip_amount, 20) [MW]; c2 = +1·trackerr [MW] (实际施加 pb 到参考集合的距离)
    d_block_L = hst.L_D_BLOCK        # 初值: 每块 sum_c2 预算 = 1.2 (⇔ 0.02 MWh/块)
    d_block_L_min = hst.L_D_BLOCK_MIN    # curriculum 收紧地板 = 0.15
    soc_tol_upper = float(getattr(args, "soc_tol", hc.SOC_TOL))   # 块末残差豁免

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
        df_test_written = 0          # 已增量落盘的 df_test9 行数
        df_episode_written = 0       # 已增量落盘的 df_episode 行数

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
            c1_soc_track = 0.0   # 本块末结算的 SOC 跟踪惩罚 (块末赋值, 记账用)
            block_ctx_H = []  # reset sub-step context for new 5-step block
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
            # 上层信号 → SOC 目标 (下层映射; 块内保持不变)
            soc_target_H = signal_to_soc_target(applied_action_H[5])
        prGreen('--H-- ori a_t = {}'.format(round_list(original_action_H, 3)))
        prPurple('--H-- soc_target = {:.3f} MWh (signal a6 = {:.3f})'.format(
            soc_target_H, applied_action_H[5]))

        # L 层观测蒸馏 (5维): [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]
        # 可行域 (RBB, 与 pb 无关, 预计算) + H 给定 SOC 目标 + 自身 SOC + 风速;
        # 不再直接看 pr/P_G/ptur —— 分层: H 给目标, L 在约束内执行.
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
        # L 层原始意图的 ±charge 越界量不再单独计费: r2 只含 RBB 窗口裁剪量
        #   (越界量与封顶项已随旧口径删除, 见 myfunc / hsrl_soctarget)
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
            soc_target_H, float(observation2[2])))
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
            # 上层【块末结算】: 目标 SOC 与实际 SOC 的差值惩罚 (豁免带 tol = --soc_tol, 默认 0.1)
            #   c1_soc_track = W_GAP * max(0, |soc_target − SOC_块末| − soc_tol)
            # 计入约束口 (与波动/不可行缺口同一预算 d_block_H), 由 lam1 对偶驱动
            # H 给出下层实际可达的目标; 块内逐步 c1 不入账本项 (一次性结算)。
            c1_soc_track = hst.gap_penalty(soc_target_H, float(observation2[2]),
                                           tol=soc_tol_upper)
            if hc.mode() == 'uniform':
                sum_r1 -= c1_soc_track   # uniform: 追踪偏好进 r (固定价 W_GAP=10 元/MWh)
            else:
                sum_c1 = sum_c1 + c1_soc_track
            # episode 结算成本同口径累计 (df_episode.episode_cost = 逐步波动成本 + 块末跟踪项)
            episode_cost += c1_soc_track
            # TD3PRLE: pass accumulated sub-step context for reward relabeling
            # ---- critic 输出层 bias(价值基线): warmup 后首个块末一次性应用 ----
            bias_init.on_block_end(step, sum_r1, sum_c1, agent_H)

            agent_H.observe_jump(
                norm_observation_H, original_action_H,
                sum_r1, sum_c1, norm_observation2, done,
                context=list(block_ctx_H)  # pass a copy
            )
            # H 层 EMA 对偶上升 —— 用 5 步块实际成本 sum_c1 更新 lam1 (与 d_block_H 同口径)
            if step > args.warmup:
                agent_H.update_ema_cost(sum_c1)
                hc.dual_update(agent_H, d_block_H)   # uniform=乘性 / legacy=加性
            # L 层 EMA 对偶上升 —— 用 5 步块实际成本 sum_c2 更新 lam2 (与 d_block_L 同口径)
            if step > args.warmup:
                agent_L.update_ema_cost(sum_c2)
                hst.dual_update_L(agent_L, d_block_L)
            # ---- L 层 curriculum: EMA 成本连续 L_CURRICULUM_BLOCKS 块满足预算 ⇒ 预算减半 ----
            #   计数改为块末每块 1 次 (此前嵌在更新循环里, "100" 的含义随 UTD 漂移)
            if agent_L.ema_cost is not None and agent_L.ema_cost <= d_block_L:
                consecutive_satisfied_L += 1
            else:
                consecutive_satisfied_L = 0
            if consecutive_satisfied_L >= hst.L_CURRICULUM_BLOCKS:
                d_block_L = max(d_block_L / 2, d_block_L_min)
                consecutive_satisfied_L = 0
                prGreen('L: d_block_L halved to {}'.format(d_block_L))
            done_L = True
        # L 层每步过渡不再存 context: c2 无风速/阈值依赖 (无需 relabel), 与 prle 一致
        hwarm.fill_bess_transition(
            agent_L, norm_observation_L, original_action_L, r2, c2,
            norm_observation2_L, done_L,
            applied_pb=float(pbrule), warmup=bool(step <= args.warmup),
            cmd_pb=float(pb_cmd_L),
            rule_pb=float(pb_rule),
            stats=bess_fill_stats, context=None)

        if step > args.warmup:
            #--------------------Update----------------------
            if episode_steps % 5 == 4:
                # H/L 层更新频率(UTD): H 每块 args.utd_h 次、L 每块 args.utd_l 次
                #   (默认 2 / 10; L 维持每块 10 次 = UTD 2:1)。块内前段与 L 交替,
                #   余量单独补足 L ⇒ 由 hc.utd_plan 拆分。
                _n_both, _n_h, _n_l = hc.utd_plan(args.utd_h, args.utd_l)
                for _ in range(_n_both):
                    agent_H.update_policy()
                    agent_L.update_policy()
                for _ in range(_n_h):
                    agent_H.update_policy()
                for _ in range(_n_l):
                    agent_L.update_policy()
                    # (L 的 curriculum 计数已移到块末: 每块 1 次, 阈值 hst.L_CURRICULUM_BLOCKS)

                # H 层 curriculum 计数: **每块 1 次**(与 UTD 解耦)。
                #   历史为每块 5 次计数(阈值 100) ⇒ 减半需 20 块, 故用
                #   hc.H_CURRICULUM_BLOCKS_5X(=20) 保持等价。
                #   EMA 成本每块只在块末更新一次 ⇒ 块内重复检查结论相同。
                if agent_H.ema_cost is not None and agent_H.ema_cost <= d_block_H:
                    consecutive_satisfied_H += 1
                else:
                    consecutive_satisfied_H = 0

                if consecutive_satisfied_H >= hc.H_CURRICULUM_BLOCKS_5X:
                    d_block_H = max(d_block_H / 2, d_block_H_min)  # H: 收紧块预算
                    consecutive_satisfied_H = 0
                    prGreen('H: d_block_H halved to {}'.format(d_block_H))
                # (L 的减半已移到块末: 每块 1 次)

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
                        # 增量落盘: 统一走 util.flush_csv_append (index=False 保持本表历史口径)
                        df_update_flushed_rows = flush_csv_append(
                            '{}/update_stats_H.csv'.format(output),
                            df_update, df_update_flushed_rows, index=False)

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
                        # c1 = 结算总成本 (含块末 SOC 跟踪项); c1_soc_track 为其分量
                        'c1_soc_track': m['c1_soc_track'],
                        'profit': m['profit'],
                        'turbine_profit': m['turbine_profit'],
                        'derate_loss': m['derate_loss'],
                        'penalty_NN_H': m['penalty_NN_H'],
                        # 存货结算 (口径修正, 不改训练路径): profit 只记现金流, 不含储存电量变动;
                        # 按外生账面价 SOC_SETTLE_PR (= hc.PR_REF = 580) 结算 ΔSOC。
                        'soc_start': m['soc_start'], 'soc_end': m['soc_end'],
                        'dsoc': m['dsoc'], 'profit_econ': m['profit_econ'],
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
                        'turbine_profit': overall['turbine_profit'],
                        'derate_loss': overall['derate_loss'],
                        'penalty_NN_H': overall['penalty_NN_H'],
                        'soc_start': overall['soc_start'], 'soc_end': overall['soc_end'],
                        'dsoc': overall['dsoc'], 'profit_econ': overall['profit_econ'],
                        'project_wind_levels': '',
                        'wind_low_max': overall['wind_low_max'],
                        'wind_high_min': overall['wind_high_min'],
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
                    writer.add_scalar('test9/overall/c1_soc_track', overall['c1_soc_track'], step)
                    writer.add_scalar('test9/overall/profit', overall['profit'], step)
                    writer.add_scalar('test9/overall/profit_econ', overall['profit_econ'], step)
                    writer.add_scalar('test9/overall/dsoc', overall['dsoc'], step)
                    writer.add_scalar('test9/overall/soc_end', overall['soc_end'], step)
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
                'soc_target': reward_details['soc_target'],
                'soc_slack': reward_details['soc_slack'],
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
                # 增量落盘: 统一走 util.flush_csv_append —— 首段覆盖写(截断旧文件),
                # 其后只追加自上次落盘以来的新行。之前每 50 步对跨 episode 只增不减的
                # 整表 df 做全量 to_csv, 是 50 整数倍 step 卡顿的根源。
                df_flushed_rows = flush_csv_append(
                    '{}/df_td3prle.csv'.format(output),
                    df, df_flushed_rows, index=False)

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
            writer.add_scalar('reward_lower/soc_track_err', reward_details['soc_track_err'], step)
            writer.add_scalar('reward_lower/clip_amount', reward_details['clip_amount'], step)

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
                        relabel_r1_substep, relabel_c1_substep,
                        discount=float(args.discount))
                    prGreen('TD3PRLE: Relabeled {} H-agent transitions'.format(n_relabeled))
                    # L 层 buffer 无需 relabel (c2 无风速/阈值依赖), 与 prle 一致

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
                # 增量落盘: 每 episode 一行, 原先每 episode 全量重写整表 (O(N²))
                df_episode_written = flush_csv_append(
                    '{}/df_episode_td3prle.csv'.format(output),
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
    # ---- 24h 相位特征 (--phase_feat) ----
    parser.add_argument('--phase_feat', default=1, type=int, choices=[0, 1, 2],
                        help='H 层观测的 24h 相位特征: 0=关(4维); 1=sin/cos(6维, 默认); '
                             '2=1+[段内剩余比例, 剩余小时/8, 下一段电价/1000](9维)')
    # 0.99
    # ---- H 层 n 步回报 / twin-target 聚合 ----
    #   冻结批离线验证 (agentutil/test_nstep_h.py 的动机, 见 2026-09 记录):
    #   1-step + min-target 在 40k 次更新后只到 MC 真值的 34 %;
    #   n=10 + mean-target 在 20k 次就到 90 % (n=10 + min 为 76 %)。
    #   实现: H 层把连续 n 个块折算进一条转移 (reward=Σγ^k r_k, 滑窗 stride-1:
    #   每个块都是某条转移的起点 ⇒ 状态覆盖 = 块数, 条目数 ≈ 块数), 每条转移自带
    #   bootstrap 系数 γ^n ⇒ 需要自助传播的深度从 1/(1−γ)=50 降到 50/n。
    #   ---- 默认值 = 5 (生产口径) ----
    #   依据: run12/run13 的 **2v2 观测** (非受控 A/B; 尚无同代码的 --nstep 1 对照臂):
    #     |dQp/da| @20k  n=5: 7.9 / 5.4  vs  n=1: 1.02 / 1.29 (5~8x, 且 n=5 不衰减)
    #     0-20k 块奖励   n=5: 181 / 160  vs  n=1: 55.5 / 90.7 (元)
    #     自洽比 @30k    n=5: 0.59/0.60  vs  n=1: 0.34 / 0.47
    #   **复现历史 run1~run10 请显式传 --nstep 1** (那条路径逐位不变)。
    parser.add_argument('--nstep', default=5, type=int,
                        help='H 层 n 步回报, 滑窗 stride-1 (默认 5 = 生产口径; '
                             '1 = 复现 run1~run10 的 1-step 历史口径; 建议 5~10)')
    parser.add_argument('--target_agg', default='min', type=str,
                        choices=['min', 'mean'],
                        help='H 层 twin profit-critic 的 target 聚合 '
                             '(min = TD3 原口径; mean = 去掉悲观偏差)')
    parser.add_argument('--bsize', default=128, type=int,
                        help='minibatch size')  #64
    parser.add_argument('--rmsize',
                        default=20000,
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
    parser.add_argument('--utd_h', default=2, type=int,
                        help='H 层每块 update_policy 次数 (默认 2 = 历史口径; 调整它不再改变 d_block_H 的减半节奏)')
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
    # L 层折扣 (步级语义) 与 H 层解耦: L 层每 5 步块即终止 (done_L=True),
    # bootstrap 视野 ≤5 步, γ 只影响块内 5 步加权 → 取 0.98 (与 H 的块级 γ 同值)。
    parser.add_argument('--l_discount', default=0.98, type=float,
                        help='L 层(BESS)折扣: 步级语义; L episode=5 步块, 视野≤5 步, 默认 0.98')
    parser.add_argument('--l_epsilon', default=50000,
                        help='L 层探索衰减步数 (与 H 的 --epsilon 解耦), 默认 50000')
    parser.add_argument('--l_noise_amp', default=1.0, type=float,
                        help='L 层探索噪声半宽 [MW] (残余动作量级; 旧式 ±0.05 比所需小 ~50 倍)')
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
    if getattr(args, 'max_steps', 0) and args.max_steps > 0:
        args.train_iter_H = int(args.max_steps)
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
    # H 观测维 = 4 (基础) + 相位维 (--phase_feat; 与其余五个 TD3 方法统一口径)
    nb_states_H = 4 + PHASE_DIM[int(getattr(args, 'phase_feat', 0))]
    print('H nb_states = {} (4 base + {} phase dims, --phase_feat={})'.format(
        nb_states_H, PHASE_DIM[int(getattr(args, 'phase_feat', 0))], args.phase_feat))
    agent_H = SP_TD3_PRLE(nb_states = nb_states_H, nb_actions = 6, args = args,
                          lam_max=(_HCFG['lam_max'] if _HCFG else 500.0), lam_min=(_HCFG['lam_min'] if _HCFG else 50.0),
                          ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                          record_update_stats=True,
                          policy_delay=args.policy_delay,
                          policy_noise=args.policy_noise,
                          noise_clip=args.noise_clip,
                          cost_critic_mode=args.cost_critic_mode,
nstep=args.nstep, target_agg=args.target_agg)
    agent_H.lam = _HCFG['lam_init'] if _HCFG else 100.0
    # 值函数回归空间归一化 (s = hc.value_scale()): 只作用于 H 层 r1/c1 进入
    # critic 之前; buffer/对偶/日志/CSV 仍是原始口径 (见 hsrl_cost 末尾小节)。
    agent_H.value_scale = hc.value_scale()
    agent_H.cost_scale = hc.cost_scale()
    # ---- 风机起始点: actor 输出层 bias = U(0.05,0.45) x5 (中性宽带; 策略接管时的起点) ----
    if args.wt_bias_init:
        init_wt_action_bias(agent_H, lo=0.05, hi=0.45, sig_value=0.0)

    if _HCFG:
        agent_H.ema_lam_lr = _HCFG['ema_lam_lr']   # uniform: 乘性对偶步长
    # L 层对偶改 EMA 实际成本更新 —— lam2 不再读 critic (Q_c), 与 td3prrbb 同配置;
    #     lam2 ∈ [hst.L_LAM_MIN, hst.L_LAM_MAX] = [0.5, 2] —— r2/c2 同为 MW ⇒ λ2 是**无量纲兑换率**
    #       ("每 MW c2 至多值 2 MW r2 / 至少值 0.5 MW r2"), 关于 1.0 对数对称; 初值 1.0
    #     ema_alpha = 0.02 (时间常数 ~50 块), **乘性**对偶 λ2 ← λ2·exp(0.02·(EMA/d_block_L − 1))
    #     蒸馏观测 5 维 = [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]
    # L 层(步骤A): policy_delay/--policy_noise 独立于 H —— 放松到 delay=1+noise≈0 时节奏接近 DDPG
    # (twin critic 与 EMA 对偶仍保留, 仅去掉延迟更新与 target smoothing 对窄可行窗的抹平)
    agent_L = SP_TD3_PRLE(nb_states = 5, nb_actions = 1, args = args,
                          lam_max=hst.L_LAM_MAX, lam_min=hst.L_LAM_MIN,
                          ema_lam=True, ema_alpha=0.02, ema_lam_lr=hst.L_EMA_LAM_LR,
                          policy_delay=args.l_policy_delay,
                          policy_noise=args.l_policy_noise,
                          noise_clip=args.noise_clip,
                          cost_critic_mode=args.cost_critic_mode)
    agent_L.lam = hst.L_LAM_INIT   # λ2 初值 (单一真源; 旧 100.0 属 [2,200] 旧口径, 已越界)
    agent_L.cost_limit = hst.L_COST_LIMIT  # L层c2专用初始值（覆盖全局cost_single_limit=1000）
    agent_L.value_scale = 1.0   # L 层 r2/c2 不归一化 (值函数尺度本就 O(1))
    # ---- 残余动作: L actor 输出层零初始化 ⇒ t=0 时 Δ≡0 ⇒ 起点逐位等于规则下层 ----
    hinit.zero_actor_output(agent_L, verbose=True)
    # L 层探索半宽 [MW] (残余动作量级): 旧式 ±0.05 MW 比所需动作小 ~50 倍
    agent_L.noise_halfwidth = float(getattr(args, 'l_noise_amp', 1.0))

    # L 层折扣与 H 解耦: 步级语义, L episode = 5 步块 → bootstrap 视野 ≤5 步
    agent_L.discount = args.l_discount
    agent_L.depsilon = 1.0 / max(args.l_epsilon, 1)   # L 探索衰减 (与 H 的 --epsilon 解耦)
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
