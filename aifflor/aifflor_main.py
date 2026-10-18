#!/usr/bin/env python3
"""
AIFFLOR — **td3per 的 FLORIS 引擎版**：上层 TD3PERRB 口径, 下层 RL BESS 追 SOC 目标
====================================================================================
本文件是 td3per 的 "td3perrb 化" 改造 (原地修改):

  * 上层 H: **与 td3perrb 完全一致**
      - 动作 6 维 = 5 台风机桨距 [0,0.5] + 1 个 SOC 目标信号 a6 ∈ [-0.5, 0.5];
      - a6 → soc_target = 3 + 5·a6 ∈ [0.5, 5.5] (块级(5 步)固定);
      - 非分段奖励 r1 / c1 (收益权重与风险系数恒 1.0, W_GAP = 10);
      - 块末 SOC 目标可行性罚 gap_pen = 10·max(0, |target − SOC_末| − 0.1);
      - 观测 = Normalize(obs) 4 维 + 24h 相位特征 (--phase_feat 0/1/2 → 4/6/9 维; 默认 0 = 无时钟);
      - 对偶: lam1 ∈ [3, 100] 初值 5.0, d_block_H = 50 → 地板 5 (EMA 实际块成本);
      - 每块 2 次更新; 逐维探索噪声 (--wt_noise / --sig_noise_max / --sig_noise_min);
      - --epsilon 4000 (H 决策次数 ⇒ 20k 环境步); actor 输出层 WT bias 初始化到“中性宽带”。
  * 下层 L: **td3per 自己的 RL BESS 操作保留** (TD3 + PER + EMA 对偶 + RBB 硬裁剪),
      只把"信号 / 追踪目标"换成 SOC 目标:
      - 观测 (5 维蒸馏) = [pbrulemin/10, pbrulemax/10, **soc_target/10**, SOC/10, u/10]
        (第 3/4 维 = SOC/目标, 第 5 维 = n_rem/5; 旧观测无 n_rem 且目标在第 3 维);
      - r2 = −1·min(clip_amount, 20)            (原影子套利项随信号语义一起删除);
      - c2 = +1·trackerr [MW] (参考集合 = Soctrack范围 ∩ 平抑窗口),
        F ≠ ∅ ⇒ 该区间内任选努力水平; F = ∅ ⇒ **不存在范围, 只有纯粹一个竭力值 r***,
        顶到 r* 才不罚、两侧偏离都按 |pbrule − r*| 线性计费 (缺口由 r1 块末 gap 罚承接);
        零成本动作仍是规则下层 (旧名 soc_track_slack 保留为兼容别名);
      - lam2 ∈ [hst.L_LAM_MIN, hst.L_LAM_MAX] = [0.5, 2] —— **无量纲兑换率**
        ("每 MW c2 至多值 2 MW r2 / 至少值 0.5 MW r2", 关于 1.0 对数对称), 初值 1.0,
        d_block_L 1.2 → 地板 0.15, 每块 1 次**乘性**对偶更新 (η2 = 0.02);
      - L 层折扣/探索按 td3per 原口径: --l_discount 0.98 (步级), --l_epsilon 50000。

奖励 / 成本口径
----------------
H: r1 = pr·P_G·dt − 490·max(−pbrule,0)·dt − penalty_NN_H          (P_G = ptur − pbrule)
    − [块末] W_GAP(10)·max(0, |soc_target − SOC_末| − SOC_TOL(0.1))   (uniform: 进 r1; legacy: 进 c1)
    c1: uniform(默认 --c1-mode) = P_F + 20·(P_F−3)⁺  [MW·step]       (hsrl_cost.fluct_cost)
        legacy = 1.0·[10·(P_F−3)⁺ + 0.1·P_F] + W_GAP(10)·unfeasible_gap
L: r2 = −1·min(clip_amount, 20)   [MW] 命令与实际之差 (两段 clip 合计)
    c2 = +1·trackerr [MW] = 实际施加 pb（由 SOC 变化反解）到**参考集合**的距离
         参考集合 = Soctrack范围[δ/(g·n_rem), δ/g] ∩ 平抑窗口（两区间分治, hsrl_soctarget）
           F ≠ ∅ ⇒ 区间（L 可在「摊还端…一步填平端」间任选努力水平）
           F = ∅ ⇒ **不存在范围, 只有纯粹一个竭力值 r*** = W 里离 B 最近的端点:
                   只有顶到 r* 才不罚, 两侧偏离都按 |pbrule − r*| 线性计费（见 myfunc）

对偶上升 (两层都是 EMA 实际块成本)
----------------------------------
    H: EMA ← 0.98·EMA + 0.02·sum_c1
       uniform: 乘性 lam1 ← lam1·exp(0.02·(rate/d_block_H − 1)), rate = critic cost 的 Q
                lam1 ∈ 物理 [2.0, 50.0] 元/(MW·step)（= actor [0.2, 5.0]），初值 = 下限 2.0
                依据: 上界 = max(平抑资源边际成本 20, 峰价降载 16.33) 的 2.5×；界外波动单价 20·λ1
                d_block_H = 15.0 (= 3 MW × 5 步)
                curriculum 地板 = 0（用户第六轮）：d_block_H 可一直减半，连续 50 块不满足即自行停止
                ⚠ 历史口径 [0.967, 100] / [50, 100] 已被取代（见 hsrl.HSRL_LAM1_PHYS_MIN 的注释）
       legacy : 加性 lam1（本项目原式）
    L: EMA ← 0.98·EMA + 0.02·sum_c2;  lam2 ← clip(lam2·exp(0.02·(EMA/d_block_L − 1)), 0.5, 2), 初值 1.0 (无量纲兑换率)
       d_block_L = 1.2 (⇔ 允许搁置 0.02 MWh/块), curriculum 减半地板 0.15

--hsrl: 另一条**方法分支**（本文件内并列、互不干扰）
---------------------------------------------------
加 `--hsrl` 后走上表的 **"块总放电量"语义**（全部实现见 `aifflor/aifflor_hsrl.py`，
未加 `--hsrl` 时该模块的函数一行都不会被调用 ⇒ 旧路径逐位不变）：

  * H 第 6 维 = **BESS 在本 5 步块内的总放电量 g [MW·min]**（放电为正），
    可行域 `[−C_max, +D_max]` 由**块首 SOC + 额定功率**推出（**不含波动约束**）；
    越界线性罚 10 元/MW·min（不封顶），与 H 前 5 维同处 r1；
  * H 奖励**去掉 bessdeg**：`r1 = pr·P_G·dt − pen5 − pen6`；**块末 SOC 目标 gap 罚删除**；
  * L 动作 = **绝对 pb**（不再是"规则基线 + Δ"），只按**每步物理域**（SOC + 额定功率）裁剪；
  * L reward：`r2 = −200·越界量[MW] − 20·放电量[MW·min]`（退化 = 越界罚的 1/10；
    **不含** rulebase 波动裁剪罚）；
  * L cost（图中公式）：`c_L = 1 − exp(−|(g − Σ已放电)/(δ − t_L) − a_L|)`，
    δ = 5、t_L = 块内已过步数、a_L/Σ 都取**实际施加值**（放电为正）；
  * L 观测 6 维 = `[u/10, pr/1000, SOC/0.5, P_G(t−1)/5, g/2.5, need/2.5]`
    （第 6 维 = c_L 的参照点 `need=(g−Σ已放电)/n_rem`；残余局限见 README §3.8）
    （⚠ 不含 Σ/n_rem ⇒ 对 c_L 的参照量非 Markov；这是刻意的原方法形态，见 README）；
  * λ2 重标 `[2, 50]`（初值 10）、`d_block_L = 0.5`（地板 0.05）；
  * warmup 下层 = **摊还执行器** `clip(−(g−Σ)/n_rem, 物理域)`（c_L ≡ 0 的零成本动作）；
  * 训练中的测试改跑 **20 步协议**：电价恒 300、风速随机取一列 + 随机起点 20 步、
    开局 SOC = 0.25，触发频率与旧 9-case 相同（`episode_steps % 480 == 0` 且已过 warmup）。
"""
import os
import sys
import numpy as np
import argparse
from copy import deepcopy
import torch

# Ensure both aifflor/ and parent (HSRL) directories are on path
_AIFFLOR_DIR = os.path.dirname(os.path.abspath(__file__))
_HSRL_DIR = os.path.dirname(_AIFFLOR_DIR)
sys.path.insert(0, _AIFFLOR_DIR)
sys.path.insert(0, _HSRL_DIR)

# aifflor 与 td3per 的**唯一**区别：风场仿真器 WFSim(MATLAB) → FLORIS(纯 Python)
from aifflor.aifflor_env import AifflorPhysics, WindBatteryAIF
from aifflor import aifflor_turbine as aturb
from aifflor.aifflor_ddpg import SP_TD3
from aifflor import aifflor_hsrl as hsrl   # --hsrl 分支的全部口径 (单一真源)
from util import *
from td3per.td3per_myfunc import (
    CheckAction, Normalize, Normalize_H,
    Cal_Rew_RB, gap_penalty, soc_target_from_signal, nine_case_test,
    violation_derate, VIOL_DERATE_COEF,
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

import hsrl_cost as hc   # H 层 c1/λ/d_block 统一口径
import hsrl_agent_init as hinit   # critic 输出层 bias(价值基线)
import hsrl_soctarget as hst      # 下层 r2/c2 两区间口径 (单一真源)
import hsrl_warmup as hwarm       # warmup 调度 + 规则下层 + transition 填入
# ⚠ aifflor **不启 MATLAB**：物理引擎是 FLORIS（纯 Python，见 aifflor_env.py）。
#   `env.py` 按需惰性 import 旧 MATLAB 路径（`env._get_matlab`），本文件不再触碰它。


RUN_TAG = 'aifflor'


def init_wt_action_bias(agent, lo=0.05, hi=0.45, sig_value=0.0, verbose=True):
    """把 actor 输出层前 5 维 (WT 轴向诱导因子 a) 的 bias 初始化到“中性宽带”。

    为什么必须做 (run160/初始策略诊断):
      `model.py` 的 Actor 只把 fc4.weight 初始化成 ±init_w(0.003),
      fc4.bias 仍是 PyTorch 默认 U(−1/√fan_in)=U(±0.25) (fan_in=hidden1=16);
      而 a 与功率系数的关系是 Cp = 4a(1−a)² (见 WFSim_port.m):
        a=0 → Cp=0 (该机零出力/停机),  a=1/3 → Cp=0.593 (Betz 最大),  a=0.5 → Cp=0.5。
      于是初始策略 ≈ bias: 实测 5 个种子 → 死机 2~4 台、全场 ΣCp 仅 0.47~0.72
      (效率 16%~24%)。策略从"半停机"出发, 而 a<0 区被 CheckAction 裁掉后梯度为 0
      → 若没有恢复力就会长期停在停机态。
    bias∈[lo,hi]=[0.05,0.45] 覆盖"上游降载 → 末台单机最优"的整个区间, **故意不预设**风场最优的形状
    (它随尾流状态(风速/湍流/来流)变化, 是**策略要学的函数**, 不是可预先测定的常数)。
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
    _h = hc.h_settings()   # uniform: 统一 d=15 / legacy: 原值
    d_block_H = _h['d_block'] if _h else 50.0        # 初值: 每块 sum_c1 预算 (先松后紧; 与 td3perrb 一致)
    d_block_H_min = _h['d_block_min'] if _h else 5.0     # 地板: 10·max(0,|δ|−0.1) ≤ 5 → 含死区总 |δ| ≤ 0.6
    # L 层 EMA 对偶 —— 每 5 步块实际成本预算 d_block_L (与 sum_c2 同口径)
    #   sum_c2 = +1·trackerr 的块内累计 [MW] (参考集合见 myfunc / hsrl_soctarget)
    #   * 尽力追的 L: sum_c2 ≡ 0;  完全不动而 |δ|≥A: 每步 ~50, 每块 ~250;
    #     最坏 (反向满功率): 每步 ≤ 340。
    #   * 于是 d_block_L = 20 现读作"允许每块浪费 ≈0.02 MWh 的可达量 (≈0.4 步能力)",
    #     地板 2.5 为"≈0.05 步能力" —— 是一个**可满足**的约束, lam2 不再必然贴顶。
    d_block_L = hst.L_D_BLOCK        # 初值: 每块 sum_c2 预算 (单一真源)
    d_block_L_min = hst.L_D_BLOCK_MIN     # curriculum 收紧地板
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
    while step < num_iterations_H:
        # start of episode 重置环境
        prYellow('########### step:{} ###########'.format(step))
        if observation is None:
            observation = deepcopy(env.reset())  # [Pr,U,E,P_last]
            # episode 起始 SOC: **正式口径 = 固定 3.0 MWh**
            #   (旧口径 = 每 episode 从 U[0.5,5.5] 随机 ⇒ 传 --soc_reset_low 0.5 --soc_reset_high 5.5 复现)
            reset_E = float(np.random.uniform(float(args.soc_reset_low),
                                              float(args.soc_reset_high)))
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
                # 传电价 ⇒ 第 6 维(SOC 目标信号)按价格游走(300 上调/580 不动/980 下调);
                # 下层用规则组件**完全追踪**该目标(Δ≡0 ⇒ trackerr≡0)。
                # --warmup_sig fixed 可退回"信号恒 0"的旧口径。
                _wa = warmup_sched.action(float(observation[0]))   # None ⇒ 对照臂 (--warmup_sched random)
                applied_action_H = original_action_H = (
                    agent_H.random_action() if _wa is None else _wa)
                penalty_NN_H = 0
            else:
                original_action_H = agent_H.select_action(norm_observation)
                applied_action_H, penalty_NN_H = CheckAction(original_action_H)
            # 越界乘性系数 (每块一个 H 动作 ⇒ 块内恒定; 系数<=0 时恒 1.0 ⇒ 旧口径逐位不变)
            viol_derate = violation_derate(original_action_H, args.viol_derate)
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
        # 下层 L 观测 (5 维, 规则原配料): [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]
        # ================================================================
        norm_observation_L = build_L_obs_distilled(
            observation, pw_tru, applied_action_H, rbb_info_obs,
            5 - (episode_steps % 5))
        prPurple('--L-- obs = {}.'.format(round_list(norm_observation_L, 3)))
        # 下层每一个 5-step 块都是一个 episode
        if episode_steps % 5 == 0:
            agent_L.reset(norm_observation_L)

        # ================================================================
        # 下层 L: RL BESS (td3per 自己的操作, 每步决策)
        # ================================================================
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
            if getattr(warmup_rule, 'is_price', False):
                # 新口径 (--warmup_bess price): 下层价格条件随机充放
                #   300 低谷随机充电 +U(0.3,0.5) / 580 平峰不动 /
                #   980 高峰随机放电 −U(0.3,0.5)，再夹进 RBB 可行窗口
                #   (L 层其余逻辑不变: 残余动作 Δ = cmd − 规则基线)
                pb_cmd_L = float(warmup_rule.act(float(observation[0]),
                                                 rbb_info_obs['pbrulemin'],
                                                 rbb_info_obs['pbrulemax']))
                delta_L = np.array([pb_cmd_L - float(pb_rule)])
            elif warmup_rule is not None:
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
        # RBB: 规则可行窗口 + 硬裁剪 (安全性兜底, 不变)
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
            turbine_derate=viol_derate,
        )
        P_F = reward_details['P_F']

        sum_r1 += r1
        sum_c1 += c1_step

        # ---- 块末结算: SOC 目标可行性罚 (上层豁免 tol = --soc_tol, 默认 0.1) ----
        gap_raw = abs(soc_target - float(observation2[2]))
        gap_pen = 0.0
        if episode_steps % 5 == 4:
            gap_pen = gap_penalty(soc_target, float(observation2[2]), tol=soc_tol_upper)
            if hc.mode() == 'uniform':
                sum_r1 -= gap_pen   # uniform: 追踪偏好进 r (固定价 W_GAP=10 元/MWh)
            else:
                sum_c1 += gap_pen
            sum_gap_pen += gap_pen

        # ================================================================
        # 下层 L 奖励/成本 (SOC 目标追踪口径)
        #   r2 = −1·min(clip_amount, 20)                 (命令与实际之差 (两段 clip 合计))
        #   c2 = +1·trackerr [MW] = 实际施加 pb 到参考集合(=Soctrack范围∩平抑窗口)的距离
        #        SOC_best = SOC + g·clip(δ/g, 窗口) —— 规则 SocTracker 的动作;
        #        只罚"本步能做到却没做"的部分 (块首够不着的跳变不计费)
        # ================================================================
        clip_amount = rbb_info['clip_amount']
        is_infeasible = rbb_info['is_infeasible']
        inevitable_violation = rbb_info['inevitable_violation']

                # 不可行窗口下 RBB 会忽略命令、改施加物理边界 ⇒ 该步 r2 可打到封顶 (口径保留; 见 hst.lower_reward)
        r2 = lower_reward(clip_amount)
        c2, c2_info = lower_cost_soc_track(
            soc_target, E_current, float(observation2[2]),
            rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'],
            5 - (episode_steps % 5),
            pbrule=float(rbb_info['pbrule']),   # 放电退化罚 (L_DEG_COEF)
            return_info=True)
        c2_slack = c2_info['slack']   # 新口径: trackerr 的 MWh 等价 (= trackerr·g, 因 ΔSOC = g·pb)
        sum_c2 += c2

        # 下一状态 L 观测: 用下一状态重算可行域 (pb 无关; soc_target 块内固定)
        _, rbb_info_obs2 = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=float(pw_tru2),
            pg_prev=float(observation2[-1]), E=float(observation2[2]))
        norm_observation2_L = build_L_obs_distilled(
            observation2, pw_tru2, applied_action_H, rbb_info_obs2,
            5 - ((episode_steps + 1) % 5))

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
            # ---- critic 输出层 bias(价值基线): warmup 后首个块末一次性应用 ----
            bias_init.on_block_end(step, sum_r1, sum_c1, agent_H)

            agent_H.observe_jump(norm_observation_H, original_action_H,
                                 sum_r1, sum_c1, norm_observation2, done)
            # H 层 EMA 对偶上升 —— 用块实际成本 sum_c1 更新 lam1 (与 d_block_H 同口径)
            if step > args.warmup:
                agent_H.update_ema_cost(sum_c1)
                hc.dual_update(agent_H, d_block_H)   # uniform=乘性 / legacy=加性
            # L 层 EMA 对偶上升 —— 用块实际成本 sum_c2 更新 lam2 (与 d_block_L 同口径)
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
                # H 层更新频率(UTD): 每块 args.utd_h 次 (默认 4)
                for _ in range(args.utd_h):
                    agent_H.update_policy()

                # H 层 curriculum 计数: **每块 1 次**(与 UTD 解耦)。
                #   EMA 成本每块只在块末更新一次 ⇒ 块内重复检查结论相同, 只是把计数多推
                #   几次。故显式改成每块一次, 阈值改用块数 hc.H_CURRICULUM_BLOCKS(=50,
                #   等价于历史的"2 次/块 × 阈值 100 = 50 块"), 调 --utd_h 不再漂移节奏。
                if agent_H.ema_cost is not None and agent_H.ema_cost <= d_block_H:
                    consecutive_satisfied_H += 1
                else:
                    consecutive_satisfied_H = 0

                # L 层更新频率(UTD): 每块 args.utd_l 次 (默认 10 = td3per 口径, L 只有 1 维动作)
                for _ in range(args.utd_l):
                    agent_L.update_policy()

                if consecutive_satisfied_H >= hc.H_CURRICULUM_BLOCKS:
                    d_block_H = max(d_block_H / 2, d_block_H_min)  # H: 收紧块预算
                    consecutive_satisfied_H = 0
                    prGreen('H: d_block_H halved to {}'.format(d_block_H))
                # (L 的 curriculum 计数与减半已移到块末: 每块 1 次, 阈值 hst.L_CURRICULUM_BLOCKS)

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
                        df_update_written = flush_csv_append_batch(
                            '{}/update_stats_H.csv'.format(output),
                            df_update_rows, df_update_written)
                        df_update_rows = []

            # Test: 9-case segmented evaluation every 480 episode steps
            # (fixed scenario seed 42 -> same wind trajectory each test)
            if episode_steps % 480 == 0:
                case_metrics, overall = nine_case_test(
                    agent_H, agent_L, env_test, rbb_system, seed=42,
                    phase_mode=phase_mode, phase_table=phase_table,
                    viol_derate_coef=args.viol_derate)
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
                            # 存货结算 (口径修正, 不改训练路径): profit 只记现金流, 不含储存电量变动;
                            # 按外生账面价 SOC_SETTLE_PR (= hc.PR_REF = 580) 结算 ΔSOC。
                            'soc_start': m['soc_start'], 'soc_end': m['soc_end'],
                            'dsoc': m['dsoc'], 'profit_econ': m['profit_econ'],
                            'soc_track_err': m['soc_track_err'],
                            # r1 分解三项 (新增列, 正交于既有口径; profit/profit_econ 不变):
                            #   profit/5 - r1 == (penalty_NN_H + derate_loss)/5
                            'turbine_profit': m['turbine_profit'],
                            'derate_loss': m['derate_loss'],
                            'penalty_NN_H': m['penalty_NN_H'],
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
                        'soc_start': overall['soc_start'], 'soc_end': overall['soc_end'],
                        'dsoc': overall['dsoc'], 'profit_econ': overall['profit_econ'],
                        'soc_track_err': overall['soc_track_err'],
                        # r1 分解三项 (45 步总量): profit/45 - r1 == (penalty_NN_H + derate_loss)/45
                        'turbine_profit': overall['turbine_profit'],
                        'derate_loss': overall['derate_loss'],
                        'penalty_NN_H': overall['penalty_NN_H'],
                    })
                    df_test_written = flush_csv_append_batch(
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
                    writer.add_scalar('test9/overall/profit_econ', overall['profit_econ'], step)
                    writer.add_scalar('test9/overall/dsoc', overall['dsoc'], step)
                    writer.add_scalar('test9/overall/soc_end', overall['soc_end'], step)
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
                'original_action_L': delta_L,              # = Δ (actor 输出; 残余动作)
                'delta_L': float(delta_L[0]),
                'pb_rule': float(pb_rule),
                'pb_cmd': float(pb_cmd_L),
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
                df_written = flush_csv_append_batch(
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
                writer.add_scalar('action/pb_rule', float(pb_rule), step)
                writer.add_scalar('action/delta_L', float(delta_L[0]), step)
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
                df_episode_written = flush_csv_append_batch(
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
        flush_csv_append_batch('{}/df_{}.csv'.format(output, RUN_TAG), df_rows, df_written)
        flush_csv_append_batch('{}/df_episode_{}.csv'.format(output, RUN_TAG),
                   df_episode_rows, df_episode_written)
        flush_csv_append_batch('{}/df_test9.csv'.format(output), df_test_rows, df_test_written)
        flush_csv_append_batch('{}/update_stats_H.csv'.format(output),
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


# =============================================================================
# 诊断开关（默认全关 ⇒ 生产路径逐位不变）
#   S1 固定风速+固定电价 / S2 变风速+固定电价 / S3 变风速+变电价 / E2 一维统一降载
# =============================================================================
DIAG = {'u': -1.0, 'pr': -1.0, 'scalar': 0, 'freeze_g': 0, 'lam_fixed': -1.0,
        'pen_h': -1.0, 'center': 0}


def _diag_fix_action(a):
    """把 actor 的 6 维原始动作改写成受约束的形式（诊断用）。

    * `scalar=1`：只保留第 1 位，`s = sigmoid(a[0]) ∈ (0,1)`，前 5 维统一取
      `0.33·s` ⇒ 1 维"统一降载"参数化（实验 2）；`s` 可由 `a[0]/0.33` 还原。
    * `freeze_g=1`：第 6 维（块总放电量 g）恒 0 ⇒ BESS 不参与（实验 1）。
    """
    a = np.asarray(a, dtype=float).reshape(-1).copy()
    if DIAG['scalar']:
        s = 1.0 / (1.0 + np.exp(-float(np.clip(a[0], -30.0, 30.0))))
        a[:5] = 0.33 * s
    if DIAG['freeze_g']:
        a[5] = 0.0
    return a


# =============================================================================
# HSRL 分支的训练主循环（--hsrl；口径总表见 aifflor/aifflor_hsrl.py）
# =============================================================================
def train_hsrl(num_iterations_H,
               agent_H,
               agent_L,
               env,
               env_test,
               rbb_system,
               wind_pool,
               output,
               max_episode_length=None,
               debug=False,
               upload=False,
               use_csv=False):
    """`--hsrl` 分支的训练主循环。

    与 `train()` **并列而不共用**：HSRL 的每一条口径（块总放电量语义 / 无 bessdeg 的 H 奖励 /
    绝对 pb 的 L 动作 / 指数式 c_L / 重标后的 λ2 / 摊还执行器 warmup / 20 步测试）都在这里
    自成一体 ⇒ **未加 `--hsrl` 时旧 `train()` 一行都不被触碰**（"逐位不变"可断言）。

    单位约定（全函数一致）：
      * `pb`   ：BESS 功率 [MW]，**正 = 充电**（与 env / RBB 同号）；
      * `a_L`  ：BESS **放电量** [MW·min]，**正 = 放电**（= −pb，公式里的 a_L^t）；
      * `g`    ：H 下发的**块总放电量** [MW·min]，**正 = 放电**；
      * 步长 = 1 min ⇒ 每步的"MW"与"MW·min"数值相等。
    """
    agent_H.is_training = True
    agent_L.is_training = True
    step = episode = episode_steps = 0
    episode_reward = episode_cost = episode_cost2 = episode_profit = 0.
    episode_bess_profit = 0.
    episode_deg_discharge = 0.
    episode_deg_charge = 0.
    observation = None
    time = hsrl.DT

    # ---- H 层 EMA 对偶（口径未变: sum_c1 = 波动成本 + W_GAP·unfeasible_gap(≡0)）----
    _h = hc.h_settings()
    d_block_H = _h['d_block'] if _h else 50.0
    # curriculum 地板：**HSRL 分支用 0（用户第六轮）** ⇒ 允许一直减半、由可达性自行停止
    #   （此前沿用 `_h['d_block_min'] = D_BLOCK = 初值` ⇒ 减半是空操作、curriculum 被设计性禁用）。
    #   ⚠ 非 HSRL 的旧 train() 仍用 `_h['d_block_min']`，逐位不变。
    d_block_H_min = float(getattr(args, 'hsrl_d_block_h_min', hsrl.HSRL_D_BLOCK_H_MIN))
    # ---- L 层 EMA 对偶（HSRL 重标: c_L 无量纲 ⇒ λ2 ∈ [2,50], d_block_L = 0.1）----
    d_block_L = float(getattr(args, 'hsrl_d_block_l', hsrl.HSRL_D_BLOCK_L))
    d_block_L_min = float(getattr(args, 'hsrl_d_block_l_min', hsrl.HSRL_D_BLOCK_L_MIN))
    consecutive_satisfied_H = 0
    consecutive_satisfied_L = 0

    # ---- 越界/退化系数（命令行可覆盖；<=0 的覆盖值不生效，一律取模块常量）----
    pen6_coef = float(getattr(args, 'hsrl_pen6_coef', hsrl.HSRL_PEN6_COEF) or hsrl.HSRL_PEN6_COEF)
    viol_coef = float(getattr(args, 'hsrl_viol_coef', hsrl.HSRL_VIOL_COEF) or hsrl.HSRL_VIOL_COEF)
    deg_coef = float(getattr(args, 'hsrl_deg_coef', hsrl.HSRL_DEG_COEF) or hsrl.HSRL_DEG_COEF)
    # ---- 对偶: 分子来源（q=critic cost 的 Q / ema=实际块成本 EMA）与 twin 聚合（=twin 均值）----
    dual_src = str(getattr(args, 'hsrl_dual_src', hsrl.HSRL_DUAL_SRC))
    dual_q_agg = str(getattr(args, 'hsrl_dual_q_agg', hsrl.HSRL_DUAL_Q_AGG))
    # `src=q` 时用哪种 Q 估计: recent_batch=近端 N 条状态的批均值 × 当前策略动作(默认, S1)
    #                            point=旧的单点 Q(s0,a0)（风依赖强，A/B 对照）
    dual_q_src = str(getattr(args, 'hsrl_dual_q_src', hsrl.HSRL_DUAL_Q_SRC))
    dual_recent_n = int(getattr(args, 'hsrl_dual_recent_n', hsrl.HSRL_DUAL_RECENT_N))
    # 块首 L 观测（Q 版对偶在块末要用它评估"当前策略的期望块成本"）
    norm_observation_L0 = None
    action_L0 = None
    # 对偶诊断（块末更新，块内保持 —— 与 g 同惯例）
    q_rate_H = float('nan')
    q_rate_L = float('nan')
    q_info_H = {'q_agg': float('nan')}
    q_info_L = {'q_agg': float('nan')}
    # 近端批（S1）诊断：块末更新，块内保持
    b_info_H = {'q_agg': float('nan'), 'n_batch': 0, 'cost_recent_mean': float('nan')}
    b_info_L = {'q_agg': float('nan'), 'n_batch': 0, 'cost_recent_mean': float('nan')}
    # update 批均值（update_batch 分子）诊断：块末更新，块内保持
    ud_info_H = {'q_agg': float('nan'), 'n_update': 0}

    if use_csv:
        df_rows = []
        df_written = 0
        df_episode_rows = []
        df_episode_written = 0
        df_test_rows = []
        df_test_written = 0
        df_update_rows = []
        df_update_written = 0

    # ---- 1440-step window counters for env metrics ----
    window_violation_count = 0
    window_power_sum = 0.0

    # ---- 24h 相位特征 (--phase_feat) ----
    phase_mode = int(getattr(args, 'phase_feat', 0))
    phase_table = phase_daily_table() if phase_mode >= 2 else None
    phase_sched = day_price_schedule()
    phase_mismatch = 0
    phase_checked = 0

    # ---- warmup: 上层调度 + 下层"摊还执行器"自检 ----
    warmup_sched, warmup_rule, bess_fill_stats = hwarm.build_from_args(args, agent_H, agent_L)
    _wb = str(getattr(args, 'warmup_bess', 'rule'))
    # HSRL 下 `--warmup_bess rule`（默认）⇒ **摊还执行器**（不是 SOC 目标追踪）:
    #   命令 = clip(−(g−Σ)/n_rem, 物理域) ⇒ c_L ≡ 0、r2 = −退化项 —— 与策略期的零成本动作同源。
    #   因此旧的 `check_residual_baseline`（断言"与 SOC 目标规则逐位相同"）在 HSRL 下**不适用**，
    #   改跑 `hsrl.check_amortized_baseline`（断言"c_L≡0 + 命令恒在物理域内 + SOC 不越界"）。
    _chk = hsrl.check_amortized_baseline()
    prCyan('[HSRL] warmup 下层口径 = {} ({});  摊还执行器自检: {} 例, 其中被物理域裁剪 {} 例 '
           '(裁剪时 c_L>0 属"做不到", 未裁剪时 c_L≡0), max c_L = {:.3e}'.format(
               _wb, '摊还执行器' if _wb == 'rule' else ('价格条件随机' if _wb == 'price' else '绝对随机'),
               _chk['n_checked'], _chk['n_clipped'], _chk['max_c_unclipped']))
    assert _chk['fail'] == 0, 'HSRL 摊还执行器自检未通过: {}'.format(_chk)
    bias_init = hinit.BiasInitTracker.from_args(args)
    bias_init.value_scale = float(getattr(agent_H, 'value_scale', hc.value_scale()))
    bias_init.discount = float(agent_H.discount)

    # ---- 20 步测试的风速池（只含"天"列；FFC_speed_noise_1.csv 第 0 列是行号）----
    wind_pool = None
    try:
        wind_pool = np.asarray(env_test.wind_noise.iloc[:, 1:].to_numpy(dtype=float))
    except Exception as _e:                                   # pragma: no cover
        prYellow('[HSRL test20] 风速池不可用 ({}); 本run 跳过 20 步测试'.format(_e))
    test_seed = int(getattr(args, 'hsrl_test_seed', -1))
    test_trigger = 0

    while step < num_iterations_H:
        prYellow('########### step:{} ###########'.format(step))
        if observation is None:
            observation = deepcopy(env.reset())          # [Pr,U,E,P_G_last]
            reset_E = float(np.random.uniform(float(args.soc_reset_low),
                                              float(args.soc_reset_high)))
            env.E = reset_E
            observation[2] = reset_E
            norm_observation = Normalize_H(observation, phase_minute_of_day(0),
                                           phase_mode, phase_table)
            agent_H.reset(norm_observation)

        pm_now = phase_minute_of_day(episode_steps)
        if phase_mode > 0:
            if abs(float(observation[0]) - phase_sched[pm_now]) > 1e-6:
                phase_mismatch += 1
            phase_checked += 1

        # ================================================================
        # 块首: H 决策 —— 第 6 维 = 本块总放电量 g [MW·min]（放电为正）
        # ================================================================
        if episode_steps % 5 == 0:
            sum_r1 = 0.0
            sum_c1 = 0.0
            sum_c2 = 0.0
            sum_disch = 0.0
            E_block_start = float(observation[2])
            feas_c_max, feas_d_max = hsrl.feasible_total(E_block_start)
            if step <= args.warmup:
                _wa = warmup_sched.action(float(observation[0]))   # None ⇒ 对照臂
                if _wa is None:
                    original_action_H = np.asarray(agent_H.random_action(), dtype=float).copy()
                    # 第 6 维按 HSRL 量程重抽（U(−2.5, 2.5) MW·min）
                    original_action_H[5] = hsrl.random_g()
                else:
                    original_action_H = np.asarray(_wa, dtype=float).copy()
                    # 价格游走 sig → g: **反号 + ×5**（低谷充电 ⇒ g<0；见 hsrl 常量区）
                    original_action_H[5] = hsrl.g_from_warmup_signal(original_action_H[5])
                original_action_H = _diag_fix_action(original_action_H)
                applied_action_H, _pen5_w, _pen6_w = hsrl.check_action_H_hsrl(
                    original_action_H, E_block_start,
                    pen_coef=(DIAG['pen_h'] if DIAG['pen_h'] > 0 else None),
                    pen6_coef=pen6_coef)
                penalty_NN_H = 0.0          # warmup 期不计越界罚（与旧口径一致）
                pen6_H = 0.0
            else:
                original_action_H = agent_H.select_action(norm_observation)
                original_action_H = _diag_fix_action(original_action_H)
                applied_action_H, penalty_NN_H, pen6_H = hsrl.check_action_H_hsrl(
                    original_action_H, E_block_start,
                    pen_coef=(DIAG['pen_h'] if DIAG['pen_h'] > 0 else None),
                    pen6_coef=pen6_coef)
            g_raw = float(original_action_H[5])
            g_applied = float(applied_action_H[5])
            viol_derate = violation_derate(original_action_H, args.viol_derate)
            norm_observation_H = norm_observation
            pw_tru = env.PowerOutput(observation[1], applied_action_H)
            pb_l = list()

        prGreen('--H-- ori a_t = {} (g_raw={:.3f} MW·min)'.format(
            np.round(np.asarray(original_action_H, dtype=float), 3), g_raw))

        ptur = float(pw_tru)
        pg_prev = float(observation[-1])
        E_current = float(observation[2])
        pb_lo, pb_hi = hsrl.feasible_power(E_current)     # 每步物理域 [MW, 正=充电]

        # ================================================================
        # 下层 L 观测 (6 维): [u/10, pr/1000, SOC/0.5, P_G(t−1)/5, g/2.5, need/2.5]
        #   第 6 维 = c_L 的参照点 need = (g − Σ已放电)/(δ − t_L)（**本步开始前**的 Σ/n_rem）
        # ================================================================
        _n_rem_L = 5 - (episode_steps % 5)
        need_L = hsrl.plan_need(g_applied, sum_disch, _n_rem_L)
        norm_observation_L = hsrl.build_L_obs_hsrl(
            observation[1], observation[0], E_current, pg_prev, g_applied, need_L)
        prPurple('--L-- obs = {} (need={:.4f})'.format(
            np.round(norm_observation_L, 3), need_L))
        if episode_steps % 5 == 0:            # 下层每个 5-step 块 = 一个 episode
            agent_L.reset(norm_observation_L)
            # 块首观测 + 块首**实际执行**的动作（Q 版对偶要在块首、用实际动作评估 L 的
            # cost：块末观测已近 episode 终止，Q_c 会退化成"最后一步的成本"）
            norm_observation_L0 = np.array(norm_observation_L, dtype=float, copy=True)
            action_L0 = None

        # ================================================================
        # 下层 L 动作: **绝对 pb**，只按物理域 (SOC + 额定功率) 裁剪
        # ================================================================
        if step <= args.warmup:
            if _wb == 'price' and getattr(warmup_rule, 'is_price', False):
                pb_cmd_L = float(warmup_rule.act(float(observation[0]), pb_lo, pb_hi))
            elif _wb == 'random' or warmup_rule is None:
                pb_cmd_L = float(np.asarray(agent_L.random_action(),
                                            dtype=float).reshape(-1)[0])
            else:   # rule (默认) ⇒ 摊还执行器
                pb_cmd_L = hsrl.amortized_rule_action(
                    g_applied, sum_disch, _n_rem_L, E_current)
            original_action_L = np.array([pb_cmd_L], dtype=np.float64)
            viol_L = 0.0
            penalty_NN_L = 0.0
        else:
            original_action_L = np.asarray(agent_L.select_action(norm_observation_L),
                                           dtype=float).reshape(-1)
            pb_cmd_L = float(original_action_L[0])
            viol_L = max(0.0, pb_cmd_L - pb_hi) + max(0.0, pb_lo - pb_cmd_L)
            penalty_NN_L = viol_coef * viol_L
        prGreen('--L-- cmd = {:.4f} MW | 物理域 [{:.4f}, {:.4f}] | 越界 {:.4f} MW | '
                'n_rem = {}'.format(pb_cmd_L, pb_lo, pb_hi, viol_L, _n_rem_L))
        if episode_steps % 5 == 0:
            # 块首实际执行的动作（Q 版对偶的评估点；见 aifflor_hsrl.q_cost_rate 的说明）
            action_L0 = np.array(original_action_L, dtype=float, copy=True)

        action = np.concatenate((applied_action_H, np.array([pb_cmd_L])))
        # RBB 在 HSRL 下以 fluctuation_limit=inf 构造 ⇒ 退化为"纯物理裁剪 + 记录":
        #   pbrule ≡ clip(命令, 物理域), clip_amount ≡ viol_L, unfeasible_gap ≡ 0。
        pb_original = float(pb_cmd_L)
        pbrule, rbb_info = rbb_system.compute_pbrule(
            pb_original=pb_original, ptur=ptur, pg_prev=pg_prev, E=E_current)

        action_rbb = action.copy()
        action_rbb[-1] = pbrule
        pw_tru2, observation2, action_L_env, penalty_L, done, done_L, info = \
            env.step(action_rbb[:6], action_rbb[6:], pw_tru)
        action_rbb[6:] = action_L_env
        pb_applied = float(np.asarray(action_L_env, dtype=float).reshape(-1)[0])
        a_L_disch = hsrl.discharge_of(pb_applied)      # 放电为正 [MW·min]
        prGreen('Applied action = {}.'.format(np.round(np.asarray(action_rbb, dtype=float), 3)))
        norm_observation2 = Normalize_H(observation2,
                                        phase_minute_of_day(episode_steps + 1),
                                        phase_mode, phase_table)

        # ================================================================
        # H 奖励: r1 = pr·P_G·dt − pen5 − pen6（**无 bessdeg**；**无块末 gap 罚**）
        # ================================================================
        r1, c1_step, reward_details = hsrl.cal_rew_rb_hsrl(
            observation, observation2, action_rbb, rbb_info,
            penalty_NN=penalty_NN_H + pen6_H, time=time, ptur=ptur,
            pb_cmd=pb_original, turbine_derate=viol_derate)
        # 诊断：奖励中心化（以"标称动作的出力"为基线）⇒ 奖励只剩"比标称好多少"，
        #   动作效应不再被状态/绝对收益淹没（S1 实测 ∂Q_p/∂a 只有真值的 1e-4 量级）。
        if DIAG['center']:
            _pnom = float(env.PowerOutput(observation[1], env.nominal()))
            r1 = r1 - float(observation[0]) * (float(ptur) - _pnom) * time
        P_F = reward_details['P_F']
        sum_r1 += r1
        sum_c1 += c1_step
        gap_pen = 0.0        # HSRL: 块末 SOC 目标 gap 罚已删除（执行偏差由 c_L 承接）
        gap_raw = 0.0

        # ================================================================
        # L 奖励/成本: r2 = −200·越界 − 20·放电 ;  c_L = 1 − exp(−|need − a_L|)
        #   need = (g − Σ已放电)/(δ − t_L)，Σ 与 a_L 都用**实际施加值**（env 回传）
        # ================================================================
        c2, c2_info = hsrl.lower_cost_hsrl(g_applied, sum_disch, _n_rem_L, a_L_disch,
                                           return_info=True)
        r2 = hsrl.lower_reward_hsrl(viol_L, a_L_disch,
                                    viol_coef=viol_coef, deg_coef=deg_coef)
        deg_step = deg_coef * max(0.0, a_L_disch)
        sum_c2 += c2
        sum_disch += a_L_disch

        # 下一状态 L 观测: 第 6 维要用**更新后**的 Σ 与 n_rem 重算（need' = (g−Σ−a_L)/(n_rem−1)）
        need_L_next = hsrl.plan_need(g_applied, sum_disch, max(1, _n_rem_L - 1))
        norm_observation2_L = hsrl.build_L_obs_hsrl(
            observation2[1], observation2[0], float(observation2[2]),
            float(observation2[-1]), g_applied, need_L_next)

        prGreen('r1 = {} (turb={}, bess={}, bessdeg 已移除) | pen5 = {}, pen6 = {:.3f}'.format(
            round(r1, 3), round(reward_details['turbine_profit'], 3),
            round(reward_details['bess_profit'], 3),
            round(penalty_NN_H, 3), round(pen6_H, 3)))
        prGreen('c1 = {} (fluct={}, unfeasible_gap={})'.format(
            round(c1_step, 3), round(reward_details['c1_fluct'], 3),
            round(reward_details['unfeasible_gap'], 4)))
        prGreen('g = {:.4f} MW·min (raw {:.4f}) | Σ已放电 = {:.4f} | n_rem = {} | '
                'need = {:.4f} | a_L = {:.4f} | dev = {:+.4f}'.format(
                    g_applied, g_raw, c2_info['sum_disch_before'], _n_rem_L,
                    c2_info['need'], c2_info['a_L'], c2_info['dev']))
        prGreen('r2 = {} (越界 {:.4f} MW, 退化 {:.3f}) | c_L = {:.6f} | SOC = {:.4f}'.format(
            round(r2, 3), viol_L, round(deg_step, 3), c2, reward_details['SOC']))

        pb_l.append(pb_original)

        if P_F > 3.0 + 1e-4:
            window_violation_count += 1
        window_power_sum += observation2[-1]

        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        # ---- 块末: 两层各自结算（对偶已移到本块 update **之后**，见下方）----
        if episode_steps % 5 == 4:
            bias_init.on_block_end(step, sum_r1, sum_c1, agent_H)
            agent_H.observe_jump(norm_observation_H, original_action_H,
                                 sum_r1, sum_c1, norm_observation2, done)
            done_L = True
        hwarm.fill_bess_transition(
            agent_L, norm_observation_L, original_action_L, r2, c2,
            norm_observation2_L, done_L,
            applied_pb=float(pbrule), warmup=bool(step <= args.warmup),
            cmd_pb=float(pb_cmd_L), rule_pb=None, stats=bess_fill_stats)

        if step > args.warmup:
            if episode_steps % 5 == 4:
                for _ in range(args.utd_h):
                    agent_H.update_policy()
                if agent_H.ema_cost is not None and agent_H.ema_cost <= d_block_H:
                    consecutive_satisfied_H += 1
                else:
                    consecutive_satisfied_H = 0
                for _ in range(args.utd_l):
                    agent_L.update_policy()
                if consecutive_satisfied_H >= hc.H_CURRICULUM_BLOCKS:
                    d_block_H = max(d_block_H / 2.0, d_block_H_min)
                    consecutive_satisfied_H = 0
                    prGreen('H: d_block_H halved to {}'.format(d_block_H))

                _block_stats = None
                if agent_H.record_update_stats and len(agent_H.update_stats) > 0:
                    _stats = agent_H.update_stats
                    _mean = {k: float(np.mean([s[k] for s in _stats])) for k in _stats[0]}
                    agent_H.update_stats = []
                    _block_stats = _stats          # 供对偶的 `update_batch` 分子使用
                    if upload:
                        for _k, _v in _mean.items():
                            writer.add_scalar('update/' + _k, _v, step)
                    if use_csv:
                        df_update_rows.append({'step': step, **_mean})
                        df_update_written = flush_csv_append_batch(
                            '{}/update_stats_H.csv'.format(output),
                            df_update_rows, df_update_written)
                        df_update_rows = []

                # ================================================================
                # 对偶上升（块末，**放在本块 update 之后**）
                #   * **λ1 的分子默认 = 本块 update 已算好的 `Qc` 批均值**（`q_c_mean`/`q_c2_mean`，
                #     即 PER 批在行为动作下的 cost Q 均值）—— 用户口径：靠**缓冲的自然淘汰**
                #     （环形容量 rmsize）让这个均值随策略改进而下降，从而驱动 λ1 下降/上升；
                #     它不含"当日风况"这一项（跨批平均）⇒ 直接消掉风况引起的慢漂。
                #   * λ2 的分子 = 近端批均值 × 当前策略动作（S1；`update_batch` 要给 L 开
                #     `record_update_stats`，本轮按用户口径**只改 lam1**）。
                #   * EMA 仍每块维护（日志 / curriculum / 回退）。
                # ================================================================
                if norm_observation_L0 is None:
                    norm_observation_L0 = np.array(norm_observation_L, dtype=float, copy=True)
                if action_L0 is None:
                    action_L0 = np.array(original_action_L, dtype=float, copy=True)
                # 诊断（单点 Q 在 (s₀, 实际执行动作)；成本可忽略）
                _, q_info_H = hsrl.q_cost_rate(
                    agent_H, norm_observation_H, a_t=original_action_H, horizon='infinite',
                    agg=dual_q_agg, return_info=True)
                _, q_info_L = hsrl.q_cost_rate(
                    agent_L, norm_observation_L0, a_t=action_L0, horizon='finite',
                    agg=dual_q_agg, return_info=True)
                b_info_H = {'q_agg': float('nan'), 'n_batch': 0, 'cost_recent_mean': float('nan')}
                b_info_L = {'q_agg': float('nan'), 'n_batch': 0, 'cost_recent_mean': float('nan')}
                ud_info_H = {'q_agg': float('nan'), 'n_update': 0}
                # ---- λ1 的分子 ----
                if dual_src == 'ema':
                    q_rate_H = float(agent_H.ema_cost) if agent_H.ema_cost is not None else None
                elif dual_q_src == 'update_batch':
                    q_rate_H, ud_info_H = hsrl.update_batch_cost_rate(
                        agent_H, stats=_block_stats, horizon='infinite', agg=dual_q_agg,
                        return_info=True)
                elif dual_q_src == 'recent_batch':
                    q_rate_H, b_info_H = hsrl.policy_batch_cost_rate(
                        agent_H, n=dual_recent_n, horizon='infinite', agg=dual_q_agg,
                        return_info=True)
                else:                                   # point
                    q_rate_H, _ = hsrl.q_cost_rate(
                        agent_H, norm_observation_H, a_t=original_action_H,
                        horizon='infinite', agg=dual_q_agg, return_info=True)
                # ---- λ2 的分子（本轮固定 S1；`--hsrl_dual_src ema` 时走 EMA）----
                if dual_src == 'ema':
                    q_rate_L = float(agent_L.ema_cost) if agent_L.ema_cost is not None else None
                else:
                    q_rate_L, b_info_L = hsrl.policy_batch_cost_rate(
                        agent_L, n=dual_recent_n, horizon='finite', agg=dual_q_agg,
                        return_info=True)
                agent_H.update_ema_cost(sum_c1)
                hsrl.dual_update_H_hsrl(agent_H, d_block_H, q_rate=q_rate_H,
                                        src=dual_src, ema=agent_H.ema_cost)
                if DIAG['lam_fixed'] >= 0:
                    agent_H.lam = float(DIAG['lam_fixed'])
                agent_L.update_ema_cost(sum_c2)
                hsrl.dual_update_L_hsrl(agent_L, d_block_L, q_rate=q_rate_L,
                                        src=dual_src, ema=agent_L.ema_cost)
                _rh = float('nan') if q_rate_H is None else q_rate_H
                _rl = float('nan') if q_rate_L is None else q_rate_L
                prCyan('dual[{}/{}]: H rate={:.4f} (ratio={:.3f}, lam1={:.5f}, n_upd={}) | '
                       'L rate={:.4f} (ratio={:.3f}, lam2={:.3f}, n_recent={})'.format(
                           dual_src, dual_q_src, _rh, _rh / max(d_block_H, 1e-9),
                           agent_H.lam, ud_info_H.get('n_update', 0), _rl,
                           _rl / max(d_block_L, 1e-9), agent_L.lam,
                           b_info_L.get('n_batch', 0)))
                q_rate_H = _rh
                q_rate_L = _rl
                # L 层 curriculum: 连续满足块数 ⇒ 预算减半（阈值 = EMA 时间常数）
                if agent_L.ema_cost is not None and agent_L.ema_cost <= d_block_L:
                    consecutive_satisfied_L += 1
                else:
                    consecutive_satisfied_L = 0
                if consecutive_satisfied_L >= hsrl.HSRL_L_CURRICULUM_BLOCKS:
                    d_block_L = max(d_block_L / 2.0, d_block_L_min)
                    consecutive_satisfied_L = 0
                    prGreen('L: d_block_L halved to {}'.format(d_block_L))

            # ---- 20 步测试（触发频率与旧 9-case 相同: episode_steps % 480 == 0）----
            if episode_steps % 480 == 0 and wind_pool is not None:
                per_blk, overall, meta = hsrl.twenty_step_test_hsrl(
                    agent_H, agent_L, env_test, rbb_system, wind_pool,
                    seed=(None if test_seed < 0 else test_seed),
                    phase_mode=phase_mode, phase_table=phase_table,
                    penalty6_coef=pen6_coef,
                    viol_derate_coef=args.viol_derate,
                    wind_col=(-1 if int(args.hsrl_test_wind_col) < 0
                              else int(args.hsrl_test_wind_col)),
                    wind_start=(-1 if int(args.hsrl_test_wind_start) < 0
                                else int(args.hsrl_test_wind_start)),
                    random_wind=(str(args.hsrl_test_wind) == 'random'))
                test_trigger += 1
                prCyan('[HSRL test20] #{} step={} 风列={} 起点={} SOC {:.3f}→{:.3f} | '
                       'r1={:.2f} c1={:.2f} r2={:.2f} c2={:.4f} 越界={:.3f} 退化={:.2f} | '
                       'Σ放电={:.3f} (g 均值 {:.3f})'.format(
                           test_trigger, step, meta['wind_col'], meta['wind_start'],
                           meta['soc_start'], meta['soc_end'], overall['r1'],
                           overall['c1'], overall['r2'], overall['c2'],
                           overall['viol'], overall['deg'],
                           float(np.sum([m['sum_disch'] for m in per_blk])),
                           overall['g']))
                if use_csv:
                    _common = {'step': step, 'trigger': test_trigger,
                               'wind_col': meta['wind_col'],
                               'wind_start': meta['wind_start'], 'price': meta['price']}
                    for _i, m in enumerate(per_blk):
                        df_test_rows.append(dict(
                            _common, block=_i, **{k: m[k] for k in (
                                'r1', 'c1', 'r2', 'c2', 'deg', 'viol', 'pen5', 'pen6',
                                'cvm', 'pf_viol', 'profit', 'profit_econ', 'soc_start',
                                'soc_end', 'dsoc', 'g', 'g_raw', 'sum_disch', 'cL_max',
                                'dev_mean', 'need_mean', 'pb_sum')}))
                    df_test_rows.append(dict(
                        _common, block='OVERALL', **{k: overall[k] for k in (
                            'r1', 'c1', 'r2', 'c2', 'deg', 'viol', 'pen5', 'pen6',
                            'cvm', 'pf_viol', 'profit', 'profit_econ', 'soc_start',
                            'soc_end', 'dsoc', 'g', 'g_raw', 'sum_disch', 'cL_max',
                            'dev_mean', 'need_mean')}))
                    df_test_written = flush_csv_append_batch(
                        '{}/df_test20_hsrl.csv'.format(output),
                        df_test_rows, df_test_written)
                    df_test_rows = []
                if upload:
                    for _k in ('r1', 'c1', 'r2', 'c2', 'deg', 'viol', 'pen6',
                               'profit', 'profit_econ', 'dsoc', 'soc_end', 'g'):
                        writer.add_scalar('test20/overall/' + _k, overall[_k], step)
                    writer.add_scalar('test20/overall/cvm', overall['cvm'], step)
                    writer.add_scalar('test20/overall/pf_viol', overall['pf_viol'], step)
                    for _i, m in enumerate(per_blk):
                        for _k in ('r1', 'c1', 'r2', 'c2', 'deg', 'viol', 'g',
                                   'sum_disch', 'cL_max'):
                            writer.add_scalar('test20/block{}/{}'.format(_i, _k),
                                              m[_k], step)

        # ---- 逐步数据落盘 ----
        if use_csv:
            df_rows.append({
                'step': step,
                'observation': observation,
                'observation2': observation2,
                'phase_minute': pm_now,
                # 上层
                'original_action_H': original_action_H,
                'applied_action_H': applied_action_H,
                'g_raw': g_raw,
                'g': g_applied,
                'feas_c_max': feas_c_max,
                'feas_d_max': feas_d_max,
                'pen5_H': penalty_NN_H,
                'pen6_H': pen6_H,
                # 下层
                'norm_observation_L': norm_observation_L,
                'obs_need': float(norm_observation_L[5]),   # 第 6 维（c_L 参照点的归一化值）
                'original_action_L': original_action_L,
                'pb_cmd': pb_cmd_L,
                'pb_applied': pb_applied,
                'a_L_disch': a_L_disch,
                'pb_lo': pb_lo,
                'pb_hi': pb_hi,
                'viol_L': viol_L,
                'n_rem_L': _n_rem_L,
                'sum_disch_before': c2_info['sum_disch_before'],
                'sum_disch_after': sum_disch,
                'need': c2_info['need'],
                'dev': c2_info['dev'],
                'action_rbb': action_rbb,
                'r1': r1, 'c1': c1_step,
                'r2': r2, 'c2': c2, 'deg_L': deg_step,
                'gap_pen': gap_pen, 'gap_raw': gap_raw,
                'lam1': agent_H.lam, 'lam2': agent_L.lam,
                'lam1_phys': agent_H.lam / max(hc.lam_scale(), 1e-12),
                'dual_src': dual_src,
                'dual_q_src': dual_q_src,
                'q_cost_H': q_info_H['q_agg'], 'rate_c_H': q_rate_H,
                'ratio_c_H': q_rate_H / max(d_block_H, 1e-9),
                'q_batch_H': b_info_H.get('q_agg', float('nan')),
                'rate_batch_H': (b_info_H.get('rate', float('nan')) if dual_src == 'q'
                                 and dual_q_src == 'recent_batch' else float('nan')),
                'n_recent_H': b_info_H.get('n_batch', 0),
                'cost_recent_H': b_info_H.get('cost_recent_mean', float('nan')),
                'q_upd_H': ud_info_H.get('q_agg', float('nan')),
                'q_upd_raw_H': ud_info_H.get('q_raw', float('nan')),
                'n_upd_H': ud_info_H.get('n_update', 0),
                'q_cost_L': q_info_L['q_agg'], 'rate_c_L': q_rate_L,
                'ratio_c_L': q_rate_L / max(d_block_L, 1e-9),
                'q_batch_L': b_info_L.get('q_agg', float('nan')),
                'n_recent_L': b_info_L.get('n_batch', 0),
                'd_block_H': d_block_H, 'd_block_L': d_block_L,
                'ema_c1_H': agent_H.ema_cost, 'ema_cL_L': agent_L.ema_cost,
                # RBB (= 纯物理裁剪)
                'pb_original': rbb_info['pb_original'],
                'pbrule': rbb_info['pbrule'],
                'pbphysicsmin': rbb_info['pbphysicsmin'],
                'pbphysicsmax': rbb_info['pbphysicsmax'],
                'clip_amount': rbb_info['clip_amount'],
                'unfeasible_gap': rbb_info['unfeasible_gap'],
                'is_infeasible': float(rbb_info['is_infeasible']),
                # 风况/电价档位 (仅诊断)
                'wind_level': reward_details['wind_level'],
                'price_level': reward_details['price_level'],
                'pr': reward_details['pr'],
                'u': reward_details['u'],
                'turbine_profit': reward_details['turbine_profit'],
                'bess_profit': reward_details['bess_profit'],
                'revenue': reward_details['revenue'],
                'degradation': reward_details['degradation'],
                'degradation_ref490': reward_details['degradation_ref490'],
                'c1_fluct': reward_details['c1_fluct'],
                'SOC': reward_details['SOC'],
                'P_F': P_F,
                'soc_next': float(observation2[2]),
                'bess_profit_net': (reward_details['bess_profit']
                                    - reward_details['degradation_ref490']),
            })
            if step % 50 == 0:
                df_written = flush_csv_append_batch(
                    '{}/df_hsrl.csv'.format(output), df_rows, df_written)
                df_rows = []

        # ---- Periodic checkpoint ----
        if step > 0 and step % 2000 == 0:
            agent_H.save_checkpoint(output, step)
            agent_L.save_checkpoint(output, step)
            prCyan('Checkpoint saved at step {}'.format(step))

        if upload:
            writer.add_scalar('r1', r1, step)
            writer.add_scalar('c1', c1_step, step)
            writer.add_scalar('r2', r2, step)
            writer.add_scalar('c2', c2, step)
            writer.add_scalar('deg_L', deg_step, step)
            writer.add_scalar('viol_L', viol_L, step)
            writer.add_scalar('pen6_H', pen6_H, step)
            writer.add_scalar('g', g_applied, step)
            writer.add_scalar('g_raw', g_raw, step)
            writer.add_scalar('sum_disch', c2_info['sum_disch_before'], step)
            writer.add_scalar('need', c2_info['need'], step)
            writer.add_scalar('dev', c2_info['dev'], step)
            writer.add_scalar('wind_speed', env.u, step)
            writer.add_scalar('lam1', agent_H.lam, step)
            writer.add_scalar('lam2', agent_L.lam, step)
            writer.add_scalar('lam1_phys', agent_H.lam / max(hc.lam_scale(), 1e-12), step)
            writer.add_scalar('dual/q_cost_H', q_info_H['q_agg'], step)
            writer.add_scalar('dual/rate_c_H', q_rate_H, step)
            writer.add_scalar('dual/ratio_c_H', q_rate_H / max(d_block_H, 1e-9), step)
            writer.add_scalar('dual/q_cost_L', q_info_L['q_agg'], step)
            writer.add_scalar('dual/rate_c_L', q_rate_L, step)
            writer.add_scalar('dual/ratio_c_L', q_rate_L / max(d_block_L, 1e-9), step)
            writer.add_scalar('d1', d_block_H, step)
            writer.add_scalar('d2', d_block_L, step)
            if agent_H.ema_cost is not None:
                writer.add_scalar('ema_c1_H', agent_H.ema_cost, step)
            if agent_L.ema_cost is not None:
                writer.add_scalar('ema_cL_L', agent_L.ema_cost, step)
            writer.add_scalar('soc', float(observation[2]), step)
            writer.add_scalar('P_F', P_F, step)
            writer.add_scalar('phase/hour', pm_now / 60.0, step)
            if phase_checked > 0:
                writer.add_scalar('phase/check_mismatch_frac',
                                  phase_mismatch / float(phase_checked), step)
            writer.add_scalar('seg/price_level',
                              {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(
                                  reward_details['price_level'], 0), step)
            writer.add_scalar('seg/wind_level',
                              {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(
                                  reward_details['wind_level'], 0), step)
            writer.add_scalar('reward_upper/turbine_profit',
                              reward_details['turbine_profit'], step)
            writer.add_scalar('reward_upper/bess_profit',
                              reward_details['bess_profit'], step)
            writer.add_scalar('reward_upper/degradation_ref490',
                              reward_details['degradation_ref490'], step)
            writer.add_scalar('rbb/clip_amount', rbb_info['clip_amount'], step)
            writer.add_scalar('rbb/pbrule', float(pbrule), step)
            writer.add_scalar('rbb/pb_original', pb_original, step)
            if episode_steps % 5 == 4:
                writer.add_scalar('sum_r1', sum_r1, step)
                writer.add_scalar('sum_c1', sum_c1, step)
                writer.add_scalar('sum_c2', sum_c2, step)
                writer.add_scalar('sum_disch_block', sum_disch, step)
                for i in range(5):
                    writer.add_scalar('action/a_H_original_{}'.format(i),
                                      original_action_H[i], step)
                    writer.add_scalar('action/a_H_clipped_{}'.format(i),
                                      applied_action_H[i], step)
                writer.add_scalar('action/g', g_applied, step)
                writer.add_scalar('action/pb_rule_amortized',
                                  hsrl.amortized_rule_action(
                                      g_applied, 0.0, 1, E_block_start), step)
            if step > 0 and step % 1440 == 0:
                writer.add_scalar('env/window_violation_count',
                                  window_violation_count, step)
                writer.add_scalar('env/window_power_sum', window_power_sum, step)
                window_violation_count = 0
                window_power_sum = 0.0

        # ---- 推进 ----
        step += 1
        episode_steps += 1
        episode_reward += r1
        episode_cost += c1_step
        episode_cost2 += c2
        episode_profit += observation2[-1] * observation[0] * time \
            - 490.0 * max(-float(pbrule), 0) * time
        episode_bess_profit += reward_details['bess_profit'] \
            - 490.0 * max(-float(pbrule), 0) * time
        episode_deg_discharge += max(-float(pbrule), 0.0) * time
        episode_deg_charge += max(float(pbrule), 0.0) * time
        norm_observation = deepcopy(norm_observation2)
        observation = deepcopy(observation2)
        pw_tru = pw_tru2

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
                    'episode_profit': episode_profit,
                    'bess_profit': episode_bess_profit,
                    'deg_charge': episode_deg_charge,
                    'deg_discharge': episode_deg_discharge,
                    'wind_day': env.j,
                })
                df_episode_written = flush_csv_append_batch(
                    '{}/df_episode_hsrl.csv'.format(output),
                    df_episode_rows, df_episode_written)
                df_episode_rows = []
            observation = None
            episode_steps = 0
            episode_reward = episode_cost = episode_cost2 = episode_profit = 0.
            episode_bess_profit = 0.
            episode_deg_discharge = episode_deg_charge = 0.
            episode += 1

    if use_csv:
        flush_csv_append_batch('{}/df_hsrl.csv'.format(output), df_rows, df_written)
        flush_csv_append_batch('{}/df_episode_hsrl.csv'.format(output),
                               df_episode_rows, df_episode_written)
        flush_csv_append_batch('{}/df_test20_hsrl.csv'.format(output),
                               df_test_rows, df_test_written)
        flush_csv_append_batch('{}/update_stats_H.csv'.format(output),
                               df_update_rows, df_update_written)

    if phase_mode > 0:
        prGreen('phase check: mode={} dims={} checked={} mismatch={} ({:.4f}%)'.format(
            phase_mode, PHASE_DIM[phase_mode], phase_checked, phase_mismatch,
            100.0 * phase_mismatch / max(phase_checked, 1)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode',
                        default='train',
                        type=str,
                        help='support option: train/test')
    # aifflor: 输出目录 / **TensorBoard run 名** 前缀改为 'AIFFLOR'
    #   （产物 output/AIFFLOR-runNN，与 td3per 的 output/TD3PER-runNN 分开）。
    #   ⚠ TB 的 scalar tag 与 td3per 逐字同名（刻意的，便于跨 run 对齐比较）
    #   ⇒ 方法身份靠这个 run 名 + CSV 名 df_aifflor.csv。
    parser.add_argument('--env',
                        default='AIFFLOR',
                        type=str,
                        help='run 身份 (决定日志目录 output/<env>-runN, 即 TensorBoard 的 run 名);'
                             ' aifflor: TB 的 scalar tag 与 td3per 逐字同名(便于跨 run 对齐),'
                             ' 故方法区分靠此目录名 + CSV 名 (df_aifflor.csv)')
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
                        default=None,
                        type=int,
                        help='time without training but only filling the replay memory')
    # 0.95: H 层 γ 以【块(5 min)】为单位生效 (observe_jump 聚合 5 步后 discount 只用一次)
    #       ⇒ 半衰期 ln0.5/ln0.95 = 13.5 块 ≈ 68 min; 1/(1−γ) = 20 块 ≈ 100 min
    #       (0.98 时半衰期 34 块 ≈ 2.8 h; 0.99 才是 69 块 ≈ 5.8 h —— 旧注释此处写错过)
    parser.add_argument('--discount', default=0.95, type=float, help='')
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
                             '1 = 复现 run1~run10 的 1-step 历史口径; 建议 5)')
    parser.add_argument('--target_agg', default='min', type=str,
                        choices=['min', 'mean'],
                        help='H 层 twin profit-critic 的 target 聚合 '
                             '(min = TD3 原口径; mean = 去掉悲观偏差)')
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
    parser.add_argument('--bsize', default=256, type=int,
                        help='minibatch size')
    # ---- memory 容量 ----
    #   ✅ 口径 = **条数直接就是环境步数**（2026-10 更正，去掉历史遗留的 x5）:
    #   旧注释写 "1 条 = n 个块 = 5n 个环境步 ⇒ rmsize x 5n"，那是**基于一个错误前提**
    #   —— 以为 nstep=5 会把 transition 压缩到 rmsize/n 条。实际滑窗 stride-1
    #   **不压缩**: 每个块都是某条转移的起点, 条目数 ≈ 块数 ⇒ 浪费问题早已解决。
    #   ⇒ **rmsize = 40000 就是 40000（条 ≈ 环境步）, 不要再乘任何系数。**
    #   覆盖环境步 = rmsize（= 40000）。
    #   复现历史 run1~run10 (1-step) 请传 --rmsize 80000; 复现 run12/run13 (n=5) 传 20000。
    #   其它五个方法没有 n 步 ⇒ 维持 80000 不动 (见 agentutil/test_phase_uniform.py 的例外表)。
    parser.add_argument('--rmsize',
                        default=40000,
                        type=int,
                        help='H 层 memory 条数（≈ 覆盖的环境步数; 不乘系数）; '
                             '默认 40000 = 生产口径; 复现 run1~run10 传 80000')
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

    # ---- UTD（每块 update_policy 次数）: 启动指令可调; 默认 = 历史口径, 逐位不变 ----
    parser.add_argument('--utd_h', default=4, type=int,
                        help='H 层每块 update_policy 次数 (默认 4; 复现历史传 --utd_h 2; 调整它不再改变 d_block_H 的减半节奏)')
    parser.add_argument('--utd_l', default=10, type=int,
                        help='L 层每块 update_policy 次数 (默认 10 = 历史口径)')

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
                        help='L 层(BESS)折扣: 步级语义; 默认 0.98 (td3per 原口径; H 为块级 0.98)')
    parser.add_argument('--l_noise_amp', default=1.0, type=float,
                        help='L 层探索噪声半宽 [MW] (残余动作量级; 旧式 ±0.05 比所需小 ~50 倍)')
    parser.add_argument('--cost_critic_mode', default='max', type=str,
                        choices=['min', 'max'],
                        help='cost twin-critic aggregation: max (safe, recommended) or min')

    # RBB parameters (硬裁剪 + L 层可行域观测输入)
    parser.add_argument('--rbb_fluctuation_limit', default=3.0, type=float,
                        help='Max grid power fluctuation between steps (MW)')
    parser.add_argument('--rbb_rated_power', default=0.5, type=float,
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
    parser.add_argument('--sig_noise_min', default=0.01, type=float,
                        help='SOC 目标信号维探索噪声**地板** (eps 收敛到它, 不再降到 0); '
                             '0.01 ⇒ SOC 目标抖动 ±0.05 MWh = SOC_TOL(0.1) 的一半')
    parser.add_argument('--wt_noise_floor', default=0.01, type=float,
                        help='WT 5 维探索噪声**地板** (0 = 20k 步后这些维完全不再探索); '
                             '0.01 = 行为带 [0,0.5] 的 2%, 折算单机功率抖动约 ±2.4%')

    # ---- 24h 相位特征 (--phase_feat) ----
    parser.add_argument('--phase_feat', default=0, type=int, choices=[0, 1, 2],
                        help='H 层观测的 24h 相位特征: 0=关(4维, 默认, 时钟过拟合治理见 README); 1=sin/cos(6维); '
                             '2=1+[段内剩余比例, 剩余小时/8, 下一段电价/1000](9维)')

    # ---- 输出层 bias 初始化 (与 td3perrb 一致) ----
    parser.add_argument('--wt_bias_init', default=1, type=int, choices=[0, 1],
                        help='1: 把 actor 输出层 WT 5 维 bias 初始化到“中性宽带” '
                             '[0.05,0.45]; 0: 用 PyTorch 默认')

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
    # ---- 诊断专用（默认全关 ⇒ 生产路径逐位不变；见 documentation/diag_two_items_plan.md）----
    parser.add_argument('--critic_action_first', default=0, type=int,
                        help='1 => critic 的动作从第一层进网（fc1 = nb_states+nb_actions -> 4*hidden1；默认 0=旧口径）')
    parser.add_argument('--diag_u', default=-1.0, type=float,
                        help='>=0 ⇒ 全训练期恒定风速 [m/s]（实验 1-S1）')
    parser.add_argument('--diag_pr', default=-1.0, type=float,
                        help='>=0 ⇒ 全天恒定电价 [元/MWh]（实验 1-S1/S2）')
    parser.add_argument('--diag_scalar', default=0, type=int,
                        help='1 ⇒ actor 只输出一个标量 s=sigmoid(a0)，前 5 维统一取 0.33·s（实验 2）')
    parser.add_argument('--diag_freeze_g', default=0, type=int,
                        help='1 ⇒ 第 6 维 g ≡ 0（BESS 不参与；实验 1）')
    parser.add_argument('--diag_lambda_fixed', default=-1.0, type=float,
                        help='>=0 ⇒ 每次对偶更新后把 agent_H.lam 钉到该值（actor 空间；实验 1 用 0）')
    parser.add_argument('--diag_center_reward', default=0, type=int,
                        help='1 => 奖励中心化: r1 -= pr*(P_G-P_nom(u))*dt（把动作效应从状态基线上抬出来）')
    parser.add_argument('--pen_h', default=-1.0, type=float,
                        help='>0 ⇒ 覆盖 H 前 5 维越界罚系数 [元/单位]（实验 1 用 400 把动作压在界内）')


    # ==================================================================
    # HSRL 方法分支（--hsrl）—— 口径总表见 aifflor/aifflor_hsrl.py
    # ==================================================================
    parser.add_argument('--hsrl', action='store_true',
                        help='启用 HSRL 分支（**方法分支**）: H 第 6 维 = 本 5 步块内的'
                             '**总放电量 g [MW·min]**（放电为正；可行域 = SOC + 额定功率，'
                             '**不含波动约束**，越界线性罚 400 元/MW·min）; '
                             'H 奖励去掉 bessdeg 与块末 SOC 目标 gap 罚; '
                             'L 动作 = **绝对 pb**（只按物理域裁剪）; '
                             'r2 = −200·越界[MW] − 20·放电[MW·min]; '
                             'c_L = 1 − exp(−|(g−Σ已放电)/(δ−t_L) − a_L|); '
                             'λ2 ∈ [2, 50]（初值 10）, d_block_L = 0.5; '
                             'L 观测(6 维) = [u/10, pr/1000, SOC/0.5, P_G(t−1)/5, g/2.5, '
                             'need/2.5]（need = c_L 的参照点 (g−Σ)/n_rem）; '
                             'warmup 下层 = 摊还执行器; 训练中测试改 20 步协议。'
                             '不加该参数时旧路径**逐位不变**')
    parser.add_argument('--hsrl_pen6_coef', default=hsrl.HSRL_PEN6_COEF, type=float,
                        help='HSRL: H 第 6 维越界罚系数 [元/MW·min] (默认 %.0f, 不封顶)'
                             % hsrl.HSRL_PEN6_COEF)
    parser.add_argument('--hsrl_viol_coef', default=hsrl.HSRL_VIOL_COEF, type=float,
                        help='HSRL: L 动作越界罚系数 [元/MW] (默认 %.0f)' % hsrl.HSRL_VIOL_COEF)
    parser.add_argument('--hsrl_deg_coef', default=hsrl.HSRL_DEG_COEF, type=float,
                        help='HSRL: L 放电退化成本 [元/MW·min, 只对放电] '
                             '(默认 %.1f = 越界罚的 1/10)' % hsrl.HSRL_DEG_COEF)
    parser.add_argument('--hsrl_sig_noise_max', default=0.5, type=float,
                        help='HSRL: 第 6 维探索噪声上限 [MW·min] (默认 0.5 = 满量程 2.5 的 20%%)')
    parser.add_argument('--hsrl_sig_noise_min', default=0.025, type=float,
                        help='HSRL: 第 6 维探索噪声地板 [MW·min] (默认 0.025 = 满量程的 1%%)')
    parser.add_argument('--hsrl_d_block_l', default=hsrl.HSRL_D_BLOCK_L, type=float,
                        help='HSRL: L 层每块 sum_c_L 预算的**初值** (默认 %.2f；地板见 '
                             '--hsrl_d_block_l_min = %.1f)' % (hsrl.HSRL_D_BLOCK_L,
                                                              hsrl.HSRL_D_BLOCK_L_MIN))
    parser.add_argument('--hsrl_dual_src', default=hsrl.HSRL_DUAL_SRC,
                        choices=['q', 'ema'],
                        help='HSRL: **对偶的分子** q=critic cost 的 Q（默认；比实际块成本的 EMA '
                             '平滑得多 ⇒ λ 波动小）；ema=实际块成本 EMA（历史口径，A/B 对照）')
    parser.add_argument('--hsrl_dual_q_agg', default=hsrl.HSRL_DUAL_Q_AGG,
                        choices=['mean', 'min', 'max'],
                        help='HSRL: Q 的 twin 聚合（对偶专用，独立于 --cost_critic_mode）: '
                             'mean=双臂均值（默认，无偏；critic 回归目标已用 max，对偶再用 max '
                             '是双重高估 ⇒ 会复现"高估→λ 顶死"）；min=保守；max=与目标一致')
    parser.add_argument('--hsrl_dual_q_src', default=hsrl.HSRL_DUAL_Q_SRC,
                        choices=['update_batch', 'recent_batch', 'point'],
                        help='HSRL: src=q 时**用哪种 Q 估计**做分子（λ1）。'
                             'update_batch（默认）= 直接用 update_policy 本块已算好的 cost Q '
                             '**批均值**（q_c_mean/q_c2_mean）⇒ 靠缓冲自然淘汰让均值随策略下降、'
                             '且不含当日风况；recent_batch = 近端 N 条状态的批均值 × 当前策略动作'
                             '（λ2 固定用它）；point = 旧的单点 Q(s0,a0)（A/B 对照）')
    parser.add_argument('--hsrl_dual_recent_n', default=hsrl.HSRL_DUAL_RECENT_N, type=int,
                        help='HSRL: recent_batch（λ2，以及 λ1 选 recent_batch 时）的近端条数'
                             '（默认 %d: H 每块 1 条 ⇒ ≈1 天）' % hsrl.HSRL_DUAL_RECENT_N)
    parser.add_argument('--hsrl_lam1_min', default=hsrl.HSRL_LAM1_PHYS_MIN, type=float,
                        help='HSRL: λ1 的**物理价下限** [元/(MW·step)]（默认 %.1f；= 名义边际收益 9.67 的 '
                             '21%%、低电价边际收益 5.0 的 40%%，且 < BESS 退化边际价 20 ⇒ 界内波动不值得'
                             '动用 BESS）。当前实现把上界设为 20 的 2.5× = 50' % hsrl.HSRL_LAM1_PHYS_MIN)
    parser.add_argument('--hsrl_lam1_max', default=hsrl.HSRL_LAM1_PHYS_MAX, type=float,
                        help='HSRL: λ1 的**物理价上限** [元/(MW·step)]（默认 %.1f = max(平抑资源边际成本: '
                             'BESS 退化 20, 峰价降载 16.33) 的 2.5×；界外波动单价是 20·λ1 ⇒ '
                             '20×50 = 1000 远大于峰价 16.33）。⚠ 上界必须 > 下界' % hsrl.HSRL_LAM1_PHYS_MAX)
    parser.add_argument('--hsrl_d_block_h_min', default=hsrl.HSRL_D_BLOCK_H_MIN, type=float,
                        help='HSRL: H 层 curriculum 的**收敛下限**（地板，默认 %.1f ⇒ 可一直减半，'
                             '由"连续 50 块不满足 ema ≤ d"自行停止）。此前沿用 hc.h_settings 的 '
                             'd_block_min = 初值 15 ⇒ 减半是空操作（curriculum 被禁用）'
                             % hsrl.HSRL_D_BLOCK_H_MIN)
    parser.add_argument('--hsrl_d_block_l_min', default=hsrl.HSRL_D_BLOCK_L_MIN, type=float,
                        help='HSRL: L 层 curriculum 的**收敛下限**（地板，默认 %.1f；此前 0.05）'
                             % hsrl.HSRL_D_BLOCK_L_MIN)
    parser.add_argument('--hsrl_test_wind', default='fixed', choices=['fixed', 'random'],
                        help='HSRL: test20 的风速段 fixed=**固定段**（默认 col=%d/start=%d，'
                             '每次触发同一场景 ⇒ 配对看收敛）; random=每次触发随机取列+起点'
                             % (hsrl.HSRL_TEST_WIND_COL, hsrl.HSRL_TEST_WIND_START))
    parser.add_argument('--hsrl_test_wind_col', default=-1, type=int,
                        help='HSRL: test20 固定风速段的**列**（天），默认 -1 ⇒ 用常量 %d'
                             % hsrl.HSRL_TEST_WIND_COL)
    parser.add_argument('--hsrl_test_wind_start', default=-1, type=int,
                        help='HSRL: test20 固定风速段的**起点**（天内步），默认 -1 ⇒ 用常量 %d'
                             % hsrl.HSRL_TEST_WIND_START)
    parser.add_argument('--hsrl_test_seed', default=-1, type=int,
                        help='HSRL: 仅 --hsrl_test_wind random 时生效的采样种子 '
                             '(>0 固定; 默认 -1 = 每次触发重新随机)。采样用**独立** RandomState '
                             '⇒ 不吃训练流、不改变训练轨迹')
    hwarm.add_warmup_args(parser)
    args = parser.parse_args()
    args = hwarm.resolve_warmup(args)   # WARMUP_AUTO: 正式口径 warmup=bsize*mult, 期零网络更新
    hc.set_mode(args.c1_mode)
    hc.set_value_scale(args.value_scale)   # 值函数回归空间归一化 (单一真源)
    hc.set_cost_scale(args.cost_scale)   # cost 通道缩放 (λ 自动跟随)
    _HCFG = hc.h_settings()
    prCyan(hc.summary())
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    max_E = 0.5
    upload = True
    use_csv = True
    args.max_episode_length = 1440  # 1d = 24h*60m = 1440 steps
    args.train_iter_H = 1000 * args.max_episode_length
    if getattr(args, 'max_steps', 0) and args.max_steps > 0:
        args.train_iter_H = int(args.max_steps)

    # ---- 诊断开关落地（默认 -1/0 ⇒ 不进入任何分支，生产路径逐位不变）----
    DIAG['u'] = float(getattr(args, 'diag_u', -1.0))
    DIAG['pr'] = float(getattr(args, 'diag_pr', -1.0))
    DIAG['scalar'] = int(getattr(args, 'diag_scalar', 0))
    DIAG['freeze_g'] = int(getattr(args, 'diag_freeze_g', 0))
    DIAG['lam_fixed'] = float(getattr(args, 'diag_lambda_fixed', -1.0))
    DIAG['pen_h'] = float(getattr(args, 'pen_h', -1.0))
    DIAG['center'] = int(getattr(args, 'diag_center_reward', 0))
    if any([DIAG['u'] >= 0, DIAG['pr'] >= 0, DIAG['scalar'], DIAG['freeze_g'],
            DIAG['lam_fixed'] >= 0, DIAG['pen_h'] > 0]):
        prCyan('[DIAG] u={} pr={} scalar={} freeze_g={} lam_fixed={} pen_h={}'.format(
            DIAG['u'], DIAG['pr'], DIAG['scalar'], DIAG['freeze_g'],
            DIAG['lam_fixed'], DIAG['pen_h']))
    # 只在**未显式指定**（--seed 默认 -1）时随机化。显式 --seed >= 0 必须被尊重，
    # 否则"同 seed 配对 A/B"会被随机种子混淆（历史 bug：本行曾无条件覆盖用户传入的 --seed）。
    if args.seed is None or int(args.seed) < 0:
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

    # aifflor: 物理引擎 = FLORIS（**不启 MATLAB**；env.py 的旧 MATLAB 路径不受影响）。
    #   动作语义与 td3per **逐字相同**（轴向因子 a ∈ [0,0.5]，航向恒 0），
    #   所以 `td3per_myfunc` 的 CheckAction / 越界罚 / 动作带全部原样可用。
    #   两个 env 各持一份物理模型 —— 与旧环境"两个 WindBattery + 同一个 MATLAB 引擎"
    #   在结构上同形（互不串状态），代价是两份 FlorisModel（构建各 ~0.2 s）。
    #   尾流参数 = `agentutil/floris_calibrated_defaults.py` 的**标定值**
    #   （对 WFSim 的验证集相对误差 66.9% → 1.1%）。
    # aifflor 的物理配置（**由 `agentutil/port_aifflor.py` 的替换表决定**）：
    #   `calibrated=False` = **FLORIS 随包默认尾流参数**（可审计：不改任何参数）；
    #   `power_scale=AT.LEVEL_COEF` = 出口电平系数 1/1.6696 ≈ 0.59895，
    #     把 FLORIS 的电场出力整体抬到 WFSim 电平。
    #   依据（实测，`agentutil/analyse_level_coef_modes.py`）：
    #     · 全域 350 点（3–20 m/s，随机动作）原始 MAPE 68.476% → **7.189%**；
    #     · 重标定后 MAPE 66.962% → **0.001%**（即电平偏差被**完全**吸收）；
    #     · 波动：限幅后波动均值相对 WFSim +22.50%（标定参数为 −28.11%）；
    #     · 锚点 u=10, a=1/3：由 5.942350 变为 3.559146 MW（WFSim 参考 3.4462，+3.28%）。
    #   ⇒ 物理用原厂默认、只加一个显式系数，功率与波动都对齐且更易审计。
    aif_physics = AifflorPhysics(nt=args.nb_turbine, calibrated=False,
                                 power_scale=aturb.LEVEL_COEF)
    aif_physics_test = AifflorPhysics(nt=args.nb_turbine, calibrated=False,
                                      power_scale=aturb.LEVEL_COEF)
    env = WindBatteryAIF(max_E, args.nb_turbine, upload, None, physics=aif_physics)
    env_test = WindBatteryAIF(max_E, args.nb_turbine, upload, None,
                              physics=aif_physics_test)
    # ---- 诊断：恒定风速 / 恒定电价（默认 -1 ⇒ 不动生产路径）----
    if DIAG['u'] >= 0:
        _udf = pd.DataFrame(np.full((1440, 201), float(DIAG['u'])))
        env.wind_noise = _udf
        env_test.wind_noise = _udf.copy()
    if DIAG['pr'] >= 0:
        env.diag_const_pr = float(DIAG['pr'])
        env_test.diag_const_pr = float(DIAG['pr'])
    prCyan("口径: " + aturb.declared_scope())

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
    if args.hsrl:
        # HSRL: 波动窗口**不参与**任何裁剪与奖励 ⇒ 用 inf 关掉它, RBB 退化为
        # "纯 SOC + 额定功率裁剪 + 记录"(其 pbphysicsmin/max 与 aifflor_hsrl.feasible_power
        # 逐位同式) ⇒ `pbrule ≡ clip(命令, 物理域)`、`unfeasible_gap ≡ 0`。
        rbb_system.fluctuation_limit = float('inf')
        # 对偶的模块级默认（Q 分子 / twin 聚合）按命令行覆盖
        hsrl.HSRL_DUAL_SRC = str(args.hsrl_dual_src)
        hsrl.HSRL_DUAL_Q_AGG = str(args.hsrl_dual_q_agg)
        hsrl.HSRL_DUAL_Q_SRC = str(args.hsrl_dual_q_src)
        prCyan(hsrl.summary())
        _lo1, _hi1, _init1 = hsrl.lam1_phys_bounds(args.hsrl_lam1_min, args.hsrl_lam1_max)
        prCyan('[HSRL] 对偶: 分子 = {} | twin 聚合 = {} | λ1 物理价 ∈ [{:.3f}, {:.3f}] '
               '元/(MW·step) (actor 空间 ×{:.3g} = [{:.4f}, {:.4f}], 初值 {:.4f})'.format(
                   ('critic cost 的 Q（{}）'.format(
                       {'update_batch': '本块 update 的 Qc 批均值（靠缓冲自然淘汰跟随策略）',
                        'recent_batch': '近端 %d 条批均值 × 当前策略动作' % args.hsrl_dual_recent_n,
                        'point': '块首单点 Q(s0,a0)'}.get(args.hsrl_dual_q_src, args.hsrl_dual_q_src))
                    if args.hsrl_dual_src == 'q' else '实际块成本 EMA'),
                   args.hsrl_dual_q_agg,
                   _lo1 / max(1e-12, hc.lam_scale()), _hi1 / max(1e-12, hc.lam_scale()),
                   hc.lam_scale(), _lo1, _hi1, _init1))

    # 上层 H: TD3 + EMA 实际成本对偶 (lam1 ∈ [3, 100], 初始 5.0) —— 与 td3perrb 一致
    nb_states_H = 4 + PHASE_DIM[int(getattr(args, 'phase_feat', 0))]
    print('H nb_states = {} (4 base + {} phase dims, --phase_feat={})'.format(
        nb_states_H, PHASE_DIM[int(getattr(args, 'phase_feat', 0))], args.phase_feat))
    agent_H = SP_TD3(nb_states=nb_states_H, nb_actions=6, args=args,
                     lam_max=(_HCFG['lam_max'] if _HCFG else 100.0), lam_min=(_HCFG['lam_min'] if _HCFG else 3.0),
                     ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                     record_update_stats=True,
                     policy_delay=args.policy_delay,
                     policy_noise=args.policy_noise,
                     noise_clip=args.noise_clip,
                     cost_critic_mode=args.cost_critic_mode,
                     wt_noise=args.wt_noise,
                     sig_noise_max=args.sig_noise_max,
                     sig_noise_min=args.sig_noise_min,
                     wt_noise_floor=args.wt_noise_floor,
                     nstep=args.nstep, target_agg=args.target_agg)
    print('[H] nstep={} target_agg={} (n 步回报: 自助传播深度 50->{:.1f}; '
          '复现历史 run1~run10 请传 --nstep 1)'.format(
              args.nstep, args.target_agg, 50.0 / max(1, args.nstep)))
    if args.viol_derate > 0:
        print('[H] 越界乘性惩罚 c={:.2f}: r1 的风机收益 × exp(-{:.2f}·Σ前5维越界量); '
              '记录列(revenue/turbine_profit/degradation/profit)不变'.format(
                  args.viol_derate, args.viol_derate))
    else:
        print('[H] 越界乘性惩罚已关闭 (--viol_derate 0): 与历史 run 逐位一致')
    agent_H.lam = _HCFG['lam_init'] if _HCFG else 5.0
    if args.hsrl:
        # λ1 的**物理价**区间/初值（见 aifflor_hsrl.HSRL_LAM1_PHYS_* 与 README §3.8）
        _lo1, _hi1, _init1 = hsrl.lam1_phys_bounds(args.hsrl_lam1_min, args.hsrl_lam1_max)
        agent_H.lam_min, agent_H.lam_max, agent_H.lam = _lo1, _hi1, _init1
    # 值函数回归空间归一化 (s = hc.value_scale()): 只作用于 H 层 r1/c1 进入
    # critic 之前; buffer/对偶/日志/CSV 仍是原始口径 (见 hsrl_cost 末尾小节)。
    agent_H.value_scale = hc.value_scale()
    agent_H.cost_scale = hc.cost_scale()

    if _HCFG:
        agent_H.ema_lam_lr = _HCFG['ema_lam_lr']   # uniform: 乘性对偶步长
    if args.wt_bias_init:
        init_wt_action_bias(agent_H, lo=0.05, hi=0.45, sig_value=0.0)
    if args.hsrl:
        # HSRL: 第 6 维量纲从"归一化信号 ∈[−0.5,0.5]"换成 **[MW·min]（满量程 ±2.5）**
        # ⇒ 探索噪声同步重标（前 5 维风机维保持恒定 wt_noise 不变）。
        agent_H.explore_amp[5] = float(args.hsrl_sig_noise_max)
        agent_H.explore_floor[5] = float(args.hsrl_sig_noise_min)
        prYellow('[HSRL] 第 6 维探索噪声: amp={} -> floor={} [MW·min] (满量程 ±{:.2f})'.format(
            args.hsrl_sig_noise_max, args.hsrl_sig_noise_min, hsrl.HSRL_G_FULL))

    # 下层 L: 保留 td3per 自己的 RL BESS 操作
    #   * 观测 5 维蒸馏 = [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]
    #   * r2 = −1·min(clip_amount, 20)
    #   * c2 = +1·trackerr [MW] (实际施加 pb 到参考集合的距离; 零成本动作 = 规则 SocTracker,
    #        F=∅ 时参考集合退化为单点竭力值 ⇒ 只有顶到它才不罚; 见 hsrl_soctarget)
    #   * lam2 ∈ [hst.L_LAM_MIN, hst.L_LAM_MAX] = [0.5, 2] —— **无量纲兑换率**
    #       "每 MW c2 至多值 2 MW r2 / 至少值 0.5 MW r2", 关于 1.0 对数对称 (MAX·MIN = 1);
    #       初值 L_LAM_INIT = 1.0 (中性兑换率)
    #   * 每块 1 次**乘性**对偶: λ2 ← clip(λ2·exp(hst.L_EMA_LAM_LR·(EMA_c2/d_block_L − 1)), 0.5, 2);
    #       步级折扣 --l_discount
    # ---- HSRL 分支: 动作语义变成"绝对 pb"、cost 换成图中指数式 ⇒
    #      λ2 的区间/初值/预算随之重标 (c_L 无量纲 vs r2 元量级; 见 aifflor_hsrl.py) ----
    if args.hsrl:
        # HSRL: 观测 = [u/10, pr/1000, SOC/0.5, P_G(t−1)/5, g/2.5, need/2.5]（6 维;
        #   第 6 维 = c_L 的参照点 (g−Σ已放电)/n_rem ⇒ 策略能直接看到"本步该出多少"）
        agent_L = SP_TD3(nb_states=6, nb_actions=1, args=args,
                         lam_max=hsrl.HSRL_L_LAM_MAX, lam_min=hsrl.HSRL_L_LAM_MIN,
                         ema_lam=True, ema_alpha=0.02, ema_lam_lr=hsrl.HSRL_L_EMA_LAM_LR,
                         policy_delay=args.l_policy_delay,
                         policy_noise=args.l_policy_noise,
                         noise_clip=args.noise_clip,
                         cost_critic_mode=args.cost_critic_mode)
        agent_L.lam = hsrl.HSRL_L_LAM_INIT
        agent_L.cost_limit = hsrl.HSRL_L_COST_LIMIT
    else:
        agent_L = SP_TD3(nb_states=5, nb_actions=1, args=args,
                         lam_max=hst.L_LAM_MAX, lam_min=hst.L_LAM_MIN,
                         ema_lam=True, ema_alpha=0.02, ema_lam_lr=hst.L_EMA_LAM_LR,
                         policy_delay=args.l_policy_delay,
                         policy_noise=args.l_policy_noise,
                         noise_clip=args.noise_clip,
                         cost_critic_mode=args.cost_critic_mode)
        agent_L.lam = hst.L_LAM_INIT
        agent_L.cost_limit = hst.L_COST_LIMIT  # L层c2专用初始值（单一真源）
    agent_L.value_scale = 1.0   # L 层 r2/c2 不归一化 (值函数尺度本就 O(1))
    # ---- 残余动作: L actor 输出层零初始化 ⇒ t=0 时 Δ≡0 ⇒ 起点逐位等于规则下层 ----
    #      HSRL 下同样是零初始化, 但语义 = "起点 pb = 0 (不充不放)"。
    hinit.zero_actor_output(agent_L, verbose=True)
    # L 层探索半宽 [MW] (残余动作量级): 旧式 ±0.05 MW 比所需动作小 ~50 倍
    agent_L.explore_amp = np.full(1, float(getattr(args, 'l_noise_amp', 1.0)), dtype=np.float32)
    agent_L.explore_floor = np.zeros(1, dtype=np.float32)

    agent_L.discount = args.l_discount          # L 步级折扣 (与 H 的块级折扣解耦)
    agent_L.depsilon = 1.0 / max(args.l_epsilon, 1)  # L 探索衰减 (与 H 的 --epsilon 解耦)
    prYellow('L 层(BESS)策略超参: policy_delay={}, policy_noise={} (H 层 TD3: delay={}, noise={}); '
             '折扣: L γ={} (步级, 块内 ≤5 步) / H γ={} (块级); '
             '探索衰减: L 每 {} 次决策 / H 每 {} 次决策'.format(
                 args.l_policy_delay, args.l_policy_noise, args.policy_delay, args.policy_noise,
                 agent_L.discount, agent_H.discount, args.l_epsilon, args.epsilon))
    if not args.hsrl:
        prCyan('下层奖励/成本口径: r2 = -1·min(clip_amount, 20) [两段 clip 合计]; '
               'c2 = +1·trackerr [MW] (dist(实际施加 pb, 参考集合=Soctrack范围∩平抑窗口)); '
               'F=∅ ⇒ 无范围, 只有纯粹一个竭力值 r* (顶到才不罚); '
               'L 观测 = [pg(t-1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]')
    else:
        prCyan('下层奖励/成本口径 (HSRL): r2 = -{:.0f}·越界[MW] - {:.0f}·放电[MW·min]; '
               'c_L = 1 - exp(-|(g-Σ已放电)/(5-t_L) - a_L|); '
               'L 观测(6 维) = [u/10, pr/1000, SOC/0.5, P_G(t-1)/5, g/{:.1f}, need/{:.1f}]; '
               'warmup 下层 = {}'.format(
                   args.hsrl_viol_coef, args.hsrl_deg_coef, hsrl.HSRL_G_FULL,
                   hsrl.OBS_NEED_SCALE,
                   '摊还执行器' if str(getattr(args, 'warmup_bess', 'rule')) == 'rule'
                   else getattr(args, 'warmup_bess')))
    #TODO 是否预训练
    # agent_H.load_weights('output/1_2310S-run18')
    # agent_L.load_weights('output/1_2310S-run18')

    if args.mode == 'train':
        if args.hsrl:
            # ---- HSRL 分支: 20 步测试的风速池 = FFC_speed_noise_1.csv 的"天"列 ----
            _hsrl_wind_pool = None
            try:
                _hsrl_wind_pool = np.asarray(
                    env_test.wind_noise.iloc[:, 1:].to_numpy(dtype=float))
            except Exception as _e:
                prYellow('[HSRL] 风速池不可用 ({}); 20 步测试将被跳过'.format(_e))
            train_hsrl(args.train_iter_H,
                       agent_H,
                       agent_L,
                       env,
                       env_test,
                       rbb_system,
                       _hsrl_wind_pool,
                       args.output,
                       max_episode_length=args.max_episode_length,
                       debug=args.debug,
                       upload=upload,
                       use_csv=use_csv)
        else:
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
        prCyan('[aifflor] 训练结束 —— 物理调用统计:')
        prCyan(env.summary())
        prCyan(env_test.summary())

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
