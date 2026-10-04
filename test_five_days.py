#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五天随机风速测试脚本（单文件，兼容 9 个方法，每方法可多组模型）
================================================================

功能
----
1. 用固定随机种子（默认 42）从【训练用风速序列】中不重复随机抽取 5 天。
   训练风速池 = FFC_speed_noise_1.csv 的 iloc[:, 1..194]（195 列，第 0 列为
   无名索引列），也就是 env.py 里

       self.wind_day = self.wind_noise.iloc[:, self.j]

   的 j 取值池；训练用的 fixed_day_sequence.csv 恰好是该池的一个排列
   （seed=64 的 random.sample，覆盖 1..194 全部列），所以"从训练序列里抽"
   与"从列池里抽"完全等价。抽取方式与 env 的取列方式一致：用 iloc 位置。

2. 对每个方法跑 5 个 episode（每天 1 个 episode，1440 步 = 24h×60min），
   逐方法复刻它自己训练时的推理路径，但 **只前向、不更新网络**：
       agent.is_training = False  → 探索噪声恒为 0，动作完全确定
       agent.observe_jump / update_policy / lam 更新全部不调用
   其余形式与训练相同：H 每 5 步决策一次、L 每步决策、RBB 硬裁剪
   （action[-1] <- pbrule）、env.step() 动态电价（CalPrice 峰谷平）。

3. 【每方法多组模型】一个方法可以给多个模型目录（配合实验方法"每个方法跑
   3 组不同模型"）：按给出顺序依次跑完（run 1/2/3…），所有 run 共用同一批
   5 天风速。CSV 与 TensorBoard 都带 run 维度，并自动给出两级汇总：
       five_day_summary.csv         每方法 × 每组模型（5 天均值）
       five_day_method_summary.csv  每方法跨「全部模型 × 全部天」的 mean ± std ★主表
   给 1 个目录时行为与旧版一致（run 恒为 1）。

4. 每天记录三项指标（另有每步明细 CSV，均带 run/model 列）：
   - viol_count       功率波动越限次数      Σ 1[ |ΔP_G| > 3 MW ]
   - viol_excess_mw   功率波动越限总量      Σ max(|ΔP_G| - 3, 0)      [MW]
   - profit_rmb       利润                  Σ (P_G·pr - 490·max(-pb,0))·Δt  [元]
       电价 pr 与训练同分布 = env.CalPrice()（峰谷平 300/580/980）；
       bessdeg = 490 元/MWh 放电量（util.BESS_DEG_RMB_PER_MWH）。

5. 【CSV + TensorBoard 双份保存】默认同时写，`--no-tensorboard` 关闭。
   TB tag 结构（step 语义已标注）：
       test/{方法}/run{n}/{指标}          step = 天序号 1..N（本组模型的逐日曲线）
       method/{方法}/{指标}               step = 天序号（跨 run 均值，横向对比用）
       method/{方法}/{指标}_min|_max      step = 天序号（跨 run 极值带）
       step/{方法}/run{n}/P_F|SOC|profit_step
                                          step = 全局步（按 run 顺延，不重叠）

兼容项目与各自的推理路径
------------------------
  per            SP_DDPG        + rbb/rbb_rule_bess.RuleBasedBESS      L 基线 5 维
  prrbb          SP_DDPG        + prrbb_rule_bess.RuleBasedBESS        L 基线 5 维
  prle           PRLE_DDPG      + prrbb_rule_bess.RuleBasedBESS        L 基线 5 维
  td3per         SP_TD3         + rbb/rbb_rule_bess.RuleBasedBESS      L 5 维(规则原配料)
                 (上层 = td3perrb 口径: 非分段奖励 + SOC 目标信号 + 相位观测)
  td3prrbb       SP_TD3         + td3prrbb_rule_bess.RuleBasedBESS     L 5 维(规则原配料)
  td3prle        SP_TD3_PRLE    + td3prle_rule_bess.RuleBasedBESS      L 5 维(规则原配料)
  td3prrlrb      SP_TD3         + td3prrlrb_rule_bess  (L=规则 SocTracker, H 带相位)
  td3perrb       SP_TD3         + td3perrb_rule_bess   (同 td3prrlrb, 但奖励**非分段**)
  td3leprrlrb    SP_TD3_LEPRRLRB+ td3leprrlrb_rule_bess(L=规则 SocTracker, H 带相位)
  aifflor        SP_TD3         + rbb/rbb_rule_bess.RuleBasedBESS      L 5 维(规则原配料)
                 (FLORIS 训练 -> **WFSim 引擎**评测的跨引擎迁移臂；label=AIFFLOR@WFSIM)

关键细节（都按各自 main 的写法对齐，不混用）：
  * agent 类、CheckAction / Normalize(_H)、RuleBasedBESS 各自取自本项目模块；
  * L 层观测：
      - per / prrbb (影子电价支) : np.append(Normalize(obs), applied_action_H[-1])
                  - td3per / td3prle / td3prrbb (SOC 目标支) : 下层 = **残余动作**
              (actor 输出 Δ, 命令 = 规则基线 + Δ) + build_L_obs_distilled(
              [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5])
      - prle (SOC 目标支) : build_L_obs_distilled(
              Normalize(obs) + soc_target/10)
      - td3prrlrb / td3perrb / td3leprrlrb : 无 RL 下层，L 由 SocTracker 规则给出
              pb = clip((soc_target - E)/gain, pbrulemin, pbrulemax)
  * td3per / td3prrlrb / td3perrb / td3leprrlrb 的 H 观测 = Normalize(obs) + 24h 相位特征，
    维数（4 + 0/2/5）从 actor_6.pkl 的 fc1.weight 自动探测，无需手填。
    相位时刻 = episode 内的分钟序号（与训练一致：episode 恰好 1440 步）。
    * td3per 改造后其 L 层仍走 RL（不是规则），只是 L 观测换成 5 维规则原配料 [pg(t−1), ptur, SOC, target, n_rem]，
    且 actor 输出 Δ（命令 = 规则基线 + Δ；旧 checkpoint 的动作语义是绝对功率，不可直接加载）；
    因此 H 权重（actor_6）与 L 权重（actor_1）都必须来自改造后的 td3per 训练 run，
    改造前的 TD3PER-run* 权重语义不同（影子电价 + 4 维 H 观测），不可混用。

不启用的训练期机制（对三项指标无影响，故测试中省去）
----------------------------------------------------
  * 网络更新、PER 优先级、lam 对偶、curriculum、checkpoint；
  * PRLE / td3prle / td3leprrlrb 的 LLM 在线阈值调整：风速分档阈值只进入
    **奖励权重/风险系数**（wT/wB/c1 系数）与 buffer relabel，不进入任何
    智能体的观测或动作，因此对越限次数 / 越限总量 / 利润三项指标零影响。
    测试中阈值保持 --seg 起点（模块默认）不变。
  * 探索噪声：agent.is_training=False 下 select_action 的噪声项恒为 0。
  * 唯一保留的随机性是 --soc-init train 时的初始 SOC（默认 uniform 无随机）。

耗时提示
--------
瓶颈是 MATLAB 的 WFSim 前向仿真（约 0.3~1 s/步，与训练主循环同速）。
1 步 = 1 分钟电量，所以「1 天 × 1 组模型」= 1440 步 ≈ 十几分钟到半小时；
9 方法 × 3 组 × 5 天 = 135 个 episode ≈ 19.4 万步 ≈ 单机 30~55 小时。
建议按方法拆成多个进程并行跑（--projects 只测子集，--output 分开写目录）。
想快速抽查就用 `--projects` / `--days N` / `--episode-length N` 缩小规模。

初始 SOC
--------
默认统一 3.0 MWh（env.reset() 的原值），便于跨方法公平对比。
加 --soc-init train 可改回各方法训练时的重置方式：
per/prrbb/prle 用 3.0；td3per/td3prrbb/td3prle/td3prrlrb/td3perrb/td3leprrlrb
用 U(0.5, 5.5) 随机（用脚本自带 RNG，可复现）。

用法
----
    # 每方法 3 组模型：一个方法后面直接跟 3 个目录，依次跑 run1/run2/run3
    python test_five_days.py \\
        --td3per     output/TD3PER-run137 output/TD3PER-run139 output/TD3PER-run140 \\
        --td3prrbb   output/TD3PRRBB-run143 output/TD3PRRBB-run145 \\
        --td3prle    output/TD3PRLE-run167 output/TD3PRLE-run168 output/TD3PRLE-run169 \\
        --td3prrlrb  output/TD3PRRLRB-run164 output/TD3PRRLRB-run166 \\
        --td3perrb   output/TD3PERRB-run170 \\
        --td3leprrlrb output/TD3LEPRRLRB-run5 output/TD3LEPRRLRB-run23 \\
        --per        output/PER-run1 --prrbb output/PRRBB-run2 --prle output/PRLE-run3

    也可用 --models（同一 KEY 重复即为多组）：
        --models td3per=output/a td3per=output/b td3per=output/c
    不带任何模型参数 → 交互式输入（一个方法的多组路径用空格分隔）。
    其它：--days 5 --seed 42 --output output/test_five_days
         --projects td3per td3prrlrb        # 只测其中几个方法
         --max-runs 3                      # 每方法最多取前 3 组（0=不限制）
         --episode-length 120               # 只跑前 120 步（冒烟测试用）
         --viol-limit 3.0 --deg-cost 490
         --no-save-steps                    # 不写每步明细
         --by-day-sort profit|viol|excess   # 汇总 C（每天×每方法）的排序键（默认 profit）
         --no-tensorboard                   # 只写 CSV（默认两者都写）
         --list-projects

输出（CSV 与 TensorBoard 双份；TB 关掉时只有 CSV）
----
    <output>/five_day_days.csv             每天 × 每方法 × 每组模型（三项指标）
    <output>/five_day_summary.csv          每方法 × 每组模型的 5 天均值/标准差
    <output>/five_day_method_summary.csv   每方法跨「全部模型 × 全部天」mean±std ★主表
    <output>/five_day_by_day.csv           每天 × 每方法（跨模型均值 + 当天名次，按 --by-day-sort 排序）
    <output>/five_day_steps.csv            每步明细（可 --no-save-steps 关闭）
    <output>/tensorboard/                  TB 日志（可 --no-tensorboard 关闭）
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

import hsrl_soctarget as hst        # 下层 r2/c2 与规则基线的单一真源

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 路径：所有 *_main.py 里的相对路径（FFC_speed_noise_1.csv / test_scenarios.csv）
# 都依赖 CWD == 项目根，故脚本自身 chdir 到根目录。
# ----------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

# Windows 中文控制台默认 GBK(cp936)，个别数学符号（如 U+2212 减号）无法编码会
# 直接抛 UnicodeEncodeError；这里让无法编码的字符降级为 '?' 而不是中断运行。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors='replace')
    except Exception:
        pass

# ----------------------------------------------------------------------------
# 常量（与各项目 main 保持一致）
# ----------------------------------------------------------------------------
DT = 1.0 / 60.0                 # 步长 (小时)
MAX_E = 6.0                     # BESS 容量 (MWh)
NB_TURBINE = 5                  # 风机台数
EPISODE_LENGTH = 1440           # 1 day = 24h × 60min
FLUCT_LIMIT = 3.0               # 波动限值 (MW)
RATED_POWER = 10.0              # BESS 额定功率 (MW)
ETA_C = ETA_D = 0.98
BESS_DEG_RMB_PER_MWH = 490.0    # 放电退化成本 (元/MWh 放电量)
VIOL_TOL = 1e-6                 # 越限判定浮点容差
SOC_INIT = 3.0                  # 统一初始 SOC (MWh)
SOC_RAND_RANGE = (0.5, 5.5)     # td3 系训练时的随机 SOC 起点

DEFAULT_SEED = 42
DEFAULT_DAYS = 5
WIND_CSV = 'FFC_speed_noise_1.csv'

PROJECT_ORDER = ['per', 'prrbb', 'prle',
                 'td3per', 'td3prrbb', 'td3prle',
                 'td3prrlrb', 'td3perrb', 'td3leprrlrb',
                 'yawflor', 'aifflor']

# 每天的三项核心指标（CSV 列名 = TensorBoard tag 末级）
DAY_METRICS = ('viol_count', 'viol_excess_mw', 'profit_rmb')


# ----------------------------------------------------------------------------
# 项目注册表
# ----------------------------------------------------------------------------
_EMA = dict(ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01)
_EMA_L = dict(ema_lam=True, ema_alpha=0.02, ema_lam_lr=1.0)   # L 层: c2 口径 ÷100 后 η2 ×100
_TD3_H = dict(policy_delay=2, policy_noise=0.1, noise_clip=0.3,
              cost_critic_mode='max')
_TD3_L = dict(policy_delay=2, policy_noise=0.1, noise_clip=0.3,
              cost_critic_mode='max')

PROJECTS = {
    'per': dict(
        label='PER',
        agent=('PER_ddpg', 'SP_DDPG'),
        myfunc='myfunc',
        rule_bess=('rbb.rbb_rule_bess', 'RuleBasedBESS'),
        lower='rl', l_obs='baseline', h_obs='normalize',
        h_kwargs={**_EMA, 'lam_max': 20.0, 'lam_min': 3.0,
                  'record_update_stats': True},
        h_lam=5.0,
        l_kwargs={**_EMA_L, 'lam_max': 500.0, 'lam_min': 10.0},
        l_cost_limit=0.1,   # = 生产 agent_L.cost_limit (L 口径 ÷100)
        soc_random=False,
        ref='PER_main0.py',
    ),
    'prrbb': dict(
        label='PRRBB',
        agent=('PER_ddpg', 'SP_DDPG'),
        myfunc='prrbb.prrbb_myfunc',
        rule_bess=('prrbb.prrbb_rule_bess', 'RuleBasedBESS'),
        lower='rl', l_obs='baseline', h_obs='normalize',
        h_kwargs={**_EMA, 'lam_max': 20.0, 'lam_min': 3.0,
                  'record_update_stats': True},
        h_lam=5.0,
        l_kwargs={**_EMA_L, 'lam_max': 500.0, 'lam_min': 10.0},
        l_cost_limit=0.1,   # = 生产 agent_L.cost_limit (L 口径 ÷100)
        soc_random=False,
        ref='prrbb/prrbb_main.py',
    ),
    'prle': dict(
        label='PRLE',
        agent=('prle.prle_ddpg', 'PRLE_DDPG'),
        myfunc='prle.prle_myfunc',
        rule_bess=('prrbb.prrbb_rule_bess', 'RuleBasedBESS'),
        lower='rl', l_obs='distilled', h_obs='normalize',
        h_kwargs={**_EMA, 'lam_max': 20.0, 'lam_min': 3.0,
                  'record_update_stats': True},
        h_lam=5.0,
        l_kwargs={**_EMA_L, 'lam_max': 200.0, 'lam_min': 10.0},
        l_cost_limit=0.1,   # = 生产 agent_L.cost_limit (L 口径 ÷100)
        soc_random=False,
        ref='prle/prle_main.py',
    ),
    'td3per': dict(
        # 上层 = td3perrb 口径 (非分段奖励 + SOC 目标信号 + 相位观测);
        # 下层 = td3per 自己的 RL BESS, 5 维观测 = [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5],
        #        r2 = -1·min(clip_amount,20), c2 = +1·trackerr [MW] (到参考集合的距离; 口径见 hsrl_soctarget)
        label='TD3PER',
        residual=True,          # 下层 = 残余动作 (命令 = hst.rule_action + Δ)
        agent=('td3per.td3per_ddpg', 'SP_TD3'),
        myfunc='td3per.td3per_myfunc',
        rule_bess=('rbb.rbb_rule_bess', 'RuleBasedBESS'),
        lower='rl', l_obs='distilled', h_obs='normalize_H',
        h_kwargs={**_EMA, **_TD3_H, 'lam_max': 100.0, 'lam_min': 3.0,
                  'record_update_stats': True,
                  'wt_noise': 0.05, 'sig_noise_max': 0.2, 'sig_noise_min': 0.05},
        h_lam=5.0,
        l_kwargs={**_EMA_L, **_TD3_L, 'lam_max': 500.0, 'lam_min': 10.0},
        l_cost_limit=0.1,   # = 生产 agent_L.cost_limit (L 口径 ÷100)
        soc_random=True,
        ref='td3per/td3per_main.py',
    ),
    'td3prrbb': dict(
        label='TD3PRRBB',
        residual=True,          # 下层 = 残余动作 (命令 = hst.rule_action + Δ)
        agent=('td3prrbb.td3prrbb_ddpg', 'SP_TD3'),
        myfunc='td3prrbb.td3prrbb_myfunc',
        rule_bess=('td3prrbb.td3prrbb_rule_bess', 'RuleBasedBESS'),
        lower='rl', l_obs='distilled', h_obs='normalize_H',
        h_kwargs={**_EMA, **_TD3_H, 'lam_max': 50.0, 'lam_min': 3.0,
                  'record_update_stats': True},
        h_lam=5.0,
        l_kwargs={**_EMA_L, **_TD3_L, 'lam_max': 500.0, 'lam_min': 10.0},
        l_cost_limit=0.1,   # = 生产 agent_L.cost_limit (L 口径 ÷100)
        soc_random=True,
        ref='td3prrbb/td3prrbb_main.py',
    ),
    'td3prle': dict(
        label='TD3PRLE',
        residual=True,          # 下层 = 残余动作 (命令 = hst.rule_action + Δ)
        agent=('td3prle.td3prle_ddpg', 'SP_TD3_PRLE'),
        myfunc='td3prle.td3prle_myfunc',
        rule_bess=('td3prle.td3prle_rule_bess', 'RuleBasedBESS'),
        lower='rl', l_obs='distilled', h_obs='normalize_H',
        h_kwargs={**_EMA, **_TD3_H, 'lam_max': 500.0, 'lam_min': 50.0,
                  'record_update_stats': True},
        h_lam=100.0,
        l_kwargs={**_EMA_L, **_TD3_L, 'lam_max': 200.0, 'lam_min': 2.0},
        l_lam=100.0,
        l_cost_limit=0.1,   # = 生产 agent_L.cost_limit (L 口径 ÷100)
        soc_random=True,
        ref='td3prle/td3prle_main.py',
    ),
    'td3prrlrb': dict(
        label='TD3PRRLRB',
        agent=('td3prrlrb.td3prrlrb_ddpg', 'SP_TD3'),
        myfunc='td3prrlrb.td3prrlrb_myfunc',
        rule_bess=('td3prrlrb.td3prrlrb_rule_bess', 'RuleBasedBESS'),
        lower='rule', l_obs=None, h_obs='normalize_H',
        h_kwargs={**_EMA, **_TD3_H, 'lam_max': 100.0, 'lam_min': 3.0,
                  'record_update_stats': True,
                  'wt_noise': 0.05, 'sig_noise_max': 0.2, 'sig_noise_min': 0.05},
        h_lam=5.0,
        tracker=dict(soc_tol=0.0, eta_c=ETA_C, eta_d=ETA_D, time=DT,
                     rated_power=RATED_POWER, pb_cap=None),
        soc_random=True,
        ref='td3prrlrb/td3prrlrb_main.py',
    ),
    'td3perrb': dict(
        label='TD3PERRB',
        agent=('td3perrb.td3perrb_ddpg', 'SP_TD3'),
        myfunc='td3perrb.td3perrb_myfunc',
        rule_bess=('td3perrb.td3perrb_rule_bess', 'RuleBasedBESS'),
        lower='rule', l_obs=None, h_obs='normalize_H',
        h_kwargs={**_EMA, **_TD3_H, 'lam_max': 100.0, 'lam_min': 3.0,
                  'record_update_stats': True,
                  'wt_noise': 0.05, 'sig_noise_max': 0.2, 'sig_noise_min': 0.05},
        h_lam=5.0,
        tracker=dict(soc_tol=0.0, eta_c=ETA_C, eta_d=ETA_D, time=DT,
                     rated_power=RATED_POWER, pb_cap=None),
        soc_random=True,
        ref='td3perrb/td3perrb_main.py',
    ),
    'td3leprrlrb': dict(
        label='TD3LEPRRLRB',
        agent=('td3leprrlrb.td3leprrlrb_ddpg', 'SP_TD3_LEPRRLRB'),
        myfunc='td3leprrlrb.td3leprrlrb_myfunc',
        rule_bess=('td3leprrlrb.td3leprrlrb_rule_bess', 'RuleBasedBESS'),
        lower='rule', l_obs=None, h_obs='normalize_H',
        h_kwargs={**_EMA, **_TD3_H, 'lam_max': 100.0, 'lam_min': 3.0,
                  'record_update_stats': True},
        h_lam=5.0,
        tracker=dict(soc_tol=0.0, eta_c=ETA_C, eta_d=ETA_D, time=DT,
                     rated_power=RATED_POWER),
        soc_random=True,
        ref='td3leprrlrb/td3leprrlrb_main.py',
    ),
    # ------------------------------------------------------------------
    # 方法 10: YAWFLOR —— 偏航 + FLORIS 环境（第二个动作空间 / 第二个物理引擎）
    #   基线 = td3per，除"动作带 + 物理引擎"外逐字克隆（见 yawflor/README.md 与
    #   `agentutil/port_yawflor.py` 的替换表）。这里与 td3per 的差异只有三处：
    #     ① `env` / `env_physics`：FLORIS 后端（**不需要 MATLAB**，matlab=False）
    #     ② h_kwargs 的 `wt_noise` = 0.2（偏航满带 [-1,+1] 的 10%）
    #     ③ 动作带由 `env.nominal()`（=0）与 myfunc 的 CheckAction([-1,+1]) 决定
    #   ⚠ 跨引擎的 profit **不可直接比较**（FLORIS 绝对电平是 WFSim 的 1.74×，
    #     见 yawflor/README.md §3）⇒ 只同轨比较或比归一化指标（MWh / 超限率）。
    # ------------------------------------------------------------------
    'yawflor': dict(
        label='YAWFLOR',
        residual=True,          # 下层 = 残余动作 (命令 = hst.rule_action + Δ)
        agent=('yawflor.yawflor_ddpg', 'SP_TD3'),
        myfunc='yawflor.yawflor_myfunc',
        rule_bess=('rbb.rbb_rule_bess', 'RuleBasedBESS'),
        lower='rl', l_obs='distilled', h_obs='normalize_H',
        env=('yawflor.yawflor_env', 'WindBatteryYaw'),
        env_physics=('yawflor.yawflor_env', 'FlorisPhysicsYaw', {}),
        matlab=False,           # 纯 FLORIS：评测时不启 MATLAB 引擎
        h_kwargs={**_EMA, **_TD3_H, 'lam_max': 100.0, 'lam_min': 3.0,
                  'record_update_stats': True,
                  'wt_noise': 0.2, 'sig_noise_max': 0.2, 'sig_noise_min': 0.05},
        h_lam=5.0,
        l_kwargs={**_EMA_L, **_TD3_L, 'lam_max': 500.0, 'lam_min': 10.0},
        l_cost_limit=0.1,   # = 生产 agent_L.cost_limit (L 口径 ÷100)
        soc_random=True,
        ref='yawflor/yawflor_main.py',
    ),
    # ------------------------------------------------------------------
    # 方法 11: AIFFLOR —— **跨引擎迁移评测**（FLORIS 训练的策略 → WFSim 引擎）
    #   `aifflor` 自己的物理引擎是 FLORIS（纯 Python，见 aifflor/README.md），
    #   但**本条目刻意不给 `env` / `env_physics`** ⇒ 走默认 `env.WindBattery`
    #   + MATLAB `WFSim_port`，即"把 FLORIS 里学到的策略搬到 WFSim 上跑"。
    #   因此 label 带 `@WFSIM` 后缀，避免与"同引擎自评"的读数混淆。
    #   为什么可零口径改动：`aifflor/aifflor_ddpg.py` 与 `aifflor/aifflor_myfunc.py`
    #   是 `td3per` 对应文件的**逐字复制**（Compare-Object = 0 行差异）⇒ agent/myfunc
    #   只换模块名，h/l kwargs、动作语义、观测构造与 td3per 条目完全相同。
    #   ⚠ 跨引擎的 profit 绝对值不可直接比较（电平差 + 状态相关分量，
    #     见 aifflor/README.md §2 / §3.7）⇒ 只与 WFSim 里的 td3per 同轨比较。
    # ------------------------------------------------------------------
    'aifflor': dict(
        label='AIFFLOR@WFSIM',
        residual=True,          # 下层 = 残余动作 (命令 = hst.rule_action + Δ)
        agent=('aifflor.aifflor_ddpg', 'SP_TD3'),
        myfunc='aifflor.aifflor_myfunc',
        rule_bess=('rbb.rbb_rule_bess', 'RuleBasedBESS'),
        lower='rl', l_obs='distilled', h_obs='normalize_H',
        # 引擎 = WFSim（**不**注册 env/env_physics）⇒ matlab=True，与 td3per 同一条路径
        h_kwargs={**_EMA, **_TD3_H, 'lam_max': 100.0, 'lam_min': 3.0,
                  'record_update_stats': True,
                  'wt_noise': 0.05, 'sig_noise_max': 0.2, 'sig_noise_min': 0.05},
        h_lam=5.0,
        l_kwargs={**_EMA_L, **_TD3_L, 'lam_max': 500.0, 'lam_min': 10.0},
        l_cost_limit=0.1,   # = 生产 agent_L.cost_limit (L 口径 ÷100)
        soc_random=True,
        ref='aifflor/aifflor_main.py',
    ),
}

# 相位编码: PHASE_DIM = {0:0, 1:2, 2:5}  (两项 RLRB 项目共用同一约定)
_DIM_TO_MODE = {0: 0, 2: 1, 5: 2}


def _resolve(spec_pair):
    """('pkg.mod', 'Cls') → Cls"""
    mod_name, cls_name = spec_pair
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name), mod


# ----------------------------------------------------------------------------
# MATLAB 引擎（env.PowerOutput → engine.WFSim_port）
# 注意：myfunc.py 在 **模块顶层** 就执行 `engine = matlab.engine.start_matlab()`
# （per / td3per 的 myfunc 就是它）。为了不白起第二个引擎，这里先把共享引擎
# 起好，再把 `matlab.engine.start_matlab` 临时指向"返回同一个引擎"，这样
# myfunc 顶层那次调用会复用共享引擎；本脚本结束时统一 quit 一次即可。
# ----------------------------------------------------------------------------
_ENGINE = None


def ensure_engine():
    global _ENGINE
    if _ENGINE is None:
        import matlab.engine
        print('  启动 MATLAB 引擎 ...')
        t0 = time.time()
        _ENGINE = matlab.engine.start_matlab()
        print('  MATLAB 引擎就绪 ({:.1f}s)'.format(time.time() - t0))
        # 让后续 `import myfunc`（顶层 start_matlab）复用同一引擎
        matlab.engine.start_matlab = lambda *a, **kw: _ENGINE
    return _ENGINE


def quit_engine():
    global _ENGINE
    if _ENGINE is not None:
        try:
            _ENGINE.quit()
        except Exception:
            pass
        _ENGINE = None


def _detect_nb_states(model_dir, nb_actions=6):
    """从 actor_<nb_actions>.pkl 的第一层权重反推观测维数。"""
    import torch
    path = os.path.join(model_dir, 'actor_{}.pkl'.format(nb_actions))
    if not os.path.isfile(path):
        return None
    sd = torch.load(path, map_location='cpu')
    if not isinstance(sd, dict):
        return None
    for key in ('fc1.weight', '0.weight'):
        if key in sd and hasattr(sd[key], 'shape') and len(sd[key].shape) == 2:
            return int(sd[key].shape[1])
    for key, val in sd.items():
        if hasattr(val, 'shape') and len(val.shape) == 2:
            return int(val.shape[1])
    return None


def _detect_hidden(model_dir, nb_actions=6):
    """从 actor_<nb_actions>.pkl 的 fc1/fc2 权重反推 H 网络 (hidden1, hidden2)。

    为什么必须探测而不能写死: 一次评测里同时有
      * 六个 TD3 方法 —— 网络宽度随配置变过 (16/100、64/128 都出现过, 见 README 口径表);
      * 3 个改造前项目 (per/prrbb/prle) —— 16/100。
    rate/bsize/rmsize 只影响训练, 纯前向评测下不生效; **只有隐层维度必须与权重一致**。
    """
    import torch
    path = os.path.join(model_dir, 'actor_{}.pkl'.format(nb_actions))
    if not os.path.isfile(path):
        return None
    sd = torch.load(path, map_location='cpu')
    if not isinstance(sd, dict) or 'fc1.weight' not in sd or 'fc2.weight' not in sd:
        return None
    return int(sd['fc1.weight'].shape[0]), int(sd['fc2.weight'].shape[0])


def _build_args(device, seed, hidden=None):
    """构造 agent 构造函数需要的 args 命名空间。

    hidden : (hidden1, hidden2) or None
        H/L 隐层宽度, **按模型目录探测后覆写**(见 _detect_hidden); None 时用下面
        的默认值 (= 六个 TD3 main 的当前默认)。
    """
    ns = argparse.Namespace()
    # 默认值 = 六个 TD3 main 的当前默认; 真实维度在下面按 checkpoint 覆写
    ns.hidden1 = 16
    ns.hidden2 = 100
    ns.rate = 0.01
    ns.prate = 0.0001
    ns.warmup = 200
    ns.discount = 0.98
    ns.l_discount = 0.98
    ns.bsize = 32          # 纯前向评测, 不更新网络; 保留小值即可
    ns.rmsize = 20000
    if hidden:
        ns.hidden1, ns.hidden2 = int(hidden[0]), int(hidden[1])
    ns.window_length = 1
    ns.tau = 0.001
    ns.ou_theta = 0.15
    ns.ou_sigma = 0.2
    ns.ou_mu = 0.0
    ns.validate_episodes = 20
    ns.max_episode_length = EPISODE_LENGTH
    ns.validate_steps = 2000
    ns.output = 'output'
    ns.debug = False
    ns.init_w = 0.003
    ns.train_iter_H = 200000
    ns.epsilon = 50000
    ns.seed = seed
    ns.resume = 'default'
    ns.cuda = bool(device.type == 'cuda')
    ns.lam = 3.0
    ns.lr_lam = 0.1
    ns.cost_single_limit = 1000.0
    ns.PER_alpha = 0.6
    ns.PER_beta = 0.4
    ns.PER_epsilon = 0.01
    ns.beta_increment_per_sampling = 3e-5
    ns.nb_turbine = NB_TURBINE
    ns.rbb_fluctuation_limit = FLUCT_LIMIT
    ns.rbb_rated_power = RATED_POWER
    ns.policy_delay = 2
    ns.policy_noise = 0.1
    ns.noise_clip = 0.3
    ns.l_policy_delay = 2
    ns.l_policy_noise = 0.1
    ns.cost_critic_mode = 'max'
    ns.wt_noise = 0.05
    ns.sig_noise_max = 0.2
    ns.sig_noise_min = 0.05
    ns.soc_tol = 0.1
    ns.tracker_soc_tol = 0.0
    ns.tracker_dead_zone = 0.0
    ns.pb_cap = -1.0
    ns.phase_feat = 1
    ns.wt_bias_init = 0
    return ns


# ----------------------------------------------------------------------------
# 单项目执行器
# ----------------------------------------------------------------------------
class ProjectRunner:
    """加载一个项目的模型，并按它自己训练时的推理路径跑完整 episode。"""

    def __init__(self, key, model_dir, device, args, engine, phase_feat=None,
                 soc_init='uniform', rng=None, verbose=True):
        self.key = key
        self.spec = PROJECTS[key]
        self.model_dir = model_dir
        self.device = device
        self.args = args
        self.soc_init = soc_init
        self.episode_length = int(EPISODE_LENGTH)
        self.rng = rng if rng is not None else np.random.RandomState(DEFAULT_SEED)
        self.verbose = verbose

        myfunc_mod_name = self.spec['myfunc']
        myf = importlib.import_module(myfunc_mod_name)
        self.myf = myf
        self.check_action = myf.CheckAction
        # myfunc.py 顶层会 start_matlab()（已被 ensure_engine 指向共享引擎）
        engine = getattr(myf, 'engine', None) or engine

        # ---- H 观测维数 & 相位模式 ----
        nb_H = _detect_nb_states(model_dir, 6)
        if nb_H is None:
            raise FileNotFoundError(
                '缺少 {}/actor_6.pkl'.format(model_dir))
        self.nb_states_H = nb_H
        if self.spec['h_obs'] == 'normalize_H':
            d = nb_H - 4
            if phase_feat is not None:
                self.phase_mode = int(phase_feat)
            elif d in _DIM_TO_MODE:
                self.phase_mode = _DIM_TO_MODE[d]
            else:
                raise ValueError(
                    '{}: actor_6.pkl 观测维数 {} 无法解释为 4+相位(0/2/5)，'
                    '请用 --phase-feat 显式指定'.format(key, nb_H))
            expect = 4 + {0: 0, 1: 2, 2: 5}[self.phase_mode]
            if expect != nb_H:
                raise ValueError(
                    '{}: --phase-feat={} 对应 {} 维, 但权重是 {} 维'
                    .format(key, self.phase_mode, expect, nb_H))
        else:
            self.phase_mode = 0
            if nb_H != 4:
                raise ValueError(
                    '{}: H 观测维数应为 4, 权重却是 {} 维'.format(key, nb_H))

        # ---- 观测构造函数 ----
        self.h_obs_fn = self._make_h_obs_fn(myf)
        self.l_obs_fn = getattr(myf, 'build_L_obs_distilled', None) \
            if self.spec['l_obs'] == 'distilled' else None

        # ---- RuleBasedBESS ----
        RBC, _ = _resolve(self.spec['rule_bess'])
        self.rbb = RBC(max_E=MAX_E, min_E_factor=0.0, rated_power=RATED_POWER,
                       eta_c=ETA_C, eta_d=ETA_D, time=DT,
                       fluctuation_limit=FLUCT_LIMIT)

        # ---- 下层规则控制器（仅 RLRB 两项）----
        self.tracker = None
        if self.spec['lower'] == 'rule':
            _, rb_mod = _resolve(self.spec['rule_bess'])
            tracker_kwargs = dict(self.spec['tracker'])
            self.tracker = rb_mod.SocTracker(**tracker_kwargs)
            _, myf_mod = _resolve((myfunc_mod_name, 'Normalize'))
            self.soc_target_from_signal = myf_mod.soc_target_from_signal

        # ---- 智能体 ----
        AgentCls, _ = _resolve(self.spec['agent'])
        agent_H = AgentCls(nb_states=nb_H, nb_actions=6, args=args,
                           **self.spec['h_kwargs'])
        if self.spec.get('h_lam') is not None:
            agent_H.lam = self.spec['h_lam']
        agent_H.is_training = False
        agent_H.load_weights(model_dir)

        agent_L = None
        if self.spec['lower'] == 'rl':
            agent_L = AgentCls(nb_states=5, nb_actions=1, args=args,
                               **self.spec['l_kwargs'])
            if self.spec.get('l_lam') is not None:
                agent_L.lam = self.spec['l_lam']
            if self.spec.get('l_cost_limit') is not None:
                agent_L.cost_limit = self.spec['l_cost_limit']
            agent_L.is_training = False
            agent_L.load_weights(model_dir)
        self.agent_H, self.agent_L = agent_H, agent_L

        # ---- 环境（每项目一个，避免状态串味）----
        # 环境类**按项目注册表解析**（默认仍是 `env.WindBattery` ⇒ 9 个既有方法
        # 的调用点与行为逐位不变）；`env_physics` 非空时走新物理后端
        # （yawflor = `FlorisPhysicsYaw`，FLORIS 引擎，不需要 MATLAB）。
        env_mod_name, env_cls_name = self.spec.get('env', ('env', 'WindBattery'))
        env_cls, _ = _resolve((env_mod_name, env_cls_name))
        phy_spec = self.spec.get('env_physics')
        if phy_spec:
            phy_cls, _ = _resolve((phy_spec[0], phy_spec[1]))
            physics = phy_cls(nt=NB_TURBINE, **(phy_spec[2] or {}))
            self.env = env_cls(MAX_E, NB_TURBINE, False, engine, physics=physics)
        else:
            self.env = env_cls(MAX_E, NB_TURBINE, False, engine)

        if verbose:
            print('    [{}] 权重 OK: H nb_states={} phase_mode={} '
                  'L={}'.format(key, nb_H, self.phase_mode,
                                '规则 SocTracker' if agent_L is None else 'RL 5维'))

    # -- 观测构造 ------------------------------------------------------------
    def _make_h_obs_fn(self, myf):
        if self.spec['h_obs'] == 'normalize':
            nrm = myf.Normalize
            return lambda obs, t: nrm(obs)
        nh = myf.Normalize_H
        params = list(inspect.signature(nh).parameters)
        mode = self.phase_mode
        if 'minute' in params:
            # td3prrlrb: Normalize_H(obs, minute, mode, table)
            table = myf.phase_daily_table() if mode >= 2 else None
            return lambda obs, t: nh(obs, myf.phase_minute_of_day(t), mode, table)
        # td3leprrlrb: Normalize_H(obs, phase_minute, phase_feat, max_E)
        rep = int(getattr(myf, 'REP_MINUTE', EPISODE_LENGTH))
        return lambda obs, t: nh(obs, int(t) % rep, mode)

    # -- 一个 episode -------------------------------------------------------
    def run_day(self, wind_series, day_idx, wind_iloc, wind_col):
        env = self.env
        agent_H, agent_L = self.agent_H, self.agent_L
        CheckAction = self.check_action

        # ---- episode 起点：复刻 env.reset()，再把风速日换成抽中的那一天 ----
        obs = deepcopy(env.reset())

        env.wind_day = wind_series            # pandas Series, index 0..1439
        env.count_wind = 0
        env.u = float(wind_series.iloc[0])
        obs[1] = env.u
        # 标称动作 = `env.nominal()`（旧环境 = 5 维轴向因子 0.3333；yawflor = 偏航 n=0）。
        # 旧路径 `np.ones(nt)*0.3333` 与 `WindBattery.nominal_action_value=0.3333` **逐位相同**。
        a_temp = env.nominal()
        temp = env.PowerOutput(env.u, a_temp)
        env.P_G_prev = temp                   # 波动基准 = 当日首分钟出力
        obs[-1] = temp

        # ---- 初始 SOC ----
        if self.soc_init == 'train' and self.spec['soc_random']:
            soc0 = float(self.rng.uniform(*SOC_RAND_RANGE))
        else:
            soc0 = SOC_INIT
        env.E = soc0
        obs[2] = soc0

        norm_obs = self.h_obs_fn(obs, 0)
        agent_H.reset(norm_obs)

        # ---- 累计量 ----
        viol_count = 0
        viol_excess = 0.0
        profit = 0.0
        soc_sum = 0.0
        soc_end = soc0
        charge_mwh = 0.0
        discharge_mwh = 0.0
        max_pf = 0.0
        step_rows = []

        action_H = None
        pw_tru = None
        soc_target = None

        n_steps = self.episode_length

        for t in range(n_steps):
            # ---- 上层 H：每 5 步一次（确定性，无探索噪声）----
            if t % 5 == 0:
                original_action_H = agent_H.select_action(norm_obs)
                action_H, _pen_H = CheckAction(original_action_H)
                pw_tru = env.PowerOutput(obs[1], action_H)
                if self.spec['lower'] == 'rule':
                    soc_target = self.soc_target_from_signal(action_H[5])

            ptur = float(pw_tru)
            pg_prev = float(obs[-1])
            E_now = float(obs[2])

            # 可行窗口（与 pb 无关，可预计算）
            _, rbb_info_obs = self.rbb.compute_pbrule(
                pb_original=0.0, ptur=ptur, pg_prev=pg_prev, E=E_now)

            # ---- 下层 L ----
            if self.spec['lower'] == 'rl':
                if self.l_obs_fn is not None:      # 5 维规则原配料 (SOC 目标支)
                    norm_obs_L = self.l_obs_fn(obs, pw_tru, action_H, rbb_info_obs, 5 - (t % 5))
                else:                              # 基线 5 维
                    norm_obs_L = np.append(norm_obs, action_H[-1])
                if t % 5 == 0:
                    agent_L.reset(norm_obs_L)
                original_action_L = agent_L.select_action(norm_obs_L)
                if self.spec.get('residual', False):
                    # TD3 全系列: actor 输出 **Δ**, 命令 = 规则基线 + Δ (单一真源 hst.rule_action)
                    pb_rule = hst.rule_action(
                        hst.soc_target_from_signal(action_H[5]), E_now,
                        rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'], 5 - (t % 5),
                        eta_c=self.rbb.eta_c, eta_d=self.rbb.eta_d,
                        time=self.rbb.time, rated_power=self.rbb.rated_power)
                    applied_action_L, _pen_L = CheckAction(
                        np.array([float(pb_rule) + float(original_action_L[0])]))
                else:
                    applied_action_L, _pen_L = CheckAction(original_action_L)
            else:                                  # RLRB：规则 SocTracker
                pb_cmd = self.tracker.act(E_now, soc_target,
                                          rbb_info_obs['pbrulemin'],
                                          rbb_info_obs['pbrulemax'])
                applied_action_L = np.array([pb_cmd])

            action = np.concatenate((action_H, applied_action_L))

            # ---- RBB 硬裁剪 ----
            pb_original = float(action[-1])
            pbrule, rbb_info = self.rbb.compute_pbrule(
                pb_original=pb_original, ptur=ptur, pg_prev=pg_prev, E=E_now)
            action_rbb = action.copy()
            action_rbb[-1] = pbrule

            # ---- 环境一步（动态电价，与训练同一个 CalPrice）----
            pw_tru2, obs2, action_L_env, _penalty_L, _done, _done_L, info = \
                env.step(action_rbb[:6], action_rbb[6:], pw_tru)

            # ---- 指标 ----
            P_F = float(info['P_F'])
            pb_used = float(pbrule)
            profit_step = (float(obs2[-1]) * float(obs[0])
                           - BESS_DEG_RMB_PER_MWH * max(-pb_used, 0.0)) * DT
            excess = max(P_F - FLUCT_LIMIT, 0.0) if P_F > FLUCT_LIMIT + VIOL_TOL else 0.0
            if excess > 0.0:
                viol_count += 1
                viol_excess += excess
            profit += profit_step
            soc_sum += float(obs2[2])
            soc_end = float(obs2[2])
            if pb_used > 0:
                charge_mwh += pb_used * ETA_C * DT
            else:
                discharge_mwh += -pb_used * DT / ETA_D
            max_pf = max(max_pf, P_F)

            step_rows.append({
                'project': self.spec['label'],
                'day_idx': day_idx,
                'wind_iloc': wind_iloc,
                'step': t,
                'price': float(obs[0]),
                'wind': float(obs[1]),
                'P_G': float(obs2[-1]),
                'P_F': P_F,
                'viol': int(excess > 0.0),
                'excess_mw': excess,
                'pb_cmd': pb_original,
                'pbrule': pb_used,
                'SOC': float(obs2[2]),
                'profit_step': profit_step,
            })

            # ---- 状态推进 ----
            norm_obs = self.h_obs_fn(obs2, t + 1)
            obs = deepcopy(obs2)
            pw_tru = pw_tru2

        day = {
            'project': self.spec['label'],
            'project_key': self.key,
            'day_idx': day_idx,
            'wind_iloc': wind_iloc,
            'wind_col': wind_col,
            'wind_mean': float(wind_series.mean()),
            'wind_min': float(wind_series.min()),
            'wind_max': float(wind_series.max()),
            'viol_count': viol_count,
            'viol_excess_mw': viol_excess,
            'viol_excess_mwh': viol_excess * DT,
            'profit_rmb': profit,
            'soc_init': soc0,
            'soc_mean': soc_sum / n_steps,
            'soc_end': soc_end,
            'charge_mwh': charge_mwh,
            'discharge_mwh': discharge_mwh,
            'max_PF': max_pf,
            'n_steps': n_steps,
        }
        return day, step_rows


# ----------------------------------------------------------------------------
# 风速日抽取
# ----------------------------------------------------------------------------
def load_wind_pool():
    df = pd.read_csv(os.path.join(_ROOT, WIND_CSV))
    # iloc[:, 0] 是 pandas 写出的无名索引列，风速数据在 iloc[:, 1..]
    n_cols = df.shape[1]
    pool = list(range(1, n_cols))
    return df, pool


def sample_days(df, pool, n_days, seed):
    rng = np.random.RandomState(seed)
    if n_days > len(pool):
        raise ValueError('days={} 超过风速池大小 {}'.format(n_days, len(pool)))
    positions = rng.choice(pool, size=n_days, replace=False)
    days = []
    for pos in positions:
        series = df.iloc[:, pos].reset_index(drop=True)
        days.append({
            'iloc': int(pos),
            'col': str(df.columns[pos]),
            'series': series,
        })
    return days


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='五天随机风速测试（单文件，兼容 9 个项目，每方法可多组模型）',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in PROJECT_ORDER:
        p.add_argument('--' + key, nargs='+', default=None, metavar='MODEL_DIR',
                       help='{} 模型目录；给多个 = 多组模型（如 3 个 run）: '
                            '--{} output/a output/b output/c'
                            .format(PROJECTS[key]['label'], key))
    p.add_argument('--models', nargs='*', default=None, metavar='KEY=PATH',
                   help='另一种写法，同一 KEY 可重复以给多组模型: '
                        'td3per=output/a td3per=output/b ...')
    p.add_argument('--max-runs', type=int, default=0,
                   help='每个方法最多取前 N 组模型（0 = 不限制，默认 0）')
    p.add_argument('--projects', nargs='*', default=None, choices=PROJECT_ORDER,
                   help='只测这些方法（默认：所有给了模型路径的方法）')
    p.add_argument('--days', type=int, default=DEFAULT_DAYS,
                   help='抽取天数（默认 5，每天 1 个 episode）')
    p.add_argument('--episode-length', type=int, default=EPISODE_LENGTH,
                   help='每个 episode 的步数（默认 1440 = 完整一天；'
                        '调小仅用于冒烟测试/快速抽查，指标将只统计这些步）')
    p.add_argument('--seed', type=int, default=DEFAULT_SEED,
                   help='抽天随机种子（默认 42）')
    p.add_argument('--output', type=str, default=os.path.join('output', 'test_five_days'),
                   help='输出目录（默认 output/test_five_days）')
    p.add_argument('--viol-limit', type=float, default=FLUCT_LIMIT,
                   help='波动越限阈值 MW（默认 3.0）')
    p.add_argument('--deg-cost', type=float, default=BESS_DEG_RMB_PER_MWH,
                   help='BESS 放电退化成本 元/MWh（默认 490）')
    p.add_argument('--soc-init', choices=['uniform', 'train'], default='uniform',
                   help='初始 SOC：uniform=统一 3.0（默认）；train=镜像各项目训练重置')
    p.add_argument('--phase-feat', type=int, default=None, choices=[0, 1, 2],
                   help='RLRB 项目的相位特征模式（默认从 actor_6.pkl 自动探测）')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--no-save-steps', action='store_true',
                   help='不写每步明细 CSV')
    p.add_argument('--by-day-sort', choices=['profit', 'viol', 'excess'],
                   default='profit',
                   help='“汇总 C：每天 × 每方法”表内的排序键：'
                        'profit=利润/天降序(高者优) | viol=越限次数升序 | '
                        'excess=越限总量升序（默认 profit）')
    p.add_argument('--no-tensorboard', action='store_true',
                   help='不写 TensorBoard（默认 CSV 与 TB 双份保存）')
    p.add_argument('--list-projects', action='store_true',
                   help='列出支持的方法后退出')
    return p.parse_args()


def resolve_model_paths(args):
    """收集模型路径 → OrderedDict{key: [abs_dir, ...]}（每方法可多组模型）。

    给路径的三种方式（可混用，同一方法的多组按给出顺序即为 run 顺序）：
      1) --td3per a b c        （每方法可给多个目录）
      2) --models td3per=a td3per=b ...（同一 KEY 重复即为多组）
      3) 不带参数 → 交互式输入（多个路径用空格分隔）
    """
    paths = {}
    for key in PROJECT_ORDER:
        val = getattr(args, key, None)
        if val:
            paths.setdefault(key, []).extend(val)

    if args.models:
        for item in args.models:
            if '=' not in item:
                raise SystemExit('--models 需要 KEY=PATH 形式，收到: {}'.format(item))
            k, v = item.split('=', 1)
            k = k.strip()
            if k not in PROJECTS:
                raise SystemExit('未知方法 key: {}（可选 {}）'
                                 .format(k, '/'.join(PROJECT_ORDER)))
            paths.setdefault(k, []).append(v)

    if args.projects:
        paths = {k: v for k, v in paths.items() if k in args.projects}

    # 无任何模型参数 → 交互式输入
    if not paths:
        print('=' * 74)
        print('  五天随机风速测试  未检测到模型参数，进入交互式输入')
        print('  （直接回车跳过该方法；一个方法要跑多组模型就一次给多个路径）')
        print('=' * 74)
        for key in PROJECT_ORDER:
            spec = PROJECTS[key]
            try:
                ans = input('  {:>13s} ({:>9s}) 模型路径(可多个, 空格分隔): '
                            .format(key, spec['label'])).strip().strip('"')
            except EOFError:
                ans = ''
            if ans:
                paths[key] = [a for a in ans.replace(';', ' ').split() if a]

    if not paths:
        raise SystemExit('未提供任何模型路径，退出。')

    # ---- 逐路径校验 / 去重 / 截断 ----
    for k in list(paths):
        cleaned = []
        for raw in paths[k]:
            d = os.path.abspath(str(raw).strip().strip('"'))
            if not os.path.isdir(d):
                print('[警告] {} 的路径不是目录，跳过: {}'.format(k, d))
                continue
            if not os.path.isfile(os.path.join(d, 'actor_6.pkl')):
                print('[警告] {} 目录下没有 actor_6.pkl，跳过: {}'.format(k, d))
                continue
            if d in cleaned:
                print('[警告] {} 重复路径，已忽略: {}'.format(k, d))
                continue
            cleaned.append(d)
        if args.max_runs and len(cleaned) > args.max_runs:
            print('[提示] {} 给了 {} 组模型，--max-runs={} → 只取前 {} 组'
                  .format(k, len(cleaned), args.max_runs, args.max_runs))
            cleaned = cleaned[:args.max_runs]
        if cleaned:
            paths[k] = cleaned
        else:
            del paths[k]

    if not paths:
        raise SystemExit('所有模型路径都无效，退出。')

    ordered = [k for k in PROJECT_ORDER if k in paths]
    return {k: paths[k] for k in ordered}


def _agg_days(df_days, by):
    """按 by 分组聚合三项指标（+辅助量），供两级汇总复用。"""
    return (df_days.groupby(by, as_index=False)
            .agg(samples=('day_idx', 'count'),
                 runs=('run', 'nunique'),
                 viol_count_mean=('viol_count', 'mean'),
                 viol_count_std=('viol_count', 'std'),
                 viol_excess_mw_mean=('viol_excess_mw', 'mean'),
                 viol_excess_mw_std=('viol_excess_mw', 'std'),
                 profit_rmb_mean=('profit_rmb', 'mean'),
                 profit_rmb_std=('profit_rmb', 'std'),
                 profit_rmb_sum=('profit_rmb', 'sum'),
                 soc_mean=('soc_mean', 'mean'),
                 discharge_mwh_mean=('discharge_mwh', 'mean')))


def print_run_summary(df_run, keys):
    """每方法 × 每 run（每组模型）的 5 天均值。"""
    print()
    print('=' * 108)
    print('  汇总 A：每方法 × 每组模型（5 天均值）')
    print('=' * 108)
    print('  {:>12s} {:>4s}  {:<22s} {:>6s} {:>11s} {:>13s} {:>14s}'
          .format('方法', 'run', '模型', '天数', '越限次数/天', '越限总量/天', '利润/天(元)'))
    print('  ' + '-' * 102)
    for key in keys:
        label = PROJECTS[key]['label']
        sub = df_run[df_run['project'] == label]
        for _, r in sub.iterrows():
            print('  {:>12s} {:>4d}  {:<22s} {:>6d} {:>11.2f} {:>13.4f} {:>14.2f}'
                  .format(label, int(r['run']), str(r['model'])[:22],
                          int(r['samples']), r['viol_count_mean'],
                          r['viol_excess_mw_mean'], r['profit_rmb_mean']))


def print_method_summary(df_m, keys):
    """每方法跨「所有 run × 所有天」的 mean ± std（论文表口径）。"""
    print()
    print('=' * 108)
    print('  汇总 B：每方法（跨全部模型 × 全部天，mean ± std）★ 主表')
    print('=' * 108)
    print('  {:>12s} {:>5s} {:>7s}  {:>19s} {:>21s} {:>22s}'
          .format('方法', '模型数', '样本数', '越限次数/天', '越限总量/天(MW)', '利润/天(元)'))
    print('  ' + '-' * 102)
    for key in keys:
        label = PROJECTS[key]['label']
        sub = df_m[df_m['project'] == label]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        print('  {:>12s} {:>5d} {:>7d}  {:>8.2f} ± {:<8.2f} {:>9.4f} ± {:<9.4f} '
              '{:>10.2f} ± {:<10.2f}'
              .format(label, int(r['runs']), int(r['samples']),
                      r['viol_count_mean'], _std0(r['viol_count_std']),
                      r['viol_excess_mw_mean'], _std0(r['viol_excess_mw_std']),
                      r['profit_rmb_mean'], _std0(r['profit_rmb_std'])))
    print('=' * 108)
    print('  说明: 越限次数 = Σ 1[|ΔP_G| > {} MW]; 越限总量 = Σ max(|ΔP_G| - {}, 0) [MW]'
          .format(FLUCT_LIMIT, FLUCT_LIMIT))
    print('        利润 = Σ (P_G·pr - {}·max(-pb,0))·Δt [元], Δt = 1/60 h'
          .format(BESS_DEG_RMB_PER_MWH))
    print('        std 的样本口径 = 所有 (模型 × 天)，样本标准差 ddof=1；单样本时显示 0.00')


def _std0(v):
    """std 为空(nan)时显示 0.00，避免打印 nan。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f


# 按天排序键 → (排序列, 升序?, 显示名)
_BY_DAY_SORT = {
    'profit': ('profit_rmb_mean', False, '利润/天 ↓（高者优）'),
    'viol':   ('viol_count_mean', True,  '越限次数/天 ↑（低者优）'),
    'excess': ('viol_excess_mw_mean', True, '越限总量/天 ↑（低者优）'),
}


def build_by_day_table(df_days, keys, sort_by='profit'):
    """按【天 × 方法】排表：每行 = 该方法在当天的全部模型均值。

    与"汇总 B"（跨全部模型×全部天）不同，本表把"天"作为独立区组展示，
    因此能看到**同一天内方法之间的直接对比**（天间的巨大差异被隔离在块之间），
    并给出当天内的名次。

    返回列
    ------
    day_idx, wind_iloc, wind_mean               该天标识（第几天 / 风速列 / 日均风速）
    project                                     方法 label
    n_models, samples                           当天参与的模型组数 / 样本数
    viol_count_mean/_std                        跨模型的越限次数（mean ± std, ddof=1）
    viol_excess_mw_mean/_std                    跨模型的越限总量
    profit_rmb_mean/_std                        跨模型的利润
    soc_end_mean, discharge_mwh_mean            辅助量（末 SOC / 放电量）
    rank_profit / rank_viol / rank_excess       当天内名次（1 = 最好），三种口径都给
    """
    want = [PROJECTS[k]['label'] for k in keys]
    sub = df_days[df_days['project'].isin(want)]
    # 只聚合实际存在的列（对缺少辅助列的旧 CSV / 部分数据也鲁棒），
    # 缺列在结果里补 NaN，保持 CSV 列结构稳定。
    spec = [('n_models', 'run', 'nunique'),
            ('samples', 'run', 'size'),
            ('wind_iloc', 'wind_iloc', 'first'),
            ('wind_mean', 'wind_mean', 'mean'),
            ('viol_count_mean', 'viol_count', 'mean'),
            ('viol_count_std', 'viol_count', 'std'),
            ('viol_excess_mw_mean', 'viol_excess_mw', 'mean'),
            ('viol_excess_mw_std', 'viol_excess_mw', 'std'),
            ('profit_rmb_mean', 'profit_rmb', 'mean'),
            ('profit_rmb_std', 'profit_rmb', 'std'),
            ('soc_end_mean', 'soc_end', 'mean'),
            ('discharge_mwh_mean', 'discharge_mwh', 'mean')]
    agg = {out: (src, fn) for out, src, fn in spec if src in sub.columns}
    g = sub.groupby(['day_idx', 'project'], as_index=False).agg(**agg)
    for out, src, fn in spec:
        if out not in g.columns:
            g[out] = np.nan
    # 三种口径的名次（1 = 最好）
    g['rank_profit'] = (g.groupby('day_idx')['profit_rmb_mean']
                        .rank(ascending=False, method='min').astype(int))
    g['rank_viol'] = (g.groupby('day_idx')['viol_count_mean']
                      .rank(ascending=True, method='min').astype(int))
    g['rank_excess'] = (g.groupby('day_idx')['viol_excess_mw_mean']
                        .rank(ascending=True, method='min').astype(int))
    col, asc, _ = _BY_DAY_SORT[sort_by]
    # 主键排序；并列时再按利润降序、方法名，保证输出稳定可复现
    g = g.sort_values(['day_idx', col, 'profit_rmb_mean', 'project'],
                      ascending=[True, asc, False, True]).reset_index(drop=True)
    return g


def print_by_day_summary(df_bd, keys, sort_by='profit'):
    """打印【汇总 C】：每天一个块，块内按表现排序打印各方法（跨模型均值）。"""
    col, asc, desc = _BY_DAY_SORT[sort_by]
    rank_col = 'rank_{}'.format(sort_by)
    print()
    print('=' * 108)
    print('  汇总 C：每天 × 每方法（跨全部模型均值，当天内按表现排序）')
    print('=' * 108)
    for day_idx in sorted(df_bd['day_idx'].unique()):
        sub = df_bd[df_bd['day_idx'] == day_idx]
        if len(sub) == 0:
            continue
        wi = int(sub['wind_iloc'].iloc[0])
        wm = float(sub['wind_mean'].iloc[0])
        print()
        print('  ' + '-' * 104)
        print('  day{}   (wind_iloc={:>4d}, 日均风速 {:5.2f} m/s)      排序: {}'
              .format(int(day_idx), wi, wm, desc))
        print('  ' + '-' * 104)
        print('  {:>4s} {:>12s} {:>6s}  {:>19s} {:>21s} {:>22s}'
              .format('排名', '方法', '模型数', '越限次数/天', '越限总量/天(MW)',
                      '利润/天(元)'))
        for _, r in sub.iterrows():
            print('  {:>4d} {:>12s} {:>6d}  {:>8.2f} ± {:<8.2f} {:>9.4f} ± {:<9.4f} '
                  '{:>10.2f} ± {:<10.2f}'
                  .format(int(r[rank_col]), str(r['project']),
                          int(r['n_models']),
                          r['viol_count_mean'], _std0(r['viol_count_std']),
                          r['viol_excess_mw_mean'], _std0(r['viol_excess_mw_std']),
                          r['profit_rmb_mean'], _std0(r['profit_rmb_std'])))
    print()
    print('=' * 108)
    print('  说明: 每行 = 该方法在【当天】全部模型上的均值 ± 跨模型标准差 (ddof=1, 单模型显示 0.00);')
    print('        “排名”按表头排序键给出; CSV 同时含 rank_profit / rank_viol / rank_excess 三种口径;')
    print('        注意: 同一方法的多组模型在该天共享同一条风速序列 ⇒ 这是"同天直接对比",')
    print('              不是独立重复; 方法级显著性仍需按模型(簇)或配对检验判定。')
    print('=' * 108)


def main():
    args = parse_args()

    if args.list_projects:
        print('支持的项目（按执行顺序）:')
        for key in PROJECT_ORDER:
            s = PROJECTS[key]
            print('  {:>13s}  {:<12s}  下层={:<5s} L观测={:<9s}  参考 {}'
                  .format(key, s['label'], s['lower'],
                          s['l_obs'] or 'rule', s['ref']))
        return

    global FLUCT_LIMIT, BESS_DEG_RMB_PER_MWH, EPISODE_LENGTH
    FLUCT_LIMIT = float(args.viol_limit)
    BESS_DEG_RMB_PER_MWH = float(args.deg_cost)
    EPISODE_LENGTH = int(args.episode_length)

    t_start = time.time()
    print('=' * 96)
    print('  五天随机风速测试  （只推理，不更新网络）')
    print('=' * 96)

    # ---- 1. 模型路径 ----
    paths = resolve_model_paths(args)
    keys = list(paths.keys())
    print('\n将测试 {} 个项目: {}'.format(
        len(keys), ', '.join(PROJECTS[k]['label'] for k in keys)))

    # ---- 2. 设备 & 随机种子 ----
    import torch
    if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print('设备: GPU {}'.format(torch.cuda.get_device_name(device)))
    else:
        device = torch.device('cpu')
        print('设备: CPU')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)          # 纯前向：省显存/加速

    # ---- 3. 抽 5 天 ----
    df_wind, pool = load_wind_pool()
    days = sample_days(df_wind, pool, args.days, args.seed)
    print('\n训练风速池: {} 列 (iloc[:, 1..{}]), 用 seed={} 抽 {} 天 (不重复)'
          .format(df_wind.shape[1], df_wind.shape[1] - 1, args.seed, len(days)))
    print('  {:>4s} {:>10s} {:>8s} {:>8s} {:>8s} {:>8s}'
          .format('天', 'wind_iloc', '列名', '均值', '最小', '最大'))
    for i, d in enumerate(days):
        s = d['series']
        print('  {:>4d} {:>10d} {:>8s} {:>8.2f} {:>8.2f} {:>8.2f}'
              .format(i + 1, d['iloc'], d['col'], s.mean(), s.min(), s.max()))

    # ---- 4. MATLAB 引擎（env.PowerOutput 需要，全脚本共用一个）----
    # 只有所选方法里有"走 MATLAB 后端"的项目时才启动（`matlab=False` 的项目 =
    # yawflor：物理引擎是 FLORIS，纯 Python）⇒ 单测 yawflor 不需要 MATLAB。
    need_matlab = [k for k in keys if PROJECTS[k].get('matlab', True)]
    if need_matlab:
        engine = ensure_engine()
    else:
        engine = None
        print('  跳过 MATLAB 引擎（所选项目全部为纯 Python 后端: {}）'
              .format(', '.join(PROJECTS[k]['label'] for k in keys)))

    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    step_csv = os.path.join(out_dir, 'five_day_steps.csv')
    days_csv = os.path.join(out_dir, 'five_day_days.csv')
    summary_csv = os.path.join(out_dir, 'five_day_summary.csv')            # 每方法×每run
    method_csv = os.path.join(out_dir, 'five_day_method_summary.csv')      # 每方法(跨run)
    by_day_csv = os.path.join(out_dir, 'five_day_by_day.csv')              # 每天×每方法
    if os.path.isfile(step_csv):
        os.remove(step_csv)

    # ---- TensorBoard（与 CSV 双份保存）----
    writer = None
    tb_dir = os.path.join(out_dir, 'tensorboard')
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(tb_dir)
        print('TensorBoard: {}'.format(tb_dir))

    # 注意: run_args 现在按模型目录构造 (见下面循环) —— 各 run 的网络维度可以不同
    # (由 _detect_hidden 从权重形状读出, 不依赖 main 的当前默认值)。

    all_days = []
    n_step_rows = 0
    global_step = 0
    n_episodes = sum(len(v) for v in paths.values()) * len(days)
    print('计划: {} 个方法, 共 {} 组模型, {} 天/组 → {} 个 episode'
          .format(len(paths), sum(len(v) for v in paths.values()), len(days),
                  n_episodes))

    try:
        for key in keys:
            spec = PROJECTS[key]
            label = spec['label']
            model_dirs = paths[key]

            for run_idx, model_dir in enumerate(model_dirs):
                run_no = run_idx + 1
                model_name = os.path.basename(model_dir.rstrip('\\/'))
                print('\n' + '-' * 96)
                print('  方法 {}  ({})  第 {}/{} 组模型  <- {}'
                      .format(label, key, run_no, len(model_dirs), model_dir))
                print('-' * 96)
                run_args = _build_args(device, args.seed,
                                       _detect_hidden(model_dir, 6))
                runner = ProjectRunner(
                    key, model_dir, device, run_args, engine,
                    phase_feat=args.phase_feat, soc_init=args.soc_init,
                    rng=np.random.RandomState(args.seed), verbose=True)

                run_days = []
                for i, d in enumerate(days):
                    day_idx = i + 1
                    t0 = time.time()
                    day, rows = runner.run_day(
                        d['series'], day_idx, d['iloc'], d['col'])
                    day['run'] = run_no
                    day['model'] = model_name
                    day['model_dir'] = model_dir
                    all_days.append(day)
                    run_days.append(day)
                    print('    第{}天 (iloc[:,{}], 均值{:5.2f} m/s): '
                          '越限 {:>4d} 次, 越限总量 {:>8.4f} MW, '
                          '利润 {:>12.2f} 元   [{:.0f}s]'
                          .format(day_idx, d['iloc'], day['wind_mean'],
                                  day['viol_count'], day['viol_excess_mw'],
                                  day['profit_rmb'], time.time() - t0))

                    # ---- CSV: 每步明细（带 run/model 列）----
                    if not args.no_save_steps and rows:
                        df_rows = pd.DataFrame(rows)
                        df_rows.insert(0, 'run', run_no)
                        df_rows.insert(1, 'model', model_name)
                        df_rows.to_csv(step_csv, mode='a',
                                       header=(n_step_rows == 0), index=False)
                        n_step_rows += len(df_rows)

                    # ---- TensorBoard: 本 run 的逐日三项指标 (step = 天序号) ----
                    if writer is not None:
                        for m in DAY_METRICS:
                            writer.add_scalar(
                                'test/{}/run{}/{}'.format(label, run_no, m),
                                float(day[m]), day_idx)
                        # 逐步子曲线（每 10 步抽 1 点；x 轴按 run 顺延不重叠）
                        for r in rows[::10]:
                            gs = global_step + int(r['step'])
                            writer.add_scalar(
                                'step/{}/run{}/P_F'.format(label, run_no),
                                float(r['P_F']), gs)
                            writer.add_scalar(
                                'step/{}/run{}/SOC'.format(label, run_no),
                                float(r['SOC']), gs)
                            writer.add_scalar(
                                'step/{}/run{}/profit_step'.format(label, run_no),
                                float(r['profit_step']), gs)
                    global_step += runner.episode_length

                # ---- 每组模型跑完就落盘 ----
                df_days = pd.DataFrame(all_days)
                df_days.to_csv(days_csv, index=False)
                _agg_days(df_days, ['project', 'project_key', 'run', 'model']) \
                    .to_csv(summary_csv, index=False)
                if writer is not None:
                    writer.flush()

            # ---- 方法级：跨 run 的逐日均值/极值曲线 ----
            if writer is not None:
                sub = pd.DataFrame(all_days)
                sub = sub[sub['project'] == label]
                for day_idx in sorted(sub['day_idx'].unique()):
                    g = sub[sub['day_idx'] == day_idx]
                    for m in DAY_METRICS:
                        writer.add_scalar('method/{}/{}'.format(label, m),
                                          float(g[m].mean()), int(day_idx))
                        writer.add_scalar('method/{}/{}_min'.format(label, m),
                                          float(g[m].min()), int(day_idx))
                        writer.add_scalar('method/{}/{}_max'.format(label, m),
                                          float(g[m].max()), int(day_idx))
                writer.flush()
    finally:
        if writer is not None:
            writer.close()
        quit_engine()

    df_days = pd.DataFrame(all_days)
    df_run = _agg_days(df_days, ['project', 'project_key', 'run', 'model'])
    df_m = _agg_days(df_days, ['project', 'project_key']).sort_values(
        'project', key=lambda s: s.map(
            {PROJECTS[k]['label']: i for i, k in enumerate(PROJECT_ORDER)}))
    df_run.to_csv(summary_csv, index=False)
    df_m.to_csv(method_csv, index=False)

    # ---- 每天 × 每方法（跨模型均值，按天分块排序）----
    df_bd = build_by_day_table(df_days, keys, args.by_day_sort)
    df_bd.to_csv(by_day_csv, index=False)

    # ---- 每天明细 ----
    print('\n' + '=' * 108)
    print('  每天明细')
    print('=' * 108)
    print('  {:>12s} {:>4s} {:>10s} {:>10s} {:>12s} {:>14s} {:>9s} {:>9s}'
          .format('方法', 'run', 'wind_iloc', '越限次数', '越限总量MW',
                  '利润(元)', '末SOC', '放电MWh'))
    for key in keys:
        label = PROJECTS[key]['label']
        sub = df_days[df_days['project'] == label]
        for _, r in sub.iterrows():
            print('  {:>12s} {:>4d} {:>10d} {:>10d} {:>12.4f} {:>14.2f} '
                  '{:>9.3f} {:>9.3f}'
                  .format(label, int(r['run']), int(r['wind_iloc']),
                          int(r['viol_count']), r['viol_excess_mw'],
                          r['profit_rmb'], r['soc_end'], r['discharge_mwh']))

    print_run_summary(df_run, keys)
    print_method_summary(df_m, keys)
    print_by_day_summary(df_bd, keys, args.by_day_sort)

    print('\n输出文件:')
    print('  每天汇总     : {}'.format(days_csv))
    print('  每方法×每run : {}'.format(summary_csv))
    print('  每方法主表   : {}'.format(method_csv))
    print('  每天×每方法  : {}'.format(by_day_csv))
    if not args.no_save_steps:
        print('  每步明细     : {}  ({} 行)'.format(step_csv, n_step_rows))
    if writer is not None:
        print('  TensorBoard  : {}'.format(tb_dir))
        print('                 tensorboard --logdir="{}"'.format(tb_dir))
    print('  总耗时       : {:.1f}s'.format(time.time() - t_start))


if __name__ == '__main__':
    main()
