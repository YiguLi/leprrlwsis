"""
HSRL 分支（aifflor 专用）——「块总放电量」语义下的上下层口径**单一真源**
============================================================================

`aifflor_main.py --hsrl` 的全部新口径都收在本模块；**未加 `--hsrl` 时本模块的函数一行都不会
被调用** ⇒ 旧路径逐位不变（`agentutil/test_aifflor_contract.py` 的对照断言仍然成立）。

与旧口径的逐条差异（对应 `aifflor/README.md` 的 `--hsrl` 小节）
---------------------------------------------------------------

① **H 第 6 维语义**：不再是 SOC 目标信号 `soc_target = 0.25 + 0.41667·a6`，而是
   「**BESS 在本 5 步块内的总放电量**」`g`，单位 **MW·min**，**放电为正**。
   可行域由**块首 SOC + 额定功率**给出（**不含波动约束**）：

       D_max = min(P_rated·δ , (E − E_min)·η_d·60)          [MW·min]  放电上限
       C_max = min(P_rated·δ , (E_max − E)·60/η_c)          [MW·min]  充电上限
       g ∈ [−C_max, +D_max]           (δ = 5 步 = 5 min, dt = 1/60 h)

   推导见 `feasible_total` 的 docstring（两条约束：额定功率 + SOC 能量）。
   越界按 **线性罚** `HSRL_PEN6_COEF · 越界量` [元/MW·min]，**不封顶**。

② **H 奖励去掉 bessdeg**：`r1 = pr·P_G·dt − penalty_NN_H`（不再减 `490·max(−pbrule,0)·dt`）；
   块末 SOC 目标 gap 罚**删除**（目标语义已变成"块总放电量"，执行偏差由 L 的 c_L 承接）。

③ **L 层动作 = 绝对 pb**（不再是"规则基线 + Δ"），只按**每步物理域**裁剪
   `[pb_lo, pb_hi]`（SOC + 额定功率；**不含波动窗口**；正=充电，与 env 同号）。
   ⇒ `rbb_system` 在 HSRL 下用 `fluctuation_limit=inf` 构造，RBB 退化为"纯物理裁剪 + 记录"。

④ **L 层 reward**：`r2 = −HSRL_VIOL_COEF·越界量 − HSRL_DEG_COEF·放电量`。
   越界量 = 原始动作超出每步物理域的量 [MW]；退化只对**放电**计费 [MW·min]，
   系数 = 越界罚的 **1/10**（用户口径）。**不含** rulebase 波动裁剪罚。

⑤ **L 层 cost（图中公式）**：

       c_L^t = 1 − exp( −| (g − Σ_{t'=t−t_L}^{t−1} a_L^{t'}) / (δ − t_L) − a_L^t | )

   δ = 5（块长）、t_L = 块内已过步数（= step % 5）、δ − t_L = n_rem（含当前步的剩余步数）、
   g = H 下发的块总放电量、a_L = **实际施加**的放电量（= −pb_applied，由 env 回传的
   `P_C − P_D` 反号得到）。Σ 同样取**实际施加值**的块内累计（**不含当前步**）。
   原式出处：`未命名/PER_main0.py:116-118`
   （`c2 = -np.exp(-abs((signal - sum(pb_l[:-1]))/(5-tt) - pb)) + 1`）。
   ⇒ c_L ∈ [0, 1)：完全按"摊还计划"执行 ≡ 0；偏离 1 MW ⇒ 1 − e⁻¹ ≈ 0.632。

⑥ **L 观测**（6 维；= "H 观测 + 上层目标 + **c_L 的参照点**"）：

       [ u/10 , pr/1000 , SOC/0.5 , P_G(t−1)/5 , g/2.5 , need/2.5 ]

   第 6 维 `need = (g − Σ已放电)/(δ − t_L)` 就是 c_L 公式里的参照点（`plan_need`）——
   把它喂给策略，actor 才能**直接**知道"本步该出多少"（零成本动作 = 令 `a_L = need`）。
   `need/2.5` 用与 `g` 同一个刻度（满量程 = `δ·P_rated`），两维可直接比较。

   ⚠ **残余局限（须知）**：6 维里给了 `g` 与 `need`，但**没有** `n_rem`（也没有 Σ 本身）
   ⇒ 由 `need = (g−Σ)/n_rem` 无法反解 `(Σ, n_rem)`（一个方程两个未知量），
   于是**下一状态的参照点** `need' = (g−Σ−a_L)/(n_rem−1)` 仍不能被策略精确预测
   （同一个观测可对应不同的 `need'`；契约测试第 G 节把这条写成可执行记录）。
   若要把这块也补上，再加一维 `n_rem/5` 即可（`nb_states` 6→7）。
   注意本步**代价** `c_L = 1−exp(−|need − a_L|)` 只依赖本步的 `need` 与 `a_L`
   ⇒ 就"本步该怎么动"而言，第 6 维已经给全了。

⑦ **L 对偶重标**：c_L 无量纲、r2 是"元"量级 ⇒ λ2 ∈ [2, 50]（初值 10；乘性**对数**步长
   0.02 不变）；每块预算 `d_block_L = 0.1`（Q 口径下不再为探索噪声付费 ⇒ 由 0.5 收紧，
   把 λ2 的工作点抬回 20~50、恢复"兑现计划"的紧度），curriculum 地板 0.05。

⑧ **warmup 下层 = 摊还执行器** `amortized_rule_action`：命令 = `clip(−(g−Σ)/n_rem, 物理域)`
   ⇒ 入池动作（绝对值）的 c_L ≡ 0、r2 = −退化项（"零成本动作"自洽）。
   warmup 上层第 6 维 = `−HSRL_WARMUP_G_SCALE · sig_walk`（价格游走：低谷充电 ⇒ g<0、
   高峰放电 ⇒ g>0；符号与旧的"SOC 目标游走"**相反**，因为放电量与 SOC 变动反号）。

⑨ **训练中的测试**：`--hsrl` 下不再跑 9-case，改跑 `twenty_step_test_hsrl`
   （电价恒 300、风速用**固定**的 20 步段 `col=120/start=483`（选段依据见常量区）、
   开局 SOC = 0.25；触发频率与旧 9-case 相同：`step > warmup` 且 `episode_step % 480 == 0`）。
   **固定段**是为了"配对看收敛"（随机段会把场景差异混进曲线）；要看泛化再开
   `--hsrl_test_wind random`。
⑩ **对偶的分子改用 critic cost 的 Q**（`--hsrl_dual_src q`，默认）而不是实际块成本的 EMA：
   Q 是 bootstrap 出来的**期望**成本 ⇒ λ 的波动显著小于"单块实现值"（实测块 sum_c1
   中位数 4.5、最大 305，重尾）。取值约定：状态 = **块首**观测、动作 = actor 的**确定性**
   动作、twin 取**均值**（回归目标已用 max，对偶再用 max 是双重高估）。Q 未建立时回退 EMA。
   EMA 仍然维护（日志 / curriculum / 回退用）。λ1 的物理价区间见 `HSRL_LAM1_PHYS_*`。
"""
import numpy as np

import hsrl_cost as hc

try:                                        # 主路径: aifflor 作为包
    from aifflor.aifflor_myfunc import (
        get_wind_level, get_price_level, Cal_soc_deviation,
        DEG_COST_DISCHARGE, DEG_COST_CHARGE,
        PENALTY_NN_H_COEF, PENALTY_NN_L_COEF, W_GAP, C1_RISK_COEF,
        Normalize_H, rep_minute_for_price_level, violation_derate,
    )
except ImportError:                         # 兼容: 直接把 aifflor/ 放进 sys.path
    from aifflor_myfunc import (            # type: ignore
        get_wind_level, get_price_level, Cal_soc_deviation,
        DEG_COST_DISCHARGE, DEG_COST_CHARGE,
        PENALTY_NN_H_COEF, PENALTY_NN_L_COEF, W_GAP, C1_RISK_COEF,
        Normalize_H, rep_minute_for_price_level, violation_derate,
    )


# =============================================================================
# 1. 常量（HSRL 分支的单一真源）
# =============================================================================
# ---- 物理/时间口径（与 env.WindBattery / rbb_rule_bess 一致）----
HSRL_BLOCK = 5                  # 块长 [步] = δ（H 每 5 步决策一次）
DT = 1.0 / 60.0                 # 步长 [h] = 1 min
ETA_C = ETA_D = 0.98            # 充/放电效率
P_RATED = 0.5                   # BESS 额定功率 [MW]
E_MIN = 0.0                     # SOC 下界 [MWh]
E_MAX = 0.5                     # SOC 上界 [MWh]

# ---- H 层：第 6 维（块总放电量）----
HSRL_PEN6_COEF = PENALTY_NN_H_COEF      # 400.0 [元/MW·min] 第 6 维越界线性罚（不封顶）
HSRL_G_FULL = HSRL_BLOCK * P_RATED      # 2.5 [MW·min] 满量程（观测刻度/噪声/warmup 定标）

# ---- L 层：reward ----
HSRL_VIOL_COEF = PENALTY_NN_L_COEF      # 200.0 [元/MW]      L 动作越界罚
HSRL_DEG_COEF = HSRL_VIOL_COEF / 10.0   # 20.0  [元/MW·min]  放电退化（= 越界罚的 1/10）

# ---- L 层：cost（图中公式）----
HSRL_COST_DELTA = HSRL_BLOCK            # δ = 5（公式里的块长，单位: 步）

# ---- L 层：对偶与预算（λ2 与 d_block_L 随 c_L 的无量纲口径重标）----
HSRL_L_LAM_MIN = 2.0            # λ2 下限：1 单位 c_L 至少值 2 元 r2
HSRL_L_LAM_MAX = 50.0           # λ2 上限：至多值 50 元 r2
HSRL_L_LAM_INIT = 10.0          # 初值 = sqrt(MIN·MAX) = 10（对数对称）
HSRL_L_EMA_LAM_LR = 0.02        # 乘性对数步长（与 hc.ETA_LAM 同刻度）
HSRL_D_BLOCK_L = 0.1            # 每块 sum_c_L 预算（Q 口径下不再为噪声付费 ⇒ 由 0.5 收紧到 0.1，
                                #   即 ratio ×5 ⇒ λ2 升到 20~50，恢复"兑现计划"的紧度）
HSRL_D_BLOCK_L_MIN = 0.05       # curriculum 地板
HSRL_L_COST_LIMIT = 0.25        # agent_L.cost_limit（仅 non-EMA 分支用）
HSRL_L_CURRICULUM_BLOCKS = 50   # 预算减半所需连续满足块数（= EMA 时间常数 1/α）

# ---- 对偶的**分子**：critic cost 的 Q（而非实际块成本的 EMA）----
#: twin cost critic 的聚合方式（对偶专用，**独立于** critic 回归目标的 `--cost_critic_mode`）：
#:   critic 的 TD 目标用 **max**（安全侧，避免低估），若对偶也用 max 就是**双重高估**
#:   ⇒ 正是仓库里记载过的"高估 → λ 顶死传导"老路（见 `td3prrbb_main.py` 的相关注释）；
#:   对偶要的是**约束价格的期望**，故默认取**双臂均值**（无偏）。
HSRL_DUAL_Q_AGG = 'mean'        # 'mean' | 'min' | 'max'
HSRL_DUAL_SRC = 'q'             # 'q'（critic cost 的 Q）| 'ema'（实际块成本 EMA，A/B 对照）
#: `src='q'` 时**用哪种 Q 估计**（`--hsrl_dual_q_src`）：
#:   * `update_batch`（**默认**，用户口径）：直接用 `update_policy` **本块已算好的**
#:     cost Q 批均值（`update_stats` 里的 `q_c_mean`/`q_c2_mean`）——
#:     靠**缓冲的自然淘汰**（环形容量 = `--rmsize`）让这个均值随策略改进而下降来驱动 λ1；
#:     批是随机抽的、跨天混合 ⇒ **不含当日风况**（消掉风况引起的慢漂）。
#:   * `recent_batch`（S1，λ2 仍用它）: **近端 N 条状态的批均值 × 当前策略动作**
#:       `rate = (1−γ_H)·mean_i Q_c(s_i, π(s_i))·s_c`
#:     实证动机（`output/AIFFLOR-run73`，26.9k 块）：
#:       - 单点 `Q_c(s₀,a₀)` 与**当日风况**相关 **+0.74**（块内 ΣP_F +0.57）⇒ λ1 的慢趋势里
#:         97% 是风况驱动（λ 自身与 u 相关 +0.18）——"是否违约极大受当日风速影响"的定量证据；
#:       - 换成**批均值**后风相关降到 **−0.01**、分子 sd 缩 4.3×；若动作用**旧的行为动作**
#:         （= `update_batch` 的口径）它锚在"近期回放分布"上 ⇒ 靠自然淘汰缓慢跟随。
#:   * `point`: 旧的单点 `Q_c(s₀,a₀)`（A/B 对照，风依赖强）。
#:   * `ema`: 实际块成本 EMA（= `--hsrl_dual_src ema`，历史口径）。
HSRL_DUAL_Q_SRC = 'update_batch'
HSRL_DUAL_RECENT_N = 256        # 近端批的条数（H 每块 1 条 ⇒ ≈256 块 ≈ 21 h，覆盖一天的风况）

# ---- H 层 λ1：**物理价**区间 [元/(MW·step)]（actor 空间 = 物理价 × hc.lam_scale()=s_c/s_p）----
#: 已定稿（第三轮讨论，用户口径 D1 = 单通道只重标定区间）：
#:   * **上界**：越限每多 1 MW·step，c1 增 21（含基础项）= `(1+K_V)·λ1`，要求最大罚 > 980×2 = 1960
#:     ⇒ `λ1_max > 1960/21 ≈ 93.3` ⇒ 取 **100**（= 峰价边际收益 980/60 = 16.33 的 6.1×、
#:     名义边际收益 9.67 的 10.3×、HSRL 下 BESS 退化边际价 20 元/MW·min 的 5×
#:     ⇒ 满足仓库既有原则"λ_max 必须高于任何平抑资源的边际成本"）。
#:   * **上界 = 50 元/(MW·step)**（**2026-10 用户第五轮**）：依据 = 仓库原则"λ_max 必须高于
#:     **任何平抑资源的边际成本**"。本项目只有两种平抑资源的边际成本：
#:       - BESS 放电退化：**20 元/(MW·step)**（= 20 元/MW·min）
#:       - 风机降载（少卖电）：`pr·Δt` = **5.0 ~ 16.33 元/(MW·step)**（300~980 元/MWh）
#:     ⇒ 需求 = `λ_max > max(20, 16.33) = 20`，取 **2.5×** ⇒ **50**
#:     （= 退化价 2.5×、峰价边际收益 `980/60 = 16.33` 的 3.06×）。
#:     界外波动的单价是 `20·λ1`（见下），故判据"980 电价下越限的净梯度必须为负"写作
#:     `20·λ_max = 1000 > 16.33` ⇒ **余量 61×**（界内项再加 1 ⇒ 机械导数为 `21·λ1`，差 5%）。
#:     ⚠ **记档：用户原写法 `50×20 = 1000 略大于 980` 是量纲混淆**（980 是 元/MWh，须先乘
#:       `Δt = 1/60 h` 变成 16.33 元/(MW·step) 才能与 λ1 比）；仓库更早的
#:       `(1+K_V)·λ_max > 980×2 = 1960 ⇒ λ_max > 93.3` 同样混淆量纲，**已废止**（它给出 100）。
#:   * **下界 = 2 元/(MW·step)**（**用户第五轮第二次重标**；此前依次为 0.967 → 物理 50 → 现在 2）：
#:       - `2/16.33 = 12%`（峰价边际收益）、`2/5.0 = 40%`（低电价边际收益）、`2/9.67 = 21%`（名义）；
#:       - **关键性质：2 < BESS 退化边际价 20** ⇒ 界内波动**不值得为它动用 BESS**（不诱导无谓循环），
#:         同时界内仍**非零价**（"不完全免费"）；
#:       - 也高于量纲正确的要求 `16.33/21 = 0.78`（界内项口径 `16.33/20 = 0.82`）⇒ 判据仍成立。
#:     用户口径的单位说明（原文）：λ1 是"界内波动"的单价 [元/(MW·step)]，界外波动的单价是 20 倍。
#:   * 生产口径（`s_c=100, s_p=1000 ⇒ lam_scale=0.1`）：物理 `[2, 50]` ⇔ actor `[0.2, 5.0]`，
#:     初值 = 下限。两个界都可用 `--hsrl_lam1_min/--hsrl_lam1_max`（物理价）临时覆盖。
#:   * 上界只在**出现越限块**时起作用（实测最大块 sum_c1 = 305 ≫ 15）；本轮**不加**非对称步长、
#:     不加含越限的专用测试段（用户选择"都不要"）⇒ 自适应仍是**对称**乘性步长 0.02。
HSRL_LAM1_PHYS_MIN = 2.0        # 元/(MW·step)（用户第五轮：2）
HSRL_LAM1_PHYS_MAX = 50.0       # 元/(MW·step)（用户第五轮：max(资源边际成本)=20 的 2.5×）
HSRL_LAM1_PHYS_INIT = HSRL_LAM1_PHYS_MIN   # 初值 = 下限（用户选择）
HSRL_LAM1_ETA = 0.02            # 乘性对数步长（= hc.ETA_LAM，对称；用户选择不改非对称）


def lam1_phys_bounds(lam_min=None, lam_max=None, lam_init=None):
    """`(lam_min, lam_max, lam_init)` 的 **actor 空间**值 = 物理价 × `hc.lam_scale()`。

    * **两个界都以物理价为准**（元/(MW·step)，与 c1 同量纲；用户第五轮口径）；
      可用参数（对应 `--hsrl_lam1_min/--hsrl_lam1_max`）临时覆盖。
    * 必须在**调用时**取 `hc.lam_scale()`：本模块在 `hc.set_cost_scale()` 之前被 import，
      模块级常量会冻结成默认缩放（1.0）而不是生产值（0.1）。
    * 初值默认 = 下限（用户选择）。
    * 生产口径：物理 `[2.0, 50.0]` ⇔ actor `[0.2, 5.0]`，初值 0.2。
    """
    kl = float(hc.lam_scale())
    lo_p = HSRL_LAM1_PHYS_MIN if lam_min is None else float(lam_min)
    hi_p = HSRL_LAM1_PHYS_MAX if lam_max is None else float(lam_max)
    if hi_p < lo_p:                 # 非空保护
        hi_p = lo_p
    in_p = lo_p if lam_init is None else float(lam_init)
    in_p = min(max(in_p, lo_p), hi_p)
    return (lo_p * kl, hi_p * kl, in_p * kl)

# ---- L 层：观测刻度 ----
OBS_U_SCALE = 10.0              # 风速 u [m/s] → /10
OBS_PR_SCALE = 1000.0           # 电价 pr [元/MWh] → /1000
OBS_E_SCALE = E_MAX             # SOC E [MWh] → /0.5
OBS_PG_SCALE = 5.0              # P_G(t−1) [MW] → /5
OBS_G_SCALE = HSRL_G_FULL       # 上层目标 g [MW·min] → /2.5
OBS_NEED_SCALE = HSRL_G_FULL    # c_L 的参照点 need=(g−Σ)/n_rem [MW] → /2.5（与 g 同刻度）

# ---- warmup ----
#: warmup 上层第 6 维定标：`g = −HSRL_WARMUP_G_SCALE · sig_walk`，`sig_walk ∈ [−0.5, 0.5]`
#: ⇒ `g ∈ [−2.5, 2.5]` = 整个可行域（在额定功率处饱和，与旧口径"信号游走到 ±0.5 饱和"同形）。
#: 符号取负：`hsrl_warmup.WarmupSchedule.sig_delta` 在**低谷(300)上调**（= 充电方向），
#: 而 g 是"放电量"（充电为负）⇒ 必须反号，否则会变成"低谷大放电"。
HSRL_WARMUP_G_SCALE = 2.0 * HSRL_G_FULL     # = 5.0

# ---- 20 步测试协议 ----
HSRL_TEST_STEPS = 20            # 20 步 = 4 个块
HSRL_TEST_PRICE = 300.0         # 固定电价（低谷）
HSRL_TEST_SOC0 = 0.25           # 开局 SOC [MWh]（= 容量中点）
#: **固定风速段**（`wind_pool = env_test.wind_noise.iloc[:, 1:]`，行=天内步、列=天）：
#: 选段依据（`agentutil` 扫描 194 列 × 1421 起点）—— 均值 8~11 m/s 的窗口中
#: **逐步爬坡 Σ|Δu| 位于前 5%**：u 均值 10.35 m/s（全池均值 10.83，代表性好）、
#: min 7.70 / max 12.86、Σ|Δu| = 29.0；按平坦地形 5 台机 dP/du ≈ 1.07 MW/(m/s) 估算，
#: 单步 |ΔP| ≈ 2 MW **< 3 MW 限幅** ⇒ 有可观的波动成本可优化、但基本不出现越限
#: （越限部分 BESS 只有 0.5 MW，本来就大多不可避 ⇒ 测不出策略差异）。
#: 固定段的意义：**每次触发都是同一条场景** ⇒ test20 曲线是"配对"的，可直接看收敛。
HSRL_TEST_WIND_COL = 120        # 列（天）
HSRL_TEST_WIND_START = 483      # 起点（天内第几步）


# =============================================================================
# 2. 可行域（块级总放电量 / 每步物理功率）
# =============================================================================
def feasible_total(E, eta_c=ETA_C, eta_d=ETA_D, time=DT, rated_power=P_RATED,
                   e_min=E_MIN, e_max=E_MAX, block=HSRL_BLOCK):
    """块级可行**总放电量**域 `(C_max, D_max)` [MW·min，放电为正]。

        D_max = min(P_rated·δ , (E − E_min)·η_d · 60)       （放电上限）
        C_max = min(P_rated·δ , (E_max − E)·60 / η_c)       （充电上限）
        g ∈ [−C_max, +D_max]

    推导（**两条例约束**，都不含波动约束 —— 用户口径）：
      ① 额定功率：每步 |pb| ≤ P_rated，块内 δ 步 ⇒ 总量 ≤ P_rated·δ；
      ② SOC 能量：放电量 q [MW·min] 使电池能量下降 `q/η_d` [MW·min]
         （因 ΔE = P_D·dt/η_d，且 1 MWh = 60 MW·min ⇒ q/η_d），
         块内累计 ≤ (E − E_min)·60 ⇒ Σq ≤ (E − E_min)·η_d·60；
         充电同理（效率在分母）：`E 上升 = q·η_c` ⇒ Σq ≤ (E_max − E)·60/η_c。
      ⚠ ② 是**整块的能量约束**（不是"乘 δ 的功率约束"）：写成 `(E−E_min)·η_d/dt` 只是
        **单步**的功率上限，整块要再换算成能量（×60 / δ 的差别在 SOC 贴边时可达数倍）。
    """
    e = float(np.clip(float(E), e_min, e_max))
    per_hour = 1.0 / float(time)                    # = 60 [min/h]
    d_soc = (e - e_min) * per_hour * float(eta_d)   # [MW·min] SOC 能量约束（放电）
    c_soc = (e_max - e) * per_hour / float(eta_c)   # [MW·min] SOC 能量约束（充电）
    d_max = min(float(rated_power) * float(block), d_soc)
    c_max = min(float(rated_power) * float(block), c_soc)
    return max(0.0, c_max), max(0.0, d_max)


def feasible_power(E, eta_c=ETA_C, eta_d=ETA_D, time=DT, rated_power=P_RATED,
                   e_min=E_MIN, e_max=E_MAX):
    """每步物理可行功率域 `(pb_lo, pb_hi)` [MW，**正=充电**，与 env 同号]。

    与 `rbb_rule_bess.RuleBasedBESS.compute_pbrule` 的 `pbphysicsmin/pbphysicsmax`
    **逐位同式**（`agentutil/test_aifflor_hsrl_contract.py` 有对照断言）
    ⇒ HSRL 下把 RBB 的 `fluctuation_limit` 设为 `inf`，`pbrule` 恰好就是本函数的结果。
    """
    e = float(np.clip(float(E), e_min, e_max))
    pb_hi = min(float(rated_power), (e_max - e) / (float(eta_c) * float(time)))
    pb_lo = max(-float(rated_power), -(e - e_min) * float(eta_d) / float(time))
    return float(pb_lo), float(pb_hi)


def discharge_of(pb):
    """功率 → 放电量（**放电为正**）：`a_L = −pb`。"""
    return -float(pb)


# =============================================================================
# 3. H 层：动作检查 / 奖励
# =============================================================================
def check_action_H_hsrl(action, E, pen_coef=None, pen6_coef=None):
    """HSRL 版 H 动作检查。

    * 前 5 维（机组轴向因子）：与旧口径**逐位相同** —— clip 到 [0, 0.5]，
      越界线性罚 `PENALTY_NN_H_COEF`(=400) [元/单位]，不封顶；
    * 第 6 维（块总放电量 g [MW·min]）：clip 到**块级可行域** `[−C_max, +D_max]`，
      越界线性罚 `HSRL_PEN6_COEF`(=400) [元/MW·min]，**不封顶**
      （与旧口径"信号 clip 到 [−0.5, 0.5] 并按 400/单位罚"在形式上同构，
        只是把边界从"人为的信号带"换成"SOC + 额定功率推出来的可行域"）。

    Returns
    -------
    applied : ndarray(6)  夹后的动作（进 env 的只有前 5 维）
    pen5    : float       前 5 维越界罚 [元]
    pen6    : float       第 6 维越界罚 [元]（诊断列 pen6；与 pen5 一起进 r1）
    """
    a = np.asarray(action, dtype=float).reshape(-1)
    if a.size < 6:
        raise ValueError('check_action_H_hsrl: 需要 6 维动作, 得到 %d' % a.size)
    c5 = PENALTY_NN_H_COEF if pen_coef is None else float(pen_coef)
    c6 = HSRL_PEN6_COEF if pen6_coef is None else float(pen6_coef)

    pen5 = 0.0
    for j in a[:5]:
        ma, mi = max(0.5, float(j)), min(0.0, float(j))
        pen5 += c5 * ((ma - 0.5) + (0.0 - mi))

    c_max, d_max = feasible_total(E)
    g = float(a[5])
    pen6 = c6 * (max(0.0, g - d_max) + max(0.0, (-c_max) - g))

    applied = np.concatenate([np.clip(a[:5], 0.0, 0.5),
                              np.array([float(np.clip(g, -c_max, d_max))])])
    return applied, float(pen5), float(pen6)


def cal_rew_rb_hsrl(observation, observation2, action_rbb, rbb_info,
                    penalty_NN=0.0, time=DT, ptur=None, pb_cmd=None,
                    turbine_derate=1.0):
    """HSRL 版 H 奖励/成本 —— 与 `aifflor_myfunc.Cal_Rew_RB` 的唯一差异是**去掉 bessdeg**：

        r1 = turbine_derate·pr·ptur·dt + (−pr·pbrule·dt) − penalty_NN
           = pr·P_G·dt − penalty_NN                    （P_G = ptur − pbrule）

    `penalty_NN` = 前 5 维越界罚 + 第 6 维越界罚（由 main 相加后传入）。
    `c1` 口径**不变**（仍走 `hc.c1_apply`；HSRL 下波动窗口不可行 ⇒ `unfeasible_gap ≡ 0`）。
    details 里 `degradation` 恒为 0（该项已不在奖励内），`degradation_ref490` 保留为**诊断列**。
    """
    if ptur is None:
        raise ValueError('cal_rew_rb_hsrl: ptur must be provided')

    pr_old = float(observation[0])
    u = float(observation[1])
    P_G_new = float(observation2[-1])
    P_G_old = float(observation[-1])
    SOC_new = float(observation2[2])

    pbrule = float(rbb_info['pbrule'])
    unfeasible_gap = float(rbb_info['unfeasible_gap'])
    clip_amount = float(rbb_info['clip_amount'])
    P_F = abs(P_G_new - P_G_old)

    wind_level = get_wind_level(u)
    price_level = get_price_level(pr_old)
    SOC_decision = float(observation[2])

    # ==== r1（H 机会侧）: **无退化项** ====
    w_wind = 1.0
    w_bess = 1.0
    turbine_profit = w_wind * pr_old * ptur * time
    bess_profit = w_bess * (-pr_old * pbrule * time)
    r1 = turbine_profit * turbine_derate + bess_profit - float(penalty_NN)

    # ==== c1（H 风险侧）: 与旧口径逐位相同（含 unfeasible_gap 项，HSRL 下恒 0）====
    c1_risk_coeff = C1_RISK_COEF
    if P_F > 3.0 + 1e-4:
        c1 = c1_risk_coeff * (10.0 * (P_F - 3.0) + 0.1 * P_F)
    else:
        c1 = c1_risk_coeff * (0.1 * P_F)
    c1 = c1 + W_GAP * unfeasible_gap
    c1 = hc.c1_apply(c1, P_F)

    details = {
        'wind_level': wind_level,
        'price_level': price_level,
        'w_wind': w_wind,
        'w_bess': w_bess,
        'pr': pr_old,
        'u': u,
        'turbine_profit': turbine_profit,
        'bess_profit': bess_profit,
        'revenue': turbine_profit + bess_profit,
        'degradation': 0.0,                      # ← 已从 H 奖励中移除（恒 0）
        'degradation_ref490': (DEG_COST_DISCHARGE * max(-pbrule, 0.0)
                               + DEG_COST_CHARGE * max(pbrule, 0.0)) * time,
        'soc_deviation': Cal_soc_deviation(SOC_new, wind_level),
        'ptur': ptur,
        'pbrule': pbrule,
        'pb_cmd': float(pb_cmd) if pb_cmd is not None else np.nan,
        'pb_rbb_clip': (abs(float(pb_cmd) - pbrule) if pb_cmd is not None else 0.0),
        'P_F': P_F,
        'SOC': SOC_new,
        'unfeasible_gap': unfeasible_gap,
        'clip_amount': clip_amount,
        'is_infeasible': rbb_info.get('is_infeasible', False),
        'inevitable_violation': rbb_info.get('inevitable_violation', 0.0),
        'penalty_NN': float(penalty_NN),
        'c1_fluct': c1,
    }
    return r1, c1, details


# =============================================================================
# 4. L 层：reward / cost / 观测 / 对偶 / warmup 基线
# =============================================================================
def check_action_L_hsrl(action, E, coef=None):
    """HSRL 版 L 动作检查：只按**每步物理域**（SOC + 额定功率）裁剪。

    Returns
    -------
    applied : ndarray(1)  夹后的 pb [MW]（正=充电）
    penalty : float       越界线性罚 `HSRL_VIOL_COEF·越界量` [元]
    viol    : float       越界量 [MW]（= 原始动作到物理域的距离；与 RBB 的
                          `clip_amount` 在 `fluctuation_limit=inf` 下逐位相同）
    """
    a = np.asarray(action, dtype=float).reshape(-1)
    c = HSRL_VIOL_COEF if coef is None else float(coef)
    pb_lo, pb_hi = feasible_power(E)
    x = float(a[0])
    viol = max(0.0, x - pb_hi) + max(0.0, pb_lo - x)
    applied = np.array([float(np.clip(x, pb_lo, pb_hi))])
    return applied, float(c) * viol, float(viol)


def lower_reward_hsrl(viol, discharge_applied, viol_coef=None, deg_coef=None):
    """HSRL 版 L 奖励：

        r2 = −viol_coef·越界量[MW] − deg_coef·放电量[MW·min]

    越界量 = 原始动作超出每步物理域的量；退化只对**放电**计费（`max(0, a_L)`）。
    **不含** rulebase 波动裁剪罚（用户口径："reward 端去掉 rulebase 波动裁剪"）。
    """
    c = HSRL_VIOL_COEF if viol_coef is None else float(viol_coef)
    d = HSRL_DEG_COEF if deg_coef is None else float(deg_coef)
    return -(c * float(viol) + d * max(0.0, float(discharge_applied)))


def plan_need(g, sum_disch, n_rem, delta=HSRL_COST_DELTA):
    """摊还需量（公式的参照点）: `(g − Σ已执行)/(δ − t_L)` [MW·min/步 = MW]。

    `sum_disch` = 本块**已执行**的实际放电量之和（MW·min，**不含当前步**）；
    `n_rem`     = 含当前步的块内剩余步数 = δ − t_L。
    """
    n = max(1, int(n_rem))
    return (float(g) - float(sum_disch)) / float(n)


def lower_cost_hsrl(g, sum_disch, n_rem, discharge_applied,
                    delta=HSRL_COST_DELTA, return_info=False):
    """图中公式：`c_L = 1 − exp(−|need − a_L|)`，`need = (g − Σ)/(δ − t_L)`。

    * 完全按摊还计划执行（`a_L == need`）⇒ `c_L ≡ 0`；
    * 偏离 `x` ⇒ `1 − e^{−|x|}`（饱和，∈ [0, 1)）；
    * `n_rem = 1` 时 `need` = 本块剩余总量（最后一步必须把总放电量补齐）。
    """
    n = max(1, int(n_rem))
    need = plan_need(g, sum_disch, n, delta)
    dev = need - float(discharge_applied)
    c = 1.0 - float(np.exp(-abs(dev)))
    if return_info:
        return c, dict(g=float(g), sum_disch_before=float(sum_disch), n_rem=n,
                       delta=float(delta), need=float(need),
                       a_L=float(discharge_applied), dev=float(dev),
                       c_L=float(c), abs_dev=abs(float(dev)))
    return c


def build_L_obs_hsrl(u, pr, E, pg_prev, g, need):
    """HSRL 版 L 观测（6 维 = "H 观测 + 上层目标 + c_L 的参照点"）：

        [ u/10 , pr/1000 , SOC/0.5 , P_G(t−1)/5 , g/2.5 , need/2.5 ]

    `need = plan_need(g, Σ已放电, n_rem)` = c_L 公式里的参照点（**必填**，由调用方按
    "本步开始前"的 Σ 与 n_rem 算好；下一状态的观测要用**更新后**的 Σ、n_rem 重算）。

    ⚠ 残余局限：`need = (g−Σ)/n_rem` 一个方程两个未知量（Σ 与 n_rem 都不可观测）
    ⇒ 下一状态的 `need'` 仍不能精确预测（本步的代价则只依赖本步的 need/a_L，已经给全）。
    """
    return np.array([float(u) / OBS_U_SCALE,
                     float(pr) / OBS_PR_SCALE,
                     float(E) / OBS_E_SCALE,
                     float(pg_prev) / OBS_PG_SCALE,
                     float(g) / OBS_G_SCALE,
                     float(need) / OBS_NEED_SCALE], dtype=np.float64)


def amortized_rule_action(g, sum_disch, n_rem, E, eta_c=ETA_C, eta_d=ETA_D,
                          time=DT, rated_power=P_RATED, delta=HSRL_COST_DELTA):
    """HSRL 分支的 warmup 下层规则（**摊还执行器**，绝对值命令）[MW，正=充电]：

        pb = clip( −(g − Σ)/n_rem , 物理域 )

    它与 c_L 的参照点**同源** ⇒ 命令不被物理域裁掉时 `c_L ≡ 0`
    （"零成本动作 = 规则"这条既有原则在 HSRL 下继续成立）。
    """
    need = plan_need(g, sum_disch, n_rem, delta)
    pb = -need
    pb_lo, pb_hi = feasible_power(E, eta_c=eta_c, eta_d=eta_d, time=time,
                                  rated_power=rated_power)
    return float(np.clip(pb, pb_lo, pb_hi))


def g_from_warmup_signal(sig, scale=HSRL_WARMUP_G_SCALE):
    """warmup 上层第 6 维：`g = −scale·sig_walk`（见常量区的符号说明）。"""
    return -float(scale) * float(sig)


def random_g(rng=None, full=None):
    """warmup 对照臂（`--warmup_sched random`）的第 6 维采样：`U(−2.5, +2.5)` MW·min。"""
    r = np.random if rng is None else rng
    f = HSRL_G_FULL if full is None else float(full)
    return float(r.uniform(-f, f))


def dual_update_L_hsrl(agent, d_block, q_rate=None, src=None, ema=None,
                       lam_min=HSRL_L_LAM_MIN, lam_max=HSRL_L_LAM_MAX,
                       eta=HSRL_L_EMA_LAM_LR):
    """HSRL 版 L 对偶上升（乘性，与 `hst.dual_update_L` 同形）：

        rate ← q_rate（critic cost 的 Q 折成的"每块成本速率"）或 ema（实际块成本 EMA）
        λ2   ← clip( λ2 · exp(η2·(rate/d_block − 1)) , lam_min , lam_max )

    **分子默认改用 critic cost 的 Q**（`q_rate`，见 `q_cost_rate`）而不是实际块成本的 EMA：
    Q 是 bootstrap 出来的**期望**成本，比"单块的实现值"平滑得多 ⇒ λ 的**波动显著变小**
    （实际块成本重尾：实测策略期块 sum_c1 中位数 4.5 而最大 305）。
    代价是引入 critic 的**偏差**（尤其 twin 取 max 时高估）⇒ 故对偶默认用双臂**均值**
    （`HSRL_DUAL_Q_AGG`），并在 Q 尚未建立（None）时**回退** EMA，保证不停摆。

    `src='ema'` 或 `q_rate=None` ⇒ 退回原口径（A/B 对照臂，`--hsrl_dual_src ema`）。
    调用方需先 `agent.update_ema_cost(sum_c)`（EMA 仍然维护：日志 / curriculum / 回退都用它）。
    """
    _src = HSRL_DUAL_SRC if src is None else str(src)
    if _src == 'q' and q_rate is not None:
        rate = float(q_rate)
    else:
        rate = None if ema is None else float(ema)
    if rate is None or agent is None:
        return None
    ratio = rate / max(float(d_block), 1e-9)
    lam = float(agent.lam) * float(np.exp(float(eta) * (ratio - 1.0)))
    agent.lam = float(np.clip(lam, float(lam_min), float(lam_max)))
    return agent.lam


def summary():
    """启动时打印一行 HSRL 口径（与 `hc.summary()` 同风格）。"""
    return ('[HSRL] H6 = 块总放电量 g ∈ [−C_max, +D_max] MW·min (SOC+额定功率, 不含波动) | '
            'H r1 = pr·P_G·dt − pen5 − pen6 (无 bessdeg, 无块末 gap 罚) | '
            'L 动作 = 绝对 pb, 仅物理域裁剪 | r2 = −{:.0f}·越界 − {:.0f}·放电 | '
            'c_L = 1 − exp(−|(g−Σ)/n_rem − a_L|) | λ2 ∈ [{:.0f}, {:.0f}] 初值 {:.0f} | '
            'd_block_L = {} | L 观测(6 维) = [u/10, pr/1000, E/0.5, P_G/5, g/{:.1f}, '
            'need/{:.1f}] | 对偶分子 = {} (twin 聚合 = {})'.format(
                HSRL_VIOL_COEF, HSRL_DEG_COEF, HSRL_L_LAM_MIN, HSRL_L_LAM_MAX,
                HSRL_L_LAM_INIT, HSRL_D_BLOCK_L, HSRL_G_FULL, OBS_NEED_SCALE,
                ('critic cost 的 Q' if HSRL_DUAL_SRC == 'q' else '实际块成本 EMA'),
                HSRL_DUAL_Q_AGG))


# =============================================================================
# 4a. 对偶的分子: critic cost 的 Q → "每块成本速率"
# =============================================================================
def _cost_scale_of(agent):
    """agent 的 cost 通道回归缩放 s_c（与 `update_policy` 的取值口径一致）。"""
    sc = getattr(agent, 'cost_scale', None)
    if sc:
        return float(sc)
    vs = getattr(agent, 'value_scale', 1.0)
    return float(vs) if vs else 1.0


def cost_rate_from_q(q_raw, gamma, block=HSRL_BLOCK, horizon='infinite'):
    """把 critic cost 的**原始口径 Q** 折算成"每块成本速率"[与 c 同口径]。

    Q 是**折扣回报**，与"每块预算 d_block"不同量纲 ⇒ 必须按层的视界折算：

    * `horizon='infinite'`（H 层：γ 按**块**、视界 1/(1−γ)=20 块 ≪ 一天 288 块）：
          `rate = (1 − γ)·Q`          （几何级数 ⇒ 稳态每块速率）
    * `horizon='finite'`（L 层：γ 按**步**、episode = δ 步且块末 `done_L=True`）：
          `Q = Σ_{k<δ} γ^k·c_step`，常值 c_step 时 `Q = c_step·(1−γ^δ)/(1−γ)`
          ⇒ `rate = Q/K`，`K = (1−γ^δ)/(δ(1−γ))`（把"每步成本"折成"每块成本"）
    """
    g = float(gamma)
    if horizon == 'infinite':
        return (1.0 - g) * float(q_raw)
    n = int(block)
    k = (1.0 - g ** n) / (n * max(1e-9, 1.0 - g))
    return float(q_raw) / max(1e-9, k)


def q_cost_rate(agent, s_t, a_t=None, horizon='infinite', agg=None, gamma=None,
                block=HSRL_BLOCK, cost_scale=None, return_info=False):
    """用 **critic cost 的 Q** 估计"每块成本速率"（对偶的分子，替代实际 EMA）。

    取值约定（写死，便于 A/B 与审计）：
      * **状态** = 块首观测 `s_t`（H 是块级决策态；L 用**块的第一步**的观测 —— 块末观测
        已接近 episode 终止，`Q_c` 会退化成"最后一步的成本"，严重低估整块成本）；
      * **动作** = 缺省 `actor(s_t)`（确定性）；**调用方应传本块第一步"实际执行"的动作**
        `a_t` —— critic 的 TD 目标是**带探索噪声的实际 cost**，用无噪动作评估会**系统性低估**
        （`c_L = 1−exp(−|·|)`、`c1 = P_F + 20(P_F−3)⁺` 都是凸的 ⇒ 噪声下 E[c] > c(E)）。
        实测：L 层 rate 0.317（无噪）vs 1.79（实际 EMA，差 5.6×），改用实际动作后收敛到同量级。
        用 Q 而不是实际值的好处仍在：噪声的**单次实现**被 critic 的**期望**取代。
      * **critic** = 在线 `critic_c / critic_c2` 的 twin 聚合（默认 `mean`，见
        `HSRL_DUAL_Q_AGG` 的说明：回归目标已用 max，对偶再用 max 是双重高估）；
      * **口径还原**：critic 吃的是 `c/s_c` ⇒ 原始口径 `Q = q·s_c`（`_cost_scale_of`）。
    """
    import torch
    a_np = np.asarray(s_t, dtype=np.float32).reshape(1, -1)
    dev = getattr(agent, 'device', None)
    s = torch.as_tensor(a_np) if dev is None else torch.as_tensor(a_np, device=dev)
    a_given = None
    if a_t is not None:
        a_np2 = np.asarray(a_t, dtype=np.float32).reshape(1, -1)
        a_given = (torch.as_tensor(a_np2) if dev is None
                   else torch.as_tensor(a_np2, device=dev))
    with torch.no_grad():
        act = agent.actor(s) if a_given is None else a_given
        q1 = float(agent.critic_c([s, act]).reshape(-1)[0])
        q2 = float(agent.critic_c2([s, act]).reshape(-1)[0])
    _agg = HSRL_DUAL_Q_AGG if agg is None else str(agg)
    q_agg = {'mean': 0.5 * (q1 + q2), 'min': min(q1, q2), 'max': max(q1, q2)}[_agg]
    sc = _cost_scale_of(agent) if cost_scale is None else float(cost_scale)
    q_raw = q_agg * sc
    g = float(agent.discount if gamma is None else gamma)
    rate = cost_rate_from_q(q_raw, g, block=block, horizon=horizon)
    if return_info:
        return rate, dict(q1=q1, q2=q2, q_agg=q_agg, agg=_agg, s_c=sc, q_raw=q_raw,
                          gamma=g, horizon=horizon, rate=rate,
                          action_given=bool(a_given is not None))
    return rate


def recent_states(agent, n=HSRL_DUAL_RECENT_N):
    """回放缓冲里**最近 n 条**转移的 (states, actions, costs)（按**写入顺序**）。

    直接读 `memory._storage` / `memory._next_idx`（`PER_buffer.ReplayBuffer` 的口径）：
    `_storage` 是列表、`_next_idx` 是环形写指针 ⇒ 未环绕时取尾部、已环绕时按模取环尾。
    **不调用 `memory.sample()`** ⇒ 不消耗随机流、不改变训练轨迹（可复现性不被破坏）。
    """
    mem = getattr(agent, 'memory', None)
    st = list(getattr(mem, '_storage', []) or [])
    L = len(st)
    if L == 0:
        return (np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    k = min(max(1, int(n)), L)
    m = int(getattr(mem, '_maxsize', L) or L)
    if L < m:                                   # 未环绕：写入顺序 == 列表顺序
        idxs = list(range(L - k, L))
    else:                                       # 已环绕：_next_idx 指向最老的一条
        nx = int(getattr(mem, '_next_idx', 0))
        idxs = [int((nx - k + i) % m) for i in range(k)]
    s = np.asarray([st[i][0] for i in idxs], dtype=np.float32)
    a = np.asarray([st[i][1] for i in idxs], dtype=np.float32)
    c = np.asarray([float(st[i][2 + 1]) for i in idxs], dtype=np.float32)   # 第 4 项 = cost
    return s, a, c


def policy_batch_cost_rate(agent, n=HSRL_DUAL_RECENT_N, horizon='infinite', agg=None,
                           gamma=None, block=HSRL_BLOCK, cost_scale=None, return_info=False):
    """**S1 分子**：近端 n 条状态的批均值，动作取**当前策略** `π(s_i)`：

        rate = K_horizon · mean_i Q_c(s_i, π(s_i)) · s_c
        （H: K=(1−γ_H)；L: K=1/K_finite，K_finite=(1−γ_L^δ)/(δ(1−γ_L))）

    三个性质（`output/AIFFLOR-run73` 上离线验过前两条）：
      * **风无关**：状态取近端 n 条（n=256 ⇒ H 覆盖 ≈1 天）⇒ 当日风况被平均掉；
      * **反映策略进步**：动作是 `π(s_i)` 而非旧的行为动作 ⇒ 策略改进时分子随之下降；
      * **与 actor 的惩罚项同泛函** ⇒ 对偶的固定点 = "actor 自己的平均成本 Q = 预算"。
    """
    import torch
    s_np, _a_stored, _c_np = recent_states(agent, n)
    if s_np.shape[0] == 0:
        if return_info:
            return 0.0, dict(q_agg=float('nan'), n_batch=0, rate=0.0, empty=True)
        return 0.0
    dev = getattr(agent, 'device', None)
    s = torch.as_tensor(s_np) if dev is None else torch.as_tensor(s_np, device=dev)
    with torch.no_grad():
        act = agent.actor(s)
        q1 = agent.critic_c([s, act]).reshape(-1)
        q2 = agent.critic_c2([s, act]).reshape(-1)
    _agg = HSRL_DUAL_Q_AGG if agg is None else str(agg)
    if _agg == 'mean':
        q_pair = 0.5 * (q1 + q2)
    elif _agg == 'min':
        q_pair = torch.minimum(q1, q2)
    else:
        q_pair = torch.maximum(q1, q2)
    q_mean = float(q_pair.mean())
    sc = _cost_scale_of(agent) if cost_scale is None else float(cost_scale)
    q_raw = q_mean * sc
    g = float(agent.discount if gamma is None else gamma)
    rate = cost_rate_from_q(q_raw, g, block=block, horizon=horizon)
    if return_info:
        return rate, dict(q_agg=q_mean, q_raw=q_raw, s_c=sc, gamma=g, horizon=horizon,
                          n_batch=int(s_np.shape[0]), agg=_agg, rate=rate,
                          cost_recent_mean=float(_c_np.mean()) if _c_np.size else float('nan'))
    return rate


def update_batch_cost_rate(agent, stats=None, horizon='infinite', agg=None, gamma=None,
                           block=HSRL_BLOCK, cost_scale=None, return_info=False):
    """**`update_batch` 分子**（λ1 默认）：直接用 `update_policy` **本块已算好的** cost Q 批均值。

    `update_stats` 里每条对应一次 `update_policy`，其中

      * `q_c_mean` / `q_c2_mean` = twin cost critic 在该次采样的 **PER 批**上、**行为动作**
        下的 Q 均值（`mean_i Q_c(s_i, a_i)`，`a_i` 来自 buffer）；
      * ⚠ 这两个数在记录时**已经乘回 `s_c`**（见 `aifflor_ddpg.update_policy` 末尾的口径还原）
        ⇒ 它们**已是原始口径**，这里**不再乘 `s_c`**（重复乘会差 100 倍，是很隐蔽的坑）。

    口径含义（用户选择）：该批均值随**缓冲的自然淘汰**（环形容量 = `--rmsize`）而更新 ——
    旧 transition 被覆盖后均值跟着当前行为分布走 ⇒ 作为"策略进步"的代理；且批是随机抽的、
    跨天混合 ⇒ **不含当日风况**（消掉风况引起的慢漂）。代价：它同时带 PER 过采偏差与
    行为动作（探索噪声）成分，且跟随速度受 `rmsize` 限制。

    统计缺失（`record_update_stats` 关 / 本块还没更新 / warmup 期）⇒ 返回 `None`，
    由调用方回退到 EMA（`dual_update_*_hsrl` 的 `ema=` 分支）。
    """
    st = getattr(agent, 'update_stats', None) if stats is None else stats
    if not st:
        if return_info:
            return None, dict(q_agg=float('nan'), n_update=0, empty=True)
        return None
    keys = set(st[0].keys())
    if 'q_c_mean' not in keys:
        if return_info:
            return None, dict(q_agg=float('nan'), n_update=0, empty=True)
        return None
    q1 = float(np.mean([float(s['q_c_mean']) for s in st]))
    q2 = float(np.mean([float(s['q_c2_mean']) for s in st])) if 'q_c2_mean' in keys else q1
    _agg = HSRL_DUAL_Q_AGG if agg is None else str(agg)
    q_raw = {'mean': 0.5 * (q1 + q2), 'min': min(q1, q2), 'max': max(q1, q2)}[_agg]
    sc = _cost_scale_of(agent) if cost_scale is None else float(cost_scale)
    g = float(agent.discount if gamma is None else gamma)
    rate = cost_rate_from_q(q_raw, g, block=block, horizon=horizon)
    if return_info:
        # `q_agg` 统一报成**网络（归一化）口径**（÷s_c），便于与点估计/近端批的列直接比较
        return rate, dict(q_agg=q_raw / max(1e-12, sc), q_raw=q_raw, q1=q1, q2=q2, agg=_agg,
                          s_c=sc, gamma=g, horizon=horizon, rate=rate,
                          n_update=len(st), empty=False)
    return rate


def dual_update_H_hsrl(agent, d_block, q_rate=None, src=None, ema=None,
                       lam_min=None, lam_max=None, eta=None):
    """HSRL 版 H 对偶上升（乘性，分子默认 = critic cost 的 Q）—— 与 L 版同形。

    ⚠ λ1 的**区间**必须与物理价口径一致（`λ_phys = λ_actor/lam_scale = λ_actor·s_p/s_c`）：
    区间/初值由 `HSRL_LAM1_PHYS_*` 给出，见那里的推导（用户口径：越限多出的 1 MW 的
    最大惩罚要 > 980×2 元）。
    """
    _src = HSRL_DUAL_SRC if src is None else str(src)
    if _src == 'q' and q_rate is not None:
        rate = float(q_rate)
    else:
        rate = None if ema is None else float(ema)
    if rate is None or agent is None:
        return None
    _lo, _hi, _ = lam1_phys_bounds()
    lo = _lo if lam_min is None else float(lam_min)
    hi = _hi if lam_max is None else float(lam_max)
    _eta = HSRL_LAM1_ETA if eta is None else float(eta)
    ratio = rate / max(float(d_block), 1e-9)
    lam = float(agent.lam) * float(np.exp(_eta * (ratio - 1.0)))
    agent.lam = float(np.clip(lam, lo, hi))
    return agent.lam


# =============================================================================
# 4b. warmup 摊还执行器的启动自检（替代残余动作项目的 check_residual_baseline）
# =============================================================================
def check_amortized_baseline(n_cases=400, seed=12345):
    """HSRL 的 warmup 下层规则（摊还执行器）自检 —— 随机 (E, g, Σ, n_rem) 上的硬断言：

    ① 命令 `clip(−(g−Σ)/n_rem, 物理域)` **恒落在物理域内**；
    ② 命令**未被裁剪**时 `c_L ≡ 0`（"零成本动作 = 规则"这条原则在 HSRL 下继续成立）；
       被裁剪时 `c_L > 0` 只可能来自"物理上做不到"（诊断量 `n_clipped`）；
    ③ 该命令经 env 口径的一步 SOC 更新（`ΔSOC = (η_c·max(pb,0) − max(−pb,0)/η_d)·dt`）
       后 SOC 仍在 `[E_min, E_max]` 内。

    为什么必须做：warmup 期入池的 (s, a, r2, c_L) 会被网络当成"示范"。若摊还执行器与
    c_L 的参照点不同源（例如 u/η 的口径漂移、n_rem 口径错位），warmup 行就会自相矛盾
    （教网络"某个 a 是零成本"而实际不是）。⇒ 运行期直接断言，不做"提示后继续"。
    """
    rg = np.random.default_rng(seed)
    n_checked = n_clipped = 0
    max_c_unclipped = 0.0
    max_c_clipped = 0.0
    fail = 0
    for _ in range(int(n_cases)):
        E = float(rg.uniform(E_MIN, E_MAX))
        n_rem = int(rg.integers(1, HSRL_BLOCK + 1))
        c_max, d_max = feasible_total(E)
        g = float(rg.uniform(-c_max, d_max))
        sum_disch = float(rg.uniform(-c_max, d_max))
        pb = amortized_rule_action(g, sum_disch, n_rem, E)
        pb_lo, pb_hi = feasible_power(E)
        if not (pb_lo - 1e-12 <= pb <= pb_hi + 1e-12):
            fail += 1
        need = plan_need(g, sum_disch, n_rem)
        c_L = lower_cost_hsrl(g, sum_disch, n_rem, discharge_of(pb))
        unclipped = (pb_lo - 1e-12 <= -need <= pb_hi + 1e-12)
        if unclipped:
            if c_L > 1e-12:
                fail += 1
            max_c_unclipped = max(max_c_unclipped, c_L)
        else:
            n_clipped += 1
            max_c_clipped = max(max_c_clipped, c_L)
        e_new = E + (max(pb, 0.0) * ETA_C - max(-pb, 0.0) / ETA_D) * DT
        if not (E_MIN - 1e-9 <= e_new <= E_MAX + 1e-9):
            fail += 1
        n_checked += 1
    return dict(n_checked=n_checked, n_clipped=n_clipped,
                max_c_unclipped=max_c_unclipped, max_c_clipped=max_c_clipped,
                fail=fail)


# =============================================================================
# 5. 训练中的测试（20 步；电价恒 300、风速随机段、开局 SOC = 0.25）
# =============================================================================
def twenty_step_test_hsrl(agent_H, agent_L, env_test, rbb_system, wind_pool,
                          seed=None, phase_mode=0, phase_table=None,
                          n_steps=HSRL_TEST_STEPS, price=HSRL_TEST_PRICE,
                          soc_start=HSRL_TEST_SOC0, penalty6_coef=None,
                          viol_derate_coef=None,
                          wind_col=None, wind_start=None, random_wind=False):
    """HSRL 分支的 20 步测试（替换旧 9-case）。

    协议（用户口径）：电价恒 `price`(=300)；风速 = **固定的 20 步段**
    （默认 `HSRL_TEST_WIND_COL=120 / HSRL_TEST_WIND_START=483`，见那里选段依据）；
    开局 SOC = `soc_start`(=0.25 MWh)。`wind_pool` 由 main 传入
    （`env_test.wind_noise.iloc[:, 1:].to_numpy()`，即只含"天"列）。

    **为什么固定段**：每次触发都用**同一条场景** ⇒ test20 曲线是"配对"的，可直接看收敛
    （随机段会把场景间差异混进曲线，看不出策略是否变好）。要看泛化再开 `random_wind=True`：
    这时用**独立** `RandomState` 抽（列+起点），不吃训练流的随机数、也不改变训练轨迹；
    `seed>0` 则每次都抽到同一段（等价于固定，但由种子决定）。

    r1 的记账与**训练同口径**：`r1 = pr·P_G·dt − (pen5 + pen6)`（块级动作越界罚按**步**摊入，
    与 `train_hsrl`/旧 `train()` 的写法一致 ⇒ 一个块内被计 5 次），并同样支持
    `--viol_derate`（默认 0 = 关闭）。

    Returns
    -------
    per_block : list[dict]  长度 = n_steps/5 的逐块指标
    overall   : dict        20 步总量/均值
    meta      : dict        wind_col / wind_start / wind_seq / price / soc_start / soc_end / dsoc
    """
    pool = np.asarray(wind_pool, dtype=float)
    if pool.ndim != 2:
        raise ValueError('twenty_step_test_hsrl: wind_pool 必须是 2 维 (行=天内的步, 列=天)')
    n_row, n_col = pool.shape
    if n_row < n_steps:
        raise ValueError('twenty_step_test_hsrl: wind_pool 只有 {} 行, 不足 {} 步'
                         .format(n_row, n_steps))
    if random_wind:
        rs = np.random.RandomState(None if (seed is None or int(seed) < 0) else int(seed))
        col = int(rs.randint(0, n_col))
        start = int(rs.randint(0, n_row - n_steps + 1))
    else:
        col = int(HSRL_TEST_WIND_COL if wind_col is None else wind_col)
        start = int(HSRL_TEST_WIND_START if wind_start is None else wind_start)
    col = int(np.clip(col, 0, n_col - 1))
    start = int(np.clip(start, 0, n_row - n_steps))
    wind_seq = pool[start:start + n_steps, col].copy()
    price_seq = np.full(n_steps, float(price), dtype=float)

    obs = env_test.ScenarioReset(wind_seq, price_seq)
    env_test.E = float(soc_start)                  # ⚠ env.ScenarioReset 硬编码 E=5.0，这里改回
    obs[2] = float(soc_start)

    time = DT
    minute0 = int(rep_minute_for_price_level(get_price_level(price)))
    norm_obs = Normalize_H(obs, minute0, phase_mode, phase_table)
    agent_H.is_training = False
    agent_L.is_training = False

    n_blk = int(n_steps // 5)
    per_blk = [{'r1': 0.0, 'c1': 0.0, 'r2': 0.0, 'c2': 0.0, 'deg': 0.0,
                'viol': 0.0, 'pen5': 0.0, 'pen6': 0.0, 'cvm': 0, 'pf_viol': 0,
                'profit': 0.0,
                'soc_start': float(soc_start), 'soc_end': float(soc_start),
                'g_raw': 0.0, 'g': 0.0, 'sum_disch': 0.0, 'pb_sum': 0.0,
                'cL_max': 0.0, 'dev_abs_sum': 0.0, 'need_sum': 0.0, 'n_step': 0}
               for _ in range(n_blk)]

    g = 0.0
    g_raw = 0.0
    sum_disch = 0.0
    pw_tru = 0.0
    action_H = None
    for step in range(int(n_steps)):
        bi = step // 5
        if step % 5 == 0:
            raw_H = np.asarray(agent_H.select_action(norm_obs), dtype=float).reshape(-1)
            action_H, pen5, pen6 = check_action_H_hsrl(raw_H, float(obs[2]),
                                                       pen6_coef=penalty6_coef)
            _derate = violation_derate(raw_H, viol_derate_coef)   # 默认 0 ⇒ 恒 1.0
            g_raw = float(raw_H[5])
            g = float(action_H[5])
            sum_disch = 0.0
            pw_tru = env_test.PowerOutput(obs[1], action_H)
            per_blk[bi]['soc_start'] = float(obs[2])
            per_blk[bi]['g_raw'] = g_raw
            per_blk[bi]['g'] = g
        n_rem = 5 - (step % 5)
        need_L = plan_need(g, sum_disch, n_rem)
        obs_L = build_L_obs_hsrl(obs[1], obs[0], obs[2], obs[-1], g, need_L)
        if step % 5 == 0:
            agent_L.reset(obs_L)
        raw_L = agent_L.select_action(obs_L)
        pb_cmd = float(np.asarray(raw_L, dtype=float).reshape(-1)[0])

        ptur = float(pw_tru)
        pb_lo, pb_hi = feasible_power(float(obs[2]))
        viol = max(0.0, pb_cmd - pb_hi) + max(0.0, pb_lo - pb_cmd)
        pbrule, rbb_info = rbb_system.compute_pbrule(
            pb_original=pb_cmd, ptur=ptur, pg_prev=float(obs[-1]), E=float(obs[2]))

        action_rbb = np.concatenate([action_H, np.array([pbrule])])
        pw_tru2, obs2, applied_L, _pen_env, _done, _done_L, _info = \
            env_test.ScenarioStep(action_rbb[:6], action_rbb[6:], pw_tru)
        pb_app = float(np.asarray(applied_L, dtype=float).reshape(-1)[0])
        a_L = discharge_of(pb_app)

        c_L, c_info = lower_cost_hsrl(g, sum_disch, n_rem, a_L, return_info=True)
        r2 = lower_reward_hsrl(viol, a_L)
        r1, c1_step, details = cal_rew_rb_hsrl(
            obs, obs2, action_rbb, rbb_info, penalty_NN=pen5 + pen6, time=time,
            ptur=ptur, pb_cmd=pb_cmd, turbine_derate=_derate)

        pc = per_blk[bi]
        pc['r1'] += r1
        pc['c1'] += c1_step
        pc['r2'] += r2
        pc['c2'] += c_L
        pc['deg'] += HSRL_DEG_COEF * max(0.0, a_L)
        pc['viol'] += viol
        pc['pen5'] += pen5
        pc['pen6'] += pen6
        if c1_step > 0:
            pc['cvm'] += 1
        if details['P_F'] > 3.0 + 1e-4:
            pc['pf_viol'] += 1
        pc['profit'] += (float(obs2[-1]) * float(obs[0]) * time
                         - DEG_COST_DISCHARGE * max(-pbrule, 0.0) * time)
        pc['soc_end'] = float(obs2[2])
        pc['pb_sum'] += float(pbrule)
        pc['cL_max'] = max(pc['cL_max'], c_L)
        pc['dev_abs_sum'] += c_info['abs_dev']
        pc['need_sum'] += c_info['need']
        pc['n_step'] += 1
        sum_disch += a_L
        pc['sum_disch'] = sum_disch

        norm_obs = Normalize_H(obs2, minute0 + step + 1, phase_mode, phase_table)
        obs = obs2
        pw_tru = pw_tru2

    agent_H.is_training = True
    agent_L.is_training = True

    # ---- 存货结算（与 9-case 同口径: 现金流 + ΔSOC·外生账面价 hc.PR_REF）----
    for m in per_blk:
        m['dsoc'] = m['soc_end'] - m['soc_start']
        m['profit_econ'] = m['profit'] + m['dsoc'] * float(hc.PR_REF)
        m['dev_mean'] = m['dev_abs_sum'] / max(m['n_step'], 1)
        m['need_mean'] = m['need_sum'] / max(m['n_step'], 1)

    _sum_keys = ('r1', 'c1', 'r2', 'c2', 'deg', 'viol', 'pen5', 'pen6', 'profit')
    _mean_keys = ('g', 'g_raw', 'sum_disch', 'cL_max', 'dev_mean', 'need_mean')
    overall = {k: float(sum(m[k] for m in per_blk)) for k in _sum_keys}
    overall.update({k: float(np.mean([m[k] for m in per_blk])) for k in _mean_keys})
    overall['cvm'] = int(sum(m['cvm'] for m in per_blk))
    overall['pf_viol'] = int(sum(m['pf_viol'] for m in per_blk))
    overall['soc_start'] = float(per_blk[0]['soc_start'])
    overall['soc_end'] = float(per_blk[-1]['soc_end'])
    overall['dsoc'] = overall['soc_end'] - overall['soc_start']
    overall['profit_econ'] = overall['profit'] + overall['dsoc'] * float(hc.PR_REF)

    meta = {'wind_col': col, 'wind_start': start, 'wind_seq': wind_seq,
            'price': float(price), 'soc_start': float(per_blk[0]['soc_start']),
            'soc_end': float(per_blk[-1]['soc_end']),
            'dsoc': overall['dsoc'], 'n_steps': int(n_steps)}
    return per_blk, overall, meta
