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

⑥ **L 观测**（5 维；用户口径 = "H 观测 + 上层目标"）：

       [ u/10 , pr/1000 , SOC/0.5 , P_G(t−1)/5 , g/2.5 ]

   ⚠ **已知局限（刻意保留的原方法形态）**：观测里**没有** Σ 与 n_rem ⇒ c_L 的参照量
   `(g−Σ)/n_rem` 对策略不可观测（单步转移对 c_L 非 Markov；PER 按单步采样会失真）。
   若要修，见 `aifflor/README.md` 的 `--hsrl` 小节（在那里记了 7 维备选）。

⑦ **L 对偶重标**：c_L 无量纲、r2 是"元"量级 ⇒ λ2 ∈ [2, 50]（初值 10；乘性**对数**步长
   0.02 不变）；每块预算 `d_block_L = 0.5`（= 允许 5 步平均 0.1 的计划偏差），
   curriculum 地板 0.05。

⑧ **warmup 下层 = 摊还执行器** `amortized_rule_action`：命令 = `clip(−(g−Σ)/n_rem, 物理域)`
   ⇒ 入池动作（绝对值）的 c_L ≡ 0、r2 = −退化项（"零成本动作"自洽）。
   warmup 上层第 6 维 = `−HSRL_WARMUP_G_SCALE · sig_walk`（价格游走：低谷充电 ⇒ g<0、
   高峰放电 ⇒ g>0；符号与旧的"SOC 目标游走"**相反**，因为放电量与 SOC 变动反号）。

⑨ **训练中的测试**：`--hsrl` 下不再跑 9-case，改跑 `twenty_step_test_hsrl`
   （电价恒 300、风速从 `FFC_speed_noise_1.csv` 随机取一列 + 随机起点抽 20 步、
   开局 SOC = 0.25；触发频率与旧 9-case 相同：`step > warmup` 且 `episode_step % 480 == 0`）。
"""
import numpy as np

import hsrl_cost as hc

try:                                        # 主路径: aifflor 作为包
    from aifflor.aifflor_myfunc import (
        get_wind_level, get_price_level, Cal_soc_deviation,
        DEG_COST_DISCHARGE, DEG_COST_CHARGE,
        PENALTY_NN_H_COEF, PENALTY_NN_L_COEF, W_GAP, C1_RISK_COEF,
        Normalize_H, rep_minute_for_price_level,
    )
except ImportError:                         # 兼容: 直接把 aifflor/ 放进 sys.path
    from aifflor_myfunc import (            # type: ignore
        get_wind_level, get_price_level, Cal_soc_deviation,
        DEG_COST_DISCHARGE, DEG_COST_CHARGE,
        PENALTY_NN_H_COEF, PENALTY_NN_L_COEF, W_GAP, C1_RISK_COEF,
        Normalize_H, rep_minute_for_price_level,
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
HSRL_D_BLOCK_L = 0.5            # 每块 sum_c_L 预算（= 5 步平均 0.1 的计划偏差）
HSRL_D_BLOCK_L_MIN = 0.05       # curriculum 地板
HSRL_L_COST_LIMIT = 0.25        # agent_L.cost_limit（仅 non-EMA 分支用）
HSRL_L_CURRICULUM_BLOCKS = 50   # 预算减半所需连续满足块数（= EMA 时间常数 1/α）

# ---- L 层：观测刻度 ----
OBS_U_SCALE = 10.0              # 风速 u [m/s] → /10
OBS_PR_SCALE = 1000.0           # 电价 pr [元/MWh] → /1000
OBS_E_SCALE = E_MAX             # SOC E [MWh] → /0.5
OBS_PG_SCALE = 5.0              # P_G(t−1) [MW] → /5
OBS_G_SCALE = HSRL_G_FULL       # 上层目标 g [MW·min] → /2.5

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


def build_L_obs_hsrl(u, pr, E, pg_prev, g):
    """HSRL 版 L 观测（5 维，用户口径 = "H 观测 + 上层目标"）：

        [ u/10 , pr/1000 , SOC/0.5 , P_G(t−1)/5 , g/2.5 ]

    ⚠ 不含 Σ / n_rem ⇒ 对 c_L 的参照量非 Markov（见模块 docstring ⑥，刻意保留）。
    """
    return np.array([float(u) / OBS_U_SCALE,
                     float(pr) / OBS_PR_SCALE,
                     float(E) / OBS_E_SCALE,
                     float(pg_prev) / OBS_PG_SCALE,
                     float(g) / OBS_G_SCALE], dtype=np.float64)


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


def dual_update_L_hsrl(agent, d_block):
    """HSRL 版 L 对偶上升（与 `hst.dual_update_L` 同形，仅区间/地板不同）：

        λ2 ← clip( λ2 · exp(η2·(EMA_c_L/d_block − 1)) , 2 , 50 )

    调用方需先 `agent.update_ema_cost(sum_c_L)`。不动点 = 预算刚好满足；比值 < 1 时回落。
    """
    if agent is None or getattr(agent, 'ema_cost', None) is None:
        return None
    ratio = float(agent.ema_cost) / max(float(d_block), 1e-9)
    lam = float(agent.lam) * float(np.exp(HSRL_L_EMA_LAM_LR * (ratio - 1.0)))
    agent.lam = float(np.clip(lam, HSRL_L_LAM_MIN, HSRL_L_LAM_MAX))
    return agent.lam


def summary():
    """启动时打印一行 HSRL 口径（与 `hc.summary()` 同风格）。"""
    return ('[HSRL] H6 = 块总放电量 g ∈ [−C_max, +D_max] MW·min (SOC+额定功率, 不含波动) | '
            'H r1 = pr·P_G·dt − pen5 − pen6 (无 bessdeg, 无块末 gap 罚) | '
            'L 动作 = 绝对 pb, 仅物理域裁剪 | r2 = −{:.0f}·越界 − {:.0f}·放电 | '
            'c_L = 1 − exp(−|(g−Σ)/n_rem − a_L|) | λ2 ∈ [{:.0f}, {:.0f}] 初值 {:.0f} | '
            'd_block_L = {} | L 观测 = [u/10, pr/1000, E/0.5, P_G/5, g/{:.1f}]'.format(
                HSRL_VIOL_COEF, HSRL_DEG_COEF, HSRL_L_LAM_MIN, HSRL_L_LAM_MAX,
                HSRL_L_LAM_INIT, HSRL_D_BLOCK_L, HSRL_G_FULL))


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
                          soc_start=HSRL_TEST_SOC0, penalty6_coef=None):
    """HSRL 分支的 20 步测试（替换旧 9-case）。

    协议（用户口径）：电价恒 `price`(=300)；风速 = 从 `wind_pool` 里**随机取一列（某天）
    + 随机起点**截取 `n_steps`(=20) 步；开局 SOC = `soc_start`(=0.25 MWh)。
    `wind_pool` 由 main 传入（`env_test.wind_noise.iloc[:, 1:].to_numpy()`，即只含"天"列）。

    `seed` 为 None / 负数 ⇒ 每次触发**重新随机**（默认；用一个**独立** RandomState，
    不吃训练流的随机数，也不改变训练轨迹）；给正数则固定可复现。

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
    rs = np.random.RandomState(None if (seed is None or int(seed) < 0) else int(seed))
    col = int(rs.randint(0, n_col))
    start = int(rs.randint(0, n_row - n_steps + 1))
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
                'viol': 0.0, 'pen6': 0.0, 'cvm': 0, 'pf_viol': 0, 'profit': 0.0,
                'soc_start': float(soc_start), 'soc_end': float(soc_start),
                'g_raw': 0.0, 'g': 0.0, 'sum_disch': 0.0, 'pb_sum': 0.0,
                'cL_max': 0.0, 'dev_abs_sum': 0.0, 'n_step': 0}
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
            action_H, _pen5, pen6 = check_action_H_hsrl(raw_H, float(obs[2]),
                                                        pen6_coef=penalty6_coef)
            g_raw = float(raw_H[5])
            g = float(action_H[5])
            sum_disch = 0.0
            pw_tru = env_test.PowerOutput(obs[1], action_H)
            per_blk[bi]['soc_start'] = float(obs[2])
            per_blk[bi]['g_raw'] = g_raw
            per_blk[bi]['g'] = g
        n_rem = 5 - (step % 5)
        obs_L = build_L_obs_hsrl(obs[1], obs[0], obs[2], obs[-1], g)
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
            obs, obs2, action_rbb, rbb_info, penalty_NN=pen6, time=time,
            ptur=ptur, pb_cmd=pb_cmd)

        pc = per_blk[bi]
        pc['r1'] += r1
        pc['c1'] += c1_step
        pc['r2'] += r2
        pc['c2'] += c_L
        pc['deg'] += HSRL_DEG_COEF * max(0.0, a_L)
        pc['viol'] += viol
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

    _sum_keys = ('r1', 'c1', 'r2', 'c2', 'deg', 'viol', 'pen6', 'profit')
    _mean_keys = ('g', 'g_raw', 'sum_disch', 'cL_max', 'dev_mean')
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
