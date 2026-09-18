"""
HSRL 下层 SOC 目标语义 —— 单一真源
====================================
本项目有**两种**下层信号语义，本模块是其中"**SOC 目标**"这一支的唯一实现：

    a6 ∈ [-0.5, 0.5]  →  soc_target = clip(3 + 5·a6, 0.5, 5.5)   (块级(5 步)固定)

另一支"影子电价"（`a6·2 ∈ (−1,1)` 作影子价、`r2` 含影子套利项、`c2` 以 SOC 安全带
为参照）留在 `per` / `prrbb` 两个项目里，本模块不涉及。

使用本模块的（下层 RL + SOC 目标）项目
--------------------------------------
* `td3per`    —— 上层 = td3perrb 口径（非分段）
* `td3prrbb`  —— 上层 = TD3 + PRRBB 分段奖励（wT/wB 风价分档保留）
* `prle`      —— 上层 = PRLE_DDPG + 分段 + LLM 阈值 relabel

三者的**下层口径逐位相同**（`agentutil/test_soctarget_uniform.py` 有对照断言），
差异只在各自的上层奖励分段与观测约定。

口径：两区间分治（r2 ↦ 平抑范围，c2 ↦ Soctrack 范围；两者同为 MW）
-----------------------------------------------------------------
每步（块内 `t = step % 5`，`n_rem = 5 − t` = 含当前步的块内剩余步数）：

    delta = soc_target − SOC_prev                       [MWh]
    g     = eta_c·dt (delta>0) / dt/eta_d (delta<0)     [MWh/MW]

    ① Soctrack 范围（尽力时间表）
         B = [ min(δ/g, δ/(g·n_rem)) , max(δ/g, δ/(g·n_rem)) ]      [MW]
             摊还端（剩余步数内推完） … 一步填平端；n_rem=1 时退化为一点
    ② 平抑范围（rule 可行窗口）
         W = [pbrulemin, pbrulemax]                                 [MW]
    ③ 参考集合
         F = B ∩ W                                  （"rule 内 best 范围"）
         若 F ≠ ∅ ⇒ 参考集合就是**区间** F（L 可在其中任选努力水平）
         若 F = ∅ ⇒ **不存在范围，只有纯粹的一个"竭力值"**（零宽单点）
                     r* = W_hi（B 在窗口上方）/ W_lo（B 在窗口下方）
                        = 该状态下尽力所能达到的最好的一点

    ④ 两个通道（单位一致：MW）
         clip_amount = |pb_cmd − pbrule|            （命令与实际之差；**含两段 clip**）
             = |pb_cmd − clip(pb_cmd, ±10)|   ← ① CheckAction 执行器 ±10 的夹裁量
             + |clip(pb_cmd, ±10) − pbrule|   ← ② RBB 平抑窗口/physics 的夹裁量
         r2 = −L_CLIP_COEF · min(clip_amount, L_CLIP_CAP)     ≤ 0   "越窗：做了不该做的"
         trackerr = dist(pbrule, 参考集合)          ≥ 0   "迟到：该做的没做"
                    F = ∅ 时 = |pbrule − r*|：只有恰好顶到 r* 才为 0，两侧偏离都线性计费
         c2 = +L_TRACK_COEF · trackerr

性质（本设计的全部行为）
------------------------
* **零成本集合 = 参考集合**：`F ≠ ∅` 时是**区间**（L 可在"摊还端 … 一步填平端"之间
  任选努力水平 ⇒ 不再被逼走最激进端；更小 |pb| ⇒ 更少退化/波动/自致窗口吞吐）；
  `F = ∅` 时是**零宽单点**（见下条）。
* **`F = ∅` ⇒ 只有那一个"竭力值"免费**：目标在平抑窗口内本来就做不到（结构性不可行）
  时，参考集合**不是区间**，只有唯一一点
      r* = clip(δ/g, W) = W 里离 Soctrack 范围 B 最近的端点
  只有 `pbrule == r*`（真顶到该端点）才 `trackerr = 0`；**任何偏离都按 `|pbrule − r*|`
  线性计费**。即：真正不可达的那段缺口不计费（由上层块末 gap 罚承接），但"没尽力"
  照罚 ⇒ `sum_c2` 度量的是"下层有没有尽力"，而不是"H 的目标有多远"，λ2 不会被顶死。
  注：单点比区间**更好学**——区间内部梯度恒 0（无信号），单点两侧都有斜率 ±1 的线性
  梯度，把策略推回该端点。
* **`trackerr(规则动作) ≡ 0`**：规则 `pb = clip(δ/g, W)` 在 `F ≠ ∅` 时落在 F 的上端、
  在 `F = ∅` 时正是那个竭力值 r* ⇒ "零成本动作 = 规则 SocTracker"这条既有原则不变。
* **`r2` 计两段 clip（不是只计窗口那一段）**：`clip_amount = |命令 − pbrule|` 天然等于
  "(±10 执行器夹裁量) + (平抑窗口夹裁量)"。RBB 系统**夹了两次**：`CheckAction` 把命令夹到
  ±rated_power(=±10)，再由 `compute_pbrule` 夹到平抑窗口 —— 两次都要计费。
  两个直接后果：
    - 各 main 必须把**命令值**（CheckAction 之前）传给 `compute_pbrule`；否则 `clip_amount`
      只含窗口那段，而 `|命令| > 10` 时 `pbrule` 与 r2 都不再随动作变化 ⇒ **该方向梯度恒 0
      （死区）**，策略被推到饱和后无法被拉回（旧 run165: max|raw|≈3719 MW 而 clip≡10）。
    - 传命令值**不改变 `pbrule`**（窗口 ⊆ ±10 ⇒ `clip(clip(cmd,±10),W) ≡ clip(cmd,W)`），
      因此环境/施加功率/其它通道全部不变，只有 r2 与记录列变化。
  封顶 `L_CLIP_CAP = 20 MW` 只截断病态饱和幅值（不改变最优动作：越界方向在 0~10 MW 仍有斜率 1）。
* **`n_rem = 1` 时 B 退化为一点**（= 一步填平）⇒ 最后一步必须精确到位，与上层块末
  gap 罚同语义；此时 `c2 = L_TRACK_COEF·g·|pbrule − pb_est|`，即旧 `slack` 口径
  （旧口径是"一步填平留点"的特例）。
* **窗口不可行**（`pbrulemin > pbrulemax`，被迫吞吐不可归责于下层）⇒ trackerr = 0。

L 层数值口径（与 `--value_scale` 无关：下层不做值函数归一化）
-------------------------------------------------------------
2026 年那轮统一曾把 `r2`/`c2` 整体 ÷100（clip 系数 100→1、c2 系数 1000→10，c2 仍是
MWh 口径）。本轮**把 c2 也换成 MW 口径**（与 r2 一致），于是预算/对偶步长必须按
"每 MWh 搁置量 = 1/g ≈ 61.2 (MW·step)"重新标定：

| 常量 | 旧（10×MWh 口径） | 新（MW 口径） | 物理等价 |
|---|---|---|---|
| `L_D_BLOCK` | 0.2 | **1.2** | 允许每块搁置 ≈0.02 MWh 的可达量（≈0.4 步能力） |
| `L_D_BLOCK_MIN` | 0.025 | **0.15** | curriculum 地板 ≈0.0025 MWh（≈0.05 步能力） |
| `L_COST_LIMIT` | 0.1 | **0.6** | `agent_L.cost_limit`（仅 non-EMA 分支用） |
| `L_EMA_LAM_LR` | 1.0 | **0.16** | 加性对偶步长 η2，保持同一物理步长（1.0/6.1） |
| `L_LAM_MIN/MAX` | 10 / 500（价格口径） | **1 / 20** | λ2 现在读作"MW 追踪缺口 : MW 越窗"的兑换率 |

⇒ `L_D_BLOCK / L_TRACK_COEF / L_CLIP_COEF` 三者在同一量纲上，λ2 的区间也是同一量纲
（`agentutil/test_l_track_range.py` 断言这些不变量）。
"""
import numpy as np

import hsrl_cost as hc

# ---------------------------------------------------------------- 目标映射
SOC_TARGET_CENTER = 3.0
SOC_TARGET_SPAN = 5.0
SOC_TARGET_MIN, SOC_TARGET_MAX = 0.5, 5.5

# ---------------------------------------------------------------- L 层数值口径（r2/c2 同为 MW）
L_CLIP_COEF = 1.0        # r2 = −L_CLIP_COEF · min(clip_amount, L_CLIP_CAP)   [MW]
L_CLIP_CAP = 20.0        # r2 封顶 [MW]：clip_amount = |命令 − pbrule| 的上限
                         #   实测"平抑窗口"那一段最大 = 20 (= |+10 − (−10)|) ⇒ 20 不截断任何
                         #   真实窗内容量；只在 ±10 之外被夹掉的量把总差异推得更大时截断
                         #   （防旧 run165 式 "raw≈900 ⇒ r2≈−900/步" 的 critic 爆量）。
                         #   注意: 封顶只压幅值不改变最优动作 —— 越界方向仍有梯度（0~10 MW 斜率 1）。
L_TRACK_COEF = 1.0       # c2 = +L_TRACK_COEF · trackerr           [MW]
L_SOC_COEF = L_TRACK_COEF   # 兼容旧名（旧口径里它是 MWh→代价的系数）
L_D_BLOCK = 1.2          # 每块 sum_c2 预算（= 允许搁置 0.02 MWh 的可达量）
L_D_BLOCK_MIN = 0.15     # curriculum 地板
L_COST_LIMIT = 0.6       # agent_L.cost_limit（仅 non-EMA 分支用）
L_EMA_LAM_LR = 0.16      # 加性对偶步长 η2（保持 ÷100 时代的物理步长）
L_LAM_MIN = 1.0          # λ2 下限（兑换率口径）
L_LAM_MAX = 20.0         # λ2 上限
L_LAM_INIT = 1.0         # λ2 初值

# ---------------------------------------------------------------- 观测刻度
OBS_SOC_SCALE = 10.0     # SOC 与目标都以 /10 进入下层观测


def soc_target_from_signal(a6):
    """上层信号 → SOC 目标: 3 + 5·a6, 夹在 [0.5, 5.5]（合法目标带）。"""
    return float(np.clip(SOC_TARGET_CENTER + SOC_TARGET_SPAN * float(a6),
                         SOC_TARGET_MIN, SOC_TARGET_MAX))


def obs_target_dim(soc_target, scale=OBS_SOC_SCALE):
    """下层观测里的目标维 = soc_target / 10（与 SOC/10 同刻度，网络可直接比较）。"""
    return float(soc_target) / float(scale)


def gap_penalty(soc_target, SOC_end, tol=None):
    """块末追踪偏好（元，进 r1）: W_GAP·max(0, |soc_target − SOC_end| − SOC_TOL)。"""
    return hc.gap_pref(float(soc_target) - float(SOC_end),
                       hc.SOC_TOL if tol is None else tol)


def applied_pb(SOC_prev, SOC_next, eta_c=0.98, time=1.0 / 60, eta_d=0.98):
    """由实际 SOC 变化反解"真正施加的" BESS 功率 (MW)。

    ΔSOC = (η_c·max(pb,0) − max(−pb,0)/η_d)·dt 是**分段线性可逆**的：
        ΔSOC ≥ 0 ⇒ pb = ΔSOC / (η_c·dt)
        ΔSOC < 0 ⇒ pb = ΔSOC · η_d / dt
    物理边界（SOC 触 0/6）时得到的是"实际生效"的功率（不是指令值）——正是追踪语义要的。
    """
    dsoc = float(SOC_next) - float(SOC_prev)
    if dsoc >= 0:
        return dsoc / (float(eta_c) * float(time))
    return dsoc * float(eta_d) / float(time)


def soc_track_range(soc_target, SOC_prev, SOC_next, pbrulemin, pbrulemax, n_rem,
                    eta_c=0.98, eta_d=0.98, time=1.0 / 60):
    """两区间分治的追踪误差 `trackerr` [MW] 及其诊断量。

    Parameters
    ----------
    soc_target : float   块级 SOC 目标 (MWh)
    SOC_prev   : float   本步开始时的 SOC (MWh)
    SOC_next   : float   本步结束后的实际 SOC (MWh) —— 用它反解"真正施加的" pb
    pbrulemin, pbrulemax : float  RBB 平抑可行窗口 (MW)
    n_rem      : int     含当前步的块内剩余步数 (= 5 − step % 5)

    Returns
    -------
    trackerr : float  (MW, ≥ 0)  = dist(真正施加的 pb, 参考集合)
        参考集合 = F = B ∩ W（一个区间）；**`F = ∅` 时退化为单点 r*（无范围）**，
        此时只有 pb_applied 恰好等于 r* 才为 0，否则 = |pb_applied − r*|。
        `infeasible_window`（被迫吞吐）时恒为 0 —— 不可归责于下层。
    info     : dict
        delta, gain, pb_applied, band_lo/hi, F_lo/hi, ref_lo/hi, ref_is_point,
        gap_empty, n_rem, d_next, d_ref, infeasible_window, trackerr, trackerr_mwh
        （`F = ∅` 时 `F_lo > F_hi` 只是"交集为空"的哨兵值，请用 ref_lo/ref_hi；
          `ref_is_point=True` ⇔ 零成本集合是一个单点，即"只有竭力值免费"）
    """
    soc_target = float(soc_target)
    SOC_prev, SOC_next = float(SOC_prev), float(SOC_next)
    pbrulemin, pbrulemax = float(pbrulemin), float(pbrulemax)
    n_rem = int(n_rem) if n_rem is not None else 1
    n_rem = max(1, min(5, n_rem))

    delta = soc_target - SOC_prev
    g = float(eta_c) * float(time) if delta > 0 else float(time) / float(eta_d)
    pb_app = applied_pb(SOC_prev, SOC_next, eta_c=eta_c, eta_d=eta_d, time=time)
    d_next = abs(soc_target - SOC_next)

    # 平抑窗口不可行（平抑 ∩ 物理 = ∅）：本步残差不可归责于下层
    # 参考集合此时无意义（整步免罚）⇒ ref_is_point=False，不冒充"竭力单点"
    if pbrulemin > pbrulemax:
        info = dict(delta=delta, gain=g, pb_applied=pb_app, n_rem=n_rem,
                    band_lo=0.0, band_hi=0.0, F_lo=0.0, F_hi=0.0,
                    ref_lo=0.0, ref_hi=0.0, ref_is_point=False, gap_empty=True,
                    d_next=d_next, d_ref=abs(delta), d_best=abs(delta),
                    pb_best=0.0, soc_best=SOC_prev, slack=0.0,
                    infeasible_window=True, trackerr=0.0, trackerr_mwh=0.0)
        return 0.0, info

    # ① Soctrack 范围：摊还端 … 一步填平端
    pb_1 = delta / g
    pb_n = delta / (g * n_rem)
    band_lo, band_hi = min(pb_1, pb_n), max(pb_1, pb_n)

    # ② ∩ 平抑范围 ⇒ 参考集合
    #    F ≠ ∅ ⇒ 区间 [F_lo, F_hi]（区间内任一点都免费）
    #    F = ∅ ⇒ **没有范围，只有唯一一个"竭力值"**：W 里离 B 最近的端点，
    #            记 ref_lo == ref_hi（零宽点）。只有恰好顶到它才免罚，两侧偏离都按
    #            |pb_app − r*| 线性计费；够不着的那段缺口不可达 ⇒ 由上层块末 gap 罚承接。
    F_lo, F_hi = max(band_lo, pbrulemin), min(band_hi, pbrulemax)
    gap_empty = F_lo > F_hi
    if gap_empty:
        ref_lo = ref_hi = pbrulemax if band_lo > pbrulemax else pbrulemin
    else:
        ref_lo, ref_hi = F_lo, F_hi
    ref_is_point = bool(ref_lo == ref_hi)      # 零成本集合是单点（F=∅，或 F 收缩成一点）

    trackerr = max(0.0, ref_lo - pb_app, pb_app - ref_hi)

    # 诊断量保持旧含义 ⇒ 与历史 run 的 CSV/TB 列直接可比：
    #   pb_best = 规则 SocTracker 的动作 = clip(δ/g, 参考集合)（等价 clip(δ/g, 平抑窗口)）
    #             F=∅ 时参考集合是零宽点 ⇒ pb_best 恒等于那个竭力值 r*
    #   d_best  = 该动作后的残差；slack = trackerr 的 MWh 等价 (= trackerr/g)
    pb_best = float(np.clip(delta / g, ref_lo, ref_hi))
    soc_best = SOC_prev + g * pb_best
    d_best = abs(soc_target - soc_best)

    info = dict(delta=delta, gain=g, pb_applied=pb_app, n_rem=n_rem,
                band_lo=band_lo, band_hi=band_hi, F_lo=F_lo, F_hi=F_hi,
                ref_lo=ref_lo, ref_hi=ref_hi, ref_is_point=ref_is_point,
                gap_empty=gap_empty,
                d_next=d_next, d_ref=d_best, d_best=d_best,
                pb_best=pb_best, soc_best=soc_best, slack=trackerr / g,
                infeasible_window=False,
                trackerr=trackerr, trackerr_mwh=trackerr / g)
    return trackerr, info


def lower_reward(clip_amount, coef=None, cap=None):
    """下层 r2 = −coef · min(clip_amount, cap)  [MW]。

    `clip_amount` = **命令值与实际施加值之差** `|pb_cmd − pbrule|`，它**天然包含两段 clip**：

        |pb_cmd − pbrule| = |pb_cmd − clip(pb_cmd, ±10)|   ← ① 执行器/CheckAction 的 ±10 夹裁量
                          + |clip(pb_cmd, ±10) − pbrule|   ← ② RBB 平抑窗口/physics 的夹裁量

    前提：各 main 必须把**命令值**（CheckAction 之前）交给 `RuleBasedBESS.compute_pbrule`。
    因为平抑窗口 ⊆ [−rated_power, +rated_power] = ±10，传命令值**不改变 `pbrule`**
    （`clip(clip(cmd,±10), W) ≡ clip(cmd, W)`，`agentutil/test_l_lower_unit.py` 有断言）；
    若改传 CheckAction 之后的值，则 |pb_cmd| > 10 时 `pbrule` 与 r2 都不再随动作变化
    ⇒ 该方向失去梯度（"±10 死区"，run165 曾出现 |raw|≈900 MW 而 clip≡10）。

    `coef=None` ⇒ 取 `L_CLIP_COEF`（=1.0），即 r2 与 c2 同为 MW 口径、系数同为 1。
    `cap=None`  ⇒ 取 `L_CLIP_CAP`（=20.0 MW，只截断病态饱和，不改真实窗内容量）；
    传 `cap=float('inf')` 可关闭封顶。
    """
    c = L_CLIP_COEF if coef is None else float(coef)
    lim = L_CLIP_CAP if cap is None else float(cap)
    return -c * min(float(clip_amount), lim)


def lower_cost_soc_track(soc_target, SOC_prev, SOC_next, pbrulemin, pbrulemax, n_rem,
                         coef=None, return_info=False, **kw):
    """下层 c2 = +coef · trackerr（两区间分治；单位 MW，与 r2 一致）。

    `coef=None` ⇒ 取 `L_TRACK_COEF`（=1.0）。
    `n_rem` = 含当前步的块内剩余步数（主循环 `5 - episode_steps % 5`）——**必填**，
    以免在块中段静默地按"整块 5 步"摊还。
    """
    trackerr, info = soc_track_range(soc_target, SOC_prev, SOC_next,
                                     pbrulemin, pbrulemax, n_rem, **kw)
    c = L_TRACK_COEF if coef is None else float(coef)
    cost = c * trackerr
    if return_info:
        return cost, info
    return cost
