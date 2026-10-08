"""
HSRL warmup 期调度助手（多项目共享）
====================================
本模块把 warmup 期的三件事收敛到**一处单一真源**：

1. `WarmupSchedule` —— warmup 期上层 H 的**确定性动作调度**（不再 `random_action()`）：
      * 5 个风机轴向因子各维从 **1/3** 起步（单机 `C_p = 4a(1−a)²` 的最优点 —— 只讲
        **单机物理**，**不对"风场最优在哪"表态**；`env.reset()` 的标称动作与 actor 的
        `init_wt_action_bias` 共用下面同一条带状区间）；
      * 每**块**（= 5 环境步，H 的决策周期）各维独立随机 **±0.05**，并直接 clip 到
        **[0.05, 0.45]**（**中性宽带**：只排除 a<0.05 的停机悬崖〔`C_p`<18%〕与
        a>0.45 的低效高诱导区，**不预设**风场最优的形状）；
      * 第 6 维（信号维）**恒为 0.0**：对 SOC 目标语义的项目 = `soc_target ≡ 3`，
        对影子电价语义的项目 = 中性影子价 0。两者都恰好等于 actor 的
        `init_wt_action_bias(..., sig_value=0.0)`，即 warmup 与策略起步**同分布**。
    ⚠ 2026-09-28 两轮修正（记录以免重犯）：旧值 `0.33 / [0.20,0.45] / U(0.25,0.40)` 的
    判据是"**单机** C_p 保持 86%~100% 最优"，用错了对象。第一轮改成"对准均匀动作的
    功率峰 `0.20 / [0.15,0.25] / U(0.17,0.23)`"——**同样是错的**：那个峰只是**均匀族**
    上坐标轮询的结果，而联合最优是**随尾流状态变化的单调上升斜坡**（最下游机组背风侧
    无机组，其功率 ∝ `CT(a₅)·Ue³` 且上游动作固定时 `Ue` 与 `a₅` 无关 ⇒ 应推向单机最优
    ≈1/3；需要降载留风的是上游）。这个**形状**取决于尾流效应，**是策略要学的函数**，
    不是可预先测定的常数；而第一轮的窄带还会从**上方掐住末台**。故本轮改为**中性宽带**。

    RNG 用**独立 RandomState**（不吃全局 np.random 流，避免扰动 episode SOC 随机化与探索噪声）。

2. `WarmupBessRule` —— warmup 期下层 BESS 的**规则行为体**（BESS-RL 项目专用）：
      * 语义与各项目的 `SocTracker.act` **逐位相同**（`test_warmup_schedule.py` 有对照断言）：
            pb = clip((soc_target − E)/g, pbrulemin, pbrulemax),  g = η·dt
        即"一步填平所需功率再夹进 RBB 可行窗口"；δ=0 时退化为 `clip(0, 窗口)`
        （只做窗口不允许 0 时的**被迫吞吐**）。
      * 与 SOC 目标语义的项目（td3per）天然一致：a6=0 ⇒ target=3 ⇒ 同一规则；
        与影子电价语义的项目（per/prrbb/prle/td3prrbb/td3prle）则给出"不作为"的
        规则兜底（不主动制造吞吐）。

3. `fill_bess_transition` —— BESS-RL 项目的**下层 transition 入池唯一入口**（"事后填入"）：
      * warmup 期下层动作由规则给，真实后果（r2/c2/下一观测）与该动作配对入池；
      * 记录的 a_L 是**命令值**（与 r2 的 −1·min(|cmd − pbrule|, 20) 语义一致），并顺带核对
        `|a_L − 实际施加的 pbrule|` 是否等于 RBB 裁剪量（正常情况下应为 0）；
      * 计数由 `BessFillStats` 累积，供启动/收尾日志核对"确实填了 N 条"。

动机（实测，`output/TD3PRRLRB-run3` day0）
----------------------------------------
* 旧 warmup（上层全随机 + 下层随机 pb）实测块收益 r̄ = **−82.0 元/块**，其中退化
  **−151.0 元/块**：随机 a6 使 soc_target 每块跳 ±2.5 MWh，而 RBB 窗口内每块只能移动
  0.36~0.52 MWh（85% 的块目标不可达）⇒ SocTracker 全程顶格往返，而退化按
  `490 元/MWh` 只对**放电毛量**计费。
* 该 r̄ 经 `r̄/(1−γ)`（γ=0.99）得到的 bias 是 **−8196**，与 day0 真实价值量级
  （全状态平均 `mean_j V(j) = +9983`）反号；且由于 `step == args.warmup` 与块末
  `episode_steps % 5 == 4` 不同余，旧代码的 bias 分支**从不触发**（run3 的 Q 起点
  实测 0.0497 = fc3.bias 默认初值）。
* 本调度下实测（同一 day0 序列，5 seed）：r̄ = **+70.4 元/块** ⇒ b_p ≈ **+7042**，
  退化降到 **17.5 元/块**（其中 712 元是"把 SOC 从 episode 随机初值拉到 3"的一次性
  对齐放电，之后稳态只剩 0.2 元/块），c1 = 5.7 元/块（偏保守，策略段 13.6）。
"""
import numpy as np

# ---------------------------------------------------------------- 默认参数
WT_START = 0.2           # ⚠ 起点由 1/3（单机 Cp 峰）改为 0.2：**warmup 不需要往最优方向带**
# 附带好处：0.2 正好落在 step=0.05 的整数格上 ⇒ 可达集就是 {0.05,0.10,...,0.45} 这 9 个整格点。
# 旧起点 1/3 = 0.3333 不是 0.05 的整数倍 ⇒ 整个格点偏移（实测只能踩到 0.0833/0.1333/…），
# 见 `agentutil/probe_warmup_coverage.py`。
WT_STEP = 0.05           # 每块每维 ±0.05
WT_LO, WT_HI = 0.05, 0.45   # 中性宽带: 只排除停机悬崖(a<0.05)与低效高区(a>0.45)
SIG_VALUE = 0.0          # 第 6 维: SOC 目标语义→target=3; 影子电价语义→中性 0
SOC_TARGET = 3.0         # warmup 期下层规则跟踪的目标 (MWh)
#: **episode 起始 SOC**（正式口径 = 固定 3.0 MWh）。旧口径每 episode 从 U[0.5,5.5] 随机，
#: 让前若干块的收益/退化都花在"把 SOC 拉到中性"上并引入与策略无关的方差；见 `--soc_reset_*`。
SOC_START = 3.0
ETA_C = ETA_D = 0.98
DT = 1.0 / 60.0
RATED_POWER = 10.0
SOC_TOL = 0.0            # 下层无死区 (与各项目 SocTracker 默认一致)

#: **warmup 时长口径**（正式方法统一）：未显式传 `--warmup` 时
#: `warmup = WARMUP_BSIZE_MULT × --bsize` 步。绑定理由：首个 minibatch 按 bsize 抽，
#: warmup 太短 ⇒ 缓冲条目 < 一个 batch ⇒ 同一条 transition 被复制多份
#: （H 层每 5 步才产 1 条 transition：`--warmup 200` + bsize=256 时首 batch 每条被
#:  复制 ~6 份）。warmup 期**不进行任何网络更新**（`BiasInitTracker` 运行期断言）。
WARMUP_BSIZE_MULT = 4.0

# ---------------------------------------------------------------- 偏航带（yawflor 专用）
# `yawflor`（FLORIS 引擎）的动作是**无量纲偏航指令** `n ∈ [−1, 1]`，env 内 `γ = 25°·n`
# （见 `yawflor/README.md` §4 的换算表：**按"占带宽比例"原样平移**，因此这里
# 不重犯"把风场最优写进先验"的错 —— 先验只编码**单机物理**）。
#
#   * `YAW_START = 0`   —— 单机最优点是 γ = 0°（自损最小）；旧环境对应 a = 1/3（Cp 峰）
#   * `YAW_LO/HI = ±1`  —— 中性**满带**（带宽 2；旧环境带宽 0.5，故步长/噪声按比例放大）
#   * `YAW_STEP = 0.2`  —— 带宽的 10%（与旧 `WT_STEP=0.05` 占 0.5 带宽的 10% 同比例；
#                          实测最优轨迹跳变 p99 = 5.5° ≈ 0.22 带宽 ⇒ 自洽）
#
# ⚠ 这组常量**只被 `yawflor` 使用**：旧 9 个方法继续用上面的 `WT_*`，行为逐位不变
#   （`add_warmup_args(parser, yaw=False)` 默认分支）。
YAW_START = 0.0
YAW_STEP = 0.2
YAW_LO, YAW_HI = -1.0, 1.0
YAW_WT_NOISE = 0.2       # 探索噪声半宽（带宽的 10%；旧值 0.05 = 0.5 带宽的 10%）
YAW_PENALTY_NN_H_COEF = 100.0   # 满带越界罚（元/单位）：100 × 带宽 2 = **200 元**，
                                # 与旧环境 `400 × 0.5 = 200 元` **等值**（README §4）


def yaw_band():
    """偏航带四元组 (start, step, lo, hi)（单一真源；yawflor_main 与测试共用）。"""
    return YAW_START, YAW_STEP, YAW_LO, YAW_HI


# =============================================================================
# 1. 上层 warmup 调度
# =============================================================================
class WarmupSchedule:
    """warmup 期上层动作调度（确定性 + 有界小步游走）。

    Parameters
    ----------
    nb_turbine : int        风机维数 (5)
    nb_actions : int or None H 动作总维数 (6); None 时只返回 nb_turbine 维
    start, step, lo, hi : float
                            起始值 / 每块步长 / clip 上下界
    sig_value : float       第 6 维恒定值 (默认 0.0)
    seed : int or None      独立 RNG 种子 (None = 每次运行不同, 但不吃全局流)
    enabled : bool          False ⇒ `action()` 返回 None, 由调用方回退旧行为

    用法（主循环块首）::

        a = warmup.action()          # 每个块调用一次
        if a is None:                # 对照臂 (--warmup_sched random)
            a = agent_H.random_action()
    """

    def __init__(self, nb_turbine, nb_actions=None, start=WT_START, step=WT_STEP,
                 lo=WT_LO, hi=WT_HI, sig_value=SIG_VALUE, seed=None,
                 enabled=True):
        self.nb_turbine = int(nb_turbine)
        self.nb_actions = int(nb_actions) if nb_actions else None
        self.start, self.step = float(start), float(step)
        self.lo, self.hi = float(lo), float(hi)
        self.sig_value = float(sig_value)
        self.enabled = bool(enabled)
        self.rng = np.random.RandomState(seed if seed is not None and seed > 0 else None)
        self.reset()

    def reset(self):
        self._wt = np.full(self.nb_turbine, self.start, dtype=float)
        self._first = True
        return self

    @property
    def wt(self):
        """当前块的风机动作（只读副本）。"""
        return self._wt.copy()

    def action(self):
        """返回本块的上层动作；`enabled=False` 时返回 None（调用方走旧随机臂）。"""
        if not self.enabled:
            return None
        if self._first:
            self._first = False                      # 第 0 块 = 起始点 (1/3, 信号 0)
        else:
            self._wt = self._wt + self.rng.choice([-self.step, self.step],
                                                  size=self.nb_turbine)
            self._wt = np.clip(self._wt, self.lo, self.hi)
        if self.nb_actions is None:
            return self._wt.copy()
        out = np.empty(self.nb_actions, dtype=float)
        out[:self.nb_turbine] = self._wt
        if self.nb_actions > self.nb_turbine:
            out[self.nb_turbine:] = self.sig_value
        return out


# =============================================================================
# 2. warmup 期下层规则行为体（BESS-RL 项目）
# =============================================================================
class WarmupBessRule:
    """warmup 期 BESS 的确定性规则：在 RBB 窗口内"一步填平"追踪 SOC 目标。

    与各项目 `SocTracker.act` 同语义（见 `test_warmup_schedule.py` 的逐位对照）：
        delta = soc_target − E
        |delta| ≤ soc_tol (默认 0 ⇒ 仅 delta==0): 返回 clip(0, 窗口)（被迫吞吐）
        否则: pb = clip(delta/g, ±rated_power)，再夹进 [pbrulemin, pbrulemax]；
              窗口不可行 (min > max) 时返回额定限幅后的值 (交 RBB 兜底)。
    """

    def __init__(self, soc_target=SOC_TARGET, eta_c=ETA_C, eta_d=ETA_D, time=DT,
                 rated_power=RATED_POWER, soc_tol=SOC_TOL):
        self.soc_target = float(soc_target)
        self.eta_c, self.eta_d = float(eta_c), float(eta_d)
        self.time = float(time)
        self.rated_power = float(rated_power)
        self.soc_tol = float(soc_tol)

    def act(self, E, pbrulemin, pbrulemax, soc_target=None):
        """返回本步 BESS 功率指令 (MW, 正=充电)。"""
        target = self.soc_target if soc_target is None else float(soc_target)
        delta = target - float(E)
        if abs(delta) <= self.soc_tol:
            if pbrulemin > pbrulemax:
                return 0.0
            return float(np.clip(0.0, pbrulemin, pbrulemax))
        g = self.eta_c * self.time if delta > 0 else self.time / self.eta_d
        pb = float(np.clip(delta / g, -self.rated_power, self.rated_power))
        if pbrulemin > pbrulemax:
            return pb
        return float(np.clip(pb, pbrulemin, pbrulemax))

    def action(self, E, pbrulemin, pbrulemax, soc_target=None):
        """同上，返回长度为 1 的数组（直接拼进 L 动作位）。"""
        return np.array([self.act(E, pbrulemin, pbrulemax, soc_target)])


# ---------------------------------------------------------------- 价格条件随机规则
#: warmup 期"价格条件随机充放"的幅度区间 (MW) 与三档电价 (元/MWh)
WARMUP_BESS_LO, WARMUP_BESS_HI = 0.3, 0.5
PRICE_LOW, PRICE_MID, PRICE_HIGH = 300.0, 580.0, 980.0


class WarmupBessPriceRule:
    """warmup 期下层 BESS 的**价格条件随机**行为（探索用；不再追踪 SOC 目标）。

    规格（用户口径，2026-10-08）::

        电价 300 (低谷) ⇒ 随机充电  +U(0.3, 0.5) MW
        电价 580 (平峰) ⇒ 不动       0
        电价 980 (高峰) ⇒ 随机放电  −U(0.3, 0.5) MW

    方向约定与本仓库一致：**正 = 充电**（同 `WarmupBessRule.act`）。命令最后夹进 RBB 可行
    窗口 `[pbrulemin, pbrulemax]` —— "L 层逻辑不变"：窗口/物理限幅/r2 通道/入池口径全部沿用；
    窗口不可行 (min > max) 时返回未夹的原始值，交 RBB 兜底（与 `WarmupBessRule` 同处理）。

    动机：旧口径 warmup 期下层是"确定性追 SOC=3"，r2/c2 恒 ≈0，**下层行为空间没有任何探索**；
    价格条件随机充放让 L 的 buffer 里出现"低充高放/平峰不动"的真实代价与收益结构。

    ⚠ 本规则**有意打破** "Δ≡0 稳定到规则" 那条旧不变量（warmup 下层会主动偏离规则基线），
    因此 `check_residual_baseline` 对它改走**行为自检**而非"与基线逐位相同"的断言；
    要复现旧行为请用 `--warmup_bess rule`。
    """

    is_price = True

    def __init__(self, lo=WARMUP_BESS_LO, hi=WARMUP_BESS_HI, seed=None):
        self.lo, self.hi = float(lo), float(hi)
        if self.lo > self.hi:
            self.lo, self.hi = self.hi, self.lo
        self.rng = np.random.RandomState(seed if seed is not None and seed > 0 else None)
        self.n_charge = self.n_discharge = self.n_idle = 0

    def act(self, price, pbrulemin, pbrulemax):
        """返回本步 BESS 功率指令 (MW, **正=充电**)。"""
        p = float(price)
        if p <= PRICE_LOW + 1e-6:
            pb = self.rng.uniform(self.lo, self.hi)          # 低谷: 随机充电
            self.n_charge += 1
        elif p >= PRICE_HIGH - 1e-6:
            pb = -self.rng.uniform(self.lo, self.hi)         # 高峰: 随机放电
            self.n_discharge += 1
        else:
            pb = 0.0                                         # 平峰: 不动
            self.n_idle += 1
        if pbrulemin > pbrulemax:
            return float(pb)
        return float(np.clip(pb, pbrulemin, pbrulemax))

    def action(self, price, pbrulemin, pbrulemax):
        """同上，返回长度为 1 的数组。"""
        return np.array([self.act(price, pbrulemin, pbrulemax)])

    def summary(self):
        return ('[warmup-bess] 价格条件随机充放: 充电 {} 步 / 放电 {} 步 / 不动 {} 步，'
                '幅度 U({:.2f},{:.2f}) MW'.format(self.n_charge, self.n_discharge,
                                                  self.n_idle, self.lo, self.hi))


# =============================================================================
# 3. BESS-RL 项目的 transition "事后填入"
# =============================================================================
class BessFillStats:
    """统计 warmup 期规则动作入池的条数与一致性（日志/断言用）。

    `maybe_report_on_policy_step()`：warmup 结束后的第一条**非 warmup** transition
    到达时自动打印一次小结（不需要主循环额外挂打印点）。
    """

    def __init__(self):
        self.n_total = 0
        self.n_warmup = 0
        self.n_mismatch = 0        # 记录的动作**无法重建命令**的条数（应为 0 = 接线正确）
        self.n_clipped = 0         # 命令被窗口/额定夹掉的条数（**正常工况**, 即 r2 通道）
        self.max_abs_gap = 0.0     # max|命令 − 实际施加| = 两段 clip 量 (r2 通道)
        self.sum_reward = 0.0
        self.sum_cost = 0.0
        self.printed = False

    def summary(self):
        return ('[bess-fill] 入池 {} 条 (warmup 规则动作 {} 条 | 记录动作无法重建命令 {} 条 | '
                '命令被夹 {} 条, max|cmd−applied|={:.3g}) | warmup 期 r2 合计={:.1f} c2 合计={:.1f}'
                .format(self.n_total, self.n_warmup, self.n_mismatch, self.n_clipped,
                        self.max_abs_gap, self.sum_reward, self.sum_cost))

    def maybe_report_on_policy_step(self):
        """warmup → 策略期的第一次调用时打印小结（幂等）。"""
        if self.printed or self.n_warmup == 0:
            return self.printed
        self.printed = True
        try:
            from util import prCyan
        except Exception:                                   # pragma: no cover
            prCyan = print
        prCyan(self.summary() + ' (warmup 结束)')
        return True


def fill_bess_transition(agent_L, s_L, a_L, r2, c2, s_L_next, done_L,
                         applied_pb=None, warmup=False, stats=None,
                         cmd_pb=None, rule_pb=None, **kwargs):
    """BESS-RL 项目**下层 transition 入池的唯一入口**。

    warmup 期下层由规则行为体驱动，"事后填入"= 用**规则的命令动作**与它产生的
    **真实后果**（r2/c2/下一观测，均由真实 pbrule/SOC' 算得）配对写入 L 的 buffer；
    不再写入 agent 自己抽的随机动作。

    Parameters
    ----------
    agent_L : 下层 agent（须有 observe_jump）
    a_L     : 记入 buffer 的动作
        残余动作的项目（td3per/td3prrbb/td3prle/td3prrlrb 等）里 a_L = **Δ**；
        影子电价的项目（per/prrbb）里 a_L = 命令值本身。
    applied_pb : float or None
        该步**实际施加**的 BESS 功率（RBB 夹窗口后的 pbrule）。
    cmd_pb : float or None
        下发给 RBB 的**有效命令**（残余动作下 = 规则基线 + Δ）。给了它才可能核对
        "记录的动作能否重建命令"；不给则退回拿 a_L 当命令（旧语义）。
    rule_pb : float or None
        残余动作下的**规则基线**（MW）。核对式 = `|rule_pb + a_L − cmd_pb| == 0`：
        少了它就会去比 |cmd − Δ| —— 那在残余语义下**永远**不等，计数会变成噪声。
    warmup  : bool  是否处于 warmup 期（用于统计）
    stats   : BessFillStats or None
    **kwargs : 透传给 observe_jump（如 prle/td3prle 的 context）

    注意：这些项目的 r2/c2 定义使 warmup 期的奖励**天然接近 0**
    （规则动作正是 c2 的零参照；影子价恒 0 ⇒ shadow_profit ≡ 0），
    因此本机制保证的是"环境侧行为正常 + H 层基线可用 + L buffer 内是真实行"，
    **不等价于对 L 的行为克隆**。
    """
    if stats is not None:
        if not warmup:
            stats.maybe_report_on_policy_step()
        stats.n_total += 1
        if warmup:
            stats.n_warmup += 1
            stats.sum_reward += float(r2)
            stats.sum_cost += float(c2)
            a_rec = float(np.asarray(a_L).reshape(-1)[0])
            # ① 记录的动作能否**重建**下发给 RBB 的命令（抓接线 bug: 记错动作/漏加基线）
            if cmd_pb is not None:
                _ref = float(cmd_pb)
                _rec = (0.0 if rule_pb is None else float(rule_pb)) + a_rec
            else:
                _ref = None if applied_pb is None else float(applied_pb)
                _rec = a_rec
            if _ref is not None and abs(_rec - _ref) > 1e-9:
                stats.n_mismatch += 1
            # ② 命令 vs 实际施加 = **两段 clip 量**（就是 r2 的那一段）。
            #    残余动作让规则基线端贴墙, 因此 >0 是**正常工况**而非错误 ⇒ 单独计数。
            if applied_pb is not None:
                _gap = abs((float(cmd_pb) if cmd_pb is not None else a_rec)
                           - float(applied_pb))
                stats.max_abs_gap = max(stats.max_abs_gap, _gap)
                if cmd_pb is not None and _gap > 1e-9:
                    stats.n_clipped += 1
    return agent_L.observe_jump(s_L, a_L, r2, c2, s_L_next, done_L, **kwargs)


def add_warmup_args(parser, yaw=False):
    """把 warmup 调度的命令行开关挂到各 main 的 parser 上（9 个 main + yawflor 共用）。

    `yaw=True` ⇒ 默认值换成**偏航带**（`YAW_START/STEP/LO/HI`，无量纲 `n∈[−1,1]`），
    文案也换成偏航语义。旧 9 个 main 不传该参数 ⇒ 与历史 run 逐位一致。
    """
    if yaw:
        d_start, d_step, d_lo, d_hi = YAW_START, YAW_STEP, YAW_LO, YAW_HI
        # ⚠ help 文本会被 argparse 做一次 `% params` 展开 ⇒ 字面百分号必须写 `%%`
        #   （否则 `--help` 直接 ValueError: unsupported format character）
        sched_help = ('warmup 期上层动作调度: fixed=偏航 5 维从 **0** 起每块随机 ±0.2 并 '
                      'clip 到 [−1,+1] + 信号维恒 0 (本方案); '
                      'random=旧的 random_action() 对照臂')
        h_start = 'warmup 偏航动作起始值 (默认 0 = 单机最优点 γ=0°)'
        h_step = 'warmup 偏航动作每块随机步长 (默认 ±0.2 = 满带宽 2 的 10%%)'
        h_lo = 'warmup 偏航动作 clip 下界 (默认 −1)'
        h_hi = 'warmup 偏航动作 clip 上界 (默认 +1)'
    else:
        d_start, d_step, d_lo, d_hi = WT_START, WT_STEP, WT_LO, WT_HI
        sched_help = ('warmup 期上层动作调度: fixed=1/3 起每块±0.05 clip[0.05,0.45]'
                      ' + 信号维恒 0 (本方案); random=旧的 random_action() 对照臂')
        h_start = 'warmup 风机动作起始值 (默认 1/3 = 单机 Cp 最优)'
        h_step = 'warmup 风机动作每块随机步长 (默认 ±0.05)'
        h_lo = 'warmup 风机动作 clip 下界 (默认 0.05)'
        h_hi = 'warmup 风机动作 clip 上界 (默认 0.45)'
    parser.add_argument('--warmup_bsize_mult', default=WARMUP_BSIZE_MULT, type=float,
                        help='warmup 时长 = 该系数 × --bsize（默认 %.1f；仅当未显式传 '
                             '--warmup 时生效）。warmup 期**不进行任何网络更新**'
                             '（由 BiasInitTracker 断言）。要复现历史口径请显式传 '
                             '--warmup 200' % WARMUP_BSIZE_MULT)
    parser.add_argument('--warmup_sched', default='fixed', choices=['fixed', 'random'],
                        help=sched_help)
    parser.add_argument('--warmup_wt_start', default=d_start, type=float, help=h_start)
    parser.add_argument('--warmup_wt_step', default=d_step, type=float, help=h_step)
    parser.add_argument('--warmup_wt_lo', default=d_lo, type=float, help=h_lo)
    parser.add_argument('--warmup_wt_hi', default=d_hi, type=float, help=h_hi)
    parser.add_argument('--warmup_bess', default='price',
                        choices=['price', 'rule', 'random'],
                        help='BESS-RL 项目 warmup 期下层行为（**正式口径 = price**）: '
                             'price=价格条件随机充放 (300 充 +U(0.3,0.5) / 580 不动 / '
                             '980 放 −U(0.3,0.5)，再夹 RBB 窗口); '
                             'rule=旧的确定性 SOC 目标跟踪 (Δ≡0, 下层零探索); '
                             'random=更旧的 agent_L.random_action()')
    parser.add_argument('--warmup_bess_lo', default=WARMUP_BESS_LO, type=float,
                        help='价格条件随机充放的幅度下界 (MW, 默认 %.2f)' % WARMUP_BESS_LO)
    parser.add_argument('--warmup_bess_hi', default=WARMUP_BESS_HI, type=float,
                        help='价格条件随机充放的幅度上界 (MW, 默认 %.2f)' % WARMUP_BESS_HI)
    # ---- episode 起始 SOC（正式口径：固定 3.0）----
    # 放在本函数里是因为所有 main 都已调用 `add_warmup_args`（避免再插一行调用点）；
    # 语义上它属于"warmup/episode 起点"口径：旧口径每 episode 从 U[0.5,5.5] 随机，
    # 会把前若干块的收益/退化都花在"把 SOC 拉到中性"上，且引入了与策略无关的方差。
    parser.add_argument('--soc_reset_low', default=SOC_START, type=float,
                        help='episode 起始 SOC 采样下界 (MWh)。**正式口径 %.1f/%.1f = 固定 %.1f**；'
                             '要复现旧的"每 episode 从 U[0.5,5.5] 随机"请传 '
                             '--soc_reset_low 0.5 --soc_reset_high 5.5'
                             % (SOC_START, SOC_START, SOC_START))
    parser.add_argument('--soc_reset_high', default=SOC_START, type=float,
                        help='episode 起始 SOC 采样上界 (MWh, 默认 %.1f)' % SOC_START)
    parser.add_argument('--warmup_soc_target', default=SOC_TARGET, type=float,
                        help='warmup 期规则下层的 SOC 目标 (默认 3.0 MWh)')
    return parser


def _check_price_rule(rule, say, n_cases=240, seed=12345):
    """价格条件随机规则的**行为自检**（替代"与策略期基线逐位相同"的断言）。

    断言：① 300 恒充电、980 恒放电、580 恒为 0；② 幅度落在 [lo, hi] 且**不是常数**；
          ③ 窗口可行时命令必在窗口内；④ 窗口不可行 (min>max) 时返回未夹的原始值。
    """
    ch = np.array([rule.act(PRICE_LOW, -10.0, 10.0) for _ in range(n_cases)])
    di = np.array([rule.act(PRICE_HIGH, -10.0, 10.0) for _ in range(n_cases)])
    idl = np.array([rule.act(PRICE_MID, -10.0, 10.0) for _ in range(n_cases)])
    assert np.all(ch > 0) and np.all(di < 0) and np.all(idl == 0.0), \
        '价格条件规则方向错: 300 应恒充电 / 980 应恒放电 / 580 应恒为 0'
    assert ch.min() >= rule.lo - 1e-12 and ch.max() <= rule.hi + 1e-12, '充电幅度越界'
    assert di.min() >= -rule.hi - 1e-12 and di.max() <= -rule.lo + 1e-12, '放电幅度越界'
    assert ch.std() > 0.01 and di.std() > 0.01, '幅度是常数（没随机）'
    for lo, hi in [(-3.0, 3.0), (-0.2, -0.1), (0.1, 0.2), (2.0, 1.0)]:
        for p in (PRICE_LOW, PRICE_MID, PRICE_HIGH):
            v = float(rule.act(p, lo, hi))
            if lo > hi:
                assert -rule.hi - 1e-12 <= v <= rule.hi + 1e-12, '不可行窗口应返回原始值'
            else:
                assert lo - 1e-12 <= v <= hi + 1e-12, \
                    '命令未夹进窗口: v=%.6f W=[%.3f,%.3f]' % (v, lo, hi)
    say('[warmup] 价格条件随机充放自检通过: 充电 %+.3f±%.3f / 放电 %+.3f±%.3f / 平峰 %.1f MW'
        '（每档 n=%d）' % (ch.mean(), ch.std(), di.mean(), di.std(), idl[0], n_cases))
    return dict(n_checked=3 * n_cases, rule_arm=True, price_rule=True,
                charge_mean=float(ch.mean()), discharge_mean=float(di.mean()),
                amp_lo=rule.lo, amp_hi=rule.hi)


def check_residual_baseline(warmup_rule, sched, args, baseline_fn,
                            soc_target_from_signal=None, n_cases=240, seed=12345):
    """**残余动作**项目的启动自检: warmup 规则臂必须与策略期基线是**同一条规则**。

    残余动作项目 (`pb_cmd = 规则基线 + Δ`, 见 `hsrl_soctarget.rule_action`) 的 warmup
    规则臂写进 buffer 的是 `Δ ≡ 0`, 它的"命令"就是**基线本身**。所以:

      * 若 `WarmupBessRule` 与 `baseline_fn` 的参数有漂移 (eta / time / rated_power /
        soc_tol 不一致), warmup 期的 (s, Δ=0, r2=0, c2=0) 行就自相矛盾 —— 它在教网络
        "Δ=0 是最优", 而 Δ=0 实际并不等于策略期的零成本动作。这类漂移**必须当期报错**,
        所以本函数直接抛 AssertionError, 不做"提示后继续"。
      * `--warmup_soc_target` 在残余动作下**不生效**: 基线跟随 H 信号给出的目标
        (固定调度下信号维恒 0 ⇒ `soc_target_from_signal(0)` = 3.0), 这正是策略期基线
        的做法, 也是 `Δ≡0` 语义自洽的前提。给了别的值就提示一句。
      * 上层若走 `--warmup_sched random` 臂, 信号维不再恒 0 ⇒ 规则将跟踪**每块的随机
        目标** (改造前的实现固定追 3.0)。这是行为变化, 提示一句。

    Parameters
    ----------
    warmup_rule : WarmupBessRule or None   None ⇒ 对照臂 (`--warmup_bess random`)
    sched       : WarmupSchedule
    args        : argparse.Namespace (读 --warmup_soc_target / --warmup_sched)
    baseline_fn : callable(target, E, lo, hi) -> float
        策略期基线 (各 main 传 `lambda: hst.rule_action(..., eta_c=rbb_system.eta_c, ...)`)
    soc_target_from_signal : callable or None  `a6 -> soc_target` (单一真源)

    Returns
    -------
    dict(n_checked=int, target_at_signal0=float or None, sched_signal_fixed=bool, rule_arm=bool)
    """
    try:
        from util import prYellow
    except Exception:                                       # pragma: no cover
        prYellow = print
    if getattr(warmup_rule, 'is_price', False):
        # 新口径：warmup 下层**主动偏离**规则基线去探索价格响应 ⇒ 旧的"Δ≡0 与基线逐位相同"
        # 断言在语义上不再成立，改走行为自检（方向/幅度/窗口）。
        return _check_price_rule(warmup_rule, prYellow, n_cases=n_cases, seed=seed)
    sched_signal_fixed = bool(getattr(sched, 'enabled', True))
    if warmup_rule is None:                # 对照臂: 绝对随机功率, 与基线无关
        return dict(n_checked=0, target_at_signal0=None,
                    sched_signal_fixed=sched_signal_fixed, rule_arm=False)

    lo_hi = [(-3.0, 3.0), (-6.5, -0.4), (0.3, 2.5), (2.0, 1.0),
             (-10.0, 10.0), (0.0, 0.0), (2.0, 1.0)]   # 含不可行窗口 (2>1) 与 0 窗口
    rg = np.random.default_rng(seed)
    n_checked = 0
    for k in range(n_cases):
        E = float(rg.uniform(0.2, 5.8))
        tgt = E if k % 7 == 0 else float(rg.uniform(0.5, 5.5))   # 含 δ=0 死区样本
        lo, hi = lo_hi[int(rg.integers(0, len(lo_hi)))]
        a = float(warmup_rule.act(E, lo, hi, tgt))               # warmup 规则臂的命令
        b = float(baseline_fn(tgt, E, lo, hi))                   # 策略期基线
        assert abs(a - b) < 1e-12, (
            'warmup 规则臂与残余基线不是同一条规则 (Δ≡0 将不再等于策略期零成本动作): '
            'E={:.6f} target={:.6f} W=[{:.3f},{:.3f}] warmup_rule={:.9f} baseline={:.9f} '
            '| 检查 eta_c/eta_d/time/rated_power/soc_tol 是否一致'.format(E, tgt, lo, hi, a, b))
        n_checked += 1

    tgt0 = None
    if soc_target_from_signal is not None:
        tgt0 = float(soc_target_from_signal(SIG_VALUE))
        given = float(getattr(args, 'warmup_soc_target', tgt0))
        if abs(given - tgt0) > 1e-12 and sched_signal_fixed:
            prYellow('[warmup] 残余动作: 规则臂跟随 H 信号目标 (= {:.3f}), '
                     '--warmup_soc_target={:.3f} 不生效 (策略期基线也这么做, Δ≡0 才自洽)'
                     .format(tgt0, given))
    if not sched_signal_fixed:
        prYellow('[warmup] --warmup_sched random: 信号维不再恒 0 ⇒ 规则臂跟踪**每块的随机 '
                 'SOC 目标** (改造前固定追 {:.1f}); 若要复现旧行为请用 --warmup_sched fixed'
                 .format(float(getattr(warmup_rule, 'soc_target', 3.0))))
    return dict(n_checked=n_checked, target_at_signal0=tgt0,
                sched_signal_fixed=sched_signal_fixed, rule_arm=True)


def build_from_args(args, agent_H, agent_L=None):
    """按 args 构造 (WarmupSchedule, warmup 下层规则, BessFillStats)。

    下层规则按 `--warmup_bess` 三选一:
      * `price`（正式口径）⇒ `WarmupBessPriceRule`（价格条件随机充放，**有探索**）
      * `rule`            ⇒ `WarmupBessRule`（确定性追 SOC 目标，Δ≡0）
      * `random`          ⇒ None（更旧的 agent_L.random_action() 对照臂）
    """
    nb_t = int(getattr(agent_H, 'nb_turbine', 5))
    nb_a = int(getattr(agent_H, 'nb_actions', nb_t + 1))
    sched = WarmupSchedule(
        nb_turbine=nb_t, nb_actions=nb_a,
        start=float(getattr(args, 'warmup_wt_start', WT_START)),
        step=float(getattr(args, 'warmup_wt_step', WT_STEP)),
        lo=float(getattr(args, 'warmup_wt_lo', WT_LO)),
        hi=float(getattr(args, 'warmup_wt_hi', WT_HI)),
        seed=int(getattr(args, 'seed', 0) or 0),
        enabled=(getattr(args, 'warmup_sched', 'fixed') != 'random'))
    _mode = str(getattr(args, 'warmup_bess', 'price'))
    if _mode == 'random':
        rule = None
    elif _mode == 'price':
        rule = WarmupBessPriceRule(lo=float(getattr(args, 'warmup_bess_lo', WARMUP_BESS_LO)),
                                   hi=float(getattr(args, 'warmup_bess_hi', WARMUP_BESS_HI)),
                                   seed=int(getattr(args, 'seed', 0) or 0))
    else:
        rule = WarmupBessRule(soc_target=float(getattr(args, 'warmup_soc_target', SOC_TARGET)),
                              time=DT,
                              rated_power=float(getattr(args, 'rbb_rated_power', RATED_POWER)))
    stats = BessFillStats() if agent_L is not None else None
    return sched, rule, stats


# =============================================================================
# 6. warmup 时长口径（单一真源）：warmup = --warmup_bsize_mult × --bsize 步
# =============================================================================
def resolve_warmup(args, verbose=True):
    """把 `args.warmup` 解析成**步数**并打印口径；返回 args（就地改写）。

    口径（**正式方法统一**）：
      * 未显式传 `--warmup`（默认 None）⇒ `warmup = warmup_bsize_mult × bsize`
        （默认系数 4 ⇒ bsize=256 时 1024 步、bsize=64 时 256 步）。
      * 显式传 `--warmup N` ⇒ **尊重 N**（复现历史 run 用 `--warmup 200`）。
      * warmup 期**零网络更新**：由 `BiasInitTracker.on_block_end` 在运行期断言
        （`agent.update_count` 必须仍为 0），并在触发点打印证明。

    为什么把 warmup 与 bsize 绑定：首个 minibatch 是按 `bsize` 抽的，warmup 太短
    ⇒ 缓冲里条目少于一个 batch ⇒ **同一条 transition 被复制多份**（H 层每 5 步才产
    1 条 transition：`--warmup 200` 时 H 缓冲仅 40 条，而 bsize=256 ⇒ 首 batch 里
    每条被复制 ~6 份）。绑到 4×bsize 后 H 缓冲 ≈ 4·bsize/5 条、L 缓冲 = 4·bsize 条。
    """
    bsize = int(getattr(args, 'bsize', 0) or 0)
    mult = float(getattr(args, 'warmup_bsize_mult', WARMUP_BSIZE_MULT) or 0.0)
    given = getattr(args, 'warmup', None)
    auto = int(round(mult * max(bsize, 0)))
    if given is None:
        args.warmup = auto
        src = 'bsize×%.1f（默认）' % mult
    else:
        args.warmup = int(given)
        src = '显式 --warmup %d（覆盖 bsize×%.1f=%d）' % (args.warmup, mult, auto)
    if args.warmup < 0:
        raise ValueError('warmup 不能为负: %r' % (args.warmup,))
    if verbose:
        n_h = args.warmup // 5      # H 层每 5 个环境步产 1 条 transition
        print('[warmup] 时长 = {} 步  ← {}  (--bsize={} ⇒ 首个 batch 前缓冲: '
              'H≈{} 条 / L={} 条) | warmup 期**零网络更新**（运行期断言）'.format(
                  args.warmup, src, bsize, n_h, args.warmup))
    return args
