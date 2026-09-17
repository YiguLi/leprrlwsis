"""
HSRL warmup 期调度助手（多项目共享）
====================================
本模块把 warmup 期的三件事收敛到**一处单一真源**：

1. `WarmupSchedule` —— warmup 期上层 H 的**确定性动作调度**（不再 `random_action()`）：
      * 5 个风机轴向因子各维从 **0.33** 起步（Cp = 4a(1−a)² 的最优点，也是
        `env.reset()` 的标称动作 a=1/3 与 actor 的 `init_wt_action_bias(0.25~0.40)` 中心）；
      * 每**块**（= 5 环境步，H 的决策周期）各维独立随机 **±0.05**，并直接 clip 到
        **[0.20, 0.45]**（该带内 Cp 仍为最优的 86%~100%，E[ΣCp]=2.82 ≈ 95% 最优；
        clip 到 [0,0.5] 则 89% 的样本会出现某维 ≤0.05、ΣCp 掉到 83% 最优）；
      * 第 6 维（信号维）**恒为 0.0**：对 SOC 目标语义的项目 = `soc_target ≡ 3`，
        对影子电价语义的项目 = 中性影子价 0。两者都恰好等于 actor 的
        `init_wt_action_bias(..., sig_value=0.0)`，即 warmup 与策略起步**同分布**。
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
      * 记录的 a_L 是**命令值**（与 r2 里 −100·clip_amount 的语义一致），并顺带核对
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
WT_START = 0.33          # Cp 最优点 (= 1/3; env.reset 的标称动作)
WT_STEP = 0.05           # 每块每维 ±0.05
WT_LO, WT_HI = 0.20, 0.45   # 直接 clip 的安全带 (Cp 保持 86%~100% 最优)
SIG_VALUE = 0.0          # 第 6 维: SOC 目标语义→target=3; 影子电价语义→中性 0
SOC_TARGET = 3.0         # warmup 期下层规则跟踪的目标 (MWh)
ETA_C = ETA_D = 0.98
DT = 1.0 / 60.0
RATED_POWER = 10.0
SOC_TOL = 0.0            # 下层无死区 (与各项目 SocTracker 默认一致)


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
            self._first = False                      # 第 0 块 = 起始点 (0.33, 信号 0)
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


# =============================================================================
# 3. BESS-RL 项目的 transition "事后填入"
# =============================================================================
class BessFillStats:
    """统计 warmup 期规则动作入池的条数与一致性（日志/断言用）。"""

    def __init__(self):
        self.n_total = 0
        self.n_warmup = 0
        self.n_mismatch = 0
        self.max_abs_gap = 0.0
        self.sum_reward = 0.0
        self.sum_cost = 0.0

    def summary(self):
        return ('[bess-fill] 入池 {} 条 (其中 warmup 规则动作 {} 条, 与施加值不一致 {} 条, '
                'max|a−applied|={:.3g}) | warmup 期 r2 合计={:.1f} c2 合计={:.1f}'
                .format(self.n_total, self.n_warmup, self.n_mismatch,
                        self.max_abs_gap, self.sum_reward, self.sum_cost))


def fill_bess_transition(agent_L, s_L, a_L, r2, c2, s_L_next, done_L,
                         applied_pb=None, warmup=False, stats=None):
    """BESS-RL 项目**下层 transition 入池的唯一入口**。

    warmup 期下层由规则行为体驱动，"事后填入"= 用**规则的命令动作**与它产生的
    **真实后果**（r2/c2/下一观测，均由真实 pbrule/SOC' 算得）配对写入 L 的 buffer；
    不再写入 agent 自己抽的随机动作。

    Parameters
    ----------
    agent_L : 下层 agent（须有 observe_jump）
    a_L     : 记入 buffer 的动作 —— 命令值（与 r2 中 −100·clip_amount 的口径一致）
    applied_pb : float or None
        该步**实际施加**的 BESS 功率（RBB 夹窗口后的 pbrule）；给出时核对
        |a_L − applied_pb| 是否为 0（非 0 仅应在"窗口不可行"的罕见步出现）并计数。
    warmup  : bool  是否处于 warmup 期（用于统计）
    stats   : BessFillStats or None
    """
    if stats is not None:
        stats.n_total += 1
        if warmup:
            stats.n_warmup += 1
            stats.sum_reward += float(r2)
            stats.sum_cost += float(c2)
            if applied_pb is not None:
                gap = abs(float(np.asarray(a_L).reshape(-1)[0]) - float(applied_pb))
                stats.max_abs_gap = max(stats.max_abs_gap, gap)
                if gap > 1e-9:
                    stats.n_mismatch += 1
    return agent_L.observe_jump(s_L, a_L, r2, c2, s_L_next, done_L)


def add_warmup_args(parser):
    """把 warmup 调度的命令行开关挂到各 main 的 parser 上（9 个 main 共用）。"""
    parser.add_argument('--warmup_sched', default='fixed', choices=['fixed', 'random'],
                        help='warmup 期上层动作调度: fixed=0.33 起每块±0.05 clip[0.20,0.45]'
                             ' + 信号维恒 0 (本方案); random=旧的 random_action() 对照臂')
    parser.add_argument('--warmup_wt_start', default=WT_START, type=float,
                        help='warmup 风机动作起始值 (默认 0.33 = Cp 最优)')
    parser.add_argument('--warmup_wt_step', default=WT_STEP, type=float,
                        help='warmup 风机动作每块随机步长 (默认 ±0.05)')
    parser.add_argument('--warmup_wt_lo', default=WT_LO, type=float,
                        help='warmup 风机动作 clip 下界 (默认 0.20)')
    parser.add_argument('--warmup_wt_hi', default=WT_HI, type=float,
                        help='warmup 风机动作 clip 上界 (默认 0.45)')
    parser.add_argument('--warmup_bess', default='rule', choices=['rule', 'random'],
                        help='BESS-RL 项目 warmup 期下层行为: rule=RBB窗口+SOC目标跟踪'
                             ' (规则行为体, 无随机); random=旧的 agent_L.random_action()')
    parser.add_argument('--warmup_soc_target', default=SOC_TARGET, type=float,
                        help='warmup 期规则下层的 SOC 目标 (默认 3.0 MWh)')
    return parser


def build_from_args(args, agent_H, agent_L=None):
    """按 args 构造 (WarmupSchedule, WarmupBessRule, BessFillStats)。"""
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
    rule = (WarmupBessRule(soc_target=float(getattr(args, 'warmup_soc_target', SOC_TARGET)),
                           time=DT, rated_power=float(getattr(args, 'rbb_rated_power', RATED_POWER)))
            if getattr(args, 'warmup_bess', 'rule') != 'random' else None)
    stats = BessFillStats() if agent_L is not None else None
    return sched, rule, stats
