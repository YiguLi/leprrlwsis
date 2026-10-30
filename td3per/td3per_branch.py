# -*- coding: utf-8 -*-
"""`td3per` 同块首宽动作覆盖（**branch**）—— 口径唯一真源。

来源：`aifflor/aifflor_branch.py`（同一份计划
`documentation/aifflor_branch_wide_coverage_plan.md`），本文件是它在 td3per 上的
**语义换算版**：动作空间、块长、观测构造完全相同，只有"第 6 维是什么"与
"下层怎么冻结/执行"按 td3per 自己的口径改写。

一句话
------
**每个块（5 步）从同一个「完整块首 state」上再跑若干个测试动作**，把它们各自的
块级 `(s, a, Σr1, Σc1, s', done)` 当成**正常 TD 数据**写进同一个 replay；
但 actor / lower / λ / EMA / curriculum / CSV / TB / test **只用 actor 分支**。

为什么要它（计划 §1.2）
----------------------
在线 TD3 学不到 `∂c1/∂a` 的根因不是网络容量，而是**行为策略的动作分布太窄**：
同一个块首 state 上 actor 只留下 1 个动作，动作边际在 value 的方差里只占 ~1%
（风速/电价解释了 ~96%）。branch 在同一 state 上补 4 个动作 ⇒ 状态被固定住，
动作成为唯一变量 ⇒ critic 才有可能辨识它。

三条硬约束（计划 §3.1 / §4.4 / §9）
----------------------------------
1. **同块首 state 必须完整相同**：不仅同一串风速，还要
   `(price, u_idx, E, P_G_prev)`。本模块用 `snapshot_env` / `restore_env`
   精确保存/恢复这 6 个标量（`count_wind` / `count_price` 也要，否则 5 步里
   推进的风速/电价比 actor 的块错位）。
2. **branch 不推进主 episode**：`restore_env` 保证主轨迹的
   `episode_steps / E / P_G_prev / count_*` 一位不变；branch 也不写任何行为日志。
3. **branch 只覆盖 1 个块**：不进 n-step 链，`n_boot = γ`
   （由 `SP_TD3.push_branch_transition` 保证，见 `td3per_ddpg.py`）。

与 aifflor 的三处**语义差异**（都是 td3per 自己的口径，不是自由度）
------------------------------------------------------------------
| 项 | aifflor（HSRL 分支） | 本文件（td3per） |
|---|---|---|
| 第 6 维 | 块总放电量 `g` [MW·min] | **SOC 目标信号** `a6 ∈ [-0.5,0.5]` → `soc_target`（`td3per_myfunc.soc_target_from_signal`） |
| 块成本 | `hsrl.cal_rew_rb_hsrl` | **`Cal_Rew_RB`**（同一个函数，含 `hc.c1_apply` 的 uniform/legacy 口径） |
| 块末 gap | HSRL 无 gap 罚 | **与主循环同式**：`10·max(0, |soc_target−SOC_末|−soc_tol)`；uniform 进 r1，legacy 进 c1 |
| 下层 | `amortized_rule_action(g, …)` | `--branch_lower freeze`（默认，`pb` 命令 ≡ 0）或 `rule`（`hst.rule_action`，零成本基线） |

第 6 维怎么取（`--branch_signal`）
---------------------------------
* `soc`（**默认**）：`a6` 反解成"目标 = 本块块首 SOC"⇒ 冻结下层时块末 gap ≡ 0
  ⇒ 块成本 `c1` **只由 5 维风机动作决定**（对上层 critic 最干净的反事实）。
* `target`：沿用 actor 本块的第 6 维 ⇒ branch 与 actor 的目标相同（gap 是同一常数）。
* `zero`：`a6 = 0` ⇒ 目标 = SOC 带中点（`SOC_TARGET_CENTER`）。
* `pm1`：随机 `±0.5` ⇒ 目标 = 可行带两端（对应 aifflor 的 `--branch_goal_pm1`）。
* `random`：`U(-0.5, 0.5)`。

⚠ `--branch_lower freeze` 的真实语义（**不要读成 `pb≡0`**）
---------------------------------------------------------
`pb` 的**命令**恒为 0，但 RBB 仍会把它夹进"平抑窗口 ∩ 物理窗口"：
窗口不含 0 时电池必须做**最小幅度的被迫吞吐**（否则 `|P_G(t)−P_G(t−1)|` 会更大）。
`aifflor` 的文档写的是 `pb≡0 ⇒ E 不变`，那在窗口含 0 时成立、否则不成立；
本文件把 `soc_start/soc_end` 逐条记进 `df_branch_*.csv`，让"冻结得多干净"可审计。
要**严格** `pb≡0` 请用 `--branch_lower none`（绕开 RBB，直接 `P_G = P_tur`）。

动作集（计划 §3.2 / §8.4）
--------------------------
| mode | 生成式 | 覆盖的东西 |
|---|---|---|
| `prior` | `[0.2,0.2,0.2,0.2,1/3] + 逐维 U(−0.05,0.05)` | 标称/中性工作点 |
| `low` | 逐维 `U(0.00, 0.10)` | 强降载边界（`a<0.05` 占一半） |
| `high` | 逐维 `U(0.40, 0.50)` | 满发边界（`a>0.45` 占一半） |
| `uniform` | 逐维 `U(0, 0.5)` | 全域无偏 |
| `ramp` | 逐维沿机组线性 `U(0.02,0.12) → U(0.38,0.48)` | 上游重降载的斜坡形状 |
| `alt` | 奇偶机组交替 `lo/hi`（各 `U(0.02,0.12)`/`U(0.38,0.48)`） | 异向动作 |
| `single` | 其余机组 = prior，随机一台夹到 `U(0.02,0.10)` 或 `U(0.40,0.48)` | 单台扰动（隔离局部效应） |

* 边界带（`low`/`high`）**刻意取 `[0,0.10]` / `[0.40,0.50]`**，而不是计划书
  字面的 `[0.05]+U[0,0.05]`：后者对 `a<0.05` 的覆盖率为 **0%**，达不到计划 §7
  的"边界覆盖各 ≥5%"。两个带都可用 `--branch_low_lo/hi`、`--branch_high_lo/hi` 覆盖。
* 所有动作**按构造就在 `[0,0.5]` 内** ⇒ 不需要"触边只截噪声"那一步；仍走一次
  `np.clip` 做合法化（正常为 0 改动）。

成本（计划 §5）
-------------
一个 branch 块 = `len(modes) × BLOCK` 个额外环境步。默认 4 个测试动作 ⇒
每触发一次 +20 步。`--branch_every 10` ⇒ 每 10 块 +20 步 ⇒ **1.4×**（含 actor 的
5 步是 50 步/10 块 + 20 = 70 ⇒ 1.4×）。

⚠ 不要在 branch 里调用 `env.reset()` / `env.step` 之外的任何推进接口：
`restore_env` 只认 `pr/u/E/P_G_prev/count_wind/count_price` 这 6 个标量。
"""
import os
import sys

import numpy as np

# ---- 路径自举 ---------------------------------------------------------------
#   * 作为包被 import（`from td3per import td3per_branch`）：ROOT 已在 sys.path；
#   * 直接执行 `python td3per/td3per_branch.py`：本文件只打印 USAGE，但 import
#     `td3per_myfunc` 需要仓库根 ⇒ 这里把两边都补上（与 aifflor_branch 同款）。
_HERE = os.path.dirname(os.path.abspath(__file__))       # td3per/
_ROOT = os.path.dirname(_HERE)                           # 仓库根
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:                                            # 主路径：td3per 作为包
    from td3per.td3per_myfunc import (
        Cal_Rew_RB, gap_penalty, soc_target_from_signal,
        violation_derate, VIOL_DERATE_COEF,
        SOC_TOL, SOC_TARGET_CENTER, SOC_TARGET_SPAN,
    )
except ImportError:                             # 兼容：直接把 td3per/ 放进 sys.path
    from td3per_myfunc import (                 # type: ignore
        Cal_Rew_RB, gap_penalty, soc_target_from_signal,
        violation_derate, VIOL_DERATE_COEF,
        SOC_TOL, SOC_TARGET_CENTER, SOC_TARGET_SPAN,
    )

try:
    import hsrl_cost as hc               # uniform / legacy 两种 c1 口径
    import hsrl_soctarget as hst         # 规则下层（--branch_lower rule）的单一真源
    import hsrl_warmup as hwarm          # WT_PRIOR：与 warmup / bias 初始化同一先验
except ImportError:                     # pragma: no cover - 正常情况下 ROOT 已在 path
    hc = hst = hwarm = None


# =============================================================================
# 1. 常量（口径单一真源）
# =============================================================================
BLOCK = 5                                # 5 步/块（td3per 的 H 决策周期）

#: replay 的 `source` 标签（**只有采样/诊断用，绝不进 critic state**，计划 §4.1）
SRC_ACTOR = 0
SRC_BRANCH = 1
SRC_NAMES = {SRC_ACTOR: 'actor', SRC_BRANCH: 'branch'}

#: replay 的 `mode` 标签（branch 内部按它均衡采样，计划 §4.2）
MODE_ACTOR = 0
MODE_PRIOR = 1
MODE_LOW = 2
MODE_HIGH = 3
MODE_UNIFORM = 4
MODE_RAMP = 5
MODE_ALT = 6
MODE_SINGLE = 7
MODE_ORDER = ('actor', 'prior', 'low', 'high', 'uniform', 'ramp', 'alt', 'single')
MODE_NAME = {i: n for i, n in enumerate(MODE_ORDER)}
MODE_BY_NAME = {n: i for i, n in enumerate(MODE_ORDER)}
#: `--branch_actions` 的默认值（计划 §3.2 的四个测试动作）
BRANCH_MODES_DEFAULT = (MODE_PRIOR, MODE_LOW, MODE_HIGH, MODE_UNIFORM)
DEFAULT_BRANCH_ACTIONS = 'prior,low,high,uniform'
#: 可以出现在 `--branch_actions` 里的名字（`actor` 不算"测试动作"，列出来只为报错清晰）
GENERATOR_MODES = ('prior', 'low', 'high', 'uniform', 'ramp', 'alt', 'single')

#: 5 台风机的标称/中性动作 —— **与 `hsrl_warmup.WT_PRIOR` 逐位相同**（单一真源：
#: warmup 采样、`init_wt_action_bias`、branch 的 `prior` 模式共用同一个先验形状）。
PRIOR5 = np.array((hwarm.WT_PRIOR if hwarm is not None else
                   (0.20, 0.20, 0.20, 0.20, 1.0 / 3.0)), dtype=float)
PRIOR_JITTER = 0.05          # prior 的逐维抖动半宽 U(−δ,+δ)
LOW_BAND = (0.00, 0.10)      # 强降载带（含 a<0.05 半边 ⇒ 满足 §7 的边界覆盖 ≥5%）
HIGH_BAND = (0.40, 0.50)     # 满发带（含 a>0.45 半边）
SINGLE_LO = (0.02, 0.10)     # `single` 模式的单台扰动量程（低侧）
SINGLE_HI = (0.40, 0.48)     # `single` 模式的单台扰动量程（高侧）

A_LO, A_HI = 0.0, 0.5        # 学习带（与 `CheckAction` / `env.PowerOutput` 一致）
SIG_LO, SIG_HI = -0.5, 0.5   # 第 6 维（SOC 目标信号）的合法带

#: 对偶/预算的口径：branch 数据**不**允许进入这些通道（计划 §4.3 / §4.4）
ACTOR_ONLY_CHANNELS = ('lambda', 'ema', 'curriculum', 'csv', 'tensorboard', 'test')

#: `snapshot_env` 保存的字段（这几个标量 + 固定的 `wind_day` 就完全决定了后续 5 步）
SNAP_FIELDS = ('pr', 'u', 'E', 'P_G_prev', 'count_wind', 'count_price')
SNAP_INT_FIELDS = ('count_wind', 'count_price')

#: `--branch_lower` 的取值
LOWER_MODES = ('freeze', 'rule', 'none')
#: `--branch_signal` 的取值
SIGNAL_MODES = ('soc', 'target', 'zero', 'pm1', 'random')

#: 单步时长 [h]
TIME_STEP = 1.0 / 60


# =============================================================================
# 2. 命令行
# =============================================================================
def add_branch_args(parser):
    """把 branch 的全部开关挂到 argparse（**默认全部关闭/不改变历史行为**）。"""
    g = parser.add_argument_group('branch（同块首宽动作覆盖，计划见 documentation/）')
    g.add_argument('--branch_actions', default=DEFAULT_BRANCH_ACTIONS, type=str,
                   help='每个 branch 块生成的测试动作，逗号分隔（可用: %s；'
                        '默认 %s；`actor` 不计入）'
                        % (','.join(GENERATOR_MODES), DEFAULT_BRANCH_ACTIONS))
    g.add_argument('--branch_every', default=0, type=int,
                   help='每 K 个块生成一次 branch（0 = **关闭**，默认；推荐从 10 起步 '
                        '= 计划 §5 的 1.4× 档；4 = 2.0× 档）')
    g.add_argument('--branch_pilot_blocks', default=0, type=int,
                   help='前 N 个块**每块都 branch**（= 计划 §5 阶段 A 的 pilot/预训练；'
                        '配合 --branch_every 使用，N 之后回到每 K 块一次）')
    g.add_argument('--branch_lower', default='freeze', choices=list(LOWER_MODES),
                   help='branch 内下层 BESS 的口径: freeze(默认)=pb 命令 ≡ 0 '
                        '(RBB 仍会施加平抑窗口强制的最小吞吐); '
                        'rule=hst.rule_action 零成本基线; '
                        'none=严格 pb≡0 (绕开 RBB, P_G=Ptur, SOC 一字不变)')
    g.add_argument('--branch_signal', default='soc', choices=list(SIGNAL_MODES),
                   help='branch 的第 6 维（SOC 目标信号）: soc(默认)=目标=本块块首 SOC '
                        '(冻结下层时 gap≡0 ⇒ c1 只由 5 维风机动作决定); '
                        'target=沿用 actor 本块的信号; zero=0; pm1=±0.5; random=U(-0.5,0.5)')
    g.add_argument('--branch_critic_frac', default=0.45, type=float,
                   help='critic 批里 branch 的占比（计划 §4.2 的 40–50%%；'
                        '默认 0.45；>0.5 会被夹到 0.5，避免 actor 被淹没）')
    g.add_argument('--branch_sup_beta', default=0.0, type=float,
                   help='可选的 branch 辅助监督权重 β0（计划 §6.2；'
                        '0 = 关闭（默认）。开启后 β 从 β0 线性降到 0.2·β0）')
    g.add_argument('--branch_sup_blocks', default=20000, type=int,
                   help='辅助监督 β 的线性衰减跨度（块数；只统计 branch 块）')
    g.add_argument('--branch_csv', default=1, type=int,
                   help='1（默认）= 写 <output>/df_branch_<RUN_TAG>.csv（逐 branch transition）')
    g.add_argument('--branch_low_lo', default=LOW_BAND[0], type=float,
                   help='low 模式带下界（默认 %g）' % LOW_BAND[0])
    g.add_argument('--branch_low_hi', default=LOW_BAND[1], type=float,
                   help='low 模式带上界（默认 %g）' % LOW_BAND[1])
    g.add_argument('--branch_high_lo', default=HIGH_BAND[0], type=float,
                   help='high 模式带下界（默认 %g）' % HIGH_BAND[0])
    g.add_argument('--branch_high_hi', default=HIGH_BAND[1], type=float,
                   help='high 模式带上界（默认 %g）' % HIGH_BAND[1])
    g.add_argument('--branch_prior_jitter', default=PRIOR_JITTER, type=float,
                   help='prior 模式的逐维抖动半宽（默认 %g）' % PRIOR_JITTER)
    g.add_argument('--branch_seed', default=-1, type=int,
                   help='branch 的**独立** RandomState 种子（>0 固定；默认 -1 = 由 --seed 派生）。'
                        '独立随机流 ⇒ 不吃训练流，开关 branch 不改变 actor 轨迹的随机数')
    return parser


def parse_modes(text):
    """`'prior,low'` → `[MODE_PRIOR, MODE_LOW]`（去重、保序、跳过 `actor`）。"""
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        items = [str(t) for t in text]
    else:
        items = str(text).replace(';', ',').split(',')
    out = []
    for raw in items:
        name = str(raw).strip().lower()
        if not name or name == 'actor':
            continue
        if name not in GENERATOR_MODES:
            raise ValueError('--branch_actions 里有未知 mode %r（可用: %s）'
                             % (raw, ','.join(GENERATOR_MODES)))
        m = MODE_BY_NAME[name]
        if m not in out:
            out.append(m)
    return out


# =============================================================================
# 3. 块首 state 的快照 / 恢复（计划 §3.1）
# =============================================================================
def snapshot_env(env):
    """保存**完整块首 state** 的 6 个标量。

    为什么是这 6 个就够：`env.step()` 的推进只由
      `CalPrice()`（读 `count_price`）、`CalWindSpeed()`（读 `count_wind` + 固定的
      `wind_day`）、`GetBatteryActions()`（读 `E` / `P_G_prev`）决定；
    其余（`wind_day`、`j`、`episode_count`、`max_E`、`eta_*`）在块内不变。
    ⇒ 恢复这 6 个标量 + 环境对象本身不变 = 逐位回到块首。
    """
    snap = {}
    for k in SNAP_FIELDS:
        if not hasattr(env, k):
            raise AttributeError('snapshot_env: env 缺字段 %r（branch 需要完整块首 state）' % k)
        if k in SNAP_INT_FIELDS:
            snap[k] = int(getattr(env, k))
        else:
            snap[k] = float(getattr(env, k))
    return snap


def restore_env(env, snap):
    """把环境恢复到 `snapshot_env` 的时刻。

    ⚠ 若 env 带物理预取缓存（`aifflor` 的 `physics._pf_state`），必须**置为失效**
    而不是恢复成快照 —— branch 跑完后缓存里是 branch 动作的键。td3per 的
    `WindBattery`（MATLAB 后端）没有这个缓存 ⇒ 下面两行是 no-op。
    """
    for k, v in snap.items():
        setattr(env, k, v)
    if hasattr(env, '_pf_state'):
        env._pf_state = None
    return env


def snapshot_diff(env, snap):
    """诊断用：返回当前 env 与快照不一致的字段（`restore` 之后应为空）。"""
    bad = []
    for k, v in snap.items():
        cur = getattr(env, k, None)
        if k in SNAP_INT_FIELDS:
            cur = int(cur)
        else:
            cur = float(cur)
        if cur != v:
            bad.append((k, v, cur))
    return bad


# =============================================================================
# 4. 测试动作生成（计划 §3.2 / §8.4）
# =============================================================================
def _band(rng, lo, hi, n):
    return np.clip(rng.uniform(float(lo), float(hi), size=int(n)), A_LO, A_HI)


def branch_action(mode, rng, nt=5, prior=None, prior_jitter=PRIOR_JITTER,
                  low_band=LOW_BAND, high_band=HIGH_BAND):
    """生成一个 branch 的 5 维风机动作（**按构造就在 `[0,0.5]` 内**）。

    Parameters
    ----------
    mode : int          `MODE_PRIOR` / `MODE_LOW` / ...（`MODE_ACTOR` 会抛异常）
    rng : np.random.RandomState   独立的随机流（不吃训练流）

    Returns
    -------
    a5 : ndarray(5)     风机轴向因子
    """
    nt = int(nt)
    p = np.asarray(PRIOR5 if prior is None else prior, dtype=float).reshape(-1)
    if p.size < nt:
        p = np.full(nt, float(p[0]) if p.size else 1.0 / 3.0)
    p = p[:nt]
    mode = int(mode)

    if mode == MODE_PRIOR:
        j = float(prior_jitter)
        return np.clip(p + rng.uniform(-j, j, size=nt), A_LO, A_HI)
    if mode == MODE_LOW:
        return _band(rng, low_band[0], low_band[1], nt)
    if mode == MODE_HIGH:
        return _band(rng, high_band[0], high_band[1], nt)
    if mode == MODE_UNIFORM:
        return _band(rng, A_LO, A_HI, nt)
    if mode == MODE_RAMP:
        lo = rng.uniform(0.02, 0.12)
        hi = rng.uniform(0.38, 0.48)
        base = np.linspace(lo, hi, nt) if nt > 1 else np.array([lo])
        return np.clip(base + rng.uniform(-0.02, 0.02, size=nt), A_LO, A_HI)
    if mode == MODE_ALT:
        lo = rng.uniform(0.02, 0.12)
        hi = rng.uniform(0.38, 0.48)
        a = np.where(np.arange(nt) % 2 == 0, hi, lo)
        return np.clip(a + rng.uniform(-0.02, 0.02, size=nt), A_LO, A_HI)
    if mode == MODE_SINGLE:
        a = np.array(p, dtype=float, copy=True)
        j = int(rng.randint(0, max(1, nt)))
        if rng.uniform() < 0.5:
            a[j] = rng.uniform(SINGLE_LO[0], SINGLE_LO[1])
        else:
            a[j] = rng.uniform(SINGLE_HI[0], SINGLE_HI[1])
        return np.clip(a, A_LO, A_HI)
    raise ValueError('branch_action: 未知 mode %r（测试动作不含 actor）' % (mode,))


def signal_for_soc(soc, center=None, span=None):
    """把"目标 = 当前 SOC"反解成第 6 维信号（夹在 [−0.5, 0.5]）。"""
    c = SOC_TARGET_CENTER if center is None else float(center)
    s = SOC_TARGET_SPAN if span is None else float(span)
    if s <= 0:
        return 0.0
    return float(np.clip((float(soc) - c) / s, SIG_LO, SIG_HI))


# =============================================================================
# 5. 一个 branch 块：5 步推进 → 块级 transition（计划 §3.3）
# =============================================================================
def run_branch_block(env, rbb_system, a5, a6, snap, block_done, make_state,
                     minute_end=0, lower='freeze', nolow=False,
                     viol_derate_coef=VIOL_DERATE_COEF, block_len=BLOCK,
                     soc_tol=None, time=TIME_STEP, return_details=False):
    """在 `snap` 给定的**块首 state** 上跑 `block_len` 步，返回块级 transition 的原料。

    **不会**推进主 episode：函数退出前一定 `restore_env(env, snap)`（`finally`）。

    Parameters
    ----------
    a5 : array_like(5)      风机动作（已合法化；`[0,0.5]` 内）
    a6 : float              第 6 维（SOC 目标信号，`[-0.5,0.5]`）。`nolow=True` 时忽略
    snap : dict             `snapshot_env(env)` 的结果
    block_done : bool       该块末是否为 episode 终止（**只影响这条 branch 自己的 bootstrap**）
    make_state : callable   `(obs, env, minute) -> ndarray`，把块末原始观测变成
                            critic state（主循环传的是 `Normalize_H(..., phase_mode)`）。
    minute_end : int        块末的当天分钟数（相位特征用；主循环传
                            `phase_minute_of_day(episode_steps + BLOCK)`）
    lower : str             'freeze'（pb 命令 0，RBB 夹窗口）/ 'rule'（hst.rule_action）/
                            'none'（严格 pb≡0，绕开 RBB）
    nolow : bool            td3per 的 `--nolow` 分支：无第 6 维、gap 罚恒 0
    soc_tol : float         块末 gap 罚的豁免裕度（主循环传 `args.soc_tol`）
    time : float            单步时长 [h]（1/60）

    Returns
    -------
    dict: a5, a6, soc_target, r1(Σ5步，含 gap 罚), c1(Σ5步，含 gap 罚), s2(critic state),
          done, P_F(list), P_F_max, n_fold(P_F∈[2,4] 的步数), soc_start/soc_end,
          pbrule_sum, gap_pen, restored
    """
    a5 = np.asarray(a5, dtype=float).reshape(-1)[:5]
    if a5.size < 5:
        raise ValueError('run_branch_block: a5 需要 5 维, 得到 %d' % a5.size)
    a5 = np.clip(a5, A_LO, A_HI)

    a6 = 0.0 if nolow else float(np.clip(float(a6), SIG_LO, SIG_HI))
    soc_target = float(soc_target_from_signal(a6))
    tol = float(SOC_TOL if soc_tol is None else soc_tol)

    # 块首观测（**从快照重建**，不读 env —— env 可能已经被上一次 branch 动过）
    obs = np.array([float(snap['pr']), float(snap['u']),
                    float(snap['E']), float(snap['P_G_prev'])], dtype=float)
    soc_start = float(obs[2])
    derate = float(violation_derate(a5, viol_derate_coef))
    a_full = np.concatenate([a5, np.array([a6])])     # 6 维 H 动作（env 只读前 5 维）

    sum_r1 = 0.0
    sum_c1 = 0.0
    pbrule_sum = 0.0
    pf_list = []
    n_fold = 0
    gap_pen = 0.0
    ok = False
    try:
        pw = float(env.PowerOutput(float(obs[1]), a_full))
        for t in range(int(block_len)):
            E_cur = float(obs[2])
            n_rem = int(block_len) - t
            # 当前步的可行窗口（与 RBB 内部同式；`rule` 与诊断都要用）
            _, _win_info = rbb_system.compute_pbrule(
                pb_original=0.0, ptur=pw, pg_prev=float(obs[-1]), E=E_cur)
            # ---- 下层命令（branch 的"反事实下层"）----
            if lower == 'rule':
                pb_cmd = float(hst.rule_action(
                    soc_target, E_cur,
                    float(_win_info['pbrulemin']), float(_win_info['pbrulemax']),
                    n_rem, eta_c=rbb_system.eta_c, eta_d=rbb_system.eta_d,
                    time=rbb_system.time, rated_power=rbb_system.rated_power))
            else:                                   # 'freeze' / 'none'
                pb_cmd = 0.0
            pbrule, rbb_info = rbb_system.compute_pbrule(
                pb_original=pb_cmd, ptur=pw, pg_prev=float(obs[-1]), E=E_cur)
            if lower == 'none':
                # 严格冻结：绕开 RBB 的窗口夹裁，P_G = P_tur、E 一字不变。
                pbrule = 0.0
                rbb_info = dict(rbb_info)
                rbb_info['pb_original'] = 0.0
                rbb_info['pbrule'] = 0.0
                rbb_info['clip_amount'] = 0.0
            pw2, obs2, aL_env, _pen_L, _done, _done_L, _info = env.step(
                a_full, np.array([float(pbrule)]), pw)
            r1, c1s, det = Cal_Rew_RB(
                obs, obs2, a_full, rbb_info,
                penalty_NN_H=0.0, time=time, ptur=pw, pb_cmd=pb_cmd,
                turbine_derate=derate)
            sum_r1 += float(r1)
            sum_c1 += float(c1s)
            pbrule_sum += float(pbrule)
            _pf = float(det['P_F'])
            pf_list.append(_pf)
            if 2.0 <= _pf <= 4.0:
                n_fold += 1
            obs = np.asarray(obs2, dtype=float).reshape(-1)
            pw = float(pw2)

        # ---- 块末 SOC 目标 gap 罚（与主循环同式；nolow 恒 0）----
        gap_pen = 0.0
        if not nolow:
            gap_pen = float(gap_penalty(soc_target, float(obs[2]), tol=tol))
            if hc is not None and hc.mode() == 'uniform':
                sum_r1 -= gap_pen          # uniform: 追踪偏好进 r
            else:
                sum_c1 += gap_pen          # legacy: 进 c

        # ⚠ 必须在 restore 之前：make_state 可能读 env 的块末状态（相位/风速索引）
        s2 = np.asarray(make_state(obs, env, int(minute_end)), dtype=float)
        ok = True
    finally:
        restore_env(env, snap)

    out = {
        'a5': a5, 'a6': a6, 'soc_target': soc_target,
        'r1': float(sum_r1), 'c1': float(sum_c1), 's2': s2,
        'done': bool(block_done),
        'P_F': pf_list, 'P_F_max': (max(pf_list) if pf_list else 0.0),
        'n_fold': int(n_fold), 'n_steps': int(block_len),
        'soc_start': soc_start, 'soc_end': float(obs[2]),
        'pbrule_sum': float(pbrule_sum), 'gap_pen': float(gap_pen),
        'restored': bool(ok),
    }
    if return_details:
        out['obs2'] = np.asarray(obs, dtype=float).copy()
    return out


# =============================================================================
# 6. 控制器：何时 branch / 用哪些动作 / 汇总诊断
# =============================================================================
class BranchController(object):
    """决定"这个块要不要 branch"、生成动作、累计诊断。

    随机流**独立**（`RandomState(seed)`）⇒ 开关 branch 不改变 actor 轨迹的随机数序列
    （这是 plan §0/§4 的可复现性要求；也便于 A/B 配对）。
    """

    def __init__(self, args, seed=None):
        self.modes = parse_modes(getattr(args, 'branch_actions', DEFAULT_BRANCH_ACTIONS))
        self.every = max(0, int(getattr(args, 'branch_every', 0) or 0))
        self.pilot_blocks = max(0, int(getattr(args, 'branch_pilot_blocks', 0) or 0))
        self.enabled = bool(self.every > 0 or self.pilot_blocks > 0) and bool(self.modes)
        self.lower = str(getattr(args, 'branch_lower', 'freeze') or 'freeze')
        if self.lower not in LOWER_MODES:
            raise ValueError('--branch_lower 必须是 %s，收到 %r'
                             % ('/'.join(LOWER_MODES), self.lower))
        self.signal = str(getattr(args, 'branch_signal', 'soc') or 'soc')
        if self.signal not in SIGNAL_MODES:
            raise ValueError('--branch_signal 必须是 %s，收到 %r'
                             % ('/'.join(SIGNAL_MODES), self.signal))
        self.nolow = bool(int(getattr(args, 'nolow', 0) or 0))
        crit_frac = float(getattr(args, 'branch_critic_frac', 0.45) or 0.0)
        self.critic_frac = float(np.clip(crit_frac, 0.0, 0.5))
        self.sup_beta0 = float(getattr(args, 'branch_sup_beta', 0.0) or 0.0)
        self.sup_blocks = max(1, int(getattr(args, 'branch_sup_blocks', 20000) or 1))
        self.low_band = (float(getattr(args, 'branch_low_lo', LOW_BAND[0])),
                         float(getattr(args, 'branch_low_hi', LOW_BAND[1])))
        self.high_band = (float(getattr(args, 'branch_high_lo', HIGH_BAND[0])),
                          float(getattr(args, 'branch_high_hi', HIGH_BAND[1])))
        self.prior_jitter = float(getattr(args, 'branch_prior_jitter', PRIOR_JITTER))
        self.nt = int(getattr(args, 'nb_turbine', 5) or 5)
        _sd = int(getattr(args, 'branch_seed', -1) or -1)
        if _sd < 0:
            _sd = int(seed if seed is not None else getattr(args, 'seed', 0) or 0) + 104729
        self.rng = np.random.RandomState(_sd)
        self.seed = int(_sd)

        self.block_id = 0            # 已生成的 branch 块序号（写进 replay 的 block_id）
        self.n_blocks = 0            # 触发过 branch 的块数
        self.n_trans = 0             # 写入 replay 的 branch transition 条数
        self.n_branch_calls = 0      # 被调用 trigger() 的块数
        self.stats = {m: dict(n=0, sum_c1=0.0, sum_r1=0.0, sum_pf=0.0, n_steps=0,
                              n_fold=0, n_lo=0, n_hi=0, n_a=0, sum_pf_max=0.0)
                      for m in self.modes}

    # ---------------- 触发 ----------------
    def trigger(self, step):
        """`step` = 全局环境步（块首时刻）。返回本块是否 branch。

        规则（计划 §5）：前 `pilot_blocks` 块**每块** branch（阶段 A pilot），
        之后每 `every` 块一次（阶段 B online）。
        """
        if not self.enabled:
            return False
        b = int(step) // BLOCK
        self.n_branch_calls += 1
        if b < self.pilot_blocks:
            return True
        if self.every <= 0:
            return False
        return (b % self.every == 0)

    # ---------------- 第 6 维 ----------------
    def signal_value(self, soc_start, actor_signal=None):
        """branch 的第 6 维（SOC 目标信号）。`--branch_signal` 见模块 docstring。"""
        if self.nolow:
            return 0.0
        if self.signal == 'target' and actor_signal is not None:
            return float(np.clip(float(actor_signal), SIG_LO, SIG_HI))
        if self.signal == 'zero':
            return 0.0
        if self.signal == 'pm1':
            return float(SIG_HI if self.rng.uniform() < 0.5 else SIG_LO)
        if self.signal == 'random':
            return float(self.rng.uniform(SIG_LO, SIG_HI))
        return signal_for_soc(soc_start)          # 'soc'（默认）

    # ---------------- 动作 ----------------
    def action(self, mode):
        return branch_action(mode, self.rng, nt=self.nt,
                             prior_jitter=self.prior_jitter,
                             low_band=self.low_band, high_band=self.high_band)

    # ---------------- 诊断 ----------------
    def note(self, mode, res):
        """累计一个 branch transition 的诊断（**只写 branch 专用 CSV/TB**）。"""
        st = self.stats.get(int(mode))
        if st is None:
            return
        a5 = np.asarray(res['a5'], dtype=float).reshape(-1)
        st['n'] += 1
        st['sum_c1'] += float(res['c1'])
        st['sum_r1'] += float(res['r1'])
        st['n_steps'] += int(res['n_steps'])
        st['n_fold'] += int(res['n_fold'])
        st['sum_pf'] += float(np.sum(res['P_F']))
        st['sum_pf_max'] += float(res['P_F_max'])
        st['n_lo'] += int(np.sum(a5 < 0.05))
        st['n_hi'] += int(np.sum(a5 > 0.45))
        st['n_a'] += int(a5.size)
        self.n_trans += 1

    def begin_block(self):
        """分配一个 block_id（同一块首的所有 branch 共享）。"""
        self.block_id += 1
        self.n_blocks += 1
        return int(self.block_id)

    def sup_beta(self, n_branch_blocks):
        """辅助监督权重 β 的线性衰减（计划 §6.2：β0 → 0.2·β0）。"""
        if self.sup_beta0 <= 0.0:
            return 0.0
        t = min(1.0, float(max(0, n_branch_blocks)) / float(self.sup_blocks))
        return float(self.sup_beta0 * (1.0 - 0.8 * t))

    def coverage(self):
        """动作覆盖诊断（计划 §7 的"边界覆盖 ≥5%"两个口径）。"""
        n_a = sum(st['n_a'] for st in self.stats.values())
        n_lo = sum(st['n_lo'] for st in self.stats.values())
        n_hi = sum(st['n_hi'] for st in self.stats.values())
        n_steps = sum(st['n_steps'] for st in self.stats.values())
        n_fold = sum(st['n_fold'] for st in self.stats.values())
        return {
            'frac_lo': (n_lo / n_a if n_a else 0.0),      # a < 0.05 的比例（**动作维**口径）
            'frac_hi': (n_hi / n_a if n_a else 0.0),      # a > 0.45
            'frac_fold': (n_fold / n_steps if n_steps else 0.0),   # P_F ∈ [2,4] 的**步**比例
            'n_a_dims': int(n_a), 'n_steps': int(n_steps),
        }

    def per_mode_mean(self):
        """每个 mode 的 (mean_c1, mean_r1, mean_P_F) —— 诊断 branch 是否真的产生差异。"""
        out = {}
        for m, st in self.stats.items():
            n = st['n']
            ns = st['n_steps']
            out[MODE_NAME[m]] = {
                'n': n,
                'c1': (st['sum_c1'] / n if n else float('nan')),
                'r1': (st['sum_r1'] / n if n else float('nan')),
                'P_F': (st['sum_pf'] / ns if ns else float('nan')),
            }
        return out

    def summary(self):
        cov = self.coverage()
        lines = ['[branch] actions={} | every={} | pilot={} | lower={} | signal={} | '
                 'critic_frac={:.2f} | seed={}'.format(
                     ','.join(MODE_NAME[m] for m in self.modes), self.every,
                     self.pilot_blocks, self.lower, self.signal,
                     self.critic_frac, self.seed)]
        if not self.enabled:
            lines.append('[branch] **未启用**（--branch_every 0 且 --branch_pilot_blocks 0）')
            return '\n'.join(lines)
        pm = self.per_mode_mean()
        for name, d in pm.items():
            lines.append('  %-8s n=%-6d  mean_c1=%9.4f  mean_r1=%9.2f  mean_P_F=%.3f'
                         % (name, d['n'], d['c1'], d['r1'], d['P_F']))
        lines.append('  blocks=%d (transitions=%d) | a<0.05 %.1f%% | a>0.45 %.1f%% | '
                     'P_F∈[2,4] %.1f%%'.format(
                         self.n_blocks, self.n_trans, 100 * cov['frac_lo'],
                         100 * cov['frac_hi'], 100 * cov['frac_fold']))
        return '\n'.join(lines)


def declared_scope():
    """一句话口径声明（打到 train.log 里，便于事后审计）。"""
    return ('[branch] 每个块从**同一完整块首 state**（pr/u/E/P_G_prev/count_wind/count_price）'
            '生成 actor + {} 个测试动作；所有 transition 都进同一 replay 当 TD 数据，'
            '但 actor/lower/λ/EMA/curriculum/CSV/TB/test **只用 actor source**；'
            'branch 只覆盖 1 个块（n_boot=γ），不进 n-step 链'.format(
                len(parse_modes(DEFAULT_BRANCH_ACTIONS))))


USAGE = """\
`td3per_branch` 是 **口径模块**（纯函数 + 常量 + 控制器），**不是训练入口**。

训练入口是 `td3per/td3per_main.py`（`--branch_*` 都属于它；注意它 import 期会
`matlab.engine.start_matlab()`，需要 MATLAB 运行时）：

    python td3per/td3per_main.py --branch_every 10
    python td3per/td3per_main.py --branch_actions prior,low,high,uniform \\
        --branch_pilot_blocks 2000 --branch_every 0 --max_steps 10000

全部 {n_flags} 个 branch 开关：
    python td3per/td3per_main.py --help        # 搜 `branch（同块首宽动作覆盖`

契约测试（纯静态，不跑仿真）：
    python agentutil/test_td3per_branch_contract.py

本模块对外提供的接口：
    snapshot_env / restore_env / snapshot_diff      块首 state 快照与恢复
    branch_action / parse_modes / signal_for_soc    7 种测试动作生成器 + 信号维
    run_branch_block                                在快照上跑 5 步 → 块级 transition
    BranchController                                触发 / 信号 / 诊断 / β 衰减
    add_branch_args                                 CLI（由 td3per_main 调用）
"""


def _n_branch_flags():
    """数一下 `add_branch_args` 挂了多少个 `--branch_*`（只给 USAGE 用）。"""
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    add_branch_args(p)
    return sum(1 for a in p._actions for o in a.option_strings
               if o.startswith('--branch'))


if __name__ == '__main__':
    print(USAGE.format(n_flags=_n_branch_flags()))
    sys.exit(2)
