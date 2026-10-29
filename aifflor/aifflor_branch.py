# -*- coding: utf-8 -*-
"""`aifflor` 同块首宽动作覆盖（**branch**）—— 口径唯一真源。

计划书：`documentation/aifflor_branch_wide_coverage_plan.md`
（本文件 = 该计划 §3/§5/§8.1/§8.4 的可执行实现；§8.2/§8.3 见
 `PER_buffer.py` 的 `set_meta`/`sample_pool` 与 `aifflor_ddpg.py` 的
 `push_branch_transition` / `update_policy`。）

一句话
------
**每个块（5 步）从同一个「完整块首 state」上再跑若干个测试动作**，把它们各自的
块级 `(s, a, Σr1, Σc1, s', done)` 当成**正常 TD 数据**写进同一个 replay；
但 `actor / lower / λ / EMA / curriculum / CSV / TB / test` 只用 actor 分支。

为什么要它（计划 §1.2）
-----------------------
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
   （由 `SP_TD3.push_branch_transition` 保证，见 `aifflor_ddpg.py`）。

口径对照（HSRL 语义，**与计划书里的 SOC 目标口径不同，此处已换算**）
-------------------------------------------------------------------
计划书写于"第 6 维 = SOC 目标信号 `±1`"的年代；`--hsrl` 之后第 6 维是
**块总放电量 `g` [MW·min]**（`aifflor_hsrl.HSRL_*`）。本模块据此换算：

| 计划书 | 本实现 |
|---|---|
| 第 6 维 `goal = ±1` | `--branch_goal_pm1 1` ⇒ `g = ±满可行量`（`+D_max` 放电 / `−C_max` 充电） |
| `--branch_freeze_lower 1` | **下层 BESS 冻结在 `pb ≡ 0`** ⇒ `P_G = P_tur`、`E` 不变 ⇒ 块内 `c1` 只由**风机动作**决定（对上层 critic 是最干净的反事实） |
| 冻结下层时的 `g` | 恒 `0.0`：`pb≡0` 时 `g` 对物理**没有任何作用**，留着只会污染动作维 |

⇒ 默认组合 = `--branch_freeze_lower 1 --branch_goal_pm1 0`：
**上下层解耦，branch 只改 5 维风机动作**。

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
* 所有动作**按构造就在 `[0,0.5]` 内** ⇒ 不需要"触边只截噪声"那一步；仍走
  `hsrl.check_action_H_hsrl` 做一次合法化 + 记越界（正常为 0）。
* `a = 0` 会撞 FLORIS 的高斯尾流死点，但 `AifflorPhysics.a_clip` 已把
  物理层抬到 `A_PHYS_LO = 0.025`（见 `aifflor/README.md` §3.5）⇒ 安全。

成本（计划 §5）
--------------
一个 branch 块 = `len(modes) × BLOCK` 个额外环境步。默认 4 个测试动作 ⇒
每触发一次 +20 步。`--branch_every 10` ⇒ 每 10 块 +20 步 ⇒ **1.4×**（含 actor 的 5 步
是 50 步/10 块 + 20 = 70 ⇒ 1.4×）。计划书给的 K=10 ⇒ 1.3× 是"每块 5 动作"口径的
近似值，实际按上式算。

⚠ 不要在 branch 里调用 `env.reset()` / `env.step` 之外的任何推进接口：
`restore_env` 只认 `pr/u/E/P_G_prev/count_wind/count_price` 这 6 个标量。
"""
import numpy as np

try:                                            # 主路径：aifflor 作为包
    from aifflor import aifflor_hsrl as hsrl
    from aifflor.aifflor_myfunc import violation_derate, VIOL_DERATE_COEF
except ImportError:                             # 兼容：直接把 aifflor/ 放进 sys.path
    import aifflor_hsrl as hsrl                 # type: ignore
    from aifflor_myfunc import violation_derate, VIOL_DERATE_COEF   # type: ignore


# =============================================================================
# 1. 常量（口径单一真源）
# =============================================================================
BLOCK = hsrl.HSRL_BLOCK                  # 5 步/块（= δ）

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

#: 5 台风机的标称/中性动作（`init_wt_action_bias` 的 lo/hi 中点带里的 "prior"）
PRIOR5 = np.array([0.2, 0.2, 0.2, 0.2, 1.0 / 3.0], dtype=float)
PRIOR_JITTER = 0.05          # prior 的逐维抖动半宽 U(−δ,+δ)
LOW_BAND = (0.00, 0.10)      # 强降载带（含 a<0.05 半边 ⇒ 满足 §7 的边界覆盖 ≥5%）
HIGH_BAND = (0.40, 0.50)     # 满发带（含 a>0.45 半边）
SINGLE_LO = (0.02, 0.10)     # `single` 模式的单台扰动量程（低侧）
SINGLE_HI = (0.40, 0.48)     # `single` 模式的单台扰动量程（高侧）

A_LO, A_HI = 0.0, 0.5        # 学习带（与 `CheckAction` / `env.py` 一致）

#: 对偶/预算的口径：branch 数据**不**允许进入这些通道（计划 §4.3 / §4.4）
ACTOR_ONLY_CHANNELS = ('lambda', 'ema', 'curriculum', 'csv', 'tensorboard', 'test')

#: `snapshot_env` 保存的字段（这几个标量 + 固定的 `wind_day` 就完全决定了后续 5 步）
SNAP_FIELDS = ('pr', 'u', 'E', 'P_G_prev', 'count_wind', 'count_price')
SNAP_INT_FIELDS = ('count_wind', 'count_price')


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
                        '= 计划 §5 的 1.3×~1.4× 档；4 = 1.9× 档）')
    g.add_argument('--branch_pilot_blocks', default=0, type=int,
                   help='前 N 个块**每块都 branch**（= 计划 §5 阶段 A 的 pilot/预训练；'
                        '配合 --branch_every 使用，N 之后回到每 K 块一次）')
    g.add_argument('--branch_freeze_lower', default=1, type=int,
                   help='1（默认）= branch 内下层 BESS 冻结在 pb≡0 ⇒ P_G=Ptur、E 不变，'
                        'c1 只由风机动作决定；0 = 用摊还执行器兑现该 branch 的 g')
    g.add_argument('--branch_goal_pm1', default=0, type=int,
                   help='1 = branch 的第 6 维 g 随机取 ±满可行量（+D_max 放电 / −C_max 充电）；'
                        '0（默认）= g≡0。仅在 --branch_freeze_lower 0 时有物理作用')
    g.add_argument('--branch_critic_frac', default=0.45, type=float,
                   help='critic 批里 branch 的占比（计划 §4.2 的 40–50%%；'
                        '默认 0.45；>0.5 会被夹到 0.5，避免 actor 被淹没）')
    g.add_argument('--branch_sup_beta', default=0.0, type=float,
                   help='可选的 branch 辅助监督权重 β0（计划 §6.2；'
                        '0 = 关闭（默认）。开启后 β 从 β0 线性降到 0.2·β0）')
    g.add_argument('--branch_sup_blocks', default=20000, type=int,
                   help='辅助监督 β 的线性衰减跨度（块数；只统计 branch 块）')
    g.add_argument('--branch_csv', default=1, type=int,
                   help='1（默认）= 写 <output>/df_branch_hsrl.csv（逐 branch transition）')
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
    """把环境恢复到 `snapshot_env` 的时刻，并把物理预取缓存**置为失效**。

    ⚠ 为什么必须让 `_pf_state = None` 而不是恢复成快照：
    `AifflorPhysics.prefetch` 是**整体替换** `_cache` 的（见 `aifflor_env.py`），
    branch 跑完之后 `_cache` 里是 branch 动作的键 ⇒ 若只把 `_pf_state` 恢复回去，
    "缓存有效"这个断言就是假的（正是 README §3.5 记载过的那类坑）。
    置 `None` 只让下一次 `PowerOutput` 重新预取一次（代价 ≈ 一次批量求值）。
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


# =============================================================================
# 5. 一个 branch 块：5 步推进 → 块级 transition（计划 §3.3）
# =============================================================================
def run_branch_block(env, rbb_system, a5, g, snap, block_done, make_state,
                     minute_end=0, freeze_lower=True,
                     viol_derate_coef=VIOL_DERATE_COEF, block_len=BLOCK,
                     return_details=False):
    """在 `snap` 给定的**块首 state** 上跑 `block_len` 步，返回块级 transition 的原料。

    **不会**推进主 episode：函数退出前一定 `restore_env(env, snap)`（`finally`）。
    唯一的外部状态副作用是 `env.physics._cache` 被 branch 的动作重填，
    且 `_pf_state` 被置为 `None`（⇒ 下一次 `PowerOutput` 重新预取，见 `restore_env`）。

    Parameters
    ----------
    a5 : array_like(5)      风机动作（已合法化；`[0,0.5]` 内）
    g : float               第 6 维（块总放电量 MW·min）。`freeze_lower=True` 时无物理作用
    snap : dict             `snapshot_env(env)` 的结果
    block_done : bool       该块末是否为 episode 终止（**只影响这条 branch 自己的 bootstrap**）
    make_state : callable   `(obs, env, minute) -> ndarray`，把块末原始观测变成
                            critic state（主循环传的是 `Normalize_H + _append_wind5_H`）。
                            ⚠ 必须在 `restore_env` **之前**调用 —— 它要读块末的
                            `env.count_wind`（五风速 `u2..u5` 依赖它）。
    minute_end : int        块末的当天分钟数（相位特征用；主循环传 `episode_steps + BLOCK`）
    freeze_lower : bool     True ⇒ `pb ≡ 0`（下层冻结，`E` 不变）；False ⇒ 摊还执行器

    Returns
    -------
    dict: a5, g, r1(Σ5步), c1(Σ5步), s2(critic state), done, P_F(list), P_F_max,
          n_fold(P_F∈[2,4] 的步数), soc_start/soc_end, sum_disch, pbrule_sum
    """
    a5 = np.asarray(a5, dtype=float).reshape(-1)[:5]
    if a5.size < 5:
        raise ValueError('run_branch_block: a5 需要 5 维, 得到 %d' % a5.size)
    a5 = np.clip(a5, A_LO, A_HI)

    # 块首观测（**从快照重建**，不读 env —— env 可能已经被上一次 branch 动过）
    obs = np.array([float(snap['pr']), float(snap['u']),
                    float(snap['E']), float(snap['P_G_prev'])], dtype=float)
    soc_start = float(obs[2])
    derate = float(violation_derate(a5, viol_derate_coef))

    sum_r1 = 0.0
    sum_c1 = 0.0
    sum_disch = 0.0
    pbrule_sum = 0.0
    pf_list = []
    n_fold = 0
    ok = False
    try:
        pw = float(env.PowerOutput(float(obs[1]), a5))
        for t in range(int(block_len)):
            E_cur = float(obs[2])
            n_rem = int(block_len) - t
            if freeze_lower:
                pb_cmd = 0.0
            else:
                pb_cmd = float(hsrl.amortized_rule_action(g, sum_disch, n_rem, E_cur))
            pbrule, rbb_info = rbb_system.compute_pbrule(
                pb_original=pb_cmd, ptur=pw, pg_prev=float(obs[-1]), E=E_cur)
            action_rbb = np.concatenate([a5, np.array([float(pbrule)])])
            pw2, obs2, aL_env, _pen_L, _done, _done_L, _info = env.step(
                a5, np.array([float(pbrule)]), pw)
            r1, c1s, det = hsrl.cal_rew_rb_hsrl(
                obs, obs2, action_rbb, rbb_info, penalty_NN=0.0, time=hsrl.DT,
                ptur=pw, pb_cmd=pb_cmd, turbine_derate=derate)
            sum_r1 += float(r1)
            sum_c1 += float(c1s)
            sum_disch += float(hsrl.discharge_of(
                float(np.asarray(aL_env, dtype=float).reshape(-1)[0])))
            pbrule_sum += float(pbrule)
            _pf = float(det['P_F'])
            pf_list.append(_pf)
            if 2.0 <= _pf <= 4.0:
                n_fold += 1
            obs = np.asarray(obs2, dtype=float).reshape(-1)
            pw = float(pw2)
        # ⚠ 必须在 restore 之前：make_state 要读块末的 env.count_wind（u2..u5）
        s2 = np.asarray(make_state(obs, env, int(minute_end)), dtype=float)
        ok = True
    finally:
        restore_env(env, snap)

    out = {
        'a5': a5, 'g': float(g), 'r1': float(sum_r1), 'c1': float(sum_c1),
        's2': s2, 'done': bool(block_done),
        'P_F': pf_list, 'P_F_max': (max(pf_list) if pf_list else 0.0),
        'n_fold': int(n_fold), 'n_steps': int(block_len),
        'soc_start': soc_start, 'soc_end': float(obs[2]),
        'sum_disch': float(sum_disch), 'pbrule_sum': float(pbrule_sum),
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
        self.freeze_lower = bool(int(getattr(args, 'branch_freeze_lower', 1) or 0))
        self.goal_pm1 = bool(int(getattr(args, 'branch_goal_pm1', 0) or 0))
        crit_frac = float(getattr(args, 'branch_critic_frac', 0.45) or 0.0)
        self.critic_frac = float(np.clip(crit_frac, 0.0, 0.5))
        self.sup_beta0 = float(getattr(args, 'branch_sup_beta', 0.0) or 0.0)
        self.sup_blocks = max(1, int(getattr(args, 'branch_sup_blocks', 20000) or 1))
        self.low_band = (float(getattr(args, 'branch_low_lo', LOW_BAND[0])),
                         float(getattr(args, 'branch_low_hi', LOW_BAND[1])))
        self.high_band = (float(getattr(args, 'branch_high_lo', HIGH_BAND[0])),
                          float(getattr(args, 'branch_high_hi', HIGH_BAND[1])))
        self.prior_jitter = float(getattr(args, 'branch_prior_jitter', PRIOR_JITTER))
        self.nt = int(getattr(args, 'na_box', 5) or 5)
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
    def goal(self, feas_c_max, feas_d_max):
        """branch 的第 6 维 g [MW·min]。

        冻结下层（默认）⇒ 恒 0：`pb≡0` 时 g 没有物理作用，留着只是污染动作维。
        否则按 `--branch_goal_pm1` 取 ±满可行量，或 0。
        """
        if self.freeze_lower:
            return 0.0
        if not self.goal_pm1:
            return 0.0
        if self.rng.uniform() < 0.5:
            return float(feas_d_max)
        return -float(feas_c_max)

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
        lines = ['[branch] actions={} | every={} | pilot={} | freeze_lower={} | goal_pm1={} | '
                 'critic_frac={:.2f} | seed={}'.format(
                     ','.join(MODE_NAME[m] for m in self.modes), self.every,
                     self.pilot_blocks, int(self.freeze_lower), int(self.goal_pm1),
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
