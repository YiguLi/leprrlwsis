# -*- coding: utf-8 -*-
"""`aifflor` branch（同块首宽动作覆盖）的**契约测试** —— 纯静态/纯算术，**不跑仿真**。

覆盖范围（对应 `documentation/aifflor_branch_wide_coverage_plan.md` 的验收口径）
---------------------------------------------------------------------------
| 段 | 内容 |
|---|---|
| A | 常量一致性（`aifflor_ddpg.SRC_*/MODE_ACTOR`、`PER_buffer` 的元数据字段） |
| B | `parse_modes` / `branch_action`：范围、边界带覆盖、mode 差异 |
| C | `snapshot_env` / `restore_env`：6 个标量往返 + 预取缓存失效 |
| D | `run_branch_block`：**独立重算** 5 步 Σr1/Σc1（不用 env.step，走纯数组公式）+ 主轨迹不受影响 |
| E | `BranchController.trigger`：pilot / every 语义 |
| F | PER 分池采样：计数、池正确性、**环形覆盖**后的计数正确 |
| G | `recent_states` 的 source 过滤（λ 的分子只看 actor） |
| H | `push_branch_transition` 的 `n_boot=γ` vs actor 的 `γ^n` |
| I | `update_policy`：branch 路径的形状/掩码/actor 只吃 actor 行；`branch_mix=0` 时走历史路径 |

运行
----
```powershell
python agentutil/test_aifflor_branch_contract.py
```

⚠ 本文件**不**推进任何 FLORIS / WFSim 仿真：段 D 用 `env.WindBattery` + 一个
**多项式假物理后端**（`StubPhysics`），`PowerOutput = 1e-4·u³·ΣCp(a)`，与 FLORIS 无关。
"""
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块
sys.path.insert(0, _HERE)                              # agentutil 内互相 import

import numpy as np                                              # noqa: E402

import hsrl_cost as hc                                          # noqa: E402
from PER_buffer import PrioritizedReplayBuffer                  # noqa: E402
from env import WindBattery                                      # noqa: E402
from aifflor import aifflor_hsrl as hsrl                        # noqa: E402
from aifflor import aifflor_branch as abr                       # noqa: E402
from aifflor import aifflor_ddpg as addpg                       # noqa: E402
from aifflor.aifflor_myfunc import C1_RISK_COEF                  # noqa: E402

FAIL = []
N_CK = [0]


def ck(cond, msg):
    N_CK[0] += 1
    if not cond:
        FAIL.append(msg)
        print('  [FAIL] %s' % msg)
    return bool(cond)


def ck_close(a, b, tol, msg):
    return ck(abs(float(a) - float(b)) <= float(tol),
              '%s (得到 %.9g, 期望 %.9g, tol %.3g)' % (msg, a, b, tol))


# =============================================================================
# A. 常量一致性
# =============================================================================
def section_a():
    print('\n[A] 常量一致性')
    ck(addpg.SRC_ACTOR == abr.SRC_ACTOR, 'SRC_ACTOR 一致')
    ck(addpg.SRC_BRANCH == abr.SRC_BRANCH, 'SRC_BRANCH 一致')
    ck(addpg.MODE_ACTOR == abr.MODE_ACTOR, 'MODE_ACTOR 一致')
    ck(abr.BLOCK == hsrl.HSRL_BLOCK, 'BLOCK == HSRL_BLOCK')
    ck(set(abr.MODE_NAME) == set(range(len(abr.MODE_ORDER))), 'MODE_NAME 连续编号')
    ck(all(n in abr.MODE_BY_NAME for n in abr.MODE_ORDER), 'MODE_BY_NAME 反查齐全')
    for f in ('src', 'mode', 'block_id', 'lower_ver'):
        ck(f in PrioritizedReplayBuffer.META_FIELDS, 'PER 有元数据字段 %r' % f)
    ck(PrioritizedReplayBuffer.META_FIELDS['src'][0] == abr.SRC_ACTOR,
       'src 的默认值 = actor')
    ck(abr.parse_modes(abr.DEFAULT_BRANCH_ACTIONS) == list(abr.BRANCH_MODES_DEFAULT),
       '默认动作集 = prior/low/high/uniform')


# =============================================================================
# B. 动作生成
# =============================================================================
def section_b():
    print('\n[B] parse_modes / branch_action')
    ck(abr.parse_modes('prior,low') == [abr.MODE_PRIOR, abr.MODE_LOW], 'parse 基本')
    try:
        abr.parse_modes('prior,bogus')
        ck(False, '未知 mode 应抛 ValueError')
    except ValueError:
        ck(True, '未知 mode 抛 ValueError')
    ck(abr.parse_modes('actor,prior,prior') == [abr.MODE_PRIOR], 'actor 被跳过 + 去重')
    ck(abr.parse_modes('') == [], '空串 ⇒ 空集')

    rng = np.random.RandomState(7)
    nn = 2000
    bands = {abr.MODE_PRIOR: None, abr.MODE_LOW: abr.LOW_BAND,
             abr.MODE_HIGH: abr.HIGH_BAND, abr.MODE_UNIFORM: (0.0, 0.5),
             abr.MODE_RAMP: None, abr.MODE_ALT: None, abr.MODE_SINGLE: None}
    all_a = {m: [] for m in bands}
    for m in bands:
        for _ in range(nn):
            a = abr.branch_action(m, rng)
            if not ck(a.shape == (5,), 'mode %s 形状 (5,)' % abr.MODE_NAME[m]):
                break
            if not ck(np.all(a >= 0.0) and np.all(a <= 0.5),
                      'mode %s 在 [0,0.5] 内' % abr.MODE_NAME[m]):
                break
            all_a[m].append(a)
    for m, band in bands.items():
        if band is None:
            continue
        A = np.array(all_a[m])
        ck(A.min() >= band[0] - 1e-12 and A.max() <= band[1] + 1e-12,
           'mode %s 落在带 %s' % (abr.MODE_NAME[m], band))

    # 边界覆盖（计划 §7：a<0.05 / a>0.45 各 ≥5%）
    A = np.concatenate([np.array(v) for k, v in all_a.items()
                        if k in abr.BRANCH_MODES_DEFAULT]).reshape(-1)
    frac_lo = float(np.mean(A < 0.05))
    frac_hi = float(np.mean(A > 0.45))
    ck(frac_lo >= 0.05, 'a<0.05 覆盖 %.1f%% ≥ 5%%' % (100 * frac_lo))
    ck(frac_hi >= 0.05, 'a>0.45 覆盖 %.1f%% ≥ 5%%' % (100 * frac_hi))

    # mode 之间必须**真的不同**（否则 branch 只是重复同一种动作）
    means = {k: float(np.mean(np.array(v))) for k, v in all_a.items()}
    ck(means[abr.MODE_LOW] < means[abr.MODE_PRIOR] < means[abr.MODE_HIGH],
       '均值序: low < prior < high')
    ck(abs(means[abr.MODE_LOW] - means[abr.MODE_HIGH]) > 0.2, 'low/high 的均值差 > 0.2')
    # single：只改一台
    for _ in range(50):
        a = abr.branch_action(abr.MODE_SINGLE, rng)
        d = np.abs(a - abr.PRIOR5)
        if not ck(int(np.sum(d > 0.05)) <= 1, 'single 只改一台'):
            break
    # ramp：单调（允许 ±0.02 抖动 ⇒ 用大样本的均值判单调）
    R = np.array(all_a[abr.MODE_RAMP]).mean(axis=0)
    ck(np.all(np.diff(R) > 0), 'ramp 均值逐台单调递增')
    # 确定性（同一 rng 状态 ⇒ 同一动作）
    r1 = np.random.RandomState(11).uniform(size=8)
    a1 = abr.branch_action(abr.MODE_UNIFORM, np.random.RandomState(11))
    _ = r1
    a2 = abr.branch_action(abr.MODE_UNIFORM, np.random.RandomState(11))
    ck(np.array_equal(a1, a2), '同种子 ⇒ 同动作（独立随机流可复现）')


# =============================================================================
# 假物理后端 + 真 env（段 C/D 用）
# =============================================================================
class StubPhysics(object):
    """多项式假物理：`P = 1e-4·u³·Σ Cp(a)`，`Cp(a)=4a(1−a)²`。**与 FLORIS 无关。**"""

    def __init__(self, nt=5):
        self.nt = int(nt)
        self.n_calls = 0

    def PowerOutput(self, u0, act_wind):
        self.n_calls += 1
        a = np.asarray(act_wind, dtype=float).reshape(-1)[:self.nt]
        a = np.clip(a, 0.0, 0.5)
        cp = 4.0 * a * (1.0 - a) ** 2
        return float(1e-4 * float(u0) ** 3 * float(np.sum(cp)))


def make_env(seed=0, e0=0.25, cw=10):
    """真 `env.WindBattery` + 假物理，停在某个"块首"。"""
    np.random.seed(seed)
    env = WindBattery(max_E=0.5, num_turbine=5, upload=False, engine=None,
                      physics=StubPhysics(5))
    env.reset()
    env.diag_const_pr = 300.0          # 恒定电价（诊断钩子）⇒ 可以纯数组重算 r1
    env.count_wind = int(cw)
    env.u = float(env.wind_day[env.count_wind])
    env.E = float(e0)
    env.count_price = 0
    env.pr = 300.0
    env.P_G_prev = float(env.PowerOutput(env.u, env.nominal()))
    return env


# =============================================================================
# C. 快照 / 恢复
# =============================================================================
def section_c():
    print('\n[C] snapshot_env / restore_env')
    env = make_env()
    snap = abr.snapshot_env(env)
    ck(set(snap) == set(abr.SNAP_FIELDS), '快照字段齐全')
    ck(isinstance(snap['count_wind'], int) and isinstance(snap['pr'], float),
       '快照类型（count_* 为 int，其余 float）')
    # 推进 5 步（任意动作），再看 restore 能否逐位回到块首
    a5 = np.full(5, 0.3)
    for _ in range(5):
        pw = float(env.PowerOutput(env.u, a5))
        env.step(a5, np.array([0.0]), pw)
    ck(env.count_wind != snap['count_wind'], '推进后 count_wind 确实变了')
    env._pf_state = ('dirty', None)
    abr.restore_env(env, snap)
    ck(abr.snapshot_diff(env, snap) == [], 'restore 后 6 个字段逐位相同')
    ck(env._pf_state is None, 'restore 把预取状态置 None（缓存失效）')


# =============================================================================
# D. run_branch_block：独立重算 + 主轨迹不受影响
# =============================================================================
def _expected_branch_block(env, a5, snap, block_len=5, pr=300.0, dt=1.0 / 60.0):
    """**独立**重算：纯数组公式（不调用 env.step / cal_rew_rb_hsrl）。"""
    n = len(env.wind_day)
    cw = int(snap['count_wind'])
    a5 = np.clip(np.asarray(a5, dtype=float).reshape(-1)[:5], 0.0, 0.5)
    cp = 4.0 * a5 * (1.0 - a5) ** 2

    def P(u):
        return float(1e-4 * float(u) ** 3 * float(np.sum(cp)))

    # 逐步语义（与 `env.WindBattery.step` 逐位对齐）：
    #   * `env.step(applied, act_bat, pw_tru)` 里 `P_G = pw_tru − P_C + P_D`
    #     ⇒ **本步**的 P_G 用的是**本步**的风机功率 P_t（不是下一步的）；
    #   * `observation2 = [pr, u_{t+1}, E, P_G(t)]` ⇒ 下一步的 `observation[-1]` 就是 P_G(t)。
    #   ⇒ block 里第 t 步的 P_F = |P_t − P_{t−1}|，其中 P_{−1} = **块首快照的 P_G_prev**
    #     （这正是 snapshot 必须含 P_G_prev 的原因：否者第 1 步的 P_F 就错了）。
    pv = []
    for i in range(int(block_len)):
        u_t = (float(snap['u']) if i == 0
               else float(np.asarray(env.wind_day)[(cw + i) % n]))
        pv.append(P(u_t))
    r1 = 0.0
    for i in range(int(block_len)):
        r1 += pr * pv[i] * dt
    prev = float(snap['P_G_prev'])
    pf = []
    for i in range(int(block_len)):
        pf.append(abs(pv[i] - prev))
        prev = pv[i]
    c1 = 0.0
    for v in pf:
        legacy = (C1_RISK_COEF * (10.0 * (v - 3.0) + 0.1 * v) if v > 3.0 + 1e-4
                  else C1_RISK_COEF * (0.1 * v))
        c1 += hc.c1_apply(legacy, v)
    return r1, c1, pf


def section_d():
    print('\n[D] run_branch_block（冻结下层）')
    hc.set_mode('uniform')
    hc.set_smooth(0, 100.0, 0.05)          # 关平滑 ⇒ c1 可纯数组重算
    env = make_env()
    snap = abr.snapshot_env(env)
    after_snap = dict(snap)

    def make_state(obs, e, minute):
        # 与主循环同形：Normalize_H(obs) [+ phase]；这里用最简单的一档
        from aifflor.aifflor_myfunc import Normalize_H
        return Normalize_H(obs, minute, 0, None)

    a5 = np.array([0.05, 0.12, 0.20, 0.33, 0.47])
    res = abr.run_branch_block(env, _StubRBB(), a5, 0.0, snap, False,
                               make_state, minute_end=5, freeze_lower=True)

    ck(res['restored'], '函数报告已恢复')
    ck(abr.snapshot_diff(env, after_snap) == [], 'branch 后 env 逐位回到块首')
    r1_exp, c1_exp, pf_exp = _expected_branch_block(env, a5, snap)
    ck_close(res['r1'], r1_exp, 1e-9, 'Σr1 与独立重算一致')
    ck_close(res['c1'], c1_exp, 1e-9, 'Σc1 与独立重算一致')
    ck(np.allclose(res['P_F'], np.asarray(pf_exp), atol=1e-12),
       '逐步 P_F 与独立重算一致')
    ck(res['n_steps'] == 5 and len(res['P_F']) == 5, '块长 5')
    ck_close(res['soc_start'], res['soc_end'], 0.0, 'freeze_lower ⇒ SOC 不变')
    ck_close(res['pbrule_sum'], 0.0, 0.0, 'freeze_lower ⇒ pbrule 恒 0')
    ck_close(res['sum_disch'], 0.0, 0.0, 'freeze_lower ⇒ 无放电')
    ck(res['done'] is False, 'done 透传')
    ck(res['s2'].shape == (4,), 's2 维数 = 4（Normalize_H, phase 0）')

    # 不同动作 ⇒ 不同的块成本（否则 branch 没有"反事实"意义）
    res2 = abr.run_branch_block(env, _StubRBB(), np.full(5, 0.45), 0.0, snap, False,
                                make_state, minute_end=5, freeze_lower=True)
    ck(res2['c1'] != res['c1'], '不同动作 ⇒ 不同 Σc1')
    ck(abr.snapshot_diff(env, after_snap) == [], '第二次 branch 后 env 仍还原')

    # 摊还执行器模式：SOC 会变、pbrule != 0；c1 仍有限
    res3 = abr.run_branch_block(env, _StubRBB(), a5, hsrl.random_g(np.random.RandomState(3)),
                                snap, False, make_state, minute_end=5,
                                freeze_lower=False)
    ck(abr.snapshot_diff(env, after_snap) == [], 'freeze_lower=0 后 env 也还原')
    ck(res3['pbrule_sum'] != 0.0, 'freeze_lower=0 ⇒ pbrule 非 0')
    ck(np.isfinite(res3['c1']) and np.isfinite(res3['r1']), 'freeze_lower=0 ⇒ 无 NaN')

    # done=True 透传（episode 末的 branch 只影响它自己的 bootstrap）
    res4 = abr.run_branch_block(env, _StubRBB(), a5, 0.0, snap, True,
                                make_state, minute_end=1439, freeze_lower=True)
    ck(res4['done'] is True, 'done=True 透传')


class _StubRBB(object):
    """与 HSRL 下的 RBB 同形（`fluctuation_limit=inf` ⇒ 纯物理裁剪）。"""

    def __init__(self, max_E=0.5, rated=0.5, eta_c=0.98, eta_d=0.98):
        self.max_E, self.min_E, self.rated = max_E, 0.0, rated
        self.eta_c, self.eta_d = eta_c, eta_d
        self.time = 1.0 / 60.0

    def compute_pbrule(self, pb_original, ptur, pg_prev, E):
        lo, hi = hsrl.feasible_power(E)
        pb = float(pb_original)
        pbrule = float(np.clip(pb, lo, hi))
        return pbrule, {'pb_original': pb, 'pbrule': pbrule,
                        'pbphysicsmin': lo, 'pbphysicsmax': hi,
                        'clip_amount': abs(pb - pbrule), 'unfeasible_gap': 0.0,
                        'is_infeasible': False, 'inevitable_violation': 0.0,
                        'soc': E}


# =============================================================================
# E. BranchController 的触发语义
# =============================================================================
def _ns(**kw):
    base = dict(branch_actions='prior,low', branch_every=10, branch_pilot_blocks=0,
                branch_freeze_lower=1, branch_goal_pm1=0, branch_critic_frac=0.45,
                branch_sup_beta=0.0, branch_sup_blocks=1000, branch_csv=1,
                branch_low_lo=0.0, branch_low_hi=0.1, branch_high_lo=0.4,
                branch_high_hi=0.5, branch_prior_jitter=0.05, branch_seed=5,
                na_box=5, seed=1)
    base.update(kw)
    return types.SimpleNamespace(**base)


def section_e():
    print('\n[E] BranchController.trigger')
    c = abr.BranchController(_ns(branch_every=10, branch_pilot_blocks=0))
    ck(c.enabled, 'every=10 ⇒ 启用')
    hits = [b for b in range(40) if c.trigger(b * 5)]
    ck(hits == [0, 10, 20, 30], 'every=10 ⇒ 块号 %r' % (hits,))
    c = abr.BranchController(_ns(branch_every=4, branch_pilot_blocks=8))
    hits = [b for b in range(40) if c.trigger(b * 5)]
    ck(hits[:8] == list(range(8)), 'pilot 前 8 块每块都 branch')
    ck(all(b % 4 == 0 for b in hits[8:]), 'pilot 之后回到 every=4')
    c = abr.BranchController(_ns(branch_every=0, branch_pilot_blocks=0))
    ck(not c.enabled and not c.trigger(0), 'every=0 且 pilot=0 ⇒ 关闭')
    c = abr.BranchController(_ns(branch_every=0, branch_pilot_blocks=3))
    ck(c.enabled and c.trigger(0) and c.trigger(10) and not c.trigger(15),
       'pilot=3/every=0 ⇒ 只有前 3 块')
    # freeze_lower 时 g 恒 0（g 没有物理作用，留着只污染动作维）
    c = abr.BranchController(_ns(branch_every=1, branch_freeze_lower=1, branch_goal_pm1=1))
    ck([c.goal(2.0, 2.0) for _ in range(20)] == [0.0] * 20, 'freeze_lower ⇒ g 恒 0')
    c = abr.BranchController(_ns(branch_every=1, branch_freeze_lower=0, branch_goal_pm1=1))
    gs = [c.goal(1.5, 2.5) for _ in range(200)]
    ck(set(gs) <= {2.5, -1.5}, 'goal_pm1 ⇒ ±满可行量（+D_max / −C_max）')
    ck(2.5 in gs and -1.5 in gs, 'goal_pm1 两侧都出现')
    c = abr.BranchController(_ns(branch_every=1, branch_freeze_lower=0, branch_goal_pm1=0))
    ck(c.goal(1.5, 2.5) == 0.0, 'goal_pm1=0 ⇒ g=0')
    # β 衰减 1.0 → 0.2
    c = abr.BranchController(_ns(branch_sup_beta=1.0, branch_sup_blocks=100))
    ck_close(c.sup_beta(0), 1.0, 1e-12, 'β(0) = β0')
    ck_close(c.sup_beta(100), 0.2, 1e-12, 'β(满) = 0.2·β0')
    ck_close(c.sup_beta(1000), 0.2, 1e-12, 'β 超过跨度后保持地板')
    # 独立随机流：BranchController 不消耗 np.random 全局流
    np.random.seed(123)
    st0 = np.random.get_state()[1].copy()
    c = abr.BranchController(_ns(branch_every=1))
    for _ in range(20):
        c.action(abr.MODE_UNIFORM)
    ck(np.array_equal(st0, np.random.get_state()[1]), 'branch 不吃全局随机流')


# =============================================================================
# F. PER 池采样
# =============================================================================
def section_f():
    print('\n[F] PER 分池采样')
    mem = PrioritizedReplayBuffer(size=64, alpha=0.6)
    for i in range(40):
        mem.append(np.zeros(4), np.zeros(6), 0.0, 0.0, np.zeros(4), 0.0)
        mem.set_meta(i, src=(abr.SRC_BRANCH if i % 5 else abr.SRC_ACTOR),
                     mode=(abr.MODE_LOW if i % 2 else abr.MODE_HIGH))
    ck(mem.has_meta, '元数据已建立')
    ck(mem.n_actor + mem.n_branch == 40, 'actor+branch = 40')
    ck(mem.n_actor == 8 and mem.n_branch == 32, '计数正确 (8 actor / 32 branch)')

    out = mem.sample_pool(abr.SRC_ACTOR, 8)
    ck(len(out) == 8, 'actor 池采样条数正确')
    ck(all(mem._src[i] == abr.SRC_ACTOR for i in out), 'actor 池只含 actor')
    out = mem.sample_pool(abr.SRC_BRANCH, 12, by_mode=True)
    ck(len(out) == 12, 'branch 池采样条数正确')
    ck(all(mem._src[i] == abr.SRC_BRANCH for i in out), 'branch 池只含 branch')
    modes = [int(mem._meta['mode'][i]) for i in out]
    ck(modes.count(abr.MODE_LOW) == 6 and modes.count(abr.MODE_HIGH) == 6,
       'by_mode ⇒ 两个 mode 各 6 条 (得到 %r)' % (modes,))

    # 池占比极小时也要凑满 batch（均匀兜底，不静默缩短）
    mem2 = PrioritizedReplayBuffer(size=256, alpha=0.6)
    for i in range(200):
        mem2.append(np.zeros(4), np.zeros(6), 0.0, 0.0, np.zeros(4), 0.0)
        mem2.set_meta(i, src=(abr.SRC_ACTOR if i == 0 else abr.SRC_BRANCH))
    ck(len(mem2.sample_pool(abr.SRC_ACTOR, 16)) == 16,
       '池只有 1 条时也能凑满 16（均匀兜底）')

    # ---- 环形覆盖：被复用的槽必须重算计数 ----
    mem3 = PrioritizedReplayBuffer(size=8, alpha=0.6)
    for i in range(8):
        mem3.append(np.zeros(4), np.zeros(6), 0.0, 0.0, np.zeros(4), 0.0)
        mem3.set_meta(i, src=abr.SRC_BRANCH)
    ck(mem3.n_branch == 8 and mem3.n_actor == 0, '满环全 branch')
    for i in range(4):                       # 覆盖前 4 个 ⇒ 应当变成 actor
        idx = mem3._next_idx
        mem3.append(np.zeros(4), np.zeros(6), 0.0, 0.0, np.zeros(4), 0.0)
        mem3.set_meta(idx, src=abr.SRC_ACTOR)
    ck(mem3.n_branch == 4 and mem3.n_actor == 4,
       '环形覆盖后计数正确 (得到 actor=%d branch=%d)' % (mem3.n_actor, mem3.n_branch))

    # ---- 无元数据 ⇒ 单池（历史路径）----
    mem4 = PrioritizedReplayBuffer(size=32, alpha=0.6)
    for i in range(20):
        mem4.append(np.zeros(4), np.zeros(6), 0.0, 0.0, np.zeros(4), 0.0)
    ck(not mem4.has_meta, '未打标签 ⇒ has_meta=False')
    ck(mem4.n_actor == 0 and mem4.n_branch == 0, '未打标签 ⇒ 计数为 0')
    ck(len(mem4.pool_idxes(abr.SRC_ACTOR)) == 20, '未打标签 ⇒ pool_idxes 退回全体')
    out = mem4.sample(16, 0.4, 3e-5)
    ck(len(out) == 8 and len(out[-1]) == 16, '历史 sample 仍返回 8 元组')

    # ---- 权重公式：无元数据时 sample 与 sample_by_idxes 完全一致 ----
    idxes = mem4._sample_proportional(8)
    w = mem4.sample_by_idxes(idxes, 0.4)[-2]
    p_min = mem4._it_min.min() / mem4._it_sum.sum()
    max_w = (p_min * len(mem4._storage)) ** (-0.4)
    ref = [( (mem4._it_sum[i] / mem4._it_sum.sum()) * len(mem4._storage) ) ** (-0.4) / max_w
           for i in idxes]
    ck(np.allclose(w, np.array(ref)), 'importance weight 公式与历史一致')


# =============================================================================
# G. recent_states 的 source 过滤
# =============================================================================
def section_g():
    print('\n[G] recent_states 的 actor 过滤')
    class _A(object):
        pass
    a = _A()
    a.memory = PrioritizedReplayBuffer(size=8, alpha=0.6)
    for i in range(8):
        s = np.full(4, float(i))
        a.memory.append(s, np.zeros(6), 0.0, float(i), s, 0.0)
        a.memory.set_meta(i, src=(abr.SRC_BRANCH if i % 2 else abr.SRC_ACTOR))
    s, act, c = hsrl.recent_states(a, n=3)
    ck(s.shape[0] == 3, '过滤后条数 = 3')
    ck(np.allclose(s[:, 0], [2.0, 4.0, 6.0]), '只取最近 3 条 **actor**（得到 %r）'
       % (s[:, 0].tolist(),))
    # 无元数据 ⇒ 历史口径（最近 n 条，不过滤）
    b = _A()
    b.memory = PrioritizedReplayBuffer(size=8, alpha=0.6)
    for i in range(6):
        s_ = np.full(4, float(i))
        b.memory.append(s_, np.zeros(6), 0.0, float(i), s_, 0.0)
    s2, _, _ = hsrl.recent_states(b, n=3)
    ck(np.allclose(s2[:, 0], [3.0, 4.0, 5.0]), '无元数据 ⇒ 最近 3 条（历史口径）')


# =============================================================================
# H. n_boot：branch = γ，actor = γ^n
# =============================================================================
def _td3_args(**kw):
    base = dict(cuda=False, seed=-1, hidden1=8, hidden2=16, init_w=0.003,
                prate=1e-4, rate=1e-3, rmsize=512, PER_alpha=0.6, PER_beta=0.4,
                PER_epsilon=0.01, beta_increment_per_sampling=0.0,
                ou_theta=0.15, ou_sigma=0.2, ou_mu=0.0, bsize=32, tau=0.005,
                discount=0.95, epsilon=4000, cost_single_limit=1000.0,
                nb_turbine=5, na_box=5, critic_arch='v1', critic_act='relu',
                critic_out_init='3e3', critic_feat_power=0, critic_feat_power_k=1.357e-3,
                cost_clip=1, farm_level=0, scalar_wt=0, aux_model=0,
                critic_action_first=0, lam=3.0, lr_lam=0.1)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _make_agent(**kw):
    import torch
    torch.manual_seed(0)
    a = addpg.SP_TD3(nb_states=4, nb_actions=6, args=_td3_args(),
                     critic_nb_states=8, nstep=5, target_agg='min',
                     policy_delay=1, cost_critic_mode='max',
                     record_update_stats=True,
                     branch_mix=kw.pop('branch_mix', 0.45),
                     branch_sup_beta=kw.pop('branch_sup_beta', 0.0), **kw)
    return a


def _fill(agent, n_actor=64, n_branch=64, seed=0):
    rng = np.random.RandomState(seed)
    for i in range(n_actor):
        s = rng.normal(size=8).astype(np.float32)
        s2 = rng.normal(size=8).astype(np.float32)
        agent._push_transition(s, agent.random_action(), float(rng.normal()),
                               abs(float(rng.normal())), s2, 0.0,
                               boot=float(agent.discount) ** agent.nstep)
    for i in range(n_branch):
        s = rng.normal(size=8).astype(np.float32)
        s2 = rng.normal(size=8).astype(np.float32)
        a = np.concatenate([rng.uniform(0.0, 0.5, 5), [0.0]]).astype(np.float32)
        agent.push_branch_transition(s, a, float(rng.normal()),
                                     abs(float(rng.normal())), s2, 0.0,
                                     mode=int(i % 4) + 1, block_id=i // 4,
                                     lower_ver=i // 16)


def section_h():
    print('\n[H] n_boot 与来源标签')
    agent = _make_agent()
    _fill(agent, n_actor=8, n_branch=8)
    mem = agent.memory
    ck(mem.n_actor == 8 and mem.n_branch == 8, '标签计数 8/8')
    ck(agent.n_branch_transitions == 8, 'n_branch_transitions 计数')
    gam = float(agent.discount)
    b_idx = [i for i in range(16) if mem._src[i] == abr.SRC_BRANCH]
    a_idx = [i for i in range(16) if mem._src[i] == abr.SRC_ACTOR]
    ck(all(abs(float(agent._boot[i]) - gam) < 1e-7 for i in b_idx),
       'branch 的 n_boot = γ')
    ck(all(abs(float(agent._boot[i]) - gam ** agent.nstep) < 1e-7 for i in a_idx),
       'actor 的 n_boot = γ^n')
    ck(int(mem._meta['block_id'][b_idx[0]]) == 0, 'block_id 落盘')
    ck(int(mem._meta['mode'][b_idx[1]]) == 2, 'mode 落盘')
    # 覆盖写：环形缓冲复用到的 slot 必须被改成 actor
    small = _make_agent()
    small.memory._maxsize = 4          # 人为缩小（不回填 storage，仅探测计数逻辑）
    for i in range(4):
        small.push_branch_transition(np.zeros(8, np.float32), np.zeros(6, np.float32),
                                     0.0, 0.0, np.zeros(8, np.float32), 0.0, mode=1)
    ck(small.memory.n_branch == 4, '4 条 branch')
    small._push_transition(np.zeros(8, np.float32), np.zeros(6, np.float32), 0.0, 0.0,
                           np.zeros(8, np.float32), 0.0, boot=gam ** 5)
    ck(small.memory.n_branch == 3 and small.memory.n_actor == 1,
       '覆盖第一个槽后 branch=3 / actor=1')


# =============================================================================
# I. update_policy：branch 路径
# =============================================================================
def section_i():
    print('\n[I] update_policy（branch 分支）')
    import torch
    agent = _make_agent(branch_mix=0.45)
    _fill(agent, n_actor=200, n_branch=100)

    seen = []
    _orig = agent.actor.forward

    def _rec(x):
        seen.append(int(x.shape[0]))
        return _orig(x)

    agent.actor.forward = _rec

    n_ok = 0
    for _ in range(6):
        agent.update_policy()
        if np.isfinite(agent.lam) and np.isfinite(float(agent.td_mean_p)):
            n_ok += 1
    ck(n_ok == 6, '6 次 update 全部有限（无 NaN）')
    ck(agent.n_updates_with_branch == 6, '6 次都走了 branch 池采样')
    ck(0.30 <= agent.last_batch_branch_frac <= 0.55,
       'branch 占比在 [0.30, 0.55]（得到 %.3f）' % agent.last_batch_branch_frac)
    n_actor_expect = agent.batch_size - int(round(0.45 * agent.batch_size))
    ck(len(seen) >= 6, 'actor 至少被前向 6 次')
    ck(all(k == n_actor_expect for k in seen),
       'actor loss **只用 actor 行**（每批 %d 行；实际 %r）' % (n_actor_expect, seen))

    # 关闭 branch ⇒ 走历史整池采样路径，且不崩
    agent2 = _make_agent(branch_mix=0.0)
    _fill(agent2, n_actor=200, n_branch=100)
    ck(agent2._tag_replay is False, 'branch_mix=0 ⇒ 不打标签')
    for _ in range(3):
        agent2.update_policy()
    ck(np.isfinite(agent2.lam), 'branch_mix=0 时 update 正常')
    ck(agent2.n_updates_with_branch == 0, 'branch_mix=0 ⇒ 不走分池')

    # 可选辅助监督 β：开启后仍应有限，且不改变 actor 行数约束
    agent3 = _make_agent(branch_mix=0.45, branch_sup_beta=0.5)
    _fill(agent3, n_actor=200, n_branch=100)
    for _ in range(3):
        agent3.update_policy()
    ck(np.isfinite(agent3.lam) and np.isfinite(float(agent3.td_mean_c)),
       'β>0 时 update 正常（无 NaN）')

    # update_stats 的 q_c_mean 必须是 actor-only（λ1 的分子）
    ck(len(agent.update_stats) == 6, 'update_stats 每条 update 一条')
    k0 = set(agent.update_stats[0].keys())
    ck('q_c_mean' in k0 and 'q_c2_mean' in k0, 'update_stats 键未变（CSV 列不变）')


# =============================================================================
# main
# =============================================================================
def main():
    print('=' * 78)
    print('aifflor branch 契约测试（纯静态，不跑仿真）')
    print('=' * 78)
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()
    section_g()
    section_h()
    section_i()
    print('\n' + '=' * 78)
    if FAIL:
        print('**失败 %d / %d**' % (len(FAIL), N_CK[0]))
        for m in FAIL:
            print('  - %s' % m)
        return 1
    print('全部通过：%d 条断言 ✅' % N_CK[0])
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
