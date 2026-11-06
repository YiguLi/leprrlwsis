# -*- coding: utf-8 -*-
"""`td3per` 的 **branch + v6** 契约测试（**纯静态 / 纯算术，不跑仿真、不需要 MATLAB**）。

被钉住的口径
------------
1. **v6 critic**（`model.Critic(arch='v6')`）在 td3per 里的接线：
   * 参数量 / 前向形状 / `infer_arch` / 输出层名字（`out`，供 `hsrl_agent_init`）；
   * `SP_TD3` 的回落值 = **v1**（直接构造的脚本/探针保持历史网络）；
   * `load_weights` 能按 checkpoint 的 `state_dict` 自动重建 arch（v1↔v6 双向往返）；
   * `--critic_arch v1` 时四个 critic 与改造前**逐位相同**（同 seed 参数相等）。
2. **branch**（`td3per/td3per_branch.py` + `td3per_ddpg.py` 的分池采样）：
   * 常量与 `td3per_ddpg` / `PER_buffer` 的元数据字段一致；
   * 7 种动作生成器（形状/边界/覆盖/可复现）+ 信号维反解；
   * 块首快照-恢复往返（6 个标量逐位）；
   * `run_branch_block` 的 `Σr1/Σc1` 与**独立重算**逐位一致（真 `env.WindBattery` +
     假物理 + 真 `RuleBasedBESS`），`freeze/rule/none` 三种下层都测，`nolow` 的 gap≡0；
   * 主轨迹逐位还原（快照 6 字段 + P_G_prev/count_* 不变）；
   * `BranchController` 的 pilot/every 语义、β 线性衰减 1→0.2、**branch 不消耗全局 RNG**；
   * PER 分池采样（计数 / 池正确 / by_mode 均分 / 环形覆盖后计数正确）；
   * `push_branch_transition`：`n_boot=γ`（actor 是 `γ^n`）、不进 `_pending`、标签落盘；
   * `update_policy`：`branch_mix=0` 走历史 `memory.sample(...)`；`branch_mix=0.45` 时
     **actor loss 每批只吃 actor 行**（forward 钩子断言行数）、批内 branch 占比落
     [0.30,0.55]、`update_stats['q_c_mean']` 只算 actor 行。

跑法::

    python agentutil/test_td3per_branch_v6_contract.py

任一断言失败 → 退出码 1。
"""
import argparse
import copy
import os
import sys

import numpy as np

# ---- 路径自举（可被任意 CWD 调用）----------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import torch                                                # noqa: E402

import model as M                                           # noqa: E402
import PER_buffer as PB                                     # noqa: E402
import hsrl_agent_init as hinit                             # noqa: E402
from td3per import td3per_branch as tbr                     # noqa: E402
from td3per.td3per_ddpg import (SP_TD3, SRC_ACTOR, SRC_BRANCH,  # noqa: E402
                                MODE_ACTOR)
from td3per.td3per_myfunc import (                          # noqa: E402
    Cal_Rew_RB, gap_penalty, soc_target_from_signal,
    SOC_TARGET_CENTER, SOC_TARGET_SPAN, SOC_TOL,
    violation_derate, Normalize_H,
)
from rbb.rbb_rule_bess import RuleBasedBESS                 # noqa: E402
import hsrl_soctarget as hst                                # noqa: E402

N_OK = 0


def ok(cond, msg):
    global N_OK
    if not cond:
        raise AssertionError('FAIL: ' + msg)
    N_OK += 1


def close(a, b, tol=0.0):
    """逐位（tol=0）或带容差比较。"""
    return abs(float(a) - float(b)) <= float(tol)


# =============================================================================
# 构造 SP_TD3 需要的 args（= td3per_main 的默认值切片）
# =============================================================================
def make_args(**kw):
    ns = argparse.Namespace()
    ns.cuda = False
    ns.seed = -1
    ns.hidden1 = 16
    ns.hidden2 = 100
    ns.init_w = 0.003
    ns.prate = 1e-4
    ns.rate = 0.01
    ns.rmsize = 2048
    ns.PER_alpha = 0.6
    ns.PER_beta = 0.4
    ns.PER_epsilon = 0.01
    ns.beta_increment_per_sampling = 3e-5
    ns.ou_theta = 0.15
    ns.ou_mu = 0.0
    ns.ou_sigma = 0.2
    ns.bsize = 32
    ns.tau = 0.001
    ns.discount = 0.95
    ns.epsilon = 4000
    ns.lam = 3.0
    ns.lr_lam = 0.1
    ns.cost_single_limit = 1000.0
    ns.nb_turbine = 5
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def make_agent(nb_states=4, nb_actions=6, **kw):
    a = make_args(**{k: v for k, v in kw.items() if k not in (
        'critic_arch', 'branch_mix', 'branch_sup_beta', 'nstep', 'target_agg',
        'record_update_stats')})
    return SP_TD3(nb_states=nb_states, nb_actions=nb_actions, args=a,
                  nstep=kw.get('nstep', 1), target_agg=kw.get('target_agg', 'min'),
                  critic_arch=kw.get('critic_arch', None),
                  branch_mix=kw.get('branch_mix', 0.0),
                  branch_sup_beta=kw.get('branch_sup_beta', 0.0),
                  record_update_stats=kw.get('record_update_stats', True))


# =============================================================================
# 假物理（不启 MATLAB；只做 u³·ΣCp 的确定性映射）
# =============================================================================
class FakePhysics(object):
    k = 1.357e-3

    def PowerOutput(self, u0, act_wind):
        a = np.asarray(act_wind, dtype=float).reshape(-1)[:5]
        g = 4.0 * a * (1.0 - a) ** 2
        return float(self.k * max(float(u0), 0.0) ** 3 * float(g.sum()) / 5.0)


class CountingPhysics(FakePhysics):
    """记录调用次数：用来证明 `run_branch_block` 没有多推/少推进。"""

    def __init__(self):
        self.n = 0

    def PowerOutput(self, u0, act_wind):
        self.n += 1
        return FakePhysics.PowerOutput(self, u0, act_wind)


def make_env(pr=580.0, u=10.0, E=0.25, P_G_prev=0.2, count_wind=10,
             count_price=600, physics=None):
    """真 `env.WindBattery` + 假物理；6 个标量手工钉住（不调用 reset）。"""
    from env import WindBattery
    phy = physics if physics is not None else FakePhysics()
    env = WindBattery(0.5, 5, False, None, physics=phy)
    env.pr = float(pr)
    env.u = float(u)
    env.E = float(E)
    env.P_G_prev = float(P_G_prev)
    env.count_wind = int(count_wind)
    env.count_price = int(count_price)
    # wind_day 必须存在（CalWindSpeed 读它）；给一条常值序列即可
    import pandas as pd
    env.wind_day = pd.Series(np.full(400, float(u)))
    return env


# =============================================================================
# [A] 常量一致性
# =============================================================================
def sec_a_constants():
    print('\n[A] 常量 / 元数据字段一致性')
    ok(tbr.BLOCK == 5, 'BLOCK = 5（与 td3per 的 H 决策周期一致）')
    ok(tbr.SRC_ACTOR == SRC_ACTOR == 0 and tbr.SRC_BRANCH == SRC_BRANCH == 1,
       'SRC_ACTOR/SRC_BRANCH 与 td3per_ddpg 逐位一致')
    ok(tbr.MODE_ACTOR == MODE_ACTOR == 0, 'MODE_ACTOR 与 td3per_ddpg 一致')
    ok(tbr.SRC_NAMES == {0: 'actor', 1: 'branch'}, 'SRC_NAMES 标签')
    ok(tbr.MODE_ORDER[0] == 'actor' and len(tbr.MODE_ORDER) == 8, 'MODE_ORDER 8 项且 actor 在 0')
    ok(tbr.MODE_NAME[tbr.MODE_RAMP] == 'ramp', 'MODE_NAME 映射')
    ok(tuple(tbr.BRANCH_MODES_DEFAULT) == (tbr.MODE_PRIOR, tbr.MODE_LOW,
                                          tbr.MODE_HIGH, tbr.MODE_UNIFORM),
       '默认动作集 = prior,low,high,uniform')
    ok(tbr.parse_modes(tbr.DEFAULT_BRANCH_ACTIONS) == list(tbr.BRANCH_MODES_DEFAULT),
       'parse_modes(默认字符串) == 默认动作集')
    # replay 元数据字段（PER_buffer 的单一真源）
    for f in ('src', 'mode', 'block_id', 'lower_ver'):
        ok(f in PB.PrioritizedReplayBuffer.META_FIELDS, 'PER 元数据字段 %r 存在' % f)
    ok(PB.PrioritizedReplayBuffer.META_FIELDS['src'][1] == np.int8, 'src dtype = int8')
    # 单步时长 / 学习带
    ok(close(tbr.TIME_STEP, 1.0 / 60, 1e-15), 'TIME_STEP = 1/60')
    ok(tbr.A_LO == 0.0 and tbr.A_HI == 0.5, '动作带 [0,0.5]')
    ok(tbr.SIG_LO == -0.5 and tbr.SIG_HI == 0.5, '信号带 [-0.5,0.5]')
    ok(np.allclose(tbr.PRIOR5, [0.2, 0.2, 0.2, 0.2, 1.0 / 3.0]), 'PRIOR5 先验形状')
    ok(np.allclose(tbr.PRIOR5, np.asarray(hst.SOC_TARGET_CENTER * 0 + tbr.PRIOR5)),
       'PRIOR5 有限值')


# =============================================================================
# [B] v6 结构
# =============================================================================
def sec_b_v6_structure():
    print('\n[B] v6 critic 结构（参数量/前向/infer_arch/输出层）')
    n_v1 = sum(p.numel() for p in M.Critic(4, 6, hidden1=16, hidden2=100, arch='v1').parameters())
    n_v6 = sum(p.numel() for p in M.Critic(4, 6, hidden1=16, hidden2=100, arch='v6').parameters())
    ok(n_v1 == 14441, 'v1 (ns=4,na=6,h=16/100) 参数量 14,441（历史口径）')
    ok(n_v6 == 150081, 'v6 (ns=4,na=6) 参数量 150,081')
    ok(n_v6 > n_v1, 'v6 比 v1 宽')
    # 宽度与 hidden1/hidden2 无关
    n_v6b = sum(p.numel() for p in M.Critic(4, 6, hidden1=400, hidden2=300,
                                            arch='v6').parameters())
    ok(n_v6b == n_v6, 'v6 的宽度写死在 V6_DIMS ⇒ --hidden1/--hidden2 不生效')
    # 前向形状（含 H 的相位维与 L 的 5→1）
    for ns, na in ((4, 6), (6, 6), (9, 6), (5, 1)):
        net = M.Critic(ns, na, arch='v6')
        x = torch.zeros(7, ns)
        a = torch.zeros(7, na)
        y = net([x, a])
        ok(tuple(y.shape) == (7, 1), 'v6 forward ns=%d na=%d → (7,1)' % (ns, na))
        ok(net.in_actions == na, 'v6 的 in_actions = nb_actions（无 feat_power）')
    # infer_arch 双向往返
    sd_v6 = M.Critic(4, 6, arch='v6').state_dict()
    sd_v1 = M.Critic(4, 6, arch='v1').state_dict()
    ok(M.infer_arch(sd_v6, 4, 6) == 'v6', 'infer_arch(v6 sd) = v6')
    ok(M.infer_arch(sd_v1, 4, 6) == 'v1', 'infer_arch(v1 sd) = v1')
    ok(M.infer_arch(M.Critic(5, 1, arch='v6').state_dict(), 5, 1) == 'v6',
       'infer_arch(L 的 v6 sd) = v6')
    # V6_DIMS 与文档一致
    ok(M.V6_DIMS['s'] == (32, 32) and M.V6_DIMS['a'] == (64, 64)
       and M.V6_DIMS['merge'] == (128, 256, 256, 128), 'V6_DIMS = 文档口径')
    # 输出层名字（hsrl_agent_init._out_head 认 fc3/out）
    v6 = M.Critic(4, 6, arch='v6')
    ok(hinit._out_head(v6) is v6.out, '_out_head(v6) = out（bias 初始化不会静默 no-op）')
    ok(hinit._out_head(M.Critic(4, 6, arch='v1')) is not None, '_out_head(v1) 非空')


# =============================================================================
# [C] SP_TD3 的接线 / 回落 / 自动重建
# =============================================================================
def sec_c_wiring():
    print('\n[C] SP_TD3 接线：回落 v1 / 显式 v6 / load_weights 自动重建')
    a1 = make_agent()                      # 不给 critic_arch ⇒ 回落 v1
    ok(a1.critic_arch == 'v1', 'SP_TD3 未给 critic_arch 时回落 **v1**（探针/契约测试不被改义）')
    a6 = make_agent(critic_arch='v6')
    ok(a6.critic_arch == 'v6', '显式 critic_arch=v6 生效')
    for name in ('critic_p', 'critic_p2', 'critic_c', 'critic_c2',
                 'critic_target_p', 'critic_target_p2',
                 'critic_target_c', 'critic_target_c2'):
        ok(getattr(a6, name).arch == 'v6', 'v6: %s.arch = v6' % name)
    # v1 与改造前逐位相同：同 seed 两个 agent 的 critic 参数相等
    torch.manual_seed(1234)
    t1 = make_agent()
    torch.manual_seed(1234)
    t2 = make_agent()
    for (k1, v1), (k2, v2) in zip(t1.critic_p.state_dict().items(),
                                   t2.critic_p.state_dict().items()):
        ok(k1 == k2 and torch.equal(v1, v2), 'v1 critic 同 seed 逐位相同 (%s)' % k1)
    # 参数尺寸自证：v1 的 fc3 存在、v6 的 out 存在
    ok(hasattr(t1.critic_p, 'fc3') and not hasattr(t1.critic_p, 'out'),
       'v1 critic 用 fc3 输出（历史命名）')
    ok(hasattr(a6.critic_p, 'out') and hasattr(a6.critic_p, 'm4'),
       'v6 critic 用 out/m4（双流 + 4 层 merge）')

    # ---- load_weights 自动重建：v6 agent 读 v1 checkpoint（再反过来）----
    #   不落盘：把 torch.load 换成按文件名查表（沙箱下运行期新建目录/写盘会被拒，
    #   而这条路径要测的正是"推断 arch → 重建 → 装载"，与磁盘无关）。
    def _fake_loads(src_agent, nb_actions=6):
        names = {'actor_{}.pkl'.format(nb_actions): src_agent.actor.state_dict()}
        for base in ('critic_p', 'critic_p2', 'critic_c', 'critic_c2'):
            names['{}_{}.pkl'.format(base, nb_actions)] = \
                getattr(src_agent, base).state_dict()
        return names

    def _load_weights_via(src_agent, target_agent):
        table = _fake_loads(src_agent, target_agent.nb_actions)
        orig = torch.load
        torch.load = lambda path, *a, **k: table[os.path.basename(str(path))]
        try:
            target_agent.load_weights('/virtual/ckpt')
        finally:
            torch.load = orig

    probe = make_agent(critic_arch='v6')
    _load_weights_via(t1, probe)
    ok(probe.critic_arch == 'v1', 'load_weights 按 checkpoint 推断出 v1 并重建')
    for k, v in t1.critic_p.state_dict().items():
        ok(torch.equal(v, probe.critic_p.state_dict()[k]),
           '重建后 v1 权重逐位还原 (%s)' % k)
    # 反向：v1 agent 读 v6 checkpoint
    probe2 = make_agent()                  # v1
    _load_weights_via(a6, probe2)
    ok(probe2.critic_arch == 'v6', 'load_weights 按 checkpoint 推断出 v6 并重建')
    for k, v in a6.critic_p.state_dict().items():
        ok(torch.equal(v, probe2.critic_p.state_dict()[k]),
           '重建后 v6 权重逐位还原 (%s)' % k)
    # 同构 checkpoint ⇒ 不重建（v1 agent 读 v1 checkpoint，arch 不变）
    probe3 = make_agent()
    _load_weights_via(t1, probe3)
    ok(probe3.critic_arch == 'v1', '同构 checkpoint ⇒ arch 不变（不重建）')
    # 无法推断时返回 None（不 panic）
    ok(make_agent()._resolve_arch({'nope': torch.zeros(1)}) is None,
       '_resolve_arch 无法推断时返回 None')


# =============================================================================
# [D] 动作生成 + 信号维
# =============================================================================
def sec_d_actions():
    print('\n[D] 7 种测试动作生成器 + 信号维反解')
    rng = np.random.RandomState(7)
    for mode in (tbr.MODE_PRIOR, tbr.MODE_LOW, tbr.MODE_HIGH, tbr.MODE_UNIFORM,
                 tbr.MODE_RAMP, tbr.MODE_ALT, tbr.MODE_SINGLE):
        for _ in range(50):
            a = tbr.branch_action(mode, rng, nt=5)
            ok(a.shape == (5,), '%s: 形状 (5,)' % tbr.MODE_NAME[mode])
            ok(np.all(a >= 0.0) and np.all(a <= 0.5), '%s: 落在 [0,0.5]' % tbr.MODE_NAME[mode])
    # 覆盖率（计划 §7：边界各 ≥5%）
    rng = np.random.RandomState(11)
    lo = np.concatenate([tbr.branch_action(tbr.MODE_LOW, rng) for _ in range(400)])
    hi = np.concatenate([tbr.branch_action(tbr.MODE_HIGH, rng) for _ in range(400)])
    ok(np.mean(lo < 0.05) >= 0.40, 'low 带里 a<0.05 占 ~50%%（实测 %.2f）' % np.mean(lo < 0.05))
    ok(np.mean(hi > 0.45) >= 0.40, 'high 带里 a>0.45 占 ~50%%（实测 %.2f）' % np.mean(hi > 0.45))
    # 均值序 prior < low < high
    rng = np.random.RandomState(13)
    m_prior = np.mean([tbr.branch_action(tbr.MODE_PRIOR, rng).mean() for _ in range(300)])
    m_low = np.mean([tbr.branch_action(tbr.MODE_LOW, rng).mean() for _ in range(300)])
    m_high = np.mean([tbr.branch_action(tbr.MODE_HIGH, rng).mean() for _ in range(300)])
    ok(m_low < m_prior < m_high, 'mean(a): low %.3f < prior %.3f < high %.3f'
       % (m_low, m_prior, m_high))
    # single：只改一台
    rng = np.random.RandomState(17)
    for _ in range(200):
        a = tbr.branch_action(tbr.MODE_SINGLE, rng)
        ok(np.sum(np.abs(a - tbr.PRIOR5) > 1e-9) == 1, 'single 只动一台')
    # ramp 单调
    rng = np.random.RandomState(19)
    for _ in range(200):
        a = tbr.branch_action(tbr.MODE_RAMP, rng)
        ok(np.all(np.diff(a) > 0), 'ramp 逐台单调上升')
    # 同种子可复现（独立随机流）
    r1 = np.random.RandomState(23)
    r2 = np.random.RandomState(23)
    seq1 = [tbr.branch_action(m, r1) for m in tbr.BRANCH_MODES_DEFAULT]
    seq2 = [tbr.branch_action(m, r2) for m in tbr.BRANCH_MODES_DEFAULT]
    ok(all(np.array_equal(x, y) for x, y in zip(seq1, seq2)), '同种子逐位可复现')
    # MODE_ACTOR 不是测试动作
    try:
        tbr.branch_action(tbr.MODE_ACTOR, rng)
        ok(False, 'branch_action(MODE_ACTOR) 必须抛异常')
    except ValueError:
        ok(True, 'branch_action(MODE_ACTOR) 抛 ValueError')
    # parse_modes 去重/保序/跳过 actor
    ok(tbr.parse_modes('low,prior,low,actor') == [tbr.MODE_LOW, tbr.MODE_PRIOR],
       'parse_modes 去重保序且跳过 actor')
    try:
        tbr.parse_modes('nope')
        ok(False, '未知 mode 必须报错')
    except ValueError:
        ok(True, 'parse_modes 对未知 mode 报错')
    # 信号维反解：目标 = 当前 SOC ⇒ gap ≡ 0
    for soc in (0.05, 0.25, 0.4, 0.46):
        a6 = tbr.signal_for_soc(soc)
        tgt = soc_target_from_signal(a6)
        ok(close(gap_penalty(tgt, soc, SOC_TOL), 0.0, 1e-12),
           'signal_for_soc(%.2f): 目标 %.4f 的 gap ≡ 0' % (soc, tgt))
    ok(tbr.signal_for_soc(10.0) == 0.5 and tbr.signal_for_soc(-10.0) == -0.5,
       'signal_for_soc 夹在 ±0.5')


# =============================================================================
# [E] 快照 / 恢复
# =============================================================================
def sec_e_snapshot():
    print('\n[E] 块首快照-恢复往返')
    env = make_env()
    snap = tbr.snapshot_env(env)
    ok(set(snap) == set(tbr.SNAP_FIELDS), '快照字段 = SNAP_FIELDS 6 个')
    ok(isinstance(snap['count_wind'], int) and isinstance(snap['count_price'], int),
       'count_* 存成 int')
    # 模拟 branch 推进 5 步
    a = np.full(5, 0.3)
    pw = env.PowerOutput(env.u, a)
    for _ in range(5):
        pw, _obs, _al, _pl, _d, _dl, _info = env.step(a, np.array([0.1]), pw)
    ok(tbr.snapshot_diff(env, snap) != [], '推进后与快照不同（否则测试无意义）')
    tbr.restore_env(env, snap)
    ok(tbr.snapshot_diff(env, snap) == [], 'restore 后 6 个标量逐位还原')
    try:
        tbr.snapshot_env(object())
        ok(False, '缺字段必须报错')
    except AttributeError:
        ok(True, 'snapshot_env 对缺字段报 AttributeError')


# =============================================================================
# [F] run_branch_block 与独立重算逐位一致
# =============================================================================
def _reference_block(env, rbb, a5, a6, snap, lower, nolow, soc_tol, make_state,
                     minute_end, block_len=5):
    """**独立重算**（不调用 run_branch_block）：同样用 6 个标量重建块首。"""
    obs = np.array([snap['pr'], snap['u'], snap['E'], snap['P_G_prev']], dtype=float)
    a6 = 0.0 if nolow else max(-0.5, min(0.5, float(a6)))
    tgt = soc_target_from_signal(a6)
    der = violation_derate(a5, 0.0)
    a_full = np.concatenate([np.asarray(a5, float), [a6]])
    r_sum = c_sum = 0.0
    pfs = []
    pw = env.PowerOutput(obs[1], a_full)
    for t in range(block_len):
        E_cur = obs[2]
        n_rem = block_len - t
        _, info0 = rbb.compute_pbrule(pb_original=0.0, ptur=pw,
                                      pg_prev=obs[-1], E=E_cur)
        if lower == 'rule':
            cmd = hst.rule_action(tgt, E_cur, info0['pbrulemin'], info0['pbrulemax'],
                                  n_rem, eta_c=rbb.eta_c, eta_d=rbb.eta_d,
                                  time=rbb.time, rated_power=rbb.rated_power)
        else:
            cmd = 0.0
        pbrule, info = rbb.compute_pbrule(pb_original=cmd, ptur=pw,
                                          pg_prev=obs[-1], E=E_cur)
        if lower == 'none':
            pbrule = 0.0
            info = dict(info, pbrule=0.0, pb_original=0.0, clip_amount=0.0)
        pw2, obs2, _al, _pl, _d, _dl, dinfo = env.step(
            a_full, np.array([pbrule]), pw)
        r1, c1, det = Cal_Rew_RB(obs, obs2, a_full, info, penalty_NN_H=0.0,
                                 time=1.0 / 60, ptur=pw, pb_cmd=cmd,
                                 turbine_derate=der)
        r_sum += r1
        c_sum += c1
        pfs.append(det['P_F'])
        obs = np.asarray(obs2, dtype=float)
        pw = pw2
    gap = 0.0
    if not nolow:
        gap = gap_penalty(tgt, obs[2], SOC_TOL if soc_tol is None else soc_tol)
        import hsrl_cost as hc
        if hc.mode() == 'uniform':
            r_sum -= gap
        else:
            c_sum += gap
    s2 = make_state(obs, env, minute_end)
    return dict(r1=r_sum, c1=c_sum, s2=np.asarray(s2, float), P_F=pfs,
                gap=gap, soc_end=float(obs[2]), tgt=tgt)


def sec_f_run_block():
    print('\n[F] run_branch_block：Σr1/Σc1 与独立重算逐位一致（freeze/rule/none + nolow）')
    rbb = RuleBasedBESS(max_E=0.5, min_E_factor=0.0, rated_power=0.5,
                        eta_c=0.98, eta_d=0.98, time=1.0 / 60,
                        fluctuation_limit=3.0)
    # 覆盖"窗口含 0 / 不含 0 / 不可行"三类块首状态
    cases = [
        dict(pr=300.0, u=10.0, E=0.25, P_G_prev=0.20),
        dict(pr=980.0, u=6.5, E=0.06, P_G_prev=0.90),   # 窗口在 0 以上（需充电）
        dict(pr=580.0, u=14.0, E=0.45, P_G_prev=0.10),  # 窗口在 0 以下（需放电）
        dict(pr=580.0, u=12.0, E=0.02, P_G_prev=2.00),  # 窗口不可行（gap>0）
    ]
    rng = np.random.RandomState(31)
    for ci, cs in enumerate(cases):
        for lower in ('freeze', 'rule', 'none'):
            for nolow, a6 in ((False, 0.0), (False, 0.24), (True, 0.0)):
                a5 = tbr.branch_action(tbr.MODE_UNIFORM, rng)
                env = make_env(**cs)
                snap = tbr.snapshot_env(env)
                minute = 123
                mk = lambda o, e, m: Normalize_H(o, m, 0, None)  # noqa: E731
                got = tbr.run_branch_block(
                    env, rbb, a5, a6, snap, False, mk, minute_end=minute,
                    lower=lower, nolow=nolow, soc_tol=SOC_TOL)
                # 主轨迹必须逐位还原
                ok(tbr.snapshot_diff(env, snap) == [],
                   '[%d/%s/nolow=%d] 主轨迹 6 字段逐位还原' % (ci, lower, nolow))
                exp = _reference_block(env, rbb, a5, a6, snap, lower, nolow,
                                       SOC_TOL, mk, minute)
                ok(close(got['r1'], exp['r1']), '[%d/%s/nolow=%d] Σr1 逐位一致'
                   % (ci, lower, nolow))
                ok(close(got['c1'], exp['c1']), '[%d/%s/nolow=%d] Σc1 逐位一致'
                   % (ci, lower, nolow))
                ok(np.array_equal(got['s2'], exp['s2']), '[%d/%s/nolow=%d] s2 逐位一致'
                   % (ci, lower, nolow))
                ok(len(got['P_F']) == tbr.BLOCK, '每块 5 个 P_F')
                ok(close(got['P_F_max'], max(exp['P_F'])), 'P_F_max')
                ok(close(got['gap_pen'], exp['gap']), 'gap 罚一致')
                ok(close(got['soc_end'], exp['soc_end']), 'soc_end 一致')
                if nolow:
                    ok(got['gap_pen'] == 0.0, '[nolow] gap ≡ 0')
                    ok(got['a6'] == 0.0, '[nolow] a6 ≡ 0')
                if lower == 'none':
                    ok(got['soc_end'] == snap['E'],
                       '[none] 严格冻结: SOC 一字不变（%r vs %r）'
                       % (got['soc_end'], snap['E']))
                    ok(close(got['pbrule_sum'], 0.0), '[none] Σpb ≡ 0')
    # 不同动作产生不同 Σc1（branch 的"反事实"确有内容）
    env = make_env()
    snap = tbr.snapshot_env(env)
    mk = lambda o, e, m: Normalize_H(o, m, 0, None)  # noqa: E731
    r_a = tbr.run_branch_block(env, rbb, np.full(5, 0.05), 0.0, snap, False, mk,
                               lower='none')
    r_b = tbr.run_branch_block(env, rbb, np.full(5, 0.48), 0.0, snap, False, mk,
                               lower='none')
    ok(not close(r_a['c1'], r_b['c1'], 1e-9),
       'a=0.05 与 a=0.48 的 Σc1 不同（%.4f vs %.4f）' % (r_a['c1'], r_b['c1']))
    ok(not close(r_a['r1'], r_b['r1'], 1e-9), 'a=0.05 与 a=0.48 的 Σr1 不同')
    # 5 步推进 = 5 次 PowerOutput（首步 1 次 + 每步 1 次）
    phy = CountingPhysics()
    env = make_env(physics=phy)
    snap = tbr.snapshot_env(env)
    tbr.run_branch_block(env, rbb, np.full(5, 0.3), 0.0, snap, False, mk,
                         lower='none')
    ok(phy.n == 1 + tbr.BLOCK, 'PowerOutput 调用 %d 次 = 1 + BLOCK（不多推）' % phy.n)


# =============================================================================
# [G] 控制器
# =============================================================================
def sec_g_controller():
    print('\n[G] BranchController：trigger / signal / β / 覆盖 / 不消耗全局 RNG')
    a = make_args(branch_actions='prior,low', branch_every=4,
                  branch_pilot_blocks=3, branch_seed=99, branch_lower='freeze',
                  branch_signal='soc', branch_critic_frac=0.8,
                  branch_sup_beta=0.2, branch_sup_blocks=100,
                  branch_low_lo=0.0, branch_low_hi=0.1,
                  branch_high_lo=0.4, branch_high_hi=0.5,
                  branch_prior_jitter=0.05, nb_turbine=5, nolow=False)
    c = tbr.BranchController(a)
    ok(c.enabled, 'enabled（every>0）')
    ok(c.critic_frac == 0.5, 'critic_frac 被夹到 ≤0.5（0.8 → 0.5）')
    ok(c.modes == [tbr.MODE_PRIOR, tbr.MODE_LOW], 'modes 解析')
    # pilot：前 3 块每块都 branch
    tr = [c.trigger(k * 5) for k in range(12)]
    ok(tr[:3] == [True, True, True], 'pilot 前 3 块全 branch')
    ok(tr[3:] == [(k % 4 == 0) for k in range(3, 12)], 'pilot 之后每 4 块一次')
    # 关闭
    c0 = tbr.BranchController(make_args(branch_every=0, branch_pilot_blocks=0))
    ok(not c0.enabled and c0.trigger(0) is False, 'every=0/pilot=0 ⇒ 关闭')
    ok(c0.summary().find('未启用') >= 0, 'summary 说明未启用')
    # 信号维
    ok(c.signal_value(0.25) == 0.0, 'signal=soc: SOC=center ⇒ a6=0')
    ok(close(soc_target_from_signal(c.signal_value(0.05)), 0.05, 1e-9),
       'signal=soc: 目标 = 块首 SOC')
    for mode, chk in (('zero', lambda v: v == 0.0),
                      ('pm1', lambda v: abs(abs(v) - 0.5) < 1e-12),
                      ('random', lambda v: -0.5 <= v <= 0.5)):
        cc = tbr.BranchController(make_args(branch_every=1, branch_actions='prior',
                                            branch_signal=mode))
        vals = [cc.signal_value(0.25) for _ in range(20)]
        ok(all(chk(v) for v in vals), 'signal=%s 取值合法' % mode)
    ct = tbr.BranchController(make_args(branch_every=1, branch_actions='prior',
                                        branch_signal='target'))
    ok(ct.signal_value(0.25, actor_signal=0.3) == 0.3, 'signal=target 沿用 actor 信号')
    cn = tbr.BranchController(make_args(branch_every=1, branch_actions='prior',
                                        branch_signal='pm1', nolow=True))
    ok(cn.signal_value(0.25) == 0.0, 'nolow ⇒ 信号维恒 0')
    # β 线性衰减 1 → 0.2
    ok(c.sup_beta(0) == 0.2, 'β(0 blocks) = β0')
    ok(close(c.sup_beta(100), 0.2 * 0.2, 1e-12), 'β(满跨度) = 0.2·β0')
    ok(close(c.sup_beta(50), 0.2 * 0.6, 1e-12), 'β(半程) = 中点（线性）')
    ok(tbr.BranchController(make_args(branch_every=1, branch_sup_beta=0.0))
       .sup_beta(10) == 0.0, 'β0=0 ⇒ 恒 0（关闭）')
    # 诊断累计
    res = dict(a5=np.array([0.01, 0.02, 0.3, 0.48, 0.49]), c1=1.5, r1=2.5,
               n_steps=5, n_fold=3, P_F=[1.0, 2.0, 3.0, 4.0, 5.0], P_F_max=5.0)
    c.note(tbr.MODE_PRIOR, res)
    ok(c.n_trans == 1 and c.stats[tbr.MODE_PRIOR]['n'] == 1, 'note 累计一条')
    cov = c.coverage()
    ok(cov['n_a_dims'] == 5 and cov['n_steps'] == 5, 'coverage 分母正确')
    ok(abs(cov['frac_lo'] - 0.4) < 1e-12, 'a<0.05 占 2/5')
    ok(abs(cov['frac_hi'] - 0.4) < 1e-12, 'a>0.45 占 2/5')
    ok(abs(cov['frac_fold'] - 0.6) < 1e-12, 'P_F∈[2,4] 占 3/5')
    ok(c.begin_block() == 1 and c.block_id == 1, 'begin_block 递增')
    # **branch 不消耗全局 RNG**
    np.random.seed(2024)
    _ = np.random.random(3)
    state = np.random.get_state()
    cc = tbr.BranchController(make_args(branch_every=1, branch_actions='prior,low',
                                        branch_seed=5))
    for m in cc.modes:
        cc.action(m)
    cc.signal_value(0.2)
    now = np.random.get_state()
    ok(state[0] == now[0] and np.array_equal(state[1], now[1]) and state[2] == now[2],
       'branch 用独立 RandomState ⇒ 全局 np.random 状态一字不变')


# =============================================================================
# [H] PER 分池采样
# =============================================================================
def sec_h_replay():
    print('\n[H] PER 分池采样（计数 / 池正确 / by_mode 均分 / 环形覆盖）')
    mem = PB.PrioritizedReplayBuffer(size=64, alpha=0.6)
    s = np.zeros(4)
    a = np.zeros(6)
    # 20 条 actor
    for i in range(20):
        mem.append(s, a, 1.0, 0.5, s, False)
        mem.set_meta(i, src=SRC_ACTOR, mode=MODE_ACTOR)
    ok(mem.n_actor == 20 and mem.n_branch == 0, '只有 actor 时计数正确')
    ok(mem.sample_pool(SRC_BRANCH, 5) == [],
       '空池 ⇒ 返回 []（`_sample_batch` 只在两池都非空时才分池，不会因此缩批）')
    # 20 条 branch，4 个 mode
    for i in range(20, 40):
        mem.append(s, a, 1.0, 0.5, s, False)
        mem.set_meta(i, src=SRC_BRANCH, mode=[1, 2, 3, 4][i % 4], block_id=i // 4)
    ok(mem.n_actor == 20 and mem.n_branch == 20, 'actor/branch 各 20')
    ok(set(mem.pool_idxes(SRC_BRANCH)) == set(range(20, 40)), 'branch 池 = 20..39')
    ok(set(mem.pool_idxes(SRC_ACTOR)) == set(range(0, 20)), 'actor 池 = 0..19')
    idx = mem.sample_pool(SRC_BRANCH, 16, by_mode=True)
    ok(len(idx) == 16 and all(i in range(20, 40) for i in idx), 'by_mode 采 16 条且都在池内')
    modes = [int(mem._meta['mode'][i]) for i in idx]
    cnt = {m: modes.count(m) for m in set(modes)}
    ok(max(cnt.values()) - min(cnt.values()) <= 1, 'mode 均衡（计数差 ≤1）: %s' % cnt)
    # 无元数据时 pool_idxes = 全体
    mem2 = PB.PrioritizedReplayBuffer(size=8, alpha=0.6)
    for i in range(5):
        mem2.append(s, a, 1.0, 0.5, s, False)
    ok(len(mem2.pool_idxes(SRC_ACTOR)) == 5 and not mem2.has_meta,
       '无元数据 ⇒ pool = 全体（历史路径不变）')
    # 环形覆盖：覆写一槽 branch→actor 后计数正确
    mem.set_meta(25, src=SRC_ACTOR, mode=MODE_ACTOR)
    ok(mem.n_actor == 21 and mem.n_branch == 19, '覆写后计数 = 21/19')
    mem.set_meta(25, src=SRC_BRANCH, mode=1)
    ok(mem.n_actor == 20 and mem.n_branch == 20, '再次覆写回 branch 计数复原')


# =============================================================================
# [I] push_branch_transition / n_boot
# =============================================================================
def sec_i_push():
    print('\n[I] push_branch_transition：n_boot=γ、不进 _pending、标签落盘')
    ag = make_agent(nstep=5, branch_mix=0.45)
    gam = ag.discount
    s = np.zeros(4)
    a = np.zeros(6)
    # branch 不进 n-step 链
    pending0 = len(ag._pending)
    idx = ag.push_branch_transition(s, a, 1.0, 2.0, s, False, mode=2,
                                    block_id=7, lower_ver=3)
    ok(len(ag._pending) == pending0, 'branch 不写 _pending')
    ok(idx == 0 and ag.n_branch_transitions == 1, 'push 返回下标，计数 +1')
    ok(close(ag._boot[idx], gam, 1e-6), 'branch 的 n_boot = γ（不是 γ^n）')
    ok(int(ag.memory._src[idx]) == SRC_BRANCH, 'src=branch')
    ok(int(ag.memory._meta['mode'][idx]) == 2, 'mode 标签落盘')
    ok(int(ag.memory._meta['block_id'][idx]) == 7, 'block_id 标签落盘')
    ok(int(ag.memory._meta['lower_ver'][idx]) == 3, 'lower_ver 标签落盘')
    # 终止的 branch ⇒ boot = 0
    idx2 = ag.push_branch_transition(s, a, 1.0, 2.0, s, True, mode=2)
    ok(ag._boot[idx2] == 0.0, 'done 的 branch ⇒ boot = 0')
    # actor（n 步）⇒ boot = γ^n
    for k in range(5):
        ag.observe_jump(s, a, 1.0, 0.5, s, False)
    ok(len(ag.memory) == 2 + 1, '滑窗 stride-1：5 个块产出 1 条 n 步转移')
    actor_idx = [i for i in range(len(ag.memory)) if int(ag.memory._src[i]) == SRC_ACTOR]
    ok(len(actor_idx) == 1, 'actor 转移带 src=actor 标签（_tag_actor 生效）')
    ok(close(ag._boot[actor_idx[0]], gam ** 5, 1e-6), 'actor 的 n_boot = γ^5')
    # branch_mix=0 ⇒ 不打标签
    ag0 = make_agent(nstep=1, branch_mix=0.0)
    ag0.observe_jump(s, a, 1.0, 0.5, s, False)
    ok(not ag0.memory.has_meta, 'branch_mix=0 ⇒ 完全不写元数据（历史路径）')
    ok(not ag0._tag_replay, '_tag_replay=False')


# =============================================================================
# [J] update_policy：分池 + actor 掩码
# =============================================================================
class _Hook(object):
    def __init__(self):
        self.rows = []

    def __call__(self, module, inputs):
        self.rows.append(int(inputs[0].shape[0]))


def sec_j_update():
    print('\n[J] update_policy：branch_mix=0 走历史；>0 时 actor loss 只吃 actor 行')
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- (1) branch_mix=0 ⇒ 调用历史 memory.sample ----
    ag = make_agent(nstep=1, bsize=16)
    called = {'sample': 0, 'pool': 0}
    _orig_sample = ag.memory.sample

    def _spy_sample(*a, **k):
        called['sample'] += 1
        return _orig_sample(*a, **k)

    ag.memory.sample = _spy_sample
    rngj = np.random.RandomState(5)

    def _row():
        # 状态/动作必须**逐条不同**：否则 critic 输出是常数，掩码断言会退化成恒等
        return rngj.randn(4), rngj.rand(6)

    for i in range(64):
        _s, _a = _row()
        ag.memory.append(_s, _a, 1.0, 0.5, _s, False)
    ag.update_policy()
    ok(called['sample'] == 1, 'branch_mix=0 ⇒ 走 mx.sample() 一次（历史路径）')
    ok(ag.last_batch_branch_frac == 0.0, 'branch_mix=0 ⇒ 批内 branch 占比 0')

    # ---- (2) branch_mix>0 ⇒ 分池 + actor 掩码 ----
    ag = make_agent(nstep=1, bsize=32, branch_mix=0.45)
    ag.policy_delay = 1          # 让本批就触发 actor 更新（默认 delay=2 要两次才触发）
    for i in range(120):
        _s, _a = _row()
        ag.memory.append(_s, _a, 1.0, 0.5, _s, False)
        ag.memory.set_meta(i, src=SRC_ACTOR, mode=MODE_ACTOR)
    for i in range(120, 200):
        _s, _a = _row()
        ag.memory.append(_s, _a, 2.0, 0.7, _s, False)
        ag.memory.set_meta(i, src=SRC_BRANCH, mode=[1, 2, 3, 4][i % 4])
    hook = _Hook()
    ag.actor.register_forward_pre_hook(hook)
    qc = {'out': []}

    def _qc_hook(module, inputs, output):
        qc['out'].append(output.detach().clone())

    ag.critic_c.register_forward_hook(_qc_hook)
    # 捕获本批的 src（`_sample_batch` 的返回值），用于**精确**核对 q_c_mean 的掩码
    captured = {}
    _orig_sb = ag._sample_batch

    def _sb_spy():
        out = _orig_sb()
        captured['src'] = np.asarray(out[-1]).reshape(-1).copy()
        return out

    ag._sample_batch = _sb_spy
    ag.update_policy()
    ok(0.30 <= ag.last_batch_branch_frac <= 0.55,
       '批内 branch 占比落 [0.30,0.55]（实测 %.3f）' % ag.last_batch_branch_frac)
    n_branch = int(np.sum(captured['src'] == SRC_BRANCH))
    n_actor = 32 - n_branch
    ok(n_branch == int(round(0.45 * 32)), 'branch 行数 = round(0.45·bsize) = 14（实测 %d）'
       % n_branch)
    ok(ag.n_updates_with_branch == 1, 'n_updates_with_branch 计数')
    ok(len(hook.rows) >= 1, 'actor 至少被前向一次')
    ok(all(r == n_actor for r in hook.rows),
       'actor loss 的每一行都是 actor 行（%s vs n_actor=%d）' % (hook.rows, n_actor))
    # q_c_mean 只算 actor 行：与 critic_c 的**本批前向**输出按同一掩码重算
    q_all = qc['out'][0].reshape(-1)
    mask = torch.tensor(captured['src'] == SRC_ACTOR)
    expect = float(q_all[mask].mean())
    got = float(ag.update_stats[-1]['q_c_mean'])
    ok(abs(got - expect) < 1e-6,
       'update_stats[q_c_mean] = actor 行均值（%.6f vs %.6f）' % (got, expect))
    ok(abs(got - float(q_all.mean())) > 1e-9,
       'q_c_mean ≠ 全批均值（否则 actor 掩码没生效）')

    # ---- (3) branch_mix>0 但池里还没有 branch ⇒ 自动回退到历史路径 ----
    ag2 = make_agent(nstep=1, bsize=8, branch_mix=0.45)
    for i in range(16):
        ag2.memory.append(np.zeros(4), np.zeros(6), 1.0, 0.5, np.zeros(4), False)
        ag2.memory.set_meta(i, src=SRC_ACTOR, mode=MODE_ACTOR)
    ag2.update_policy()
    ok(ag2.last_batch_branch_frac == 0.0 and ag2.n_updates_with_branch == 0,
       '池里无 branch ⇒ 回退历史采样（不报错、不缩批）')

    # ---- (4) 可选 1-step 辅助监督（β>0）路径不炸、键不变 ----
    ag3 = make_agent(nstep=1, bsize=16, branch_mix=0.45, branch_sup_beta=0.3)
    ag3.policy_delay = 1
    for i in range(40):
        _s, _a = _row()
        ag3.memory.append(_s, _a, 1.0, 0.5, _s, False)
        ag3.memory.set_meta(i, src=SRC_ACTOR, mode=MODE_ACTOR)
    for i in range(40, 70):
        _s, _a = _row()
        ag3.memory.append(_s, _a, 2.0, 0.7, _s, False)
        ag3.memory.set_meta(i, src=SRC_BRANCH, mode=[1, 2, 3, 4][i % 4])
    keys0 = set(ag3.update_stats[0].keys()) if ag3.update_stats else None
    ag3.update_policy()
    ok(ag3.update_stats and all(np.isfinite(v) for v in ag3.update_stats[-1].values()),
       'β>0 的辅助监督路径不产生 NaN/Inf')
    if keys0 is not None:
        ok(set(ag3.update_stats[-1].keys()) == keys0,
           'update_stats 键不变（CSV 列不因 β 改变）')
    # β 衰减到 0 后回到纯 TD（不抛异常）
    ag3.branch_sup_beta = 0.0
    ag3.update_policy()
    ok(np.isfinite(float(ag3.update_stats[-1]['td_c_mean'])), 'β=0 ⇒ 纯 TD 路径仍正常')


# =============================================================================
# [K] main 的静态接线（不 import：td3per_main 在 import 期会启 MATLAB）
# =============================================================================
def sec_k_wiring_static():
    print('\n[K] td3per_main 静态接线（不 import main）')
    path = os.path.join(ROOT, 'td3per', 'td3per_main.py')
    src = open(path, encoding='utf-8').read()
    ok('from td3per import td3per_branch as tbr' in src, 'main 导入分支模块 tbr')
    ok('tbr.add_branch_args(parser)' in src, 'CLI 里挂了 tbr.add_branch_args')
    ok("parser.add_argument('--critic_arch', default='v7'" in src,
       '--critic_arch 的 CLI 默认 = v7（2026-11-04 起）')
    ok("'v7'" in src and "'v6'" in src, 'choices 同时保留 v7 与 v6（旧 run 可复现）')
    ok('--l_critic_arch' in src and "'inherit'" in src, '--l_critic_arch 默认 inherit')
    # agent 构造传了三个 critic 参数 + branch_mix
    i_agent = src.index('agent_H = SP_TD3(')
    blk = src[i_agent:i_agent + 1600]
    for key in ('critic_arch=args.critic_arch', 'critic_act=args.critic_act',
                'critic_out_init=args.critic_out_init',
                'branch_mix=_branch_frac', 'branch_sup_beta=_branch_beta'):
        ok(key in blk, 'agent_H 构造传了 %s' % key)
    # branch 调用点在**块首**（episode_steps % 5 == 0）且在主步推进之前
    i_blk = src.index('if episode_steps % 5 == 0:')
    i_br = src.index('branch_ctl.trigger(step)')
    i_nxt = src.index("prGreen('--H-- ori a_t =", i_blk)
    ok(i_blk < i_br < i_nxt, 'branch 在块首决策段内、主步推进之前')
    ok('tbr.run_branch_block(' in src, '调用 tbr.run_branch_block')
    ok('soc_tol=soc_tol_upper' in src, 'run_branch_block 传主循环同一个 soc_tol')
    ok('agent_H.push_branch_transition(' in src, '调用 agent_H.push_branch_transition')
    ok('lower=branch_ctl.lower' in src and 'nolow=bool(args.nolow)' in src,
       'lower/nolow 由控制器与 args 传入（nolow 走 5 维）')
    ok('if args.utd_l > 0:' in src and 'agent_H.lower_ver' in src,
       'lower_ver 随 L 层更新推进（诊断口径）')
    ok('branch_ctl.sup_beta(branch_ctl.n_blocks)' in src, 'β 衰减接到 controller')
    ok('df_branch_' in src, 'branch CSV 文件名带 RUN_TAG')
    ok('branch/cover_lo' in src and 'branch/batch_frac' in src, 'TB branch/* 已接')
    # main 引用的所有 tbr.* 成员都必须存在（重命名/拼错立即暴露）
    import re
    names = sorted(set(re.findall(r'\btbr\.([A-Za-z_][A-Za-z0-9_]*)', src)))
    missing = [n for n in names if not hasattr(tbr, n)]
    ok(not missing, 'main 引用的 tbr 成员全部存在（缺失: %s）' % missing)
    ok(set(names) >= {'BLOCK', 'BranchController', 'MODE_NAME', 'add_branch_args',
                      'declared_scope', 'run_branch_block', 'snapshot_env'},
       'main 引用的关键成员齐备')
    # branch 开关必须全部挂在 argparse 上（--help 可见）
    p = argparse.ArgumentParser(add_help=False)
    tbr.add_branch_args(p)
    flags = {o for a in p._actions for o in a.option_strings}
    want = {'--branch_actions', '--branch_every', '--branch_pilot_blocks',
            '--branch_lower', '--branch_signal', '--branch_critic_frac',
            '--branch_sup_beta', '--branch_sup_blocks', '--branch_csv',
            '--branch_low_lo', '--branch_low_hi', '--branch_high_lo',
            '--branch_high_hi', '--branch_prior_jitter', '--branch_seed'}
    ok(flags == want, 'branch 的 15 个开关全部注册（%d）' % len(flags))


# =============================================================================
# 主入口
# =============================================================================
def main():
    print('=' * 78)
    print('td3per branch + v6 契约测试（纯静态 / 纯算术）')
    print('=' * 78)
    sec_a_constants()
    sec_b_v6_structure()
    sec_c_wiring()
    sec_d_actions()
    sec_e_snapshot()
    sec_f_run_block()
    sec_g_controller()
    sec_h_replay()
    sec_i_push()
    sec_j_update()
    sec_k_wiring_static()
    print('\n' + '=' * 78)
    print('全部通过：%d 条断言' % N_OK)
    print('=' * 78)


if __name__ == '__main__':
    main()
