#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""残余动作 `pb_cmd = 规则基线 + Δ` 的不变量测试 —— 不需要 MATLAB。

TD3 全系列的 RL-BESS 下层从"绝对功率"改成"**残余动作**"：
    pb_cmd = hst.rule_action(soc_target, SOC, 窗口, n_rem)  +  Δ
    Δ 由 actor 输出（零初始化 ⇒ t=0 时 Δ≡0），入 buffer 的动作是 **Δ**，
    交给 RBB 的命令是 `pb_cmd`（因此 clip_amount = |cmd − pbrule| 仍是两段合计）。

为什么要这么改（本文件把"为什么"变成可执行的断言）
----------------------------------------------------
* 规则的**绝对**功率靠观测里的窗口信息才可复现，而 5 维观测不含窗口
  ⇒ 旧参数化下 actor 学不到窗口（实测线性可读 R²=0.78, MAE 1.05 MW），
  而残余只学"相对规则偏多少"，窗口已被基线吃掉。
* 规则的残余自由空间实测为 **0**（激进方向 100% 贴墙、和缓方向 >0.5 MW 仅 2.3%）
  ⇒ "起点 = 规则"必须由**结构**保证，不能靠初始化偶然小：故零初始化输出层。
* Δ≡0 ⇒ 命令逐位等于规则动作 ⇒ `r2 ≡ 0` **且** `c2 ≡ 0`（规则是 L 奖励的唯一最优）。

分节
  A. 基线逐位等价: `hst.rule_action` ≡ 三个 `rule_bess.SocTracker.act`
  B. Δ≡0 ⇒ r2≡0 ∧ c2≡0（含不可行窗口/死区）; Δ≠0 的负向对照
  C. 零初始化输出层 ⇒ t=0 时 Δ≡0（精确 0，不是"接近 0"）
  D. 三个 TD3 main 的接线（静态断言: 命令入 RBB、Δ 入 buffer、日志列、探索半宽）
  E. 探索半宽 [MW] 的 ddpg 侧接线
  F. `fill_bess_transition(cmd_pb=…)` 用**命令**而非 Δ 做一致性核对
"""
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

import hsrl_soctarget as hst                                     # noqa: E402
import hsrl_agent_init as hinit                                  # noqa: E402
import hsrl_warmup as hwarm                                      # noqa: E402

ETA_C = ETA_D = 0.98
DT = 1.0 / 60.0
RATED = 10.0
SOC_LO, SOC_HI = 0.0, 6.0

TD3_MAINS = {
    'td3per': 'td3per/td3per_main.py',
    'td3prrbb': 'td3prrbb/td3prrbb_main.py',
    'td3prle': 'td3prle/td3prle_main.py',
}
RULE_BESS = {
    'td3prrlrb': 'td3prrlrb/td3prrlrb_rule_bess.py',
    'td3perrb': 'td3perrb/td3perrb_rule_bess.py',
    'td3leprrlrb': 'td3leprrlrb/td3leprrlrb_rule_bess.py',
}
# 零初始化输出层的期望（主循环里在建 agent 之后、训练之前）


def src(p):
    return open(p, encoding='utf-8').read()


def flat(txt):
    return re.sub(r'\s+', ' ', txt)


def soc_after(prev, pb):
    """ΔSOC = g·pb（g 按充放分段）—— 与 env / `hst.applied_pb` 同一分段线性模型。"""
    g = ETA_C * DT if pb >= 0 else DT / ETA_D
    return prev + g * float(pb)


def rbb_applied(cmd, lo, hi):
    """RBB 施加值: 窗口可行 ⇒ 夹窗口; 窗口不可行 ⇒ 按额定夹裁（交物理兜底）。"""
    if lo > hi:
        return float(np.clip(cmd, -RATED, RATED))
    return float(np.clip(cmd, lo, hi))


def cases(n=4000, seed=20260214):
    """随机状态: 覆盖常规窗口 / 不含 0 的窗口 / 越出额定的窗口 / 不可行窗口。"""
    rg = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        prev = float(rg.uniform(0.2, 5.8))
        target = float(rg.uniform(0.5, 5.5))
        m = int(rg.integers(0, 100))
        if m < 10:                                    # 不可行窗口（被迫吞吐）
            lo = float(rg.uniform(1.0, 8.0))
            hi = lo - float(rg.uniform(0.05, 3.0))
        elif m < 30:                                  # 窗口不含 0
            if rg.random() < 0.5:
                lo, hi = float(rg.uniform(0.2, 2.0)), float(rg.uniform(2.2, 8.0))
            else:
                lo, hi = -float(rg.uniform(2.2, 8.0)), -float(rg.uniform(0.2, 2.0))
        elif m < 38:                                  # 窗口越出 ±rated
            lo, hi = -float(rg.uniform(11.0, 14.0)), float(rg.uniform(11.0, 14.0))
        else:                                         # 常规
            c = float(rg.uniform(-6.0, 6.0))
            w = float(rg.uniform(0.05, 9.0))
            lo, hi = c - w / 2.0, c + w / 2.0
        out.append((target, prev, lo, hi, int(rg.integers(1, 6))))
    # 死区样本: δ ≡ 0 与 |δ| ≤ 0.1
    for _ in range(200):
        prev = float(rg.uniform(0.5, 5.5))
        out.append((prev, prev, -3.0, 3.0, 5))
        out.append((prev + 0.05, prev, -3.0, 3.0, 5))
    return out


def reachable(cs):
    """实机 RBB 的窗口 = 平抑需求 ∩ 物理可行域 ⊆ ±rated。

    `cases()` 里刻意含"窗口越出 ±rated"的样本（用来暴露 td3leprrlrb 旧拷贝的差异），
    但那种窗口**实机不可能出现**（见 `SocTracker` 的 rated 限幅与 RBB 的构造）。
    本函数按实机口径把窗口夹回 ±rated：夹完若 lo > hi，就退化成真实的"不可行窗口"
    （被迫吞吐，整步免罚），而不是"参考点在执行器能力之外"的伪罚。
    """
    out = []
    for (tgt, prev, lo, hi, n_rem) in cs:
        out.append((tgt, prev, max(lo, -RATED), min(hi, RATED), n_rem))
    return out


CASES = cases()
CASES_REACH = reachable(CASES)
print('随机状态样本: {} 组 (含 {} 组不可行窗口); 按实机口径夹回 ±rated 后不可行 {} 组'
      .format(len(CASES), sum(1 for c in CASES if c[2] > c[3]),
              sum(1 for c in CASES_REACH if c[2] > c[3])))

# ------------------------------------------------------------------ A
print('=' * 84)
print('A. 基线逐位等价: hst.rule_action ≡ SocTracker.act')
print('=' * 84)
import importlib                                                 # noqa: E402
for tag, mod_path in RULE_BESS.items():
    mod = importlib.import_module(mod_path[:-3].replace('/', '.'))
    for tol in (0.0, 0.1):
        trk = mod.SocTracker(soc_tol=tol)
        worst, n_diff, bad = 0.0, 0, 0
        for (tgt, prev, lo, hi, n_rem) in CASES:
            a = hst.rule_action(tgt, prev, lo, hi, n_rem, eta_c=ETA_C, eta_d=ETA_D,
                                time=DT, rated_power=RATED, soc_tol=tol)
            b = float(trk.act(prev, tgt, lo, hi))
            d = abs(a - b)
            worst = max(worst, d)
            if d > 0.0:
                n_diff += 1
                # 允许的差异只有一种: 旧拷贝在夹窗口**之前**没有先夹 ±rated
                if not (lo < -RATED or hi > RATED):
                    bad += 1
        print('  {:<12} soc_tol={:<4} max|Δ|={:.3e}  差异样本={:<5} 其中窗口在 ±rated 内={}'
              .format(tag, tol, worst, n_diff, bad))
        assert bad == 0, (tag, tol, '窗口内的差异说明基线不等于 SocTracker')
        if tag != 'td3leprrlrb':
            assert worst == 0.0, (tag, tol, worst)
        else:
            # td3leprrlrb 是旧拷贝: 只有窗口越出 ±rated 时才不同（该分支在实机上不可达,
            # 因 W ⊆ pbphysics ⊆ ±rated）——这里把这条"唯一差异"钉住。
            assert n_diff <= 400, (tag, tol, n_diff)

# 基线自身的性质: 恒落在窗口内（不可行窗口除外）
n_out = 0
for (tgt, prev, lo, hi, n_rem) in CASES:
    pb = hst.rule_action(tgt, prev, lo, hi, n_rem)
    if lo <= hi and not (lo - 1e-12 <= pb <= hi + 1e-12):
        n_out += 1
assert n_out == 0, n_out
print('  基线恒落在窗口内 (可行窗口 {} 组，越界 0 组)'.format(sum(1 for c in CASES if c[2] <= c[3])))

# ------------------------------------------------------------------ B
print('=' * 84)
print('B. Δ≡0 ⇒ r2≡0 且 c2≡0 (规则是 L 奖励的唯一最优); 残余自由度 = 参考集合宽度')
print('=' * 84)
# B1 实机可达窗口（W ⊆ ±rated）: 两通道**精确**归零
def pen_of(tgt, prev, lo, hi, n_rem, pb_rule, d):
    """Δ = d 时的**总罚** = c2 − r2 = trackerr + min(clip_amount, 20) ≥ 0。

    （r2 是**负**奖励 `−1·min(clip_amount,20)`，故总罚要取 `c2 − r2`。）
    """
    cmd = pb_rule + d
    app = rbb_applied(cmd, lo, hi)
    return (hst.lower_cost_soc_track(tgt, prev, soc_after(prev, app), lo, hi, n_rem)
            - hst.lower_reward(abs(cmd - app)))


worst_r2 = worst_c2 = 0.0
n_point = n_interval = n_infeas = 0
n_pt_ok = n_iv_ok = n_inf_ok = 0
for (tgt, prev, lo, hi, n_rem) in CASES_REACH:
    pb_rule = hst.rule_action(tgt, prev, lo, hi, n_rem)
    pb_app = rbb_applied(pb_rule, lo, hi)
    r2 = hst.lower_reward(abs(pb_rule - pb_app))          # RBB 的 clip_amount 口径
    c2, info = hst.lower_cost_soc_track(tgt, prev, soc_after(prev, pb_app), lo, hi,
                                        n_rem, return_info=True)
    worst_r2 = max(worst_r2, abs(r2))
    worst_c2 = max(worst_c2, abs(c2))

    if info['infeasible_window']:
        # 被迫吞吐: 整步不可归责 ⇒ 至少"减吞吐"方向完全免费; 但把命令推出 ±rated
        # （执行器能力之外）的那一侧仍按两段 clip 计费 —— 这是物理事实, 不是罚项设计。
        n_infeas += 1
        n_inf_ok += int(min(pen_of(tgt, prev, lo, hi, n_rem, pb_rule, d)
                            for d in (0.5, -0.5)) < 1e-9)
    elif info['ref_is_point']:
        # F = ∅ ⇒ 只有"竭力值"免费 ⇒ **任何** Δ≠0 都在两个通道之一被计费
        n_point += 1
        n_pt_ok += int(min(pen_of(tgt, prev, lo, hi, n_rem, pb_rule, d)
                           for d in (0.05, 0.5, 1.0, -0.05, -0.5, -1.0)) > 1e-9)
    else:
        # F = [F_lo, F_hi] ⇒ 免费残余区间**恰好**是 F − pb_rule（宽度 = F 的宽度）
        n_interval += 1
        w = info['ref_hi'] - info['ref_lo']
        e_lo = pen_of(tgt, prev, lo, hi, n_rem, pb_rule, info['ref_lo'] - pb_rule)
        e_hi = pen_of(tgt, prev, lo, hi, n_rem, pb_rule, info['ref_hi'] - pb_rule)
        o_lo = pen_of(tgt, prev, lo, hi, n_rem, pb_rule, info['ref_lo'] - pb_rule - 0.5)
        o_hi = pen_of(tgt, prev, lo, hi, n_rem, pb_rule, info['ref_hi'] - pb_rule + 0.5)
        n_iv_ok += int(max(abs(e_lo), abs(e_hi)) < 1e-9 and min(o_lo, o_hi) > 1e-9
                       and w >= 0.0)
assert worst_r2 < 1e-9, worst_r2
assert worst_c2 < 1e-9, worst_c2
n = len(CASES_REACH)
print('  Δ≡0: max|r2|={:.3e}  max|c2|={:.3e}   (两通道同时归零, {} 组样本)'
      .format(worst_r2, worst_c2, n))
print('  参考集合形态: 单点(F=∅) {:.1f}% / 区间 {:.1f}% / 窗口不可行 {:.1f}%'
      .format(100.0 * n_point / n, 100.0 * n_interval / n, 100.0 * n_infeas / n))
print('  残余自由度: 单点组"任何 Δ≠0 都被计费" {}/{}; 区间组"免费区间=F, 端点外必罚" {}/{};'
      ' 不可行组"整步平坦" {}/{}'
      .format(n_pt_ok, n_point, n_iv_ok, n_interval, n_inf_ok, n_infeas))
assert n_pt_ok == n_point, (n_pt_ok, n_point, 'F=∅ 时 Δ≠0 竟然免费')
assert n_iv_ok == n_interval, (n_iv_ok, n_interval, '免费残余区间 ≠ 参考集合')
assert n_inf_ok == n_infeas, (n_inf_ok, n_infeas, '不可行窗口步竟被计费')

# B2 不可达窗口（W ⊄ ±rated，实机不会出现）: Δ=0 也不再免费，且**只能**因为参考点
#    落在执行器能力之外（|ref| > rated）——把这条边界钉住，避免将来误当作 bug。
n_unreach_bad = 0
for (tgt, prev, lo, hi, n_rem) in CASES:
    if lo >= -RATED and hi <= RATED:
        continue
    pb_rule = hst.rule_action(tgt, prev, lo, hi, n_rem)
    pb_app = rbb_applied(pb_rule, lo, hi)
    c2, info = hst.lower_cost_soc_track(tgt, prev, soc_after(prev, pb_app), lo, hi,
                                        n_rem, return_info=True)
    if abs(c2) > 1e-9:
        n_unreach_bad += 1
        assert not info['infeasible_window'], (tgt, prev, lo, hi)
        assert abs(info['ref_lo']) > RATED + 1e-9, (tgt, prev, lo, hi, info['ref_lo'])
print('  窗口越出 ±rated 的 {} 组（实机 RBB 不会产生）: Δ=0 也不再免费，'
      '且原因恒为 |ref|>rated'.format(n_unreach_bad))

# ------------------------------------------------------------------ C
print('=' * 84)
print('C. 零初始化输出层 ⇒ t=0 时 Δ≡0')
print('=' * 84)
import torch                                                     # noqa: E402
import model                                                     # noqa: E402


class _StubAgent:
    """最小假体: 只需 .actor / .actor_target。"""


_stub = _StubAgent()
_stub.actor = model.Actor(5, 1)
_stub.actor_target = model.Actor(5, 1)
_x = torch.from_numpy(np.random.default_rng(0).normal(size=(64, 5)).astype(np.float32))
_before = float(_stub.actor(_x).abs().max().item())
assert _before > 0.0, _before
_names = hinit.zero_actor_output(_stub)
_after = float(_stub.actor(_x).abs().max().item())
_after_t = float(_stub.actor_target(_x).abs().max().item())
assert _names == ['actor', 'actor_target'], _names
assert _after == 0.0 and _after_t == 0.0, (_after, _after_t)
# 输出层是 fc4（fc3 是隐层）; 只置 bias 不够（权重仍会映射隐层激活）
assert hasattr(model.Actor(5, 1), 'fc4')
print('  新建 Actor 输出 max|·|={:.4f} → 置零后 {:.1e} (actor_target 同为 {:.1e}); 处理 {}'
      .format(_before, _after, _after_t, _names))

# ------------------------------------------------------------------ D
print('=' * 84)
print('D. 三个 TD3 main 的接线')
print('=' * 84)
for proj, main_py in TD3_MAINS.items():
    mt = src(main_py)
    f = flat(mt)
    # ① SOC 目标与基线: 与 myfunc 的 soc_target 同源 (applied H 的信号维)
    assert '_soc_tgt_L = hst.soc_target_from_signal(float(applied_action_H[5]))' in f, proj
    assert 'pb_rule = hst.rule_action( _soc_tgt_L, E_current, rbb_info_obs' in f, proj
    assert "rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'], _n_rem_L" in f, proj
    assert 'time=rbb_system.time, rated_power=rbb_system.rated_power)' in f, proj
    assert '_n_rem_L = 5 - (episode_steps % 5)' in f, proj
    # ② 三个分支: warmup-rule ⇒ Δ=0; warmup-random ⇒ 绝对随机命令; 否则 actor 输出 Δ
    assert 'delta_L = np.zeros(1, dtype=np.float64)' in f, proj
    assert 'pb_cmd_L = float(pb_rule)' in f, proj
    assert 'pb_cmd_L = float(pb_rule) + float(delta_L[0])' in f, proj
    assert 'original_action_L = delta_L' in f, proj
    assert 'applied_action_L, penalty_NN_L = CheckAction(np.array([pb_cmd_L]))' in f, proj
    # ③ 命令值（含 ±10 那段）交给 RBB; 施加值进 env
    assert 'pb_original = float(pb_cmd_L)' in f, proj
    assert 'action_rbb[-1] = pbrule' in f, proj
    assert 'pb_original = float(np.ravel(np.asarray(original_action_L' not in f, proj
    assert 'applied_action_L, penalty_NN_L = CheckAction(original_action_L)' not in f, proj
    # ④ Δ 入 buffer（动作空间 = Δ），命令值只用于一致性核对
    assert 'agent_L, norm_observation_L, original_action_L, r2, c2,' in f, proj
    assert 'cmd_pb=float(pb_cmd_L),' in f, proj
    assert 'rule_pb=float(pb_rule),' in f, proj
    assert 'hwarm.fill_bess_transition(' in f, proj
    # ⑤ 日志列
    for key in ("'original_action_L': delta_L,", "'delta_L': float(delta_L[0]),",
                "'pb_rule': float(pb_rule),", "'pb_cmd': float(pb_cmd_L),"):
        assert key in f, (proj, key)
    assert "writer.add_scalar('action/pb_rule'" in f, proj
    assert "writer.add_scalar('action/delta_L'" in f, proj
    # ⑥ 零初始化 + 探索半宽（默认 1.0 MW，旧式 ±0.05 比所需小 ~50 倍）
    assert 'zero_actor_output(agent_L' in f, proj
    assert re.search(r'--l_noise_amp.*?default=1\.0.*?type=float', f), proj
    # warmup 规则臂必须与策略期基线**同一条规则** (启动自检), 且 Δ 恒 0
    assert 'hwarm.check_residual_baseline(' in f, proj
    assert "soc_target_from_signal=hst.soc_target_from_signal" in f, proj
    if proj == 'td3per':
        assert "agent_L.explore_amp = np.full(1, float(getattr(args, 'l_noise_amp', 1.0))" in f, proj
        assert "agent_L.explore_floor = np.zeros(1, dtype=np.float32)" in f, proj
    else:
        assert "agent_L.noise_halfwidth = float(getattr(args, 'l_noise_amp', 1.0))" in f, proj
    # ⑦ 零初始化必须在训练之前
    assert mt.index('zero_actor_output(agent_L') < mt.index("if args.mode == 'train':"), proj
    # ⑧ 口径未变: 仍需单一真源
    assert 'hst.dual_update_L(' in f and 'import hsrl_soctarget as hst' in f, proj
    print('  {:<9} OK (基线/Δ/命令/buffer/日志/零初始化/探索半宽)'.format(proj))

# ------------------------------------------------------------------ E
print('=' * 84)
print('E. 探索半宽 [MW] 的 ddpg 侧接线')
print('=' * 84)
for p in ('td3prrbb/td3prrbb_ddpg.py', 'td3prle/td3prle_ddpg.py'):
    t = flat(src(p))
    assert 'self.noise_halfwidth = 0.05' in t, (p, '默认半宽必须保持 0.05 ⇒ H 层分布不变')
    assert 'hw = float(getattr(self,' in t and 'noise_halfwidth' in t, p
    assert '* 2.0 * hw * eps' in t, p
assert 'amp = np.maximum(self.explore_amp * eps, self.explore_floor)' in \
    flat(src('td3per/td3per_ddpg.py'))
print('  td3parrbb/td3prle: noise_halfwidth (默认 0.05, L 层被 main 覆写为 --l_noise_amp)')
print('  td3per: explore_amp/explore_floor 逐维幅值 (L 层 nb_actions=1)')

# ------------------------------------------------------------------ F
print('=' * 84)
print('F. fill_bess_transition: 记录的动作能重建命令; 命令被夹是正常工况而非失配')
print('=' * 84)


class _FakeL:
    def __init__(self):
        self.calls = []

    def observe_jump(self, s, a, r, c, s2, done, **kw):
        self.calls.append((np.asarray(a, dtype=float).reshape(-1), float(r), float(c)))


def _fill(stats, agent, a, cmd, rule, applied):
    hwarm.fill_bess_transition(agent, [0.0] * 5, np.array([a]), -1.0, 2.0, [0.0] * 5, True,
                               applied_pb=applied, warmup=True, stats=stats,
                               cmd_pb=cmd, rule_pb=rule)


_st = hwarm.BessFillStats()
_ag = _FakeL()
# ① 残余语义: 记录动作 Δ 与 rule_pb 一起能重建命令 ⇒ 不算失配（哪怕命令被窗口夹掉）
_fill(_st, _ag, 0.4, 2.4, 2.0, 2.0)
assert (_st.n_mismatch, _st.n_clipped, _st.n_total) == (0, 1, 1), \
    (_st.n_mismatch, _st.n_clipped, _st.n_total)
assert abs(_st.max_abs_gap - 0.4) < 1e-12, _st.max_abs_gap
# ② 记录的动作与命令对不上（接线 bug）⇒ 计失配（命令离施加值很远 ⇒ 也计"被夹"）
_fill(_st, _ag, 0.4, 5.0, 2.0, 2.0)
assert (_st.n_mismatch, _st.n_clipped) == (1, 2), (_st.n_mismatch, _st.n_clipped)
# ③ 忘了加基线（把 Δ 当命令）⇒ 也计失配
_fill(_st, _ag, 0.4, 0.4, 2.0, 2.0)
assert (_st.n_mismatch, _st.n_clipped) == (2, 3), (_st.n_mismatch, _st.n_clipped)
# ④ 命令与施加值一致（窗口没夹）⇒ n_clipped 不涨, 重建成功 ⇒ 失配不涨
_fill(_st, _ag, 0.4, 2.4, 2.0, 2.4)
assert (_st.n_mismatch, _st.n_clipped) == (2, 3), (_st.n_mismatch, _st.n_clipped)
# ⑤ 旧语义（影子电价项目: 不给 cmd_pb/rule_pb）: 记录动作就是命令
_st2 = hwarm.BessFillStats()
_fill_none = dict(warmup=True, stats=_st2, applied_pb=3.0)
hwarm.fill_bess_transition(_FakeL(), [0.0] * 5, np.array([3.0]), -1.0, 2.0, [0.0] * 5, True,
                           **_fill_none)
assert _st2.n_mismatch == 0, _st2.n_mismatch
hwarm.fill_bess_transition(_FakeL(), [0.0] * 5, np.array([2.0]), -1.0, 2.0, [0.0] * 5, True,
                           **_fill_none)
assert _st2.n_mismatch == 1, _st2.n_mismatch
# 入池动作 = Δ 本身（不是命令）
assert abs(_ag.calls[0][0][0] - 0.4) < 1e-12, _ag.calls[0]
print('  残余语义: rule_pb+a_L == cmd_pb ⇒ 不失配 (命令被夹 {}/{} 条单独计数);'
      ' Δ 当命令 / 动作对不上 ⇒ 失配'.format(_st.n_clipped, _st.n_total))
print('  旧语义 (per/prrbb): 不给 cmd_pb ⇒ 仍按 |a_L − applied| 核对; 入池动作 = Δ')

# ------------------------------------------------------------------ G
print('=' * 84)
print('G. 评测夹具同步: 加载残余动作的 checkpoint 时必须补上规则基线')
print('=' * 84)
# test_five_days.py: 三个 TD3 SOC 项目 residual=True, 其余项目不得误标
_fd = src('test_five_days.py')
assert 'import hsrl_soctarget as hst' in _fd, 'test_five_days 未接真源'
assert _fd.count('residual=True,') == 3, _fd.count('residual=True,')
for key in ('td3per', 'td3prrbb', 'td3prle'):
    i = _fd.index("'%s': dict(" % key)
    j = _fd.index('),\n', i)
    assert 'residual=True,' in _fd[i:j], (key, '未标 residual')
for key in ('per', 'prrbb', 'prle', 'td3prrlrb', 'td3perrb', 'td3leprrlrb'):
    i = _fd.index("'%s': dict(" % key)
    j = _fd.index('),\n', i)
    assert 'residual=True,' not in _fd[i:j], (key, '不应有残余动作 (影子电价/规则下层)')
_g = flat(_fd)
assert "if self.spec.get('residual', False):" in _g
assert 'pb_rule = hst.rule_action( hst.soc_target_from_signal(action_H[5]), E_now,' in _g
assert 'eta_c=self.rbb.eta_c, eta_d=self.rbb.eta_d' in _g
assert 'applied_action_L, _pen_L = CheckAction( np.array([float(pb_rule) + float(original_action_L[0])]))' in _g
# test_simple.py: 三个 TD3 SOC 项目补基线, PRLE 不补
_ts = flat(src('test_simple.py'))
assert "residual_proj = ('TD3PER', 'TD3PRRBB', 'TD3PRLE')" in _ts, 'test_simple 未标残余项目'
assert 'if proj_type in residual_proj:' in _ts
assert 'pb_rule = hst.rule_action( hst.soc_target_from_signal(action_H[-1]), float(obs[2]),' in _ts
assert 'action_L = np.array([float(pb_rule) + float(action_L[0])])' in _ts
# test_compare_projects.py 只覆盖 per/prrbb/prle (影子电价 + DDPG) ⇒ 不动
_cp = flat(src('test_compare_projects.py'))
assert 'residual' not in _cp and 'hst.rule_action' not in _cp, 'compare_projects 不在本次范围'
print('  test_five_days: 3 个项目 residual=True (其余 6 个不标) + 规则基线补齐; ')
print('  test_simple: residual_proj = TD3PER/TD3PRRBB/TD3PRLE; test_compare_projects 不在范围')

# ------------------------------------------------------------------ H
print('=' * 84)
print('H. warmup 行为生成: 规则臂 = 策略期基线 (Δ≡0), 与改造前逐位同行为')
print('=' * 84)
# ① 默认调度的 warmup 目标不变: 信号维恒 0 ⇒ soc_target = 3.0 = SOC_TARGET = 改造前的 target
_t0 = float(hst.soc_target_from_signal(hwarm.SIG_VALUE))
assert abs(_t0 - 3.0) < 1e-15 and abs(hwarm.SOC_TARGET - _t0) < 1e-15, (_t0, hwarm.SOC_TARGET)
_rule = hwarm.WarmupBessRule()                      # 默认: target=3.0, eta 0.98/0.98, dt 1/60, rated 10, soc_tol 0
# ② 公式等价: warmup 规则行为体 ≡ 单一真源基线 (同 target)
_w_same = max(abs(float(_rule.act(t, lo, hi, tgt))
                  - float(hst.rule_action(tgt, t, lo, hi, nr)))
              for (tgt, t, lo, hi, nr) in CASES)
# ③ 行为等价: 改造前的调用 (_rule.act(E, lo, hi) 固定追 3.0) vs 改造后的基线 (信号目标)
_w_old = max(abs(float(_rule.act(t, lo, hi))
                 - float(hst.rule_action(hst.soc_target_from_signal(hwarm.SIG_VALUE), t, lo, hi, nr)))
             for (tgt, t, lo, hi, nr) in CASES)
assert _w_same == 0.0, _w_same
assert _w_old == 0.0, _w_old
print('  soc_target_from_signal(0) = {:.1f} = SOC_TARGET; 同 target 下 规则行为体 ≡ hst.rule_action:'
      ' max|Δ|={:.1e}'.format(_t0, _w_same))
print('  改造前规则 (固定追 3.0) vs 改造后基线 (信号目标): max|Δ|={:.1e} ({} 组, 默认调度 ⇒ 逐位相同)'
      .format(_w_old, len(CASES)))
# ④ 启动自检: 同一条规则 ⇒ 通过; 参数漂移 ⇒ 抛错; 对照臂 ⇒ rule_arm=False
import argparse                                                 # noqa: E402
_sched_fixed = hwarm.WarmupSchedule(nb_turbine=5, nb_actions=6, enabled=True)
_sched_rand = hwarm.WarmupSchedule(nb_turbine=5, nb_actions=6, enabled=False)
_args = argparse.Namespace(warmup_soc_target=3.0)
_res = hwarm.check_residual_baseline(
    _rule, _sched_fixed, _args,
    lambda t, E, lo, hi: hst.rule_action(t, E, lo, hi, 5),
    soc_target_from_signal=hst.soc_target_from_signal)
assert _res['rule_arm'] and _res['n_checked'] > 0 and abs(_res['target_at_signal0'] - 3.0) < 1e-15, _res
assert not hwarm.check_residual_baseline(None, _sched_fixed, _args, lambda t, E, lo, hi: 0.0)['rule_arm']
for _kw in ({'eta_c': 0.90}, {'rated_power': 5.0}, {'soc_tol': 0.2}, {'time': 1.0 / 30}):
    try:
        hwarm.check_residual_baseline(
            _rule, _sched_fixed, _args,
            lambda t, E, lo, hi: hst.rule_action(t, E, lo, hi, 5, **_kw),
            soc_target_from_signal=hst.soc_target_from_signal)
        raise SystemExit('参数漂移 {} 未被自检抓住'.format(_kw))
    except AssertionError:
        pass
print('  启动自检: 同规则通过 ({} 组); eta_c/rated_power/soc_tol/time 任一漂移 ⇒ AssertionError;'
      ' 对照臂 ⇒ rule_arm=False'.format(_res['n_checked']))
# ⑤ 对照臂 (--warmup_bess random) 的环境行为 = 改造前: 绝对随机功率 U(-rated, rated)
_ddpg = src('td3per/td3per_ddpg.py')
assert 'def random_action' in _ddpg and 'charge = 10.' in _ddpg \
    and 'np.random.uniform(-charge, charge' in _ddpg, '对照臂随机动作口径变了'
assert 'pb_cmd_L = float(delta_L[0])' in flat(src('td3per/td3per_main.py'))
print('  对照臂: 环境侧仍是绝对随机功率 U(-10, 10) MW (random_action 未改), 仅 buffer 内改记为 Δ = cmd − rule')

print('=' * 84)
print('TEST OK')
