#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""warmup 调度 + critic bias 触发修复 的单元测试 —— 不需要 MATLAB。

验证七件事:
  1. `first_trigger_step` / `BiasInitTracker`：**修复了旧触发条件**
     （`step == warmup` 落在块首 ⇒ 默认 warmup=200 永不触发），且在首次策略更新前生效
  2. `WarmupSchedule`：起始点、步长、clip 带、可复现性、信号维 = 0（⇔ soc_target = 3）
  3. `WarmupBessRule`：与各项目 `SocTracker.act` **逐位同语义**
  4. `fill_bess_transition`：真实 transition 入池 + 计数（BESS-RL 项目的"事后填入"）
  5. 旧触发条件的反例（warmup=200 下 `step == warmup` 与块末永不同时成立）
  6. **偏航带契约**（yawflor，P2）：起点 = 0、界内、步长 = 带宽 10%、两端可达、
     信号维恒 0 —— **不断言任何"最优/效率"**（风场最优是策略要学的函数）
  7. **满带越界罚等值**（P2）：旧 `400×0.5` == 新 `100×2` == 200 元；下层 1 维不变
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块 (hsrl_*, myfunc, util, td3*/...)
sys.path.insert(0, _HERE)                              # agentutil 内互相 import
sys.path.insert(0, os.path.join(ROOT, 'td3prrlrb'))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

import hsrl_agent_init as hinit            # noqa: E402
import hsrl_warmup as hw                   # noqa: E402
from td3prrlrb_myfunc import soc_target_from_signal   # noqa: E402
from td3prrlrb_rule_bess import SocTracker            # noqa: E402

np.random.seed(0)
torch.manual_seed(0)

# ---------------------------------------------------------------- 0) 触发点
print('--- 1) bias 触发点 ---')
exp = {0: 4, 1: 4, 4: 4, 5: 9, 199: 199, 200: 204, 201: 204, 204: 204, 500: 504}
for w, e in exp.items():
    got = hinit.first_trigger_step(w)
    assert got == e, (w, got, e)
    assert got % 5 == 4 and got >= w, (w, got)
print('first_trigger_step: ' + ', '.join('{}→{}'.format(k, v) for k, v in exp.items()))

# 旧写法反例: step == warmup 且同时是块末 —— 默认 warmup=200 永不成立
old_hits = [s for s in range(0, 600) if s % 5 == 4 and s == 200]
assert old_hits == [], old_hits
assert hinit.first_trigger_step(200) == 204
print('旧条件 (step == warmup 且 step%%5==4) 在 warmup=200 下命中 0 次；新触发点=204 ✓')


# ---------------------------------------------------------------- 1) bias tracker
class _CriticStub:
    """只带 fc3 (50→1) 的最小 stub，够 init_critic_bias 用。"""

    def __init__(self):
        for name in ('critic_p', 'critic_target_p', 'critic_p2', 'critic_target_p2',
                     'critic_c', 'critic_target_c', 'critic_c2', 'critic_target_c2'):
            net = torch.nn.Module()
            net.fc3 = torch.nn.Linear(50, 1)
            setattr(self, name, net)


def _run_loop(warmup, mode='estimate', b_p=-1.0, r1_const=-82.0, c1_const=19.8,
              n_steps=900):
    """复刻主循环的块末判定（episode_steps % 5 == 4, episode 长度 1440）。"""
    tr = hinit.BiasInitTracker(mode=mode, warmup=warmup, discount=0.99,
                               b_p=b_p, verbose=False)
    agent = _CriticStub()
    fired = []
    for step in range(n_steps):
        if step % 1440 % 5 == 4:                       # 块末
            info = tr.on_block_end(step, r1_const, c1_const, agent)
            if info:
                fired.append(info)
    return tr, agent, fired


for warmup in (199, 200, 201, 204, 500):
    tr, agent, fired = _run_loop(warmup)
    assert len(fired) == 1, (warmup, len(fired))          # 恰好应用一次
    info = fired[0]
    assert info['step'] == hinit.first_trigger_step(warmup), (warmup, info)
    n_expect = (info['step'] - 4) // 5 + 1               # 4,9,...,trigger_step
    assert info['n_blocks'] == n_expect, (info['n_blocks'], n_expect)
    assert abs(info['b_p'] - (-82.0) / 0.01) < 1e-6, info['b_p']
    assert abs(float(agent.critic_p.fc3.bias.item()) - info['b_p']) < 1e-6
    first_update = min(s for s in range(0, 900) if s % 5 == 4 and s > warmup)
    assert info['step'] <= first_update, (info['step'], first_update, warmup)
    print('warmup={:<4d} → 触发 step={:<4d} 用 {} 块估计 b_p={:+.0f} (首次策略更新 step={})'
          .format(warmup, info['step'], info['n_blocks'], info['b_p'], first_update))

# 显式覆盖 + none 对照臂
tr, agent, fired = _run_loop(200, b_p=8000.0, r1_const=-82.0)
assert fired and abs(fired[0]['b_p'] - 8000.0) < 1e-9, fired
assert abs(float(agent.critic_p.fc3.bias.item()) - 8000.0) < 1e-6
tr, agent, fired = _run_loop(200, mode='none')
assert fired == [] and abs(float(agent.critic_p.fc3.bias.item())) < 1.0
print('显式 --critic_bias_p 8000 生效；mode=none 不触发 ✓')

# 值函数回归空间归一化: 样本/覆盖值都是原始口径, 写 fc3.bias 前统一 /s
tr = hinit.BiasInitTracker(mode='estimate', warmup=200, discount=0.99, verbose=False,
                           value_scale=1000.0)
agent = _CriticStub()
fired = []
for step in range(900):
    if step % 5 == 4:
        info = tr.on_block_end(step, -82.0, 19.8, agent)
        if info:
            fired.append(info)
info = fired[0]
assert abs(info['b_p_raw'] - (-8200.0)) < 1e-6, info['b_p_raw']      # 原始口径
assert abs(info['b_p'] - (-8.2)) < 1e-6, info['b_p']                 # = b_p_raw/s
assert abs(float(agent.critic_p.fc3.bias.item()) - info['b_p']) < 1e-6
tr = hinit.BiasInitTracker(mode='estimate', warmup=200, discount=0.99, verbose=False,
                           value_scale=1000.0, b_p=8000.0)
agent = _CriticStub()
for step in range(900):
    if step % 5 == 4:
        tr.on_block_end(step, -82.0, 19.8, agent)
assert abs(float(agent.critic_p.fc3.bias.item()) - 8.0) < 1e-9      # 覆盖值 8000/1000
print('s=1000: b_p_raw=-8200 → fc3.bias=-8.2; 覆盖值 8000 → 8.0 ✓')

# verbose=True 必须与 verbose=False 数值一致, 且打印分支不得抛异常。
# 回归背景: 双尺度改造把 on_block_end 里的单尺度局部变量 s 拆成 sp/sc, 漏改了
# `if self.verbose:` 分支里的格式占位符 ⇒ NameError: name 's' is not defined。
# 而 from_args **不传 verbose** ⇒ 取类默认 verbose=True ⇒ 生产每次都在触发块崩;
# 既有测试全部用 verbose=False, 所以这条分支零覆盖、CI 全绿却线上必炸。
import contextlib
import io as _io

_num = ('b_p', 'b_c', 'b_p_raw', 'b_c_raw', 'step', 'trigger_step', 'scale', 'scale_c')


def _run_verbose(verbose, b_p=-1.0, b_c=-1.0):
    buf = _io.StringIO()
    tr = hinit.BiasInitTracker(mode='estimate', warmup=200, discount=0.99,
                               verbose=verbose, value_scale=1000.0, cost_scale=100.0,
                               b_p=b_p, b_c=b_c)
    ag = _CriticStub()
    info = None
    with contextlib.redirect_stdout(buf):
        for step in range(900):
            if step % 5 == 4:
                info = tr.on_block_end(step, -82.0, 19.8, ag) or info
    return info, ag, buf.getvalue()


info_v, ag_v, out_v = _run_verbose(True)
info_q, ag_q, out_q = _run_verbose(False)
assert info_v is not None and info_q is not None
for k in _num:
    assert info_v[k] == info_q[k], (k, info_v[k], info_q[k])
assert abs(float(ag_v.critic_p.fc3.bias.item()) - info_v['b_p']) < 1e-6   # float32 张量
assert 'BiasInitTracker: 触发于' in out_v, out_v[:200]
assert 's_p=1000' in out_v and 's_c=100' in out_v, out_v[:400]
assert out_q == '', 'verbose=False 不应打印任何东西'
# 覆盖值分支 (b_p_raw/b_c_raw 为 None) 也要走一遍 verbose 打印
_, ag_o, out_o = _run_verbose(True, b_p=8000.0, b_c=300.0)
assert abs(float(ag_o.critic_p.fc3.bias.item()) - 8.0) < 1e-9
assert abs(float(ag_o.critic_c.fc3.bias.item()) - 3.0) < 1e-9
assert 'BiasInitTracker: 触发于' in out_o
print('verbose=True 打印分支可执行且数值与 verbose=False 逐项一致 '
      '(双尺度 s_p=1000/s_c=100; 覆盖值分支亦覆盖) ✓')

# 原样复刻生产链路 (traceback 的入口): main 里是 from_args(args) -> on_block_end(...),
# 而 from_args **不传 verbose** ⇒ 取类默认 True ⇒ 生产走的就是上面那个打印分支。
import types as _types

ns = _types.SimpleNamespace(critic_bias_init='estimate', warmup=200, discount=0.99,
                            critic_bias_p=-1.0, critic_bias_c=-1.0,
                            value_scale=1000.0, cost_scale=100.0)
tr_p = hinit.BiasInitTracker.from_args(ns)
assert tr_p.verbose is True, 'from_args 未显式传 verbose ⇒ 应为类默认 True (生产分支)'
buf_p = _io.StringIO()
ag_p = _CriticStub()
with contextlib.redirect_stdout(buf_p):
    for step in range(900):
        if step % 5 == 4:
            tr_p.on_block_end(step, -82.0, 19.8, ag_p)
assert 'BiasInitTracker: 触发于' in buf_p.getvalue(), buf_p.getvalue()[:200]
print('生产链路 from_args(args) → on_block_end (verbose 默认 True) 无异常 ✓')

# ---------------------------------------------------------------- 2) warmup 调度
print('--- 2) warmup 上层调度 ---')
sched = hw.WarmupSchedule(nb_turbine=5, nb_actions=6, seed=7)
a0 = sched.action()
assert np.allclose(a0[:5], 1.0 / 3) and a0[5] == 0.0, a0
seq = [a0] + [sched.action() for _ in range(40)]
A = np.array(seq)
assert A.shape == (41, 6)
assert A[:, :5].min() >= 0.05 - 1e-12 and A[:, :5].max() <= 0.45 + 1e-12
assert np.allclose(A[:, 5], 0.0)
d = np.diff(A[:, :5], axis=0)
# 每块每维要么走满 ±0.05，要么是撞到 [lo,hi] 后截断的步 (|Δ| < 0.05 且新旧值必有一端贴墙)
wall = np.isclose(A[:, :5], 0.05) | np.isclose(A[:, :5], 0.45)
assert np.all(np.abs(d) <= 0.05 + 1e-12)
short = np.abs(d) < 0.05 - 1e-12
n_clipped = int(short.sum())
assert np.all((wall[:-1] | wall[1:])[short]), '截断只应发生在贴墙处'
print('41 块: wt∈[{:.2f},{:.2f}] 动作步长 |Δ|∈[{:.3f},{:.3f}] (贴墙截断 {} 处)'
      .format(A[:, :5].min(), A[:, :5].max(), np.abs(d).min(), np.abs(d).max(), n_clipped))
# 判据 = **纯契约**, 不判"最优在哪": 风场最优是随尾流状态变化的单调上升斜坡
# (最下游机组应推向单机最优 ≈1/3, 上游降载留风), 它是**策略要学的函数**,
# 不是可预先测定的常数 —— 故这里只验中性宽带被真正走到两端 + 起点 = 单机最优点。
a_mean = A[:, :5].mean()
print('warmup wt: E[a]={:.4f} (起点 1/3), a∈[{:.2f},{:.2f}] (中性宽带 [0.05,0.45])'
      .format(a_mean, A[:, :5].min(), A[:, :5].max()))
assert A[:, :5].min() <= 0.10 and A[:, :5].max() >= 0.40, \
    '中性宽带必须被真正走到两端 (seed=7 的 41 块实测 min=0.05 / max=0.45)'
assert 0.20 < a_mean < 0.40, a_mean          # 从 1/3 出发的有界游走, 均值应在带中部附近
# 可复现 + 种子无关的隔离流
assert np.allclose(np.array([hw.WarmupSchedule(5, 6, seed=7).action() for _ in range(1)]), a0)
s_a = np.array([hw.WarmupSchedule(5, 6, seed=11).action() for _ in range(3)])
s_b = np.array([hw.WarmupSchedule(5, 6, seed=11).action() for _ in range(3)])
assert np.allclose(s_a, s_b)
st0 = np.random.get_state()
_ = [hw.WarmupSchedule(5, 6, seed=3).action() for _ in range(50)]
st1 = np.random.get_state()
assert st0[0] == st1[0] and np.array_equal(st0[1], st1[1]) and st0[2:] == st1[2:], \
    'WarmupSchedule 不得消耗全局 numpy RNG 流'
# 信号维 0 ⇔ soc_target 3
assert abs(soc_target_from_signal(0.0) - 3.0) < 1e-12
assert abs(soc_target_from_signal(hw.SIG_VALUE) - 3.0) < 1e-12
# 对照臂
assert hw.WarmupSchedule(5, 6, enabled=False).action() is None
assert hw.WarmupSchedule(5, None, seed=1).action().shape == (5,)
print('种子可复现、不消耗全局 RNG、信号维 0.0 ⇔ soc_target=3.0 ✓')

# ---------------------------------------------------------------- 3) 规则下层
print('--- 3) warmup 规则 BESS ≡ SocTracker.act ---')
rule = hw.WarmupBessRule(soc_target=3.0)
trk = SocTracker(soc_tol=0.0, eta_c=0.98, eta_d=0.98, time=1.0 / 60,
                 rated_power=10.0, pb_cap=None)
rng = np.random.RandomState(0)
n_infeas = 0
mx = 0.0
for _ in range(20000):
    E = rng.uniform(0.0, 6.0)
    if rng.rand() < 0.15:                     # 不可行窗口 (min > max)
        lo_ = rng.uniform(0.0, 4.0)
        hi_ = lo_ - rng.uniform(0.0, 2.0)
        n_infeas += 1
    else:
        lo_ = rng.uniform(-10.0, 0.0)
        hi_ = lo_ + rng.uniform(0.0, 12.0)
    got = rule.act(E, lo_, hi_)
    ref = float(trk.act(E, 3.0, lo_, hi_))
    mx = max(mx, abs(got - ref))
assert mx < 1e-12, mx
print('20000 组随机 (E, 窗口) 对照: max|Δ|={:.1e} (含 {} 组不可行窗口) ✓'.format(mx, n_infeas))
# delta == 0 分支 ⇒ clip(0, 窗口)
assert abs(rule.act(3.0, -1.0, 2.0) - 0.0) < 1e-12
assert abs(rule.act(3.0, 1.5, 4.0) - 1.5) < 1e-12         # 窗口不含 0 ⇒ 被迫吞吐
assert abs(rule.act(3.0, -0.3 / 0.98 / 60 * 0 - 10, 10)) < 1e-12
print('δ=0 → clip(0,窗口)（窗口不含 0 时被迫吞吐）✓')

# ---------------------------------------------------------------- 4) transition 填入
print('--- 4) BESS-RL transition 事后填入 ---')


class _LStub:
    def __init__(self):
        self.calls = []

    def observe_jump(self, s, a, r, c, s2, done):
        self.calls.append((np.asarray(s).copy(), np.asarray(a).copy(),
                           float(r), float(c), np.asarray(s2).copy(), done))
        return len(self.calls)


lstub, stats = _LStub(), hw.BessFillStats()
S_L = np.zeros(5)
a_rule = np.array([-3.0])
hw.fill_bess_transition(lstub, S_L, a_rule, 0.0, 0.0, S_L + 1, True,
                        applied_pb=-3.0, warmup=True, stats=stats)
hw.fill_bess_transition(lstub, S_L, np.array([-4.0]), -12.0, 3.0, S_L + 2, True,
                        applied_pb=-3.5, warmup=True, stats=stats)
hw.fill_bess_transition(lstub, S_L, np.array([2.0]), -5.0, 1.0, S_L + 3, True,
                        applied_pb=2.0, warmup=False, stats=stats)
assert len(lstub.calls) == 3
assert np.allclose(lstub.calls[0][1], a_rule)              # 记的是规则动作, 不是随机动作
assert stats.n_total == 3 and stats.n_warmup == 2, stats.summary()
assert stats.n_mismatch == 1 and abs(stats.max_abs_gap - 0.5) < 1e-12
print(stats.summary())

# ---------------------------------------------------------------- 5) 命令行开关
print('--- 5) 命令行开关 ---')
import argparse  # noqa: E402

ap = argparse.ArgumentParser()
hw.add_warmup_args(ap)
d = ap.parse_args([])
assert d.warmup_sched == 'fixed' and d.warmup_bess == 'rule'
assert abs(d.warmup_soc_target - 3.0) < 1e-12
assert (d.warmup_wt_start, d.warmup_wt_step, d.warmup_wt_lo, d.warmup_wt_hi) == \
    (1.0 / 3, 0.05, 0.05, 0.45)
r = ap.parse_args(['--warmup_sched', 'random', '--warmup_bess', 'random',
                   '--warmup_soc_target', '2.5'])
assert (r.warmup_sched, r.warmup_bess, r.warmup_soc_target) == ('random', 'random', 2.5)


class _HStub:
    nb_turbine, nb_actions = 5, 6


ns = argparse.Namespace(**{**vars(d), 'seed': 3, 'rbb_rated_power': 10.0})
sched_a, rule_a, stats_a = hw.build_from_args(ns, _HStub(), object())
assert rule_a is not None and stats_a is not None and sched_a.action() is not None
ns_r = argparse.Namespace(**{**vars(r), 'seed': 3, 'rbb_rated_power': 10.0})
sched_b, rule_b, stats_b = hw.build_from_args(ns_r, _HStub(), object())
assert rule_b is None and sched_b.action() is None          # random 对照臂
assert hw.build_from_args(ns, _HStub(), None)[2] is None    # 无 L agent ⇒ 无填入统计
print('--warmup_sched/--warmup_bess/--warmup_soc_target/--warmup_wt_* 默认(中性宽带领先验)与对照臂 ✓')

# ---------------------------------------------------------------- 6) 偏航带 (P2)
# 判据 = **纯契约**（起点/界内/步长/两端可达/信号维），与旧环境的第 2 节同构：
#   * 起点 = 单机最优点（旧 a = 1/3 / 新 n = 0）；
#   * 带宽：旧 0.5（[0.05,0.45]）→ 新 2（[−1,+1]）；
#   * 步长与探索噪声 = **带宽的 10%**（旧 0.05 / 新 0.2）—— 这是"按占带宽比例平移"的
#     唯一含义，绝不编码任何"风场最优在哪"（那是策略要学的函数，见 yawflor/README.md §4）。
print('--- 6) yawflor 偏航带契约（纯契约；不断言最优/效率）---')
Y_START, Y_STEP, Y_LO, Y_HI = hw.yaw_band()
band_y = Y_HI - Y_LO
assert (Y_START, Y_LO, Y_HI) == (0.0, -1.0, 1.0), (Y_START, Y_LO, Y_HI)
assert abs(Y_STEP - 0.10 * band_y) < 1e-12, ('步长必须 = 带宽的 10%', Y_STEP, band_y)
assert abs(hw.YAW_WT_NOISE - 0.10 * band_y) < 1e-12, hw.YAW_WT_NOISE
# 旧带必须**未被改动**（9 个既有方法的行为逐位不变）
assert (hw.WT_START, hw.WT_STEP, hw.WT_LO, hw.WT_HI) == (1.0 / 3, 0.05, 0.05, 0.45)

ap_y = argparse.ArgumentParser()
hw.add_warmup_args(ap_y, yaw=True)
dy = ap_y.parse_args([])
assert (dy.warmup_wt_start, dy.warmup_wt_step, dy.warmup_wt_lo, dy.warmup_wt_hi) == \
    (0.0, 0.2, -1.0, 1.0), vars(dy)
# 不带 yaw 的默认值不受影响（9 个 main 的调用点）
ap_o = argparse.ArgumentParser()
hw.add_warmup_args(ap_o)
do = ap_o.parse_args([])
assert (do.warmup_wt_start, do.warmup_wt_step, do.warmup_wt_lo, do.warmup_wt_hi) == \
    (1.0 / 3, 0.05, 0.05, 0.45)

ns_y = argparse.Namespace(**{**vars(dy), 'seed': 7, 'rbb_rated_power': 10.0})
sched_y, rule_y, _ = hw.build_from_args(ns_y, _HStub(), object())
a0y = sched_y.action()
assert a0y.shape == (6,) and np.allclose(a0y[:5], 0.0) and a0y[5] == 0.0, a0y
Ay = np.array([a0y] + [sched_y.action() for _ in range(120)])
assert Ay.shape == (121, 6)
assert Ay[:, :5].min() >= Y_LO - 1e-12 and Ay[:, :5].max() <= Y_HI + 1e-12
assert np.all(Ay[:, 5] == 0.0), '信号维必须恒 0（⇔ soc_target = 3）'
dy_steps = np.diff(Ay[:, :5], axis=0)
assert np.all(np.abs(dy_steps) <= Y_STEP + 1e-12), np.abs(dy_steps).max()
short_y = np.abs(dy_steps) < Y_STEP - 1e-12
wall_y = np.isclose(Ay[:, :5], Y_LO) | np.isclose(Ay[:, :5], Y_HI)
assert np.all((wall_y[:-1] | wall_y[1:])[short_y]), '截断只应发生在贴墙处'
assert Ay[:, :5].min() <= Y_LO + 0.2 and Ay[:, :5].max() >= Y_HI - 0.2, \
    '满带必须被真正走到两端 (seed=7 实测 min=-1 / max=+1)'
print('121 块（偏航带）: n∈[{:.3f},{:.3f}] 步长 |Δ|∈[{:.3f},{:.3f}] 起点 {:.1f} '
      '信号维恒 0 ✓'.format(Ay[:, :5].min(), Ay[:, :5].max(), np.abs(dy_steps).min(),
                            np.abs(dy_steps).max(), a0y[0]))
# 独立 RNG 流（与旧调度同一条约定）
st0 = np.random.get_state()
_ = [hw.WarmupSchedule(5, 6, start=Y_START, step=Y_STEP, lo=Y_LO, hi=Y_HI,
                       seed=3).action() for _ in range(50)]
st1 = np.random.get_state()
assert st0[0] == st1[0] and np.array_equal(st0[1], st1[1]) and st0[2:] == st1[2:]

# ---------------------------------------------------------------- 7) 越界罚等值 (P2)
# "满带越界 = 200 元"是**经济口径**：旧环境 400 元/单位 × 带宽 0.5 = 200；
# 新环境带宽 2 ⇒ 系数必须 ÷4 = 100。等比例越界（带宽的 25%）罚也必须相同。
print('--- 7) 满带越界罚 = 200 元（与旧环境等值）---')
import td3per.td3per_myfunc as old_mf          # noqa: E402
import yawflor.yawflor_myfunc as new_mf        # noqa: E402

assert old_mf.PENALTY_NN_H_COEF == 400.0, old_mf.PENALTY_NN_H_COEF
assert new_mf.PENALTY_NN_H_COEF == 100.0, new_mf.PENALTY_NN_H_COEF
assert abs(new_mf.PENALTY_NN_H_COEF * 2.0 - 200.0) < 1e-12
# 满带越界（越界量恰好 = 带宽）
_a, p_old_full = old_mf.CheckAction(np.array([0.5 + 0.5, 0., 0., 0., 0., 0.]))
_a, p_new_full = new_mf.CheckAction(np.array([1.0 + 2.0, 0., 0., 0., 0., 0.]))
assert p_old_full == 200.0 and p_new_full == 200.0, (p_old_full, p_new_full)
# 等比越界（带宽的 25%）= 50 元 两臂相同；下界越界同理（旧带宽 0.5 ⇒ −0.125；新带宽 2 ⇒ −1.5）
_a, p_old_q = old_mf.CheckAction(np.array([0., 0., 0., 0., -0.125, 0.]))
_a, p_new_q = new_mf.CheckAction(np.array([0., 0., 0., 0., -1.5, 0.]))
assert abs(p_old_q - 50.0) < 1e-12 and abs(p_new_q - 50.0) < 1e-12, (p_old_q, p_new_q)
# 夹带：偏航维必须夹到 [−1,+1]（旧环境夹到 [0,0.5]）
ap_new, _p = new_mf.CheckAction(np.array([5.0, -5.0, 0.0, 0.0, 0.0, 0.9]))
assert np.allclose(ap_new, np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.5])), ap_new
# 下层 1 维动作**保持旧口径**（±10 MW；200 元/单位，不进 r2）
assert abs(new_mf.PENALTY_NN_L_COEF - old_mf.PENALTY_NN_L_COEF) < 1e-12
al, pl = new_mf.CheckAction(np.array([12.0]))
assert np.allclose(al, np.array([10.0])) and pl == 400.0, (al, pl)
# --viol_derate 的越界量带也必须跟着动作带换（默认关 ⇒ 恒 1.0）
assert abs(new_mf.violation_derate(np.array([1.5, 0., 0., 0., 0.]), 1.0) -
           np.exp(-1.0 * 0.5)) < 1e-12
assert abs(new_mf.violation_derate(np.array([1.0, 0., 0., 0., 0.]), 1.0) - 1.0) < 1e-12
assert abs(new_mf.violation_derate(np.array([9.9, 0., 0., 0., 0.]), 0.0) - 1.0) < 1e-12
print('旧 400×0.5 = 新 100×2 = 200 元；等比 50 元；夹带 [−1,+1]；下层 ±10 不变 ✓')

print('TEST OK')
