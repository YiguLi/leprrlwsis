#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""warmup 调度 + critic bias 触发修复 的单元测试 —— 不需要 MATLAB。

验证五件事:
  1. `first_trigger_step` / `BiasInitTracker`：**修复了旧触发条件**
     （`step == warmup` 落在块首 ⇒ 默认 warmup=200 永不触发），且在首次策略更新前生效
  2. `WarmupSchedule`：起始点、步长、clip 带、可复现性、信号维 = 0（⇔ soc_target = 3）
  3. `WarmupBessRule`：与各项目 `SocTracker.act` **逐位同语义**
  4. `fill_bess_transition`：真实 transition 入池 + 计数（BESS-RL 项目的"事后填入"）
  5. 旧触发条件的反例（warmup=200 下 `step == warmup` 与块末永不同时成立）
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
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

# ---------------------------------------------------------------- 2) warmup 调度
print('--- 2) warmup 上层调度 ---')
sched = hw.WarmupSchedule(nb_turbine=5, nb_actions=6, seed=7)
a0 = sched.action()
assert np.allclose(a0[:5], 0.33) and a0[5] == 0.0, a0
seq = [a0] + [sched.action() for _ in range(40)]
A = np.array(seq)
assert A.shape == (41, 6)
assert A[:, :5].min() >= 0.20 - 1e-12 and A[:, :5].max() <= 0.45 + 1e-12
assert np.allclose(A[:, 5], 0.0)
d = np.diff(A[:, :5], axis=0)
# 每块每维要么走满 ±0.05，要么是撞到 [lo,hi] 后截断的步 (|Δ| < 0.05 且新旧值必有一端贴墙)
wall = np.isclose(A[:, :5], 0.20) | np.isclose(A[:, :5], 0.45)
assert np.all(np.abs(d) <= 0.05 + 1e-12)
short = np.abs(d) < 0.05 - 1e-12
n_clipped = int(short.sum())
assert np.all((wall[:-1] | wall[1:])[short]), '截断只应发生在贴墙处'
print('41 块: wt∈[{:.2f},{:.2f}] 动作步长 |Δ|∈[{:.3f},{:.3f}] (贴墙截断 {} 处)'
      .format(A[:, :5].min(), A[:, :5].max(), np.abs(d).min(), np.abs(d).max(), n_clipped))
cp = 4 * A[:, :5] * (1 - A[:, :5]) ** 2
print('E[ΣCp]={:.3f} = 最优 {:.3f} 的 {:.1f}%'.format(cp.sum(1).mean(), 5 * 16 / 27,
                                                     100 * cp.sum(1).mean() / (5 * 16 / 27)))
assert cp.sum(1).mean() > 2.75, cp.sum(1).mean()
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

print('TEST OK')
