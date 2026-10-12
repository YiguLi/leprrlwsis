# -*- coding: utf-8 -*-
r"""受控定位：H 层 buffer 里存的 cost 到底是什么？

做法：构造真实 `td3per.td3per_ddpg.SP_TD3`（小网络、桩 args），
  * 记录 `observe_jump(s, a, r, c, s2, done)` 收到的 `c`；
  * 直接读 `agent.memory` 里存的 cost 列；
  * 跑一次 `update_policy`，读 `update_stats[-1]['cost_mean']`。
三者对齐 ⇒ 就能判定"日志的块级 c1 之和"与"buffer 的 cost"是否同一个量。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402


class _Args(object):
    lam = 0.0967
    PER_beta = 0.4
    discount = 0.95
    tau = 0.001
    batch_size = 8
    rmsize = 500
    policy_noise = 0.1
    noise_clip = 0.3
    policy_delay = 2
    wt_noise = 0.05
    wt_noise_floor = 0.01
    sig_noise_min = 0.01
    sig_noise_max = 0.02
    cost_critic_mode = 'max'
    hidden1 = 16
    hidden2 = 100
    init_w = 3e-3
    rate = 0.01
    prate = 0.0001
    value_scale = 1000.0
    cost_scale = 100.0
    cuda = False
    seed = 0
    utd = 1
    PER_alpha = 0.6
    bsize = 8
    epsilon = 4000
    ou_theta = 0.15
    ou_mu = 0.0
    ou_sigma = 0.2
    lr_lam = 0.01
    cost_single_limit = 15.0
    PER_epsilon = 0.01
    beta_increment_per_sampling = 1e-5
    nb_turbine = 5


import td3per.td3per_ddpg as D  # noqa: E402

ag = D.SP_TD3(nb_states=4, nb_actions=5, args=_Args(),
              nstep=5, target_agg='min', record_update_stats=True)
ag.is_training = True
ag.value_scale = 1000.0
ag.cost_scale = 100.0

# ---- 灌入已知的 (r, c) 对：r=1000、块级 c1 之和 = 5.7（与 run57 实测一致）----
recv = []
orig = ag.observe_jump


def spy(s_t, a_t, r_t, c_t, s_t1, done):
    recv.append(float(c_t))
    return orig(s_t, a_t, r_t, c_t, s_t1, done)


ag.observe_jump = spy
rng = np.random.RandomState(0)

# ---- 直接监视 _emit_nstep 的输入输出 ----
calls = []
_orig_emit = ag._emit_nstep


def _emit_spy(start, n_used):
    seg = ag._pending[start:start + n_used]
    nxt = ag.memory._next_idx
    _orig_emit(start, n_used)
    calls.append((start, n_used,
                  [float(x[2]) for x in seg], [float(x[3]) for x in seg],
                  float(ag.memory._storage[nxt][2]), float(ag.memory._storage[nxt][3])))
    return None


ag._emit_nstep = _emit_spy

N = 40
for i in range(N):
    s = rng.uniform(0, 1, 4).astype(np.float32)
    s2 = rng.uniform(0, 1, 4).astype(np.float32)
    a = rng.uniform(0, 0.5, 5).astype(np.float32)
    ag.observe_jump(s, a, 1000.0, 5.7, s2, False)

print('灌入 N=%d 条 (r=1000, c=5.7)' % N)
print('_emit_nstep 被调 %d 次；前 3 次:' % len(calls))
for st, nu, rs, cs, sr, sc in calls[:3]:
    print('  start=%d n_used=%d  窗内 r=%s  c=%s  ⇒ 写入 r_sum=%.4f c_sum=%.4f'
          % (st, nu, np.round(rs, 2), np.round(cs, 3), sr, sc))
if calls:
    print('  ⇒ 窗内 5 条 r 之和 %.1f（应 5000）与写入值 %.1f 的比值 %.4f'
          % (sum(calls[0][2]), calls[0][4], calls[0][4] / sum(calls[0][2])))
print('灌入 N=%d 条' % N)
mem = ag.memory
print('memory 长度 = %d（rmsize=%d，未满 ⇒ 不覆盖）' % (len(mem), _Args.rmsize))
for name in ('r1', 'c1', 'cost', 'c'):
    if hasattr(mem, name):
        v = np.asarray(getattr(mem, name), dtype=float).reshape(-1)
        print('  memory.%s: len %d  mean %.4f  first %.4f' % (name, v.size, v.mean(), v[0]))
attr = [a for a in dir(mem) if not a.startswith('_')]
print('memory 公开属性: %s' % attr)

for _ in range(12):
    ag.update_policy()
st = ag.update_stats[-1]
print('\nupdate_stats[-1]:')
for k in ('reward_mean', 'cost_mean', 'weight_mean', 'q_p_mean', 'q_c_mean'):
    print('  %-12s %14.4f' % (k, st[k]))

mem = ag.memory
print('\n---- 直接查 buffer 内部 ----')
rew = np.array([d[2] for d in mem._storage], dtype=float)
cos = np.array([d[3] for d in mem._storage], dtype=float)
print('  _storage 里 reward: mean %.4f  first %s' % (rew.mean(), rew[:4]))
print('  _storage 里 cost  : mean %.4f  first %s' % (cos.mean(), cos[:4]))
s = mem.sample(_Args.bsize, 0.4, 1e-5)
print('  sample() 第 4 个返回（cost）的前 8 个: %s' % np.round(np.asarray(s[3]).reshape(-1)[:8], 4))
print('  sample() 第 2 个返回（action）形状 %s' % (np.asarray(s[1]).shape,))
print('  sample() 第 3 个返回（reward）前 4 个: %s' % np.round(np.asarray(s[2]).reshape(-1)[:4], 3))
print('\n判定：若 _storage 的 cost 已经是 25 级，则问题在上游（main 侧或 nstep 折叠）；')
print('      若 _storage 是 5.7 而 sample() 返回 ~25，则问题在 buffer 的取数。')
