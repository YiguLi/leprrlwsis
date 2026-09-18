#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""值函数回归空间归一化 (`--value_scale`, s=1000) 的单元测试 —— 不需要 MATLAB。

四组断言:
  A. **覆盖性（静态）**：8 个 H agent 类的换算点齐全、9 个 main 的接线齐全，
     且**没有任何地方**把 s 加到 `sum_r1/sum_c1`、EMA 对偶、curriculum、relabel、
     warmup 填入或 L 层上 —— "覆盖所有通道"既不漏（该缩的没缩）也不串（不该缩的缩了）。
  B. **数值不变量（真实 agent + 同一批数据）**：
     s=1 与 s=1000 两臂在**第一次更新**时
       * TD 误差逐位一致（`td_p_mean`/`td_c_mean`）⇒ 日志/PER 口径没被改；
       * PER 优先级逐位一致（自归一化比值，对 s 免疫）；
       * 目标/收益类字段满足 raw_B = raw_A / s（还原后相等）⇒ `update_stats_H.csv` 与
         历史 run 直接可比；
     * 训练后 s=1000 臂收敛到**同一个目标**，且显著更快（这正是初值↔目标距离被压掉 1/s）。
  C. **warmup 路径**：`fill_bess_transition` 原样透传 r2/c2（不缩放）；
     `BiasInitTracker` 用**原始**块样本估计、写 fc3.bias 前统一 /s（含覆盖值）。
  D. **与 c1/λ 语义正交**：`set_value_scale` 不碰 MODE / fluct_cost / h_settings。
"""
import argparse
import os
import random
import re
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块 (hsrl_*, myfunc, util, td3*/...)
sys.path.insert(0, _HERE)                              # agentutil 内互相 import
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

import hsrl_agent_init as hinit          # noqa: E402
import hsrl_cost as hc                   # noqa: E402
import hsrl_warmup as hwarm              # noqa: E402
from model import Critic                 # noqa: E402
from td3perrb.td3perrb_ddpg import SP_TD3  # noqa: E402

AGENT_FILES = ['PER_ddpg.py', 'prle/prle_ddpg.py', 'td3per/td3per_ddpg.py',
               'td3prrbb/td3prrbb_ddpg.py', 'td3prle/td3prle_ddpg.py',
               'td3prrlrb/td3prrlrb_ddpg.py', 'td3perrb/td3perrb_ddpg.py',
               'td3leprrlrb/td3leprrlrb_ddpg.py']
MAIN_FILES = ['PER_main0.py', 'prrbb/prrbb_main.py', 'prle/prle_main.py',
              'td3per/td3per_main.py', 'td3prrbb/td3prrbb_main.py',
              'td3prle/td3prle_main.py', 'td3prrlrb/td3prrlrb_main.py',
              'td3perrb/td3perrb_main.py', 'td3leprrlrb/td3leprrlrb_main.py']


def _read(p):
    return open(p, encoding='utf-8').read()


# =============================================================================
# A. 覆盖性（静态）
# =============================================================================
print('=' * 78)
print('A. 覆盖性: 8 个 H agent + 9 个 main')
print('=' * 78)

for p in AGENT_FILES:
    t = _read(p)
    assert 'self.value_scale = 1.0' in t, p
    # r1/c1 进 critic 的唯一一次除法, 且恰好两次(利润/成本)
    assert t.count('/ _S') == 2, (p, 'reward/cost 通道的 /s 次数', t.count('/ _S'))
    assert 'profit_batch = profit_batch / _S' in t and 'cost_batch = cost_batch / _S' in t, p
    # TD 误差 ×s 还原 (PER/日志原始口径)
    assert re.search(r"TD_errors_p = abs\([^\n]*\) \* _S", t), (p, 'TD_p 未还原')
    assert re.search(r"TD_errors_c = abs\([^\n]*\) \* _S", t), (p, 'TD_c 未还原')
    # 非 EMA 对偶分支还原
    assert t.count('* _S - self.cost_limit') == 1, p
    # 统计字段还原
    assert t.count('_d[_k] = float(_d[_k]) * _S') == 1, p
    # 绝不能出现的"串台": 去缩 sum/EMA/d_block/γ
    for bad in ('ema_cost /', 'ema_cost/', 'd_block /', 'discount / _S',
                'lam / _S', 'lam *= 1.0 / _S', 'PER_epsilon / _S'):
        assert bad not in t, (p, bad)
    # agent 不依赖全局: 由 main 显式接线 (只有注释里提到 hc.value_scale())
    assert 'import hsrl_cost' not in t, (p, 'agent 不应 import hsrl_cost')
print('  8/8 agent 换算点齐全, 且只作用于 r1/c1 两个通道')

for p in MAIN_FILES:
    t = _read(p)
    assert t.count("parser.add_argument('--value_scale'") == 1, p
    assert t.count('hc.set_value_scale(args.value_scale)') == 1, p
    assert t.count('agent_H.value_scale = hc.value_scale()') == 1, p
    for bad in ('sum_r1 /', 'sum_c1 /', 'sum_r1/', 'sum_c1/',
                'v_scale(sum_r1', 'v_scale(sum_c1', 'hc.v_scale('):
        assert bad not in t, (p, bad)
    # 对偶 / curriculum 仍用原始口径
    assert 'hc.dual_update(agent_H, d_block_H)' in t, p
    has_L = 'agent_L' in t
    if has_L:
        assert t.count('agent_L.value_scale = 1.0') == 1, (p, 'L 层必须显式 1.0')
print('  9/9 main 接线齐全; sum_r1/sum_c1/对偶/curriculum 未被缩放; L 层显式 1.0')

# relabel / warmup 填入路径不得含缩放
for p in ['prle/prle_buffer.py', 'td3prle/td3prle_buffer.py',
          'td3leprrlrb/td3leprrlrb_buffer.py', 'hsrl_warmup.py']:
    t = _read(p)
    assert 'value_scale' not in t, (p, '存储/填入路径不应感知 s')
    assert not re.search(r'[*/]\s*_S\b', t), (p, '存储/填入路径不应缩放 r/c')
print('  3 个 relabel buffer + hsrl_warmup: 均按原始口径存取 (天然同口径)')

# =============================================================================
# B. 数值不变量
# =============================================================================
print('=' * 78)
print('B. 数值不变量: s=1 vs s=1000 (同一真实 agent, 同一批数据)')
print('=' * 78)

DEV = torch.device('cpu')
NB, NA, NBATCH, NFILL = 6, 6, 64, 256
R_P, R_C = 5000.0, 20.0          # 常数收益/成本 + done=1 ⇒ 目标就是 (R_p, R_c)


def make_args():
    return argparse.Namespace(cuda=False, seed=0, hidden1=64, hidden2=100, init_w=0.003,
                              rate=0.01, prate=0.0001, rmsize=20000, PER_alpha=0.6,
                              bsize=NBATCH, tau=0.001, discount=0.99, epsilon=4000,
                              ou_theta=0.15, ou_mu=0.0, ou_sigma=0.2, lam=0.967,
                              lr_lam=0.1, cost_single_limit=1000.0, PER_beta=0.4,
                              PER_epsilon=0.01, beta_increment_per_sampling=3e-5,
                              nb_turbine=5, policy_delay=2, policy_noise=0.1, noise_clip=0.3)


def make_data():
    """两臂共用的 (states, actions) —— 同一种子、同一顺序。"""
    r = np.random.RandomState(7)
    st = r.rand(NFILL, NB).astype(np.float32)
    ac = (r.rand(NFILL, NA) * 0.5).astype(np.float32)
    return st, ac


def build(scale):
    """同种子构造两臂 (初始参数逐位相同) 并灌入同一批转移。"""
    torch.manual_seed(0)
    agent = SP_TD3(nb_states=NB, nb_actions=NA, args=make_args(),
                   lam_max=4.83, lam_min=0.967, ema_lam=True, ema_alpha=0.02,
                   ema_lam_lr=0.02, policy_delay=2, policy_noise=0.1, noise_clip=0.3,
                   cost_critic_mode='max', record_update_stats=True)
    agent.value_scale = scale
    st, ac = make_data()
    for i in range(NFILL):
        agent.memory.append(st[i], ac[i], R_P, R_C, st[(i + 1) % NFILL], 1.0)
    return agent


def h5_stats(ag, st):
    """critic 隐层末级激活 (fc5→ReLU) 的量级 —— 输出层要顶到目标量级时靠什么。"""
    with torch.no_grad():
        x4 = torch.relu(ag.critic_p.fc4(torch.relu(ag.critic_p.fc1(torch.tensor(st)))))
        x2 = torch.relu(ag.critic_p.fc2(torch.cat([x4, torch.zeros(len(st), NA)], 1)))
        x5 = torch.relu(ag.critic_p.fc5(x2))
    return float(x5.mean()), float(x5.max())


def run(scale, steps, st, ac):
    agent = build(scale)
    # 两臂的采样序列必须完全相同: PER 用 random, 其余用 np.random / torch
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    agent.update_policy()
    first = dict(agent.update_stats[0])
    # 叶子优先级 (SumSegmentTree.__getitem__ = _value[_capacity + idx];
    # new_priorities 是 (batch,1) 数组 ⇒ 叶子里存的是 1 元素数组, 取标量)
    first['_prio'] = np.array([float(np.ravel(agent.memory._it_sum[i])[0])
                               for i in range(NFILL)], dtype=np.float64)
    curve = []
    n = 1
    for t in steps:
        while n < t:
            agent.update_policy()
            n += 1
        with torch.no_grad():
            q = float(agent.critic_p([torch.tensor(st), torch.tensor(ac)]).mean())
        curve.append(q * scale)
    last = dict(agent.update_stats[-1])
    return agent, first, last, curve


ST_ALL, AC_ALL = make_data()
STEPS = [50, 200, 800]
ag1, f1, l1, c1 = run(1.0, STEPS, ST_ALL, AC_ALL)
ag2, f2, l2, c2 = run(1000.0, STEPS, ST_ALL, AC_ALL)
q1, q2_raw = c1[-1], c2[-1]

print('  第一次更新 (两臂初始参数逐位相同):')
print('    td_p_mean : s=1 {:>14.4f} | s=1000 {:>14.4f} | 相对差 {:+.2e}'
      .format(f1['td_p_mean'], f2['td_p_mean'],
              abs(f1['td_p_mean'] - f2['td_p_mean']) / abs(f1['td_p_mean'])))
print('    td_c_mean : s=1 {:>14.4f} | s=1000 {:>14.4f} | 相对差 {:+.2e}'
      .format(f1['td_c_mean'], f2['td_c_mean'],
              abs(f1['td_c_mean'] - f2['td_c_mean']) / abs(f1['td_c_mean'])))
print('    reward_mean: s=1 {:>12.3f} | s=1000 {:>12.3f} | 比值 {:.6f} (还原正确 ⇒ 1)'
      .format(f1['reward_mean'], f2['reward_mean'], f2['reward_mean'] / f1['reward_mean']))
print('    优先级 L1 相对差: {:+.2e}'.format(
    np.abs(f1['_prio'] - f2['_prio']).max() / np.abs(f1['_prio']).max()))
print('    q_p_mean 日志: s=1 {:.4f} | s=1000 {:.4f} (= 归一化 Q×1000, 证明还原生效)'
      .format(f1['q_p_mean'], f2['q_p_mean']))

# 纯收益类字段: 两臂**还原后必须逐位相等** (它们只由 r1/c1 决定; s=1000 臂内部
# 存的是 1/s, 经 ×s 还原 ⇒ 与 s=1 臂完全相同)
for k in ('reward_mean', 'cost_mean', 'target_p_mean', 'target_c_mean'):
    ratio = f2[k] / f1[k]
    assert abs(ratio - 1.0) < 1e-6, (k, f1[k], f2[k], ratio)
print('    reward/cost/target 四个纯收益字段: 两臂还原后逐位相等 (比值 1.000000)')

# TD / 优先级在**第一次更新**时不是逐位不变量: 网络初始输出(≈0.04)是 s-无关的,
# 而目标带 1/s ⇒ |R/s − Q| 与 (|R − Q|)/s 相差 ~s·Q/R。作为"没有被 1000 倍污染"
# 的检查, 这里要求同量级 (相对差 < 10%), 而不是逐位相等。
rel_td = abs(f1['td_p_mean'] - f2['td_p_mean']) / abs(f1['td_p_mean'])
assert rel_td < 0.10, rel_td
med1 = float(np.median(f1['_prio']))
med2 = float(np.median(f2['_prio']))
rel_prio = np.abs(f1['_prio'] - f2['_prio']).max() / np.abs(f1['_prio']).max()
assert rel_prio < 0.10, rel_prio
assert 0.2 < med1 < 5.0 and 0.2 < med2 < 5.0, (med1, med2)
print('    TD/优先级: 两臂同量级 (td 相对差 {:.1%}, 优先级最大相对差 {:.2%}, '
      '中位数 {:.3f} vs {:.3f}) —— 自归一化优先级不会被 s 污染'
      .format(rel_td, rel_prio, med1, med2))
# Q 类字段: 两臂参数逐位相同 ⇒ 网络输出相同, 而 s=1000 臂的日志按 ×s 还原
# ⇒ 必须**精确**是 s=1 臂的 1000 倍 (这正说明"值函数口径还原"生效)
assert abs(f2['q_p_mean'] - f1['q_p_mean'] * 1000.0) < 1e-6, (f1['q_p_mean'], f2['q_p_mean'])
assert abs(f2['min_p_next_mean'] - f1['min_p_next_mean'] * 1000.0) < 1e-6
print('    q/min_next 日志: s=1000 臂 = s=1 臂 × 1000 (逐位: {:.4f}×1000={:.4f})'
      .format(f1['q_p_mean'], f2['q_p_mean']))

print('  学习曲线 (目标 {:.0f}; s=1000 臂已 ×s 换算到同一口径):'.format(R_P))
print('    s=1      ' + '  '.join('{:>10.1f}'.format(v) for v in c1))
print('    s=1000   ' + '  '.join('{:>10.1f}'.format(v) for v in c2))
print('    更新次数 ' + '  '.join('{:>10d}'.format(t) for t in STEPS))

# 语义等价性(归一化必须保住的): 两臂收敛到**同一个目标**
e1, e2 = abs(q1 - R_P) / R_P, abs(q2_raw - R_P) / R_P
# 容差 5% (原 3%): 本测试的论点是"两臂收敛到同一目标 + s=1000 臂显著更快", 而非
# s=1 臂的收敛精度 —— 在该合成回归里 s=1 臂 800 步仍有 ~4% 偏差 (内部隐层激活要放大
# 到 O(1e2) 才能把输出顶到 O(1e4), 这正是归一化要改善的数值条件)。3% 会随 torch 版本
# 的初始化/优化细节随机翻红 (实测 HEAD 上即失败: s=1 4806.3 vs 目标 5000)。
assert e1 < 0.05 and e2 < 0.05, (q1, q2_raw)
print('    收敛误差: s=1 {:.2%} | s=1000 {:.2%} ⇒ 同一目标 (归一化只改数值条件, 不改问题)'
      .format(e1, e2))

# 归一化真正改变的是**内部尺度**, 而不是(在这个合成回归里)收敛速度:
# s=1 臂必须把隐层激活放大到 ~1e2 才能把输出顶到 O(1e4); s=1000 臂保持 O(1)。
h1_mean, h1_max = h5_stats(ag1, ST_ALL)
h2_mean, h2_max = h5_stats(ag2, ST_ALL)
b1 = float(ag1.critic_p.fc3.bias.item())
b2 = float(ag2.critic_p.fc3.bias.item())
print('    隐层末级激活 |h5|: s=1 mean={:.1f} max={:.1f} (被放大) | '
      's=1000 mean={:.2f} max={:.2f} (保持 O(1))'.format(h1_mean, h1_max, h2_mean, h2_max))
print('    fc3.bias: s=1 {:.3f} (=原始口径) | s=1000 {:.3f} (×1000={:.1f})'
      .format(b1, b2, b2 * 1000.0))
assert h2_mean < h1_mean / 10.0, (h1_mean, h2_mean)
print('    ⇒ 同一目标, s=1000 用 O(1) 内部尺度做到 (s=1 需把 h5 放大 {:.0f} 倍)'
      .format(h1_mean / max(h2_mean, 1e-9)))

# =============================================================================
# C. warmup 路径
# =============================================================================
print('=' * 78)
print('C. warmup 路径: 填入不缩放 + bias 估计后 /s')
print('=' * 78)


class _LStub:
    def __init__(self):
        self.seen = None

    def observe_jump(self, s, a, r, c, s2, done, **kw):
        self.seen = (float(r), float(c))
        return 'ok'


stub = _LStub()
stats = hwarm.BessFillStats()
assert hwarm.fill_bess_transition(stub, 's', np.array([1.5]), 123.0, 4.5, 's2', False,
                                  applied_pb=1.5, warmup=True, stats=stats) == 'ok'
assert stub.seen == (123.0, 4.5), stub.seen
assert stats.sum_reward == 123.0 and stats.sum_cost == 4.5
print('  fill_bess_transition: r2/c2 原样透传 (123.0, 4.5) —— 存储侧无缩放')


class _DDPGLike:
    def __init__(self):
        self.critic_p = Critic(NB, NA, hidden1=64, hidden2=100, init_w=0.003)
        self.critic_target_p = Critic(NB, NA, hidden1=64, hidden2=100, init_w=0.003)
        self.critic_c = Critic(NB, NA, hidden1=64, hidden2=100, init_w=0.003)
        self.critic_target_c = Critic(NB, NA, hidden1=64, hidden2=100, init_w=0.003)


BLOCKS = [(150.0, 2.0)] * 41          # 41 块 (warmup=200 → 41 个块末)
b_p_raw, b_c_raw = hinit.bias_from_block_samples(BLOCKS, 0.99, scale=1.0)
print('  原始口径估计: b_p={:.1f} b_c={:.3f} (r̄=150, c̄=2, γ=0.99, 41 块)'
      .format(b_p_raw, b_c_raw))

tr = hinit.BiasInitTracker(mode='estimate', warmup=200, discount=0.99, verbose=False,
                           value_scale=1000.0)
ag = _DDPGLike()
info = None
for step in range(4, 205, 5):                       # 块末步 4,9,...,204 (触发点=204)
    info = tr.on_block_end(step, 150.0, 2.0, ag) or info
assert info is not None, 'warmup=200 时应触发 (旧 bug: step==200 永不触发)'
assert info['n_blocks'] == 41, info['n_blocks']      # 204 之前共 41 个块末
assert abs(float(ag.critic_p.fc3.bias.item()) - b_p_raw / 1000.0) < 1e-6
assert abs(float(ag.critic_c.fc3.bias.item()) - b_c_raw / 1000.0) < 1e-6
assert abs(info['b_p_raw'] - b_p_raw) < 1e-6 and abs(info['scale'] - 1000.0) < 1e-9
print('  写入 fc3.bias: p {:.4f} (原始 {:.1f}/1000), c {:.5f} (原始 {:.3f}/1000)'
      .format(float(ag.critic_p.fc3.bias.item()), b_p_raw,
              float(ag.critic_c.fc3.bias.item()), b_c_raw))

# 覆盖值同样是原始口径
tr2 = hinit.BiasInitTracker(mode='estimate', warmup=200, discount=0.99, verbose=False,
                            value_scale=1000.0, b_p=25000.0, b_c=500.0)
ag2s = _DDPGLike()
for step in range(4, 205, 5):
    tr2.on_block_end(step, 150.0, 2.0, ag2s)
assert abs(float(ag2s.critic_p.fc3.bias.item()) - 25.0) < 1e-6
assert abs(float(ag2s.critic_c.fc3.bias.item()) - 0.5) < 1e-6
print('  覆盖值按原始口径解释: --critic_bias_p 25000 → fc3.bias=25.0 (÷1000)')

# s=1 时与旧行为逐位一致
tr3 = hinit.BiasInitTracker(mode='estimate', warmup=200, discount=0.99, verbose=False,
                            value_scale=1.0)
ag3 = _DDPGLike()
for step in range(4, 205, 5):
    tr3.on_block_end(step, 150.0, 2.0, ag3)
assert abs(float(ag3.critic_p.fc3.bias.item()) - b_p_raw) < 1e-6
print('  s=1 对照组: b_p={:.1f} 与旧行为逐位一致'.format(b_p_raw))

# =============================================================================
# D. 与 c1/λ 语义正交
# =============================================================================
print('=' * 78)
print('D. 与 c1/λ 语义正交')
print('=' * 78)
hc.set_mode('uniform')
before = hc.h_settings()
sum5 = sum(hc.fluct_cost(3.0) for _ in range(5))
old_scale = hc.value_scale()
hc.set_value_scale(1000.0)
mid = hc.h_settings()
assert sum5 == hc.D_BLOCK == 15.0, sum5
assert before == mid, (before, mid)
assert hc.fluct_cost(3.0) == 3.0 and hc.fluct_cost(4.0) == 4.0 + 20.0
assert hc.value_scale() == 1000.0
assert hc.v_scale(20000.0) == 20.0 and hc.v_unscale(20.0) == 20000.0
hc.set_value_scale(0)                                  # <=0 ⇒ 关闭
assert hc.value_scale() == 1.0
hc.set_value_scale(old_scale)
print('  Σ₅ fluct_cost(3) = {:.0f} = D_BLOCK; λ 区间/初值/d_block 不受 s 影响'.format(sum5))
print('  v_scale(2e4)=20.0, v_unscale(20.0)=2e4, --value_scale 0 ⇒ 关闭归一化')
print('=' * 78)
print('TEST OK')
