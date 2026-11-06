# -*- coding: utf-8 -*-
r"""`update_stats_H` 到底有没有被放大？**受控实验定死**（只读，需 torch，不启 MATLAB）。

已知事实（本会话已用真引擎/日志双向验证）：
  * 逐步日志的**块级 r1 之和** = 216.8 元（episode 级 Σr1 ≡ episode_reward，逐集 0.00 误差）
  * buffer 里存的是**跨 5 个连续块的折叠奖励** r_n = Σ_{k<5}γ^k·r_blk ⇒ 均值 ≈ 981 元
  * `update_stats_H.reward_mean` 的实测均值 = 1023（全段）/ 1085（末 300 行）

问题：`reward_mean` 是 r_n 本身（未放大），还是 r_n/s_p（被除以 1000）？
做法：构造**真实 SP_TD3**，灌入已知的块级奖励，读 `update_stats` —— 三种 s_p 配置各跑一遍。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

GAMMA, NSTEP = 0.95, 5
FOLD = float(np.sum(GAMMA ** np.arange(NSTEP)))


class Args(object):
    lam = 0.0967
    PER_beta = 0.4
    discount = GAMMA
    tau = 0.001
    bsize = 256
    rmsize = 2000
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
    cuda = False
    seed = 0
    PER_alpha = 0.6
    epsilon = 4000
    ou_theta = 0.15
    ou_mu = 0.0
    ou_sigma = 0.2
    lr_lam = 0.01
    cost_single_limit = 15.0
    PER_epsilon = 0.01
    beta_increment_per_sampling = 1e-5
    nb_turbine = 5


from td3per.td3per_ddpg import SP_TD3  # noqa: E402


def run(sp, sc, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ag = SP_TD3(nb_states=4, nb_actions=5, args=Args(), nstep=NSTEP,
                target_agg='min', record_update_stats=True)
    ag.is_training = True
    if sp is not None:
        ag.value_scale = sp
    if sc is not None:
        ag.cost_scale = sc
    rng = np.random.RandomState(7)
    N = 2000
    # 灌入**块级**奖励（均值 216.8 = 实测）与块级成本（均值 5.74 = 实测）
    R = rng.normal(216.8, 225.6, N)
    C = rng.gamma(2.0, 2.85, N)
    for i in range(N):
        s = rng.uniform(0, 1, 4).astype(np.float32)
        s2 = rng.uniform(0, 1, 4).astype(np.float32)
        a = rng.uniform(0, 0.5, 5).astype(np.float32)
        ag.observe_jump(s, a, float(R[i]), float(C[i]), s2, False)
    stored_r = np.array([d[2] for d in ag.memory._storage], dtype=float)
    stored_c = np.array([d[3] for d in ag.memory._storage], dtype=float)
    for _ in range(40):
        ag.update_policy()
    # 用**多次更新的均值**压掉 PER 加权采样的抖动
    rm = float(np.mean([s['reward_mean'] for s in ag.update_stats]))
    cm = float(np.mean([s['cost_mean'] for s in ag.update_stats]))
    return dict(sp=getattr(ag, 'value_scale', None), sc=getattr(ag, 'cost_scale', None),
                stored_r=stored_r.mean(), stored_c=stored_c.mean(),
                reward_mean=rm, cost_mean=cm)


print('=' * 100)
print('受控实验：灌入块级奖励 mean %.1f（实测值）' % 216.8)
print('  预期：buffer 存折叠值 = %.1f × Σγ^k(%.4f) = %.1f' % (216.8, FOLD, 216.8 * FOLD))
print('=' * 100)
print('  %-14s %-14s %12s %12s %14s %12s' %
      ('value_scale', 'cost_scale', '存r(折叠)', '存c(折叠)', 'reward_mean', 'cost_mean'))
for sp, sc, lab in ((1000.0, 100.0, '真源默认'), (1000.0, 1000.0, 's_c=s_p'),
                    (1.0, 1.0, '全关'),
                    (0.001, 0.001, '**探针** s=1e-3'),
                    (0.001, 1.0, '**探针** s_p=1e-3, s_c=1')):
    d = run(sp, sc)
    print('  %-14s %-14s %12.1f %12.3f %14.4f %12.4f   [%s]'
          % (d['sp'], d['sc'], d['stored_r'], d['stored_c'], d['reward_mean'],
             d['cost_mean'], lab))
print()
print('  判读（s=1e-3 是关键探针）：')
print('   · 若 reward_mean 在 s_p=1e-3 时暴涨 ~1000 倍 ⇒ 还原逻辑确实执行 `×s_p`')
print('     ⇒ s_p=1000 的那一行如果把值**压小**了，说明真源默认下日志被**缩小**而非放大；')
print('   · 若 reward_mean 在 s_p=1e-3 时几乎不变 ⇒ 还原块**根本没执行**（_SP=_SC=1）')
print('     ⇒ 日志即原始口径，不存在"放大"。')
