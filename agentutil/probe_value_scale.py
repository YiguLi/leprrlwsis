# -*- coding: utf-8 -*-
"""诊断: 归一化两臂的收敛速度与机制 (独立脚本, 不依赖测试文件)。

目标: 弄清"把值函数输出搬到 O(1e4) 量级"到底受什么限制 ——
末层 bias 走 lr? 还是隐层放大 (Σ|∂Q/∂θ| 随训练增长)?
"""
import argparse
import os
import random
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块 (hsrl_*, myfunc, util, td3*/...)
sys.path.insert(0, _HERE)                              # agentutil 内互相 import

from td3perrb.td3perrb_ddpg import SP_TD3

NB, NA, NBATCH, NFILL = 6, 6, 64, 256
R_P, R_C = 5000.0, 20.0
STATE_DEP = 0.0          # >0 ⇒ 目标随状态变化 (r = R_P + STATE_DEP*1000*state[0])


def make_args():
    return argparse.Namespace(cuda=False, seed=0, hidden1=64, hidden2=100, init_w=0.003,
                              rate=0.01, prate=0.0001, rmsize=20000, PER_alpha=0.6,
                              bsize=NBATCH, tau=0.001, discount=0.99, epsilon=4000,
                              ou_theta=0.15, ou_mu=0.0, ou_sigma=0.2, lam=0.967,
                              lr_lam=0.1, cost_single_limit=1000.0, PER_beta=0.4,
                              PER_epsilon=0.01, beta_increment_per_sampling=3e-5,
                              nb_turbine=5, policy_delay=2, policy_noise=0.1,
                              noise_clip=0.3)


def make_data():
    r = np.random.RandomState(7)
    st = r.rand(NFILL, NB).astype(np.float32)
    ac = (r.rand(NFILL, NA) * 0.5).astype(np.float32)
    return st, ac


ST, AC = make_data()
REW = (R_P + STATE_DEP * 1000.0 * ST[:, 0]).astype(np.float32)


def build(scale):
    torch.manual_seed(0)
    ag = SP_TD3(nb_states=NB, nb_actions=NA, args=make_args(),
                lam_max=4.83, lam_min=0.967, ema_lam=True, ema_alpha=0.02,
                ema_lam_lr=0.02, policy_delay=2, policy_noise=0.1, noise_clip=0.3,
                cost_critic_mode='max', record_update_stats=True)
    ag.value_scale = scale
    for i in range(NFILL):
        ag.memory.append(ST[i], AC[i], REW[i], R_C, ST[(i + 1) % NFILL], 1.0)
    return ag


def probe(ag, tag):
    with torch.no_grad():
        xt = torch.tensor(ST)
        at = torch.tensor(AC)
        x4 = torch.relu(ag.critic_p.fc4(torch.relu(ag.critic_p.fc1(xt))))
        x2 = torch.relu(ag.critic_p.fc2(torch.cat([x4, at], 1)))
        x5 = torch.relu(ag.critic_p.fc5(x2))
        q = ag.critic_p.fc3(x5)
    w = {n: float(getattr(ag.critic_p, n).weight.detach().abs().mean())
         for n in ('fc1', 'fc4', 'fc2', 'fc5', 'fc3')}
    print('      {:<11s} bias={:>10.4f} |w| fc1={:.4f} fc4={:.4f} fc2={:.4f} '
          'fc5={:.4f} fc3={:.4f} | h5_mean={:>7.3f} h5_max={:>8.2f} Q_mean={:>11.3f}'
          .format(tag, float(ag.critic_p.fc3.bias.item()), w['fc1'], w['fc4'], w['fc2'],
                  w['fc5'], w['fc3'], float(x5.mean()), float(x5.max()), float(q.mean())))


STEPS = [50, 100, 200, 400, 800, 1600]
print('目标 r1: 均值 {:.0f} (min {:.0f} max {:.0f});  critic lr=0.01, '
      'clip_grad_norm=2.0'.format(float(REW.mean()), float(REW.min()), float(REW.max())))

for s in (1.0, 1000.0):
    ag = build(s)
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    n = 0
    vals = []
    for t in STEPS:
        while n < t:
            ag.update_policy()
            n += 1
        with torch.no_grad():
            q = float(ag.critic_p([torch.tensor(ST), torch.tensor(AC)]).mean())
        vals.append(q * s)
        if t in (200, 3200):
            probe(ag, 's={:g} t={}'.format(s, t))
    print('    s={:<8g} '.format(s) + ' '.join('{:>9.1f}'.format(v) for v in vals))
    print('      步: {}'.format(STEPS))
    target = float(REW.mean())
    hit = None
    for i in range(1, len(vals)):
        if vals[i] >= 0.9 * target:
            hit = STEPS[i - 1] + (0.9 * target - vals[i - 1]) * (STEPS[i] - STEPS[i - 1]) \
                / max(vals[i] - vals[i - 1], 1e-9)
            break
    print('      达 90% 目标 ({:.0f}) 所需步数: {}'.format(
        0.9 * target, '未达到' if hit is None else '{:.0f}'.format(hit)))
