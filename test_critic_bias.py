#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""critic 输出层 bias（价值基线）初始化的单元测试 —— 不需要 MATLAB。

验证三件事:
  1. 初始化前 Q ≈ O(0.1)（与小权重/小 bias 相符）；初始化后 Q ≈ b（基线被搬到目标量级）
  2. target 网络与 online 网络同步被设置
  3. 缺 p2/c2 的 DDPG 风格 agent 也能安全跳过（graceful）
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

import hsrl_agent_init as hinit  # noqa: E402
from td3per.td3per_ddpg import SP_TD3  # noqa: E402
from model import Critic  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

args = argparse.Namespace(cuda=False, seed=0, hidden1=16, hidden2=100, init_w=0.003,
                          rate=0.01, prate=0.0001, rmsize=20000, PER_alpha=0.6, bsize=32,
                          tau=0.001, discount=0.98, epsilon=4000, ou_theta=0.15, ou_mu=0.0,
                          ou_sigma=0.2, lam=3.0, lr_lam=0.1, cost_single_limit=1000.0,
                          PER_beta=0.4, PER_epsilon=0.01, beta_increment_per_sampling=3e-5,
                          nb_turbine=5, policy_delay=2, policy_noise=0.1, noise_clip=0.3)
agent = SP_TD3(nb_states=6, nb_actions=6, args=args, lam_max=4.83, lam_min=0.967,
               ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.02, policy_delay=2,
               policy_noise=0.1, noise_clip=0.3, cost_critic_mode='max')
S = torch.rand(256, 6)
A = torch.rand(256, 6) * 0.5

B_P, B_C = 1850.0, 40.0
q_p0 = hinit.q_probe(agent, S, A, 'critic_p')
q_c0 = hinit.q_probe(agent, S, A, 'critic_c')
print('初始化前: Q_p={:+.4f}  Q_c={:+.4f}   (critic 输出层 fc3.bias 初值={:+.4f})'
      .format(q_p0, q_c0, float(agent.critic_p.fc3.bias.item())))

done = hinit.init_critic_bias(agent, B_P, B_C)
q_p1 = hinit.q_probe(agent, S, A, 'critic_p')
q_c1 = hinit.q_probe(agent, S, A, 'critic_c')
print('初始化后: Q_p={:+.4f}  Q_c={:+.4f}   (已设置 {} 个网络)'.format(q_p1, q_c1, len(done)))

# ---- 断言 ----
assert abs(q_p0) < 1.0 and abs(q_c0) < 1.0, (q_p0, q_c0)          # 初值确实 ~0.1 量级
assert abs(q_p1 - B_P) < 5.0, (q_p1, B_P)                          # 基线已搬到位（残差=特征变化量）
assert abs(q_c1 - B_C) < 5.0, (q_c1, B_C)
assert set(done) == {'critic_p', 'critic_target_p', 'critic_p2', 'critic_target_p2',
                     'critic_c', 'critic_target_c', 'critic_c2', 'critic_target_c2'}, done
for name in done:
    net = getattr(agent, name)
    b = float(net.fc3.bias.item())
    exp = B_P if name in ('critic_p', 'critic_target_p', 'critic_p2',
                          'critic_target_p2') else B_C
    assert abs(b - exp) < 1e-6, (name, b, exp)
# 权重未被改动（只动 bias）
assert abs(float(agent.critic_p.fc3.weight.abs().mean()) - 1.0 / np.sqrt(50) / 2) < 0.02

# ---- 3) DDPG 风格（只有 p/c，没有 p2/c2）----
class _DDPGLike:
    def __init__(self):
        self.critic_p = Critic(6, 6, hidden1=16, hidden2=100, init_w=0.003)
        self.critic_target_p = Critic(6, 6, hidden1=16, hidden2=100, init_w=0.003)
        self.critic_c = Critic(6, 6, hidden1=16, hidden2=100, init_w=0.003)
        self.critic_target_c = Critic(6, 6, hidden1=16, hidden2=100, init_w=0.003)

stub = _DDPGLike()
q_before = float(stub.critic_p([S, A]).mean())
done2 = hinit.init_critic_bias(stub, 500.0, 10.0)
q_after = float(stub.critic_p([S, A]).mean())
assert set(done2) == {'critic_p', 'critic_target_p', 'critic_c', 'critic_target_c'}, done2
assert abs(q_after - 500.0) < 5.0 and abs(q_before) < 1.0
print('DDPG 风格 stub: Q_p {:+.4f} → {:+.4f}，安全跳过 p2/c2'.format(q_before, q_after))

# ---- 基线估计函数 ----
b_p, b_c = hinit.bias_from_block_samples([(200.0, 1.0), (100.0, 3.0)], 0.98)
assert abs(b_p - 150.0 / 0.02) < 1e-6 and abs(b_c - 2.0 / 0.02) < 1e-6, (b_p, b_c)
assert hinit.bias_from_block_samples([], 0.98) == (None, None)
print('bias_from_block_samples: (r̄=150, c̄=2, γ=0.98) → b_p={:.0f}, b_c={:.0f}'.format(b_p, b_c))
print('TEST OK')
