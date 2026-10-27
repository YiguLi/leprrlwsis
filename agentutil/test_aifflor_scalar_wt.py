#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_aifflor_scalar_wt.py —— aifflor baseline 动作空间的契约测试（--scalar_wt）。

覆盖 6 条契约（任一失败即 exit 1）：
  A 投影语义：`a_i = a_0`（前 5 维），第 6 维 g 不动，**不 clip**（越界交给 clip+pen5）
  B agent 层：探索噪声**之后**仍相等（训练态、ε 最大）；`_act` 的确定性输出也相等
  C 梯度：投影是线性的 ⇒ 对第 1..4 维的梯度**恒为 0**（不是"很小"）
  D 物理：env.PowerOutput(u, a) 与把第 1..4 维任意改写后的结果**逐位相同**
  E c1：`--c1_smooth 0` ⇒ c1 = |ΔP| + 20·(|ΔP|−3)⁺（硬折点，逐位相等）
  F 网络：`--critic_arch v1`（PER 结构）在 16/100 下 = 14 441 参数；actor 3 498 参数

跑法：python agentutil/test_aifflor_scalar_wt.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import model  # noqa: E402
import hsrl_cost as hc  # noqa: E402
from aifflor.aifflor_ddpg import SP_TD3  # noqa: E402

FAIL = []


def check(cond, msg):
    print(('  ✔ ' if cond else '  ✘ ') + msg)
    if not cond:
        FAIL.append(msg)


def make_args(scalar_wt=1, farm_level=0):
    return SimpleNamespace(
        seed=0, cuda=False, hidden1=16, hidden2=100, init_w=3e-3,
        prate=1e-4, rate=3e-3, rmsize=64, bsize=8, tau=1e-3, discount=0.95,
        epsilon=4000, lam=1.0, lr_lam=0.0, cost_single_limit=1000.0,
        PER_alpha=0.6, PER_beta=0.4, PER_epsilon=0.01,
        beta_increment_per_sampling=0.0, nb_turbine=5,
        ou_theta=0.15, ou_mu=0.0, ou_sigma=0.2, max_episode_length=1440,
        critic_arch='v1', critic_act='relu', critic_out_init='3e3',
        critic_feat_power=0, critic_feat_power_k=1.357e-3, cost_clip=1,
        farm_level=farm_level, scalar_wt=scalar_wt, na_box=5,
        aux_model=0, critic_action_first=0, critic_bias_init='none',
        warmup_wt_sched='lhs',
    )


# ---------------------------------------------------------------- A 投影语义
def test_projection():
    print('[A] 投影语义 a_i = a_0（前 5 维），g 不动，不 clip')
    a = torch.tensor([[0.12, 0.40, 0.01, 0.33, 0.25, -0.7],
                      [-0.20, 0.10, 0.50, 0.00, 0.45, 0.33]])
    p = model.proj_scalar_wt(a, nt=5)
    check(bool(torch.all(p[:, :5] == a[:, :1])), 'torch: 前 5 维全部等于 a[0]')
    check(bool(torch.all(p[:, 5] == a[:, 5])), 'torch: 第 6 维 g 原样保留')
    check(bool(torch.all(p[:, 0] == a[:, 0])), 'torch: 第 0 维不变')
    check(np.isclose(float(p[1, 0]), -0.20, atol=1e-6), 'torch: 负值不被裁剪（越界交给 clip+pen5）')
    one = model.proj_scalar_wt(torch.tensor([0.3, 0.1, 0.2, 0.4, 0.5, 0.1]), nt=5)
    check(bool(torch.all(one[:5] == one[0])), 'torch: 1-D 输入同样成立')
    pn = model.proj_scalar_wt_np(np.array([0.12, 0.40, 0.01, 0.33, 0.25, -0.7]))
    check(np.array_equal(pn[:5], np.full(5, 0.12)) and pn[5] == -0.7,
          'numpy: 与 torch 版语义一致（前 5 维 = a0, g 不变）')
    # 幂等
    check(np.array_equal(model.proj_scalar_wt_np(pn), pn), 'numpy: 幂等（投影两次 = 一次）')


# ---------------------------------------------------------------- B agent 层
def test_agent_keeps_uniform():
    print('[B] agent：探索噪声之后仍五台相等（训练态、ε 满幅）')
    ag = SP_TD3(nb_states=4, nb_actions=6, args=make_args(scalar_wt=1))
    ag.is_training = True
    ag.epsilon = 1.0
    s = np.array([0.58, 1.0, 0.5, 0.9], dtype=np.float32)
    bad = 0
    for _ in range(50):
        a = ag.select_action(s)
        if not (np.allclose(a[:5], a[0]) and abs(a[5] - a[5]) < 1e-12):
            bad += 1
    check(bad == 0, 'select_action × 50 次：前 5 维恒相等（噪声只在第 0 维有效）')
    # ε→0 时同维对齐
    ag.epsilon = 0.0
    a0 = ag.select_action(s)
    check(np.allclose(a0[:5], a0[0]), 'ε=0：前 5 维恒相等')
    # 对照：不开 scalar_wt 时不应相等（否则测不到东西）
    ag2 = SP_TD3(nb_states=4, nb_actions=6, args=make_args(scalar_wt=0))
    ag2.is_training, ag2.epsilon = True, 1.0
    neq = sum(0 if np.allclose(ag2.select_action(s)[:5], ag2.select_action(s)[0]) else 1
              for _ in range(10))
    check(neq > 0, '对照（scalar_wt=0）：前 5 维**不**相等（投影确实起作用）')
    # 确定性 _act
    with torch.no_grad():
        det = ag._act(ag.actor, torch.tensor(np.array([s]))).numpy().ravel()
    check(np.allclose(det[:5], det[0]), '_act（critic 评估点）前 5 维相等')


# ---------------------------------------------------------------- C 梯度恒 0
def test_zero_gradient():
    print('[C] 投影线性 ⇒ 对第 1..4 维的梯度恒为 0')
    net = model.Critic(4, 6, hidden1=16, hidden2=100, arch='v1')
    S = torch.randn(16, 4)
    A = (torch.rand(16, 6) * 0.5).requires_grad_(True)
    q = net((S, model.proj_scalar_wt(A, nt=5)))
    g, = torch.autograd.grad(q.sum(), A)
    check(float(g[:, 1:5].abs().max()) == 0.0,
          '∂Q/∂a_j (j=1..4) 的最大绝对值 = %.1e（严格 0）' % float(g[:, 1:5].abs().max()))
    check(float(g[:, 0].abs().max()) > 0.0, '∂Q/∂a_0 非零（第 0 维仍是有效自由度）')
    check(float(g[:, 5].abs().max()) > 0.0, '∂Q/∂g 非零（第 6 维仍独立）')


# ---------------------------------------------------------------- D 物理不变
def test_env_invariance():
    print('[D] 物理：投影之后出力只由 a_0 决定（第 1..4 维影响不到任何东西）')
    import aifflor.aifflor_turbine as AT
    from aifflor.aifflor_env import AifflorPhysics
    ph = AifflorPhysics(nt=5, calibrated=False, power_scale=AT.LEVEL_COEF)
    u = 9.0
    # 两个**同为 a_0=0.22、其余维完全不同**的原始动作
    a1 = np.array([0.22, 0.40, 0.05, 0.31, 0.11, 0.0])
    a2 = np.array([0.22, -0.10, 0.50, 0.00, 0.45, 0.0])
    p1 = float(ph.power(u, model.proj_scalar_wt_np(a1)[:5]))
    p2 = float(ph.power(u, model.proj_scalar_wt_np(a2)[:5]))
    check(abs(p1 - p2) == 0.0,
          '同一 a_0、其余维任改 ⇒ 出力逐位相同 (%.6f vs %.6f)' % (p1, p2))
    p_raw = float(ph.power(u, a1[:5]))
    check(abs(p_raw - p1) > 1e-6,
          '未投影的原始动作给出不同出力 %.6f vs %.6f（投影确实生效）' % (p_raw, p1))
    # 出力仍随 a_0 变化（第 0 维是真自由度）
    ps = [float(ph.power(u, model.proj_scalar_wt_np(np.array([x, 0, 0, 0, 0, 0]))[:5]))
          for x in (0.05, 0.22, 0.40)]
    check(max(ps) - min(ps) > 1e-3, '出力随 a_0 单调变化 ⇒ 第 0 维仍是有效自由度')


# ---------------------------------------------------------------- E 非平滑 c1
def test_nonsmooth_c1():
    print('[E] --c1_smooth 0 ⇒ c1 = |ΔP| + 20·(|ΔP|−3)⁺（硬折点）')
    hc.set_smooth(0)
    xs = [0.0, 0.5, 2.999, 3.0, 3.001, 5.0, 12.0]
    ok = all(abs(hc.fluct_cost(x) - (x + 20.0 * max(0.0, x - 3.0))) < 1e-12 for x in xs)
    check(ok, '逐点与硬折点公式相等（含 x=3 两侧）')
    hc.set_smooth(1, 100.0, 0.05)
    d = abs(hc.fluct_cost(3.0) - 60.0)
    check(d > 1e-6, '对照：smooth=1 在同一批点上**不同**（偏差 %.4f），开关有效' % d)
    hc.set_smooth(0)                                   # 复位


# ---------------------------------------------------------------- F PER 网络
def test_per_structure():
    print('[F] --critic_arch v1 + 16/100 = PER 的网结构')
    c = model.Critic(4, 6, hidden1=16, hidden2=100, arch='v1')
    n = sum(v.numel() for v in c.state_dict().values())
    check(n == 14441, 'critic v1(16/100) 参数量 = %d（PER 口径 14 441）' % n)
    has = {k: hasattr(c, k) for k in ('fc1', 'fc4', 'fc2', 'fc5', 'fc3')}
    check(all(has.values()) and not hasattr(c, 's1') and not hasattr(c, 'm1'),
          '层名 = PER 的 fc1→fc4→cat(a)→fc2→fc5→fc3（无 s1/m1 双流结构）')
    act = model.Actor(4, 6, hidden1=16, hidden2=100)
    na = sum(v.numel() for v in act.state_dict().values())
    check(na == 3498, 'actor(16/100) 参数量 = %d（PER 口径 3 498）' % na)


# ---------------------------------------------------------------- G 默认臂不受影响
def test_default_arm_identity():
    print('[G] 默认臂（scalar_wt=0, farm_level=0）逐位不受影响')
    ag = SP_TD3(nb_states=4, nb_actions=6, args=make_args(scalar_wt=0))
    x = torch.tensor([[0.11, 0.42, 0.03, 0.35, 0.27, -0.5]])
    check(bool(torch.equal(ag._project(x), x)), '_project 是严格恒等（update_policy 的利润塔改走 _act 后仍逐位等价）')
    check(np.array_equal(ag._project_np(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])),
                         np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])),
          '_project_np 是严格恒等')


def main():
    test_projection()
    test_agent_keeps_uniform()
    test_zero_gradient()
    test_env_invariance()
    test_nonsmooth_c1()
    test_per_structure()
    test_default_arm_identity()
    print()
    if FAIL:
        print('✘ %d 项失败:' % len(FAIL))
        for f in FAIL:
            print('   - ' + f)
        return 1
    print('baseline(--scalar_wt) 契约测试：全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
