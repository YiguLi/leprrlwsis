#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H 层 n 步回报 (--nstep) 与 twin-target 聚合 (--target_agg) 的回归测试。

背景 (冻结批离线验证, 见对话记录 2026-09)
----------------------------------------
γ=0.98 时 H 层真值 ≈ 块奖励/(1−γ) = 50×块奖励, 1-step 目标必须把奖励信息沿
50 步自助传播 ⇒ 冻结数据集上只训 critic, 40k 次更新只到 MC 真值的 34 %。
把连续 n 个块折叠成一条转移 (reward = Σγ^k r_k, 自带 bootstrap 系数 γ^n) 后
需要自助传播的深度降到 50/n: n=10 + mean-target 在 20k 次就到 90 %。

本测试不依赖 MATLAB, 只验证**实现契约**:
  A. nstep=1 路径: memory 内容与输入逐位相同, _boot 为 None (默认路径不变)
  B. nstep=n 折叠: reward/next_state/boot/截断语义正确 (含 done 与 episode 尾)
  C. target 公式: nstep=1 为 r+γv, nstep=n 为 Σγ^k r_k + γ^n v; mean 聚合正确
  D. 契约: nstep 只作用于 H (L 由 main 传默认 1), target_agg 取值受限

运行: python agentutil/test_nstep_h.py
"""
import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块
sys.path.insert(0, _HERE)                              # agentutil 内互相 import

# Windows 控制台默认 GBK, 本文件的 "✓" 会 UnicodeEncodeError ⇒ 显式转 UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

from td3per.td3per_ddpg import SP_TD3                                  # noqa: E402

GAM = 0.98
NB_S, NB_A = 6, 6


def make_args(rmsize=64, bsize=8, discount=GAM):
    return SimpleNamespace(
        seed=0, cuda=False, hidden1=8, hidden2=8, init_w=3e-3,
        prate=1e-4, rate=1e-2, rmsize=rmsize, bsize=bsize, tau=1e-3,
        discount=discount, epsilon=4000, lam=1.0, lr_lam=0.0,
        cost_single_limit=1000.0, PER_alpha=0.6, PER_beta=0.4,
        PER_epsilon=0.01, beta_increment_per_sampling=0.0, nb_turbine=5,
        ou_theta=0.15, ou_mu=0.0, ou_sigma=0.2, max_episode_length=1440,
    )


def make_agent(nstep=1, target_agg='min', rmsize=64, bsize=8):
    a = SP_TD3(nb_states=NB_S, nb_actions=NB_A, args=make_args(rmsize, bsize),
               nstep=nstep, target_agg=target_agg,
               cost_critic_mode='max', record_update_stats=True)
    a.is_training = True
    return a


def st(i):
    """确定性状态, 便于按位比对。"""
    return np.arange(NB_S, dtype=np.float32) + float(i)


def act(i):
    return np.full(NB_A, 0.1 * (i + 1), dtype=np.float32)


def row(mem, i):
    return mem._storage[i]


# ---------------------------------------------------------------- A
def test_default_path_unchanged():
    ag = make_agent(nstep=1)
    assert ag.nstep == 1 and ag._boot is None, (ag.nstep, ag._boot)
    seq = [(st(i), act(i), 1.0 * i, 0.5 * i, st(i + 1), 0.0) for i in range(5)]
    for (s, a, r, c, s2, d) in seq:
        ag.observe_jump(s, a, r, c, s2, d)
    assert len(ag.memory) == 5
    for i, (s, a, r, c, s2, d) in enumerate(seq):
        g = row(ag.memory, i)
        assert np.array_equal(g[0], s) and np.array_equal(g[1], a)
        assert g[2] == r and g[3] == c and np.array_equal(g[4], s2) and g[5] == d
    print('  A. nstep=1 默认路径: 5 条转移与输入逐位相同, _boot=None  ✓')


# ---------------------------------------------------------------- B
def test_nstep_folding():
    n = 3
    ag = make_agent(nstep=n)
    gam = ag.discount
    assert ag._boot is not None and len(ag._boot) == 64
    # 9 个无终止的块 -> 折叠成 3 条 n 步转移
    for i in range(9):
        ag.observe_jump(st(i), act(i), np.float32(1.0 + i), np.float32(0.1 * i),
                        st(i + 1), 0.0)
    assert len(ag.memory) == 3, len(ag.memory)
    for k in range(3):
        g = row(ag.memory, k)
        r_exp = sum((gam ** j) * (1.0 + (n * k + j)) for j in range(n))
        c_exp = sum((gam ** j) * (0.1 * (n * k + j)) for j in range(n))
        assert abs(g[2] - r_exp) < 1e-5, (g[2], r_exp)
        assert abs(g[3] - c_exp) < 1e-6, (g[3], c_exp)
        assert np.array_equal(g[4], st(n * k + n)), 'next_state 应为第 n 个后继'
        assert abs(ag._boot[k] - gam ** n) < 1e-6, ag._boot[k]
        assert g[5] == 0.0
    # 尾部不足 n 步: episode 结束 (done) 时逐条截断结算, boot=0
    ag2 = make_agent(nstep=n)
    ag2.observe_jump(st(0), act(0), 2.0, 0.2, st(1), 0.0)
    ag2.observe_jump(st(1), act(1), 3.0, 0.3, st(2), 1.0)     # done 在第 2 条
    assert len(ag2.memory) == 2, len(ag2.memory)              # 尾部 2 条都结算
    g0 = row(ag2.memory, 0)
    assert abs(g0[2] - (2.0 + gam * 3.0)) < 1e-6, g0[2]       # 窗口内遇 done 即截断
    assert abs(ag2._boot[0]) < 1e-9 and g0[5] == 1.0
    g1 = row(ag2.memory, 1)
    assert abs(g1[2] - 3.0) < 1e-6 and abs(ag2._boot[1]) < 1e-9 and g1[5] == 1.0
    # 下一 episode 的窗口不与上一 episode 混合
    ag2.observe_jump(st(9), act(9), 7.0, 0.7, st(10), 0.0)
    assert len(ag2.memory) == 2 and len(ag2._pending) == 1
    print('  B. nstep=3 折叠: Σγ^k r_k / next_state=第 3 后继 / boot=γ^3 / done 截断  ✓')


# ---------------------------------------------------------------- C
def test_target_formula():
    """把 4 张 target critic 换成常量, 直接读 update_stats 里的 target_p_mean。"""
    def patch(ag, v_p1, v_p2):
        def f(v):
            return lambda xs: torch.full((xs[0].shape[0], 1), float(v),
                                         dtype=torch.float32)
        ag.critic_target_p.forward = f(v_p1)
        ag.critic_target_p2.forward = f(v_p2)
        ag.critic_target_c.forward = f(0.0)
        ag.critic_target_c2.forward = f(0.0)

    # nstep=1: target = r + γ·min(v1,v2)
    ag = make_agent(nstep=1, bsize=4)
    for i in range(4):
        ag.observe_jump(st(i), act(i), np.float32(10.0), np.float32(1.0),
                        st(i + 1), 0.0)
    patch(ag, 5.0, 7.0)
    ag.update_policy()
    tgt = ag.update_stats[-1]['target_p_mean']
    exp = 10.0 + GAM * 5.0                                   # min(5,7)
    assert abs(tgt - exp) < 1e-4, (tgt, exp)
    print('  C1. nstep=1 + min: target=%.4f = r + γ·min(v1,v2)  ✓' % tgt)

    # nstep=1 + mean: target = r + γ·(v1+v2)/2
    ag = make_agent(nstep=1, target_agg='mean', bsize=4)
    for i in range(4):
        ag.observe_jump(st(i), act(i), np.float32(10.0), np.float32(1.0),
                        st(i + 1), 0.0)
    patch(ag, 5.0, 7.0)
    ag.update_policy()
    tgt = ag.update_stats[-1]['target_p_mean']
    exp = 10.0 + GAM * 6.0
    assert abs(tgt - exp) < 1e-4, (tgt, exp)
    print('  C2. nstep=1 + mean: target=%.4f = r + γ·(v1+v2)/2  ✓' % tgt)

    # nstep=3: target = Σγ^k r_k + γ^3·v
    n = 3
    ag = make_agent(nstep=n, bsize=4)
    for i in range(3 * 4):
        ag.observe_jump(st(i), act(i), np.float32(10.0), np.float32(1.0),
                        st(i + 1), 0.0)
    assert len(ag.memory) == 4
    patch(ag, 5.0, 7.0)
    ag.update_policy()
    tgt = ag.update_stats[-1]['target_p_mean']
    r_n = sum((GAM ** k) * 10.0 for k in range(n))
    exp = r_n + (GAM ** n) * 5.0
    assert abs(tgt - exp) < 1e-3, (tgt, exp)
    print('  C3. nstep=3 + min: target=%.4f = Σγ^k·10 + γ^3·min(v1,v2)  ✓' % tgt)


# ---------------------------------------------------------------- D
def test_contracts():
    ag = make_agent(nstep=1)
    assert not hasattr(ag, '_boot') or ag._boot is None
    try:
        make_agent(nstep=1, target_agg='avg')
    except AssertionError:
        pass
    else:
        raise AssertionError('target_agg 非法值未被拒绝')
    # nstep<1 视为 1 (防御)
    assert make_agent(nstep=0).nstep == 1
    print('  D. 契约: 非法 target_agg 被拒 / nstep<1 归一为 1  ✓')


if __name__ == '__main__':
    print('=' * 84)
    print('H 层 n 步回报 (--nstep) 与 target 聚合 (--target_agg) 回归')
    print('=' * 84)
    test_default_path_unchanged()
    test_nstep_folding()
    test_target_formula()
    test_contracts()
    print()
    print('TEST OK')
