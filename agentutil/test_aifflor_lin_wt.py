#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_aifflor_lin_wt.py —— 线性行为分支（--lin_wt）的契约测试。

覆盖 6 条契约（任一失败即 exit 1）：
  A 投影语义：a0 / a4 原样保留，中间三维 = 线性插值 + 两位小数四舍五入；
             第 6 维 g 不动；**不 clip**（越界交给动作检查 + pen5）
  B 梯度：投影只读首尾 ⇒ ∂/∂a_j (j=1..3) **恒为 0**；a0 / a4 / g 非零
             （取整用 straight-through，故 a0/a4 的梯度不会被 round 清零）
  C agent 层：探索噪声之后仍落在“首尾 + 线性插值”流形上；ε=0 的确定性输出同样
  D 物理：同一 (a0, a4)、其余维任改 ⇒ env.PowerOutput 逐位相同；改 a4 出力变
  E 默认臂：lin_wt=0 时 _project / _project_np 严格恒等（逐位不变）
  F 接线：aifflor_main 的两条主循环（旧 train / train_hsrl）都调用 _apply_lin_wt

跑法：python agentutil/test_aifflor_lin_wt.py
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
from aifflor.aifflor_ddpg import SP_TD3  # noqa: E402

FAIL = []


def check(cond, msg):
    print(('  ✔ ' if cond else '  ✘ ') + msg)
    if not cond:
        FAIL.append(msg)


def make_args(lin_wt=1, farm_level=0, scalar_wt=0, decimals=2):
    return SimpleNamespace(
        seed=0, cuda=False, hidden1=16, hidden2=100, init_w=3e-3,
        prate=1e-4, rate=3e-3, rmsize=64, bsize=8, tau=1e-3, discount=0.95,
        epsilon=4000, lam=1.0, lr_lam=0.0, cost_single_limit=1000.0,
        PER_alpha=0.6, PER_beta=0.4, PER_epsilon=0.01,
        beta_increment_per_sampling=0.0, nb_turbine=5,
        ou_theta=0.15, ou_mu=0.0, ou_sigma=0.2, max_episode_length=1440,
        critic_arch='v1', critic_act='relu', critic_out_init='3e3',
        critic_feat_power=0, critic_feat_power_k=1.357e-3, cost_clip=1,
        farm_level=farm_level, scalar_wt=scalar_wt,
        lin_wt=lin_wt, lin_wt_decimals=decimals, na_box=5,
        aux_model=0, critic_action_first=0, critic_bias_init='none',
        warmup_wt_sched='lhs',
    )


# ---------------------------------------------------------------- A 投影语义
def test_projection():
    print('[A] 投影语义：a0/a4 不动，中间三维线性插值 + 两位小数，g 不动，不 clip')
    a = torch.tensor([[0.12, 0.40, 0.01, 0.33, 0.25, -0.7],
                      [0.30, 0.00, 0.50, 0.40, 0.10, 0.33]])
    p = model.proj_lin_wt(a, nt=5)
    exp = np.array([[0.12, 0.15, 0.19, 0.22, 0.25, -0.7],
                    [0.30, 0.25, 0.20, 0.15, 0.10, 0.33]])
    check(np.allclose(p.numpy(), exp, atol=1e-9),
          'torch 批量: 中间维 = 线性插值并四舍五入（%s）' % np.round(p[0, :5].numpy(), 2))
    check(bool(torch.all(p[:, 0] == a[:, 0])) and bool(torch.all(p[:, 4] == a[:, 4])),
          'torch: 首尾维 a0/a4 **逐位**保留（不取整）')
    check(bool(torch.all(p[:, 5] == a[:, 5])), 'torch: 第 6 维 g 原样保留')
    one = model.proj_lin_wt(torch.tensor([0.12, 0.40, 0.01, 0.33, 0.25, -0.7]), nt=5)
    check(np.allclose(one.numpy(), exp[0], atol=1e-9), 'torch: 1-D 输入同样成立')
    # 负值：不 clip，按 away-from-zero 四舍五入
    neg = model.proj_lin_wt(torch.tensor([-0.20, 0.40, 0.01, 0.33, 0.45, 0.0]), nt=5)
    check(np.allclose(neg[:5].numpy(), [-0.20, -0.04, 0.13, 0.29, 0.45], atol=1e-6),
          'torch: 负值不被裁剪，中间维四舍五入 (-0.0375→-0.04, 0.125→0.13, 0.2875→0.29)')
    # numpy 版
    pn = model.proj_lin_wt_np(np.array([0.12, 0.40, 0.01, 0.33, 0.25, -0.7]))
    check(np.allclose(pn, exp[0], atol=1e-12), 'numpy: 与 torch 版语义一致')
    check(np.array_equal(model.proj_lin_wt_np(pn), pn), 'numpy: 幂等（投影两次 = 一次）')
    # 取整开关
    nr = model.proj_lin_wt_np(np.array([0.12, 0.0, 0.0, 0.0, 0.25, 0.0]), decimals=-1)
    check(np.allclose(nr[:5], [0.12, 0.1525, 0.185, 0.2175, 0.25], atol=1e-12),
          'decimals<0 ⇒ 关闭取整（保留完整插值精度）')
    # nt 泛化 / 退化
    p3 = model.proj_lin_wt_np(np.array([0.10, 0.90, 0.30, 0.77]), nt=3)
    check(np.allclose(p3[:3], [0.10, 0.20, 0.30], atol=1e-12) and p3[3] == 0.77,
          'nt=3: 只改中间 1 维，nt 之后维度原样')
    p2 = model.proj_lin_wt_np(np.array([0.10, 0.90]), nt=2)
    check(np.array_equal(p2, np.array([0.10, 0.90])), 'nt=2: 没有中间维，原样返回')
    # 四舍五入口径（half away from zero）
    x = np.array([0.005, -0.005, 0.015, -0.015, 0.0149, -0.0149])
    q = model._round_half_away_np(x, 2)
    check(np.allclose(q, [0.01, -0.01, 0.02, -0.02, 0.01, -0.01], atol=1e-12),
          '四舍五入 = half away from zero（0.005→0.01, -0.005→-0.01）')


# ---------------------------------------------------------------- B 梯度
def test_gradient():
    print('[B] 投影只读首尾 ⇒ ∂/∂a_j (j=1..3) 恒为 0；a0/a4/g 非零（STE）')
    torch.manual_seed(0)
    net = model.Critic(4, 6, hidden1=16, hidden2=100, arch='v1')
    S = torch.randn(32, 4)
    A = (torch.rand(32, 6) * 0.5).requires_grad_(True)
    q = net((S, model.proj_lin_wt(A, nt=5)))
    g, = torch.autograd.grad(q.sum(), A)
    check(float(g[:, 1:4].abs().max()) == 0.0,
          '∂Q/∂a_j (j=1..3) 的最大绝对值 = %.1e（严格 0）' % float(g[:, 1:4].abs().max()))
    check(float(g[:, 0].abs().max()) > 0.0, '∂Q/∂a_0 非零（首端仍是有效自由度）')
    check(float(g[:, 4].abs().max()) > 0.0, '∂Q/∂a_4 非零（末端仍是有效自由度）')
    check(float(g[:, 5].abs().max()) > 0.0, '∂Q/∂g 非零（第 6 维仍独立）')
    # STE 的直接契约：round 本身导数 0，STE 前向取整、反向恒等
    x = torch.tensor([-1.234, -0.005, 0.0, 0.005, 0.015, 0.0199],
                     dtype=torch.float64, requires_grad=True)
    y = model._round_half_away_ste(x, 2)
    gx, = torch.autograd.grad(y.sum(), x)
    check(bool(torch.all(gx == 1.0)), '_round_half_away_ste: ∂y/∂x ≡ 1（straight-through）')
    check(np.allclose(y.detach().numpy(), [-1.23, -0.01, 0.0, 0.01, 0.02, 0.02],
                      atol=1e-12), '前向仍两位小数（含 negative half-away）')
    xr = x.detach().clone().requires_grad_(True)
    yr = torch.round(xr * 100.0) / 100.0
    gr, = torch.autograd.grad(yr.sum(), xr)
    check(float(gr.abs().max()) == 0.0, '对照: 直接 torch.round ⇒ ∂y/∂x ≡ 0（STE 必需）')


# ---------------------------------------------------------------- C agent 层
def test_agent_keeps_manifold():
    print('[C] agent：探索噪声之后仍落在“首尾 + 线性插值”流形上')
    ag = SP_TD3(nb_states=4, nb_actions=6, args=make_args(lin_wt=1))
    ag.is_training = True
    ag.epsilon = 1.0
    s = np.array([0.58, 1.0, 0.5, 0.9], dtype=np.float32)
    bad = 0
    max_err = 0.0
    for _ in range(50):
        a = ag.select_action(s)
        ref = model.proj_lin_wt_np(a)
        max_err = max(max_err, float(np.abs(a[1:4] - ref[1:4]).max()))
        if not np.allclose(a[1:4], ref[1:4], atol=1e-9):
            bad += 1
    check(bad == 0, 'select_action × 50：中间维 = 首尾线性插值（最大偏差 %.1e）' % max_err)
    ag.epsilon = 0.0
    a0 = ag.select_action(s)
    check(np.allclose(a0[1:4], model.proj_lin_wt_np(a0)[1:4], atol=1e-9),
          'ε=0：确定性动作仍在流形上')
    with torch.no_grad():
        det = ag._act(ag.actor, torch.tensor(np.array([s]))).numpy().ravel()
    check(np.allclose(det[1:4], model.proj_lin_wt_np(det)[1:4], atol=1e-9),
          '_act（critic 评估点）同样在流形上')
    # 对照：不开 lin_wt 时不应满足线性关系
    ag2 = SP_TD3(nb_states=4, nb_actions=6, args=make_args(lin_wt=0))
    ag2.is_training, ag2.epsilon = True, 1.0
    neq = 0
    for _ in range(10):
        b = ag2.select_action(s)
        neq += int(not np.allclose(b[1:4], model.proj_lin_wt_np(b)[1:4], atol=1e-6))
    check(neq > 0, '对照（lin_wt=0）：中间维**不**满足线性关系（投影确实起作用）')
    # 下层 L 是 1 维动作：同一个 --lin_wt 不能把它扩成 5 维（必须自动跳过）
    agL = SP_TD3(nb_states=5, nb_actions=1, args=make_args(lin_wt=1))
    agL.is_training = True
    agL.epsilon = 1.0
    aL = agL.select_action(np.array([0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32))
    check(np.asarray(aL).size == 1 and np.isfinite(aL).all() and agL.lin_wt is False,
          '下层（nb_actions=1）：lin_wt 自动跳过，动作仍 1 维且有限')


# ---------------------------------------------------------------- D 物理
def test_env_invariance():
    print('[D] 物理：同一 (a0, a4)、其余维任改 ⇒ 出力逐位相同；改 a4 出力变')
    import aifflor.aifflor_turbine as AT
    from aifflor.aifflor_env import AifflorPhysics
    ph = AifflorPhysics(nt=5, calibrated=False, power_scale=AT.LEVEL_COEF)
    u = 9.0
    a1 = np.array([0.22, 0.40, 0.05, 0.31, 0.11, 0.0])
    a2 = np.array([0.22, -0.10, 0.50, 0.00, 0.11, 0.0])
    p1 = float(ph.power(u, model.proj_lin_wt_np(a1)[:5]))
    p2 = float(ph.power(u, model.proj_lin_wt_np(a2)[:5]))
    check(abs(p1 - p2) == 0.0,
          '同 a0/a4、其余维任改 ⇒ 出力逐位相同 (%.6f vs %.6f)' % (p1, p2))
    p_raw = float(ph.power(u, a1[:5]))
    check(abs(p_raw - p1) > 1e-6,
          '未投影的原始动作给出不同出力 %.6f vs %.6f（投影确实生效）' % (p_raw, p1))
    # 改末端 ⇒ 整条型线变化 ⇒ 出力变化
    a3 = np.array([0.22, 0.0, 0.0, 0.0, 0.40, 0.0])
    p3 = float(ph.power(u, model.proj_lin_wt_np(a3)[:5]))
    check(abs(p3 - p1) > 1e-3, '改 a4 ⇒ 出力变化 (%.6f vs %.6f)' % (p3, p1))


# ---------------------------------------------------------------- E 默认臂
def test_default_arm_identity():
    print('[E] 默认臂（lin_wt=0）逐位不受影响')
    ag = SP_TD3(nb_states=4, nb_actions=6, args=make_args(lin_wt=0))
    x = torch.tensor([[0.11, 0.42, 0.03, 0.35, 0.27, -0.5]])
    check(ag._project(x) is x, '_project 严格恒等（返回同一对象）')
    check(np.array_equal(ag._project_np(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])),
                         np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])),
          '_project_np 严格恒等')


# ---------------------------------------------------------------- F 接线
def test_main_wiring():
    print('[F] aifflor_main：两条主循环都接 _apply_lin_wt')
    p = os.path.join(ROOT, 'aifflor', 'aifflor_main.py')
    src = open(p, encoding='utf-8').read()
    n_def = src.count('def _apply_lin_wt(')
    n_call = src.count('_apply_lin_wt(') - n_def
    check(n_def == 1, '定义 _apply_lin_wt 恰好 1 处')
    check(n_call >= 4, '_apply_lin_wt 调用点 ≥4（旧 train 的 warmup/actor + HSRL 的 warmup/actor），实际 %d' % n_call)
    check("'--lin_wt'" in src and "'--lin_wt_decimals'" in src,
          'CLI 参数 --lin_wt / --lin_wt_decimals 已注册')
    check('--lin_wt 与 --scalar_wt / --farm_level 互斥' in src,
          '互斥断言已存在')


def main():
    test_projection()
    test_gradient()
    test_agent_keeps_manifold()
    test_env_invariance()
    test_default_arm_identity()
    test_main_wiring()
    print()
    if FAIL:
        print('✘ %d 项失败:' % len(FAIL))
        for f in FAIL:
            print('   - ' + f)
        return 1
    print('线性行为分支（--lin_wt）契约测试：全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
