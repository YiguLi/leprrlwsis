#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_aifflor_lin_wt.py —— 线性行为分支（--lin_wt，**结构版**）的契约测试。

结构版口径：
  * actor 输出层 = `[a0, a_end, g...]` 共 `2 + (nb_actions − na_box)` 个神经元
    （上层 6 维动作 ⇒ **3 个神经元**）；
  * 中间 `na_box−2` 维由 `model.expand_lin_wt` 从 `a0/a_end` 线性插值展开，
    并四舍五入到 `--lin_wt_decimals` 位（默认 2）；
  * critic 仍吃 6 维动作；噪声 / target smoothing 只加在自由维上；
  * checkpoint 为 `actor_3.pkl`（critic 仍 `*_6.pkl`）。

覆盖（任一失败即 exit 1）：
  A 展开/投影语义：首尾原样、中间线性插值 + 两位小数，g 原样，不 clip
  B actor 结构：3 输出头 / 参数数 / 自由维索引 / 默认臂仍 6 输出
  C 梯度：3 个输出都对 Q 有梯度；没有"中间输出单元"；STE 直接契约
  D agent 层：探索噪声 / random_action / target smoothing 都在流形上
  E checkpoint：`actor_3.pkl` + `critic_*_6.pkl` 可保存/加载
  F 物理：同一 (a0,a4)、其余维任改 ⇒ 出力逐位相同
  G 初始化映射：3 输出头的 a0/a4/g bias 映射
  H 接线：两条主循环都调用 `_apply_lin_wt`，CLI/互斥/结构版接线存在

跑法：python agentutil/test_aifflor_lin_wt.py
"""
import os
import shutil
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
TMP_DIR = os.path.join(ROOT, 'output', '_lin_wt_contract')


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


# ---------------------------------------------------------------- A 展开/投影
def test_expand_and_projection():
    print('[A] 展开/投影：首尾原样、中间线性插值 + 两位小数、g 原样、不 clip')
    raw = torch.tensor([[0.12, 0.25, -0.7],
                        [0.30, 0.10, 0.33]])
    a = model.expand_lin_wt(raw, nt=5)
    exp = np.array([[0.12, 0.15, 0.19, 0.22, 0.25, -0.7],
                    [0.30, 0.25, 0.20, 0.15, 0.10, 0.33]])
    check(a.shape == (2, 6), 'expand_lin_wt: 3 维 raw → 6 维动作')
    check(np.allclose(a.numpy(), exp, atol=1e-6),
          '3→6: 中间维 = 线性插值并四舍五入（%s）' % np.round(a[0, :5].numpy(), 2))
    check(bool(torch.all(a[:, 0] == raw[:, 0])) and bool(torch.all(a[:, 4] == raw[:, 1])),
          '首尾 a0/a4 **逐位**保留（不取整）')
    check(bool(torch.all(a[:, 5] == raw[:, 2])), '第 6 维 g 原样保留')
    # 无额外维（5 动作 H）
    a5 = model.expand_lin_wt(torch.tensor([0.10, 0.30]), nt=5)
    check(a5.shape == (5,) and np.allclose(a5.numpy(), [0.10, 0.15, 0.20, 0.25, 0.30]),
          '2 维 raw → 5 维动作（无 g 的动作空间）')
    # numpy 版
    pn = model.expand_lin_wt_np(np.array([0.12, 0.25, -0.7]), nt=5)
    check(np.allclose(pn, exp[0], atol=1e-12), 'numpy: 与 torch 版语义一致')
    # proj_lin_wt 仍用于 warmup 的 6 维动作
    p6 = model.proj_lin_wt(torch.tensor([0.12, 0.40, 0.01, 0.33, 0.25, -0.7]), nt=5)
    check(np.allclose(p6.numpy(), exp[0], atol=1e-6),
          'proj_lin_wt(6 维 warmup 动作) 与 expand 的结果一致')
    # 负值不 clip
    neg = model.expand_lin_wt(torch.tensor([-0.20, 0.45, 0.0]), nt=5)
    check(np.allclose(neg.numpy(), [-0.20, -0.04, 0.13, 0.29, 0.45, 0.0], atol=1e-6),
          '负值不被裁剪，中间维四舍五入 (-0.0375→-0.04, 0.125→0.13, 0.2875→0.29)')
    # 幂等（对展开后的完整动作再投影）
    check(np.array_equal(model.proj_lin_wt_np(pn), pn), '完整动作再投影：幂等')


# ---------------------------------------------------------------- B actor 结构
def test_actor_structure():
    print('[B] actor 结构：上层 3 输出（a0/a4/g），默认臂仍 6 输出')
    ag = SP_TD3(nb_states=4, nb_actions=6, args=make_args(lin_wt=1))
    check(ag.actor.fc4.out_features == 3,
          'lin: actor 输出层 = 3 个神经元（实际 %d）' % ag.actor.fc4.out_features)
    check(ag.actor_nb_actions == 3 and ag.actor_tag == 3,
          'lin: actor_nb_actions=3 / actor_tag=3')
    n_actor = sum(v.numel() for v in ag.actor.state_dict().values())
    check(n_actor == 3447, 'lin: actor(16/100, 3 输出) 参数量 = %d（期望 3447）' % n_actor)
    check(np.array_equal(ag.action_free_idx, np.array([0, 4, 5])),
          'lin: 自由维索引 = [0, 4, 5]（a0/a4/g）')
    check(np.allclose(ag.action_amp_free, [0.5, 0.5, 0.5]),
          'lin: target-smoothing 自由维幅度 = [0.5, 0.5, 0.5]')
    check(np.allclose(ag.explore_amp, [0.05] * 5 + [0.20]),
          'lin: 原始 explore_amp 仍 5×wt_noise + sig_noise_max（自由维取 0/4/5）')
    # 下层 L（1 维动作）：同一个 --lin_wt 必须自动跳过
    agL = SP_TD3(nb_states=5, nb_actions=1, args=make_args(lin_wt=1))
    check(agL.lin_wt is False and agL.actor.fc4.out_features == 1,
          '下层（1 维动作）：lin_wt 自动跳过，actor 仍 1 输出')
    # 默认臂 / scalar / farm：仍 6 输出
    ag0 = SP_TD3(nb_states=4, nb_actions=6, args=make_args(lin_wt=0))
    check(ag0.actor.fc4.out_features == 6 and ag0.actor_tag == 6,
          '默认臂：actor 仍 6 输出（逐位不变）')
    check(sum(v.numel() for v in ag0.actor.state_dict().values()) == 3498,
          '默认臂：actor(16/100, 6 输出) 参数量 = 3498')


# ---------------------------------------------------------------- C 梯度
def test_gradient():
    print('[C] 梯度：3 个输出都对 Q 有梯度；没有中间输出单元；STE 契约')
    torch.manual_seed(0)
    net = model.Critic(4, 6, hidden1=16, hidden2=100, arch='v1')
    S = torch.randn(32, 4)
    raw = (torch.rand(32, 3) * 0.5).requires_grad_(True)
    q = net((S, model.expand_lin_wt(raw, nt=5)))
    g, = torch.autograd.grad(q.sum(), raw)
    check(float(g[:, 0].abs().max()) > 0.0, '∂Q/∂a0 非零')
    check(float(g[:, 1].abs().max()) > 0.0, '∂Q/∂a4 非零')
    check(float(g[:, 2].abs().max()) > 0.0, '∂Q/∂g 非零')
    # STE：round 本身导数 0，STE 前向取整、反向恒等
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


# ---------------------------------------------------------------- D agent 层
def test_agent_manifold():
    print('[D] agent：探索噪声 / random_action / target smoothing 都在流形上')
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
        if a.shape != (6,) or not np.allclose(a[1:4], ref[1:4], atol=1e-9):
            bad += 1
    check(bad == 0, 'select_action × 50：6 维、中间维 = 首尾线性插值（最大偏差 %.1e）' % max_err)
    ag.epsilon = 0.0
    a0 = ag.select_action(s)
    check(np.allclose(a0[1:4], model.proj_lin_wt_np(a0)[1:4], atol=1e-9),
          'ε=0：确定性动作仍在流形上')
    # random_action（warmup）
    bad_r = 0
    for _ in range(50):
        r = ag.random_action()
        if not (r.shape == (6,) and 0.0 <= r[0] <= 0.5 and 0.0 <= r[4] <= 0.5
                and -0.5 <= r[5] <= 0.5):
            bad_r += 1
            continue
        if not np.allclose(r[1:4], model.proj_lin_wt_np(r)[1:4], atol=1e-9):
            bad_r += 1
    check(bad_r == 0, 'random_action × 50：6 维、a0/a4∈[0,0.5]、g∈[-0.5,0.5]、中间线性')
    # main 会在构造后覆写 explore_amp[5]/floor[5]（HSRL 的 g 噪声量纲）
    # ⇒ `_add_noise_free` 必须**动态**读当前数组，不能缓存 __init__ 时的值。
    ag.explore_amp[5] = 0.5
    ag.explore_floor[5] = 0.025
    ag.epsilon = 1.0
    g_noise = np.array([ag._add_noise_free(np.zeros(3))[2] for _ in range(200)])
    check(np.abs(g_noise).max() > 0.3 and np.abs(g_noise).max() <= 0.5 + 1e-9,
          'HSRL 覆写 explore_amp[5]=0.5 后，g 噪声幅度随之生效（max=%.3f）'
          % np.abs(g_noise).max())
    # target smoothing：只在自由维加噪，展开后仍在流形上
    with torch.no_grad():
        sm = ag._target_smoothed_action(torch.randn(32, 4)).numpy()
    err_sm = max(float(np.abs(row[1:4] - model.proj_lin_wt_np(row)[1:4]).max())
                 for row in sm)
    check(sm.shape == (32, 6) and err_sm < 1e-6,
          'target smoothing：32×6 且中间维仍在流形上（最大偏差 %.1e）' % err_sm)
    # 对偶的 Q 分子也必须先把 3 输出展开成 6 维
    import aifflor.aifflor_hsrl as hsrl
    rate = hsrl.q_cost_rate(ag, s, a_t=np.zeros(6))
    check(np.isfinite(rate), 'q_cost_rate(lin)：3 输出经 _act 展开后 critic 可评估')
    # 对照：默认臂的 select_action 不要求线性
    ag2 = SP_TD3(nb_states=4, nb_actions=6, args=make_args(lin_wt=0))
    ag2.is_training, ag2.epsilon = True, 1.0
    neq = 0
    for _ in range(10):
        b = ag2.select_action(s)
        neq += int(not np.allclose(b[1:4], model.proj_lin_wt_np(b)[1:4], atol=1e-6))
    check(neq > 0, '对照（lin_wt=0）：中间维**不**满足线性关系（结构版确实生效）')


# ---------------------------------------------------------------- E checkpoint
def test_checkpoint():
    print('[E] checkpoint：actor_3.pkl + critic_*_6.pkl')
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    try:
        ag = SP_TD3(nb_states=4, nb_actions=6, args=make_args(lin_wt=1))
        ag.save_model(TMP_DIR)
        files = set(os.listdir(TMP_DIR))
        check('actor_3.pkl' in files, 'lin: 保存 actor_3.pkl')
        check(all('critic_%s_6.pkl' % k in files for k in ('p', 'p2', 'c', 'c2')),
              'lin: critic 仍按 6 维动作保存（critic_*_6.pkl）')
        ag2 = SP_TD3(nb_states=4, nb_actions=6, args=make_args(lin_wt=1))
        ag2.load_weights(TMP_DIR)
        ok = all(torch.equal(ag.actor.state_dict()[k], ag2.actor.state_dict()[k])
                 for k in ag.actor.state_dict())
        check(ok, 'lin: actor_3.pkl 可加载且权重逐位一致')
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


# ---------------------------------------------------------------- F 物理
def test_env_invariance():
    print('[F] 物理：同一 (a0,a4)、其余维任改 ⇒ 出力逐位相同')
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


# ---------------------------------------------------------------- G 初始化映射
def test_bias_mapping():
    print('[G] 初始化映射：3 输出头的 a0/a4/g bias')
    rng = np.random.RandomState(0)
    v3 = model.wt_bias_values(3, lo=0.05, hi=0.45, sig_value=0.0,
                              lin_wt=True, rng=rng)
    check(v3.shape == (3,) and 0.05 <= v3[0] <= 0.45 and 0.05 <= v3[1] <= 0.45,
          '3 输出：a0/a4 bias ∈ [0.05,0.45]（%s）' % np.round(v3, 3))
    check(v3[2] == 0.0, '3 输出：g bias = sig_value = 0')
    rng = np.random.RandomState(0)
    v6 = model.wt_bias_values(6, lo=0.05, hi=0.45, sig_value=0.0,
                              lin_wt=False, rng=rng)
    check(v6.shape == (6,) and np.all((v6[:5] >= 0.05) & (v6[:5] <= 0.45)),
          '6 输出（默认臂）：前 5 维仍 ∈ [0.05,0.45]')
    check(v6[5] == 0.0, '6 输出：signal bias = 0')
    # 同 seed 可复现
    rng = np.random.RandomState(7)
    check(np.array_equal(model.wt_bias_values(3, lin_wt=True, rng=rng),
                         model.wt_bias_values(3, lin_wt=True,
                                              rng=np.random.RandomState(7))),
          '同 seed 初始化可复现')


# ---------------------------------------------------------------- H 接线
def test_main_wiring():
    print('[H] 接线：两条主循环 + CLI + 结构版 actor')
    mp = os.path.join(ROOT, 'aifflor', 'aifflor_main.py')
    src_m = open(mp, encoding='utf-8').read()
    n_def = src_m.count('def _apply_lin_wt(')
    n_call = src_m.count('_apply_lin_wt(') - n_def
    check(n_def == 1, '定义 _apply_lin_wt 恰好 1 处')
    check(n_call >= 4, '_apply_lin_wt 调用点 ≥4（旧 train + HSRL 的 warmup/actor），实际 %d' % n_call)
    check("'--lin_wt'" in src_m and "'--lin_wt_decimals'" in src_m,
          'CLI 参数 --lin_wt / --lin_wt_decimals 已注册')
    check('--lin_wt 与 --scalar_wt / --farm_level 互斥' in src_m, '互斥断言已存在')
    check('_wt_bias_values' in src_m and 'is_lin' in src_m,
          'init_wt_action_bias 已接 3 输出映射')
    dp = os.path.join(ROOT, 'aifflor', 'aifflor_ddpg.py')
    src_d = open(dp, encoding='utf-8').read()
    check('Actor(self.nb_states, self.actor_nb_actions' in src_d,
          'actor 构造使用 actor_nb_actions（结构版 3 输出）')
    check("'{}/actor_{}.pkl'.format(output, self.actor_tag)" in src_d, 'save_model 使用 actor_tag')
    check("'{}/actor_{}.pkl'.format(output, self.actor_tag)" in src_d, 'load_weights 使用 actor_tag')
    check('def _target_smoothed_action' in src_d, 'target smoothing 已抽成自由维专用方法')


def main():
    test_expand_and_projection()
    test_actor_structure()
    test_gradient()
    test_agent_manifold()
    test_checkpoint()
    test_env_invariance()
    test_bias_mapping()
    test_main_wiring()
    print()
    if FAIL:
        print('✘ %d 项失败:' % len(FAIL))
        for f in FAIL:
            print('   - ' + f)
        return 1
    print('线性行为分支（--lin_wt，结构版）契约测试：全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
