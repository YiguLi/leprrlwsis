#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_action_state_sensitivity.py —— 确定性策略**用不用状态**？有没有漂出动作带？

动机（两件事一起量）
--------------------
1. 训练日志里的 `action_excess` 是**带噪动作**（actor 输出 + 探索噪声）的越界量，
   分不清"策略自己走出去了"还是"策略贴着边界、噪声把它推出去"。这里把 CSV 里的
   真实状态面板回放给各 checkpoint 的 actor，**不加噪声**，直接看每维分布与越界率。
2. `model.Actor` 的输出层 `fc4.weight` 初始化到 ±3e-3，而 `init_wt_action_bias` 把
   `fc4.bias` 放到 U(−1,1) ⇒ **初始时动作几乎等于常数 bias**。所以还要问：
   训练 `k` 步之后，"动作随状态变化的那部分"（= 各维对状态面板的 **sd**）长大了没有？
   sd ≪ 0.1 就意味着策略仍是"固定偏航"，`test9` 的 gain 全部来自常数偏移。

判据
----
* **sd ≪ 1**（动作带半宽）⇒ 策略对状态不敏感（常数偏航）；
* `mean` 越过 ±1 且占比高 ⇒ **常量项漂出动作带**（env 侧夹住 ⇒ 有效策略贴边饱和，
  奖励里同时多出"越界罚"这一项随策略自身变化的分量）；
* 同一探针支持 **yawflor**（`--layout yawflor`，6 维观测/6 维动作）⇒ 可直接跨臂比较。
* 网络维度从 checkpoint 的**形状**反推（历史 16/100 与 64/128 两种格式都能读）。

`--decompose`：把"**为什么是常数**"的四组机制量一并打出来
---------------------------------------------------------
    [A] 逐层状态方差传递（input → fc1relu → fc2relu → fc3relu）：表征层把状态变化
        衰减了几倍；
    [B] 动作分解 a(s) = [W4·mean(h3) + b4]（**常数项**）+ W4·(h3(s) − mean(h3))（**状态项**）
        ⇒ 状态项占常数项多少（≪1 ⇒ 策略本质是常数）；
    [C] 输出层生长速度 vs **Adam 步长上限**（更新次数 × lr）：实测/上限 ≈ 100% ⇒
        不是"没梯度"，而是"每步最多走 lr，预算不够"；
    [D] critic 的 ∂Q/∂a（**逐样本**，用 `autograd.grad(q.sum())`；⚠ 用 `q.mean().backward()`
        会把梯度除以批量大小，容易误判成"critic 对动作不敏感"）。

用法
----
    python agentutil/probe_action_state_sensitivity.py <run_dir> [--layout yawprofit|yawflor] [--n 4000]
    python agentutil/probe_action_state_sensitivity.py <run_dir> --decompose [--prate 1e-4]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))              # agentutil/
ROOT = os.path.dirname(_HERE)                                   # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

try:
    sys.stdout.reconfigure(errors='replace')
except Exception:                                               # pragma: no cover
    pass

import torch                                                    # noqa: E402
from model import Actor                                         # noqa: E402

LAYOUTS = {
    # layout: (CSV 名, nb_states, nb_actions, 状态构造函数)
    'yawprofit': ('df_yawprofit.csv', 4, 5, None),
    'yawflor': ('df_yawflor.csv', 6, 6, None),
}


def _parse_vec(s):
    if isinstance(s, str):
        return np.array(s.strip().strip('[]').split(), dtype=float)
    return np.asarray(s, dtype=float)


def _latest_run():
    cands = [d for d in os.listdir('output')
             if d.startswith('YAWPROFIT') and os.path.isdir(os.path.join('output', d))
             and os.path.exists(os.path.join('output', d, 'df_yawprofit.csv'))]
    cands.sort(key=lambda d: int(d.split('-run')[-1]) if '-run' in d else -1)
    return os.path.join('output', cands[-1])


def _load_actor(path, nb_states=None):
    """从 checkpoint 的**形状**反推网络维度（历史 run 的 16/100 与 64/128 两种格式都能读）。"""
    sd = torch.load(path, map_location='cpu')
    h1 = int(sd['fc1.weight'].shape[0])
    h2 = int(sd['fc2.weight'].shape[0])
    ns = int(sd['fc1.weight'].shape[1])
    na = int(sd['fc4.weight'].shape[0])
    if nb_states is not None and ns != nb_states:
        raise SystemExit('checkpoint nb_states={} != 布局推得的 {}'.format(ns, nb_states))
    net = Actor(ns, na, hidden1=h1, hidden2=h2, init_w=3e-3)
    net.load_state_dict(sd)
    net.eval()
    return net, (h1, h2, ns, na)


def _stats(net, S):
    with torch.no_grad():
        A = net(torch.tensor(S, dtype=torch.float32)).numpy()
    out = []
    for j in range(A.shape[1]):
        a = A[:, j]
        out.append((a.mean(), a.std(), float((a > 1).mean()), float((a < -1).mean())))
    return A, out


def _decompose(run, S, nb_actions, prate=1e-4, init_w=3e-3, policy_delay=2):
    """把"为什么策略是常数"拆成四组可验证的量（见模块 docstring 的 [A]~[D]）。"""
    from model import Critic
    x = torch.tensor(S, dtype=torch.float32)
    sa = torch.load(os.path.join(run, 'actor_{}.pkl'.format(nb_actions)), map_location='cpu')
    net = Actor(int(sa['fc1.weight'].shape[1]), nb_actions,
                hidden1=int(sa['fc1.weight'].shape[0]),
                hidden2=int(sa['fc2.weight'].shape[0]), init_w=init_w)
    net.load_state_dict(sa)
    net.eval()

    with torch.no_grad():
        h1 = torch.relu(net.fc1(x))
        h2 = torch.relu(net.fc2(h1))
        h3 = torch.relu(net.fc3(h2))
        A = net.fc4(h3).numpy()

    # ---- [A] 逐层状态方差传递 ----
    print('\n[A] 逐层状态方差传递（跨 {} 个真实状态）'.format(len(S)))
    print('    {:<10} {:>10} {:>12} {:>12}'.format('layer', 'mean', 'sd(跨状态)', 'sd_out/sd_in'))
    prev = S.std(0).mean()
    print('    {:<10} {:>10.4f} {:>12.4f} {:>12}'.format('input', float(S.mean()), float(prev), '—'))
    for name, h, inp in (('fc1relu', h1, S), ('fc2relu', h2, h1), ('fc3relu', h3, h2)):
        H = h.numpy()
        i_np = inp if isinstance(inp, np.ndarray) else inp.numpy()
        print('    {:<10} {:>10.4f} {:>12.4f} {:>12.4f}'.format(
            name, float(H.mean()), float(H.std(0).mean()),
            float(H.std(0).mean() / max(1e-12, i_np.std(0).mean()))))

    # ---- [B] 动作 = 常数项 + 状态项 ----
    W4 = net.fc4.weight.data.numpy()
    b4 = net.fc4.bias.data.numpy()
    H3 = h3.numpy()
    const = W4 @ H3.mean(0) + b4
    state = (H3 - H3.mean(0)) @ W4.T
    c_sd = state.std(0)
    print('\n[B] 动作分解: 常数项 = b4 + W4·mean(h3)；状态项 = W4·(h3(s) − mean(h3))')
    print('    {:<5} {:>9} {:>12} {:>10} {:>11} {:>12}'.format(
        'dim', 'b4', 'W4·mean(h3)', '常数合计', '状态项 sd', '状态/|常数|'))
    for j in range(A.shape[1]):
        print('    {:<5} {:>9.3f} {:>12.3f} {:>10.3f} {:>11.4f} {:>11.2%}'.format(
            j, b4[j], float(W4[j] @ H3.mean(0)), const[j], c_sd[j],
            c_sd[j] / max(1e-9, abs(const[j]))))
    print('    平均: |常数|={:.3f}  状态项 sd={:.4f}  ⇒  状态/常数 = {:.2%}'.format(
        float(np.abs(const).mean()), float(c_sd.mean()),
        float(c_sd.mean() / max(1e-9, np.abs(const).mean()))))

    # ---- [C] 生长速度 vs Adam 上限 ----
    print('\n[C] 输出层 |fc4.weight| 生长 vs Adam 步长上限')
    grew = float(np.abs(W4).max()) - init_w
    upd = None
    try:
        n_blocks = len(pd.read_csv(os.path.join(run, 'update_stats_H.csv')))
        upd = n_blocks * 2 // policy_delay
    except Exception:
        pass
    print('    |W4|: mean={:.5f} max={:.5f} (init 上界 {:.5f}) ⇒ 最大生长 {:.5f}'
          .format(float(np.abs(W4).mean()), float(np.abs(W4).max()), init_w, grew))
    if upd:
        print('    actor 更新次数 ≈ {}（update_stats 行数 × 2 / policy_delay {}）；'
              'Adam 上限 = {} × prate {:.1e} = {:.5f} ⇒ **实际/上限 = {:.1%}**'
              .format(upd, policy_delay, upd, prate, upd * prate,
                      grew / max(1e-12, upd * prate)))
        print('    ⚠ ≈100% ⇒ 不是"梯度缺失"，而是"每步最多走 lr，预算不够"')

    # ---- [D] critic 对动作的敏感度（逐样本梯度）----
    try:
        sc = torch.load(os.path.join(run, 'critic_p_{}.pkl'.format(nb_actions)),
                        map_location='cpu')
    except Exception as e:                                       # pragma: no cover
        print('\n[D] 跳过（读不到 critic_p: {}）'.format(e))
        return
    cri = Critic(int(sc['fc1.weight'].shape[1]), nb_actions,
                 hidden1=int(sc['fc1.weight'].shape[0]),
                 hidden2=int(sc['fc4.weight'].shape[0]), init_w=init_w)
    cri.load_state_dict(sc)
    cri.eval()
    a_t = torch.tensor(A, dtype=torch.float32).clone().requires_grad_(True)
    q = cri([x, a_t])
    g, = torch.autograd.grad(q.sum(), a_t)      # ⚠ 用 sum(): mean() 会把梯度除以批量大小
    g = g.numpy()
    print('\n[D] critic ∂Q/∂a（逐样本；`autograd.grad(q.sum())`；critic 在 r1/s_p 空间回归）')
    for j in range(nb_actions):
        print('    dim{}: mean={:+.4f} sd={:.4f} 相对 sd={:>7.2%}'.format(
            j, g[:, j].mean(), g[:, j].std(), g[:, j].std() / max(1e-9, abs(g[:, j].mean()))))
    print('    |∂Q/∂a| 平均 = {:.4f}；相对 sd（去均值部分/均值部分）= {:.1%}'
          .format(float(np.abs(g).mean()),
                  float(np.abs(g - g.mean(0)).mean() / max(1e-9, np.abs(g.mean(0)).mean()))))


def main(run=None, layout='yawprofit', n=4000, decompose=False, prate=1e-4):
    run = run or _latest_run()
    csv_name, nb_states, nb_actions, _ = LAYOUTS[layout]
    if layout == 'yawprofit':
        from yawprofit.yawprofit_myfunc import Normalize_H
    else:
        from yawflor.yawflor_myfunc import Normalize_H

    df = pd.read_csv(os.path.join(run, csv_name))
    df = df.iloc[::max(1, len(df) // n)].reset_index(drop=True)
    obs = np.vstack(df.observation.apply(_parse_vec).values)
    S = np.vstack([Normalize_H(obs[i], int(df.phase_minute.iloc[i]), 1)
                   for i in range(len(obs))])
    print('run = {} | layout={} | 回放 {} 个真实状态 (step {}..{}) | nb_states={} nb_actions={}'
          .format(run, layout, len(S), df.step.min(), df.step.max(), nb_states, nb_actions))

    ckpts = []
    for sub in sorted((d for d in os.listdir(run) if d.startswith('checkpoint_')),
                      key=lambda d: int(d.split('_')[1])):
        p = os.path.join(run, sub, 'actor_{}.pkl'.format(nb_actions))
        if os.path.exists(p):
            ckpts.append((sub.replace('checkpoint_', 'step '), p))
    p = os.path.join(run, 'actor_{}.pkl'.format(nb_actions))
    if os.path.exists(p):
        ckpts.append(('final(=step{})'.format(df.step.max() + 1), p))

    for tag, path in ckpts:
        net, dims = _load_actor(path, nb_states=nb_states)
        A, st = _stats(net, S)
        print('\n--- {} --- (hidden={}/{}, nb_states={}, nb_actions={})'.format(
            tag, dims[0], dims[1], dims[2], dims[3]))
        print('  整体: mean={:+.3f} sd={:.3f} | 越上界维度占比={:.1%} 越下界={:.1%} | '
              '至少一维越界的样本占比={:.1%}'.format(
                  A.mean(), A.std(), float((A > 1).mean()), float((A < -1).mean()),
                  float((np.abs(A) > 1).any(axis=1).mean())))
        print('  状态敏感度（每维 sd，偏航维平均 = {:.4f}）:'.format(
            float(np.mean([s for _, s, _, _ in st[:5]]))))
        for j, (m, s, hu, hl) in enumerate(st):
            print('  dim{}: mean={:+.3f} sd={:.3f} 越界占比={:.1%} (上 {:.1%} / 下 {:.1%})'.format(
                j, m, s, hu + hl, hu, hl))

    if decompose:
        _decompose(run, S, nb_actions, prate=prate)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run', nargs='?', default=None)
    ap.add_argument('--layout', default='yawprofit', choices=list(LAYOUTS))
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--decompose', action='store_true',
                    help='额外打印"为什么是常数"的四组机制量 ([A] 逐层方差传递 / [B] 动作分解 / '
                         '[C] 生长速度 vs Adam 上限 / [D] critic ∂Q/∂a)')
    ap.add_argument('--prate', type=float, default=1e-4, help='actor lr（[C] 的 Adam 上限用）')
    a = ap.parse_args()
    main(a.run, a.layout, a.n, a.decompose, a.prate)
