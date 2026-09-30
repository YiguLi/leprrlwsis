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

用法
----
    python agentutil/probe_action_state_sensitivity.py <run_dir> [--layout yawprofit|yawflor] [--n 4000]
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


def main(run=None, layout='yawprofit', n=4000):
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
        print('  状态敏感度（每维 sd，5 个偏航维的平均 = {:.4f}）:'.format(
            float(np.mean([s for _, s, _, _ in st[:5]]))))
        for j, (m, s, hu, hl) in enumerate(st):
            print('  dim{}: mean={:+.3f} sd={:.3f} 越界占比={:.1%} (上 {:.1%} / 下 {:.1%})'.format(
                j, m, s, hu + hl, hu, hl))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run', nargs='?', default=None)
    ap.add_argument('--layout', default='yawprofit', choices=list(LAYOUTS))
    ap.add_argument('--n', type=int, default=4000)
    a = ap.parse_args()
    main(a.run, a.layout, a.n)
