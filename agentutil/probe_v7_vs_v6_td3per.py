#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读探针：确认 v7 的 checkpoint **真的换成了 v7 网络，且策略/价值函数与 v6 不同**。

不跑仿真、不 import `*_myfunc`（避免启 MATLAB）：直接按权重形状重建
`model.Actor` / `model.Critic`，在同一批固定状态上比两个 checkpoint 的输出。

用法::

    python agentutil/probe_v7_vs_v6_td3per.py \
        --v6 output/TD3PER-run19/checkpoint_24000 \
        --v7 output/td3perv7branch5-run26/checkpoint_24000
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np          # noqa: E402
import torch                # noqa: E402
import model                # noqa: E402


def build_actor(sd):
    h1 = int(sd['fc1.weight'].shape[0])
    h2 = int(sd['fc2.weight'].shape[0])
    a = model.Actor(int(sd['fc1.weight'].shape[1]), int(sd['fc4.weight'].shape[0]),
                    hidden1=h1, hidden2=h2)
    a.load_state_dict(sd)
    a.eval()
    return a


def build_critic(sd, ns, na):
    arch = model.infer_arch(sd, ns, na)
    q = model.Critic(ns, na, arch=arch)
    q.load_state_dict(sd)
    q.eval()
    return q, arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v6', required=True, help='v6 checkpoint 目录（含 actor_6.pkl）')
    ap.add_argument('--v7', required=True, help='v7 checkpoint 目录')
    a = ap.parse_args()

    # 固定状态面板（td3per 的 Normalize_H 口径: pr/1000, u/10, SOC/0.5, P_G/5）
    grid = []
    for pr in (300.0, 580.0, 980.0):
        for u in (4.0, 8.0, 12.0, 16.0, 20.0):
            for soc in (0.1, 0.5):
                for pg in (0.0, 2.0, 5.0):
                    grid.append([pr / 1000.0, u / 10.0, soc / 0.5, pg / 5.0])
    X = torch.tensor(grid, dtype=torch.float32)
    print('固定状态面板: %d 个' % X.shape[0])

    acts = {}
    for tag, d in (('v6', a.v6), ('v7', a.v7)):
        sd = torch.load(os.path.join(d, 'actor_6.pkl'), map_location='cpu')
        act = build_actor(sd)
        n = sum(p.numel() for p in act.parameters())
        shapes = [tuple(m.weight.shape) for m in (act.fc1, act.fc2, act.fc3, act.fc4)]
        with torch.no_grad():
            y = act(X).numpy()
        acts[tag] = y
        print('%s actor: %d 参数, 层 %s' % (tag, n, shapes))
        print('   动作均值 %s' % np.round(y.mean(0), 4))
        print('   动作 sd  %s' % np.round(y.std(0), 4))

    d = acts['v7'] - acts['v6']
    print('\n策略差异（同状态面板）:')
    print('  逐维 |Δa| 均值 %s' % np.round(np.abs(d).mean(0), 4))
    print('  逐维相关系数 %s' % np.round([np.corrcoef(acts['v6'][:, j], acts['v7'][:, j])[0, 1]
                                       for j in range(d.shape[1])], 3))
    print('  最大 |Δa| %.4f' % np.abs(d).max())

    print('\n价值函数（同一批 (s, a)，各用自己的 actor 动作）:')
    for tag, d_ in (('v6', a.v6), ('v7', a.v7)):
        sd = torch.load(os.path.join(d_, 'critic_p_6.pkl'), map_location='cpu')
        q, arch = build_critic(sd, 4, 6)
        with torch.no_grad():
            qv = q([X, torch.tensor(acts[tag], dtype=torch.float32)]).numpy().reshape(-1)
        print('  %s critic: arch=%s, %d 参数, Q 均值 %.4f, Q sd %.4f'
              % (tag, arch, sum(p.numel() for p in q.parameters()),
                 float(qv.mean()), float(qv.std())))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
