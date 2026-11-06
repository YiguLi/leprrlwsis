#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run90 诊断：为什么 actor 停在"界外"（顺着线性罚梯度滑行）。

结构（run90 的 checkpoint 实测）：H actor 6→16→100→16→6（无输出压缩），
H critic 6→16→100→(∥a 6)→80→50→1。

在被控对象上直接前向测三件事：
  A. actor 的原始输出落在哪（是否已深在界外）；
  B. critic 的 ∂Qp/∂a、∂Qc/∂a 在 **界内 / 刚出界 / 深远界外 / 上界外** 四区的符号与量级
     —— 若界外仍然是"界内斜率"的延续（而不是"物理被 clip ⇒ 平坦 ⇒ 斜率 0"），
     就说明 MLP 对 clip 折点无能为力，而界外唯一斜率正确的一项只有线性罚；
  C. 合成梯度 ∂Qp/∂a − λ·∂Qc/∂a 在界外的符号（负 = 继续往外滑）。

用法: python agentutil/probe_run90_action_gradient.py [--run AIFFLOR-run90] [--n 300] [--dim 3]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import torch                     # noqa: E402
import torch.nn as nn            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='AIFFLOR-run90')
ap.add_argument('--n', type=int, default=300)
ap.add_argument('--dim', type=int, default=3)
ap.add_argument('--lam', type=float, default=5.0, help='actor 空间 λ（物理 50 × 0.1）')
a = ap.parse_args()
D = os.path.join('output', a.run)
DAY = 1440.0


class Actor(nn.Module):
    def __init__(self, ns, na, h1=16, h2=100):
        super().__init__()
        self.fc1 = nn.Linear(ns, h1); self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h1); self.fc4 = nn.Linear(h1, na)

    def forward(self, x):
        x = torch.relu(self.fc1(x)); x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x)); return self.fc4(x)


class Critic(nn.Module):
    def __init__(self, ns, na, h1=16, h2=100):
        super().__init__()
        self.fc1 = nn.Linear(ns, h1); self.fc4 = nn.Linear(h1, h2)
        self.fc2 = nn.Linear(h2 + na, 80); self.fc5 = nn.Linear(80, 50); self.fc3 = nn.Linear(50, 1)

    def forward(self, x, act):
        h = torch.relu(self.fc1(x)); h = self.fc4(h)
        h = torch.relu(self.fc2(torch.cat([h, act], 1)))
        h = torch.relu(self.fc5(h)); return self.fc3(h)


def load(cls, path, ns, na):
    net = cls(ns, na)
    net.load_state_dict(torch.load(path, map_location='cpu'))
    net.eval()
    return net


cols = ["step", "original_action_H", "phase_minute", "pr", "u", "SOC", "turbine_profit", "P_F"]
ch = next(pd.read_csv(D + r"\df_hsrl.csv", usecols=cols, chunksize=400000))
O = np.vstack([np.fromstring(str(x).strip().lstrip("[").rstrip("]"), sep=" ") for x in ch["original_action_H"]])
P = ch.drop(columns=["original_action_H"]).reset_index(drop=True)
P["PG"] = P.turbine_profit * 60.0 / P.pr
ang = 2.0 * np.pi * (P.phase_minute.to_numpy() % DAY) / DAY
S = np.column_stack([P.pr / 1000.0, P.u / 10.0, P.SOC / 0.5, P.PG / 5.0, np.sin(ang), np.cos(ang)])
idx = np.linspace(len(S) - 1, int(len(S) * 0.6), a.n).astype(int)

actor = load(Actor, os.path.join(D, 'actor_6.pkl'), 4, 6)
cp = load(Critic, os.path.join(D, 'critic_p_6.pkl'), 4, 6)
cc = load(Critic, os.path.join(D, 'critic_c_6.pkl'), 4, 6)

s = torch.tensor(S[idx][:, :4], dtype=torch.float32)   # H 状态 = 4 维（无相位）
araw = actor(s).detach().numpy()

print('=' * 92)
print('A. actor 原始输出（%d 个末期状态，%s）' % (len(idx), a.run))
for i in range(5):
    print('   dim%d: mean %+7.3f  p5 %+7.3f  p95 %+7.3f  min %+6.2f  max %+6.2f | 出界步占比 %5.1f%%'
          % (i, araw[:, i].mean(), *np.percentile(araw[:, i], [5, 95]), araw[:, i].min(),
             araw[:, i].max(), 100 * (np.abs(araw[:, i] - np.clip(araw[:, i], 0, 0.5)) > 1e-9).mean()))

d = a.dim
print('\n' + '=' * 92)
print('B/C. 沿 dim%d 扫描（其余维固定为 actor 的原始输出）' % d)
grids = [('界内 [0,0.5]', np.linspace(0.02, 0.48, 24)),
         ('刚出界 [−0.3,0)', np.linspace(-0.30, -0.02, 15)),
         ('深远界外 [−2,−0.5]', np.linspace(-2.0, -0.5, 16)),
         ('上界外 (0.5,1.0]', np.linspace(0.52, 1.0, 13))]
for name, g in grids:
    vals = []
    for gv in g:
        act = torch.tensor(araw, dtype=torch.float32).clone()
        act[:, d] = float(gv)
        with torch.no_grad():
            vals.append((cp(s, act).squeeze().numpy().mean(), cc(s, act).squeeze().numpy().mean()))
    v = np.array(vals)
    dqp = np.gradient(v[:, 0], g); dqc = np.gradient(v[:, 1], g)
    net = dqp - a.lam * dqc
    print('   %-18s ∂Qp/∂a %+7.4f (|·| %.4f) | ∂Qc/∂a %+7.4f | 合成 %+7.4f ⇒ %s'
          % (name, dqp.mean(), np.abs(dqp).mean(), dqc.mean(), net.mean(),
             '继续往外滑' if net.mean() < 0 else '往内走'))
print('\n  合成 = ∂Qp/∂a − λ·∂Qc/∂a（actor 的上升方向 = 加大 a；<0 ⇒ 被推向更小的 a）')
print('  λ = %.1f（actor 空间；物理 %.0f）' % (a.lam, a.lam * 10))
