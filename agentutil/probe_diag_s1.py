#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1（恒风恒价、纯出力最大化）诊断：critic 知不知道最优？actor 跟不跟？

判据：
  * 若 critic 的 argmax 指向 `a*`（refs_s1.csv）而 actor 停在别处 ⇒ **actor 优化失败**
  * 若 critic 的 argmax 就在 actor 当前位置 ⇒ **critic 本身没学到 a→P**

用法: python agentutil/probe_diag_s1.py [--run output/DIAG-S1] [--n 200]
"""
import argparse
import glob
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
ap.add_argument('--run', default=os.path.join('output', 'DIAG-S1'))
ap.add_argument('--n', type=int, default=200)
a = ap.parse_args()
D = max(glob.glob(os.path.join(a.run, 'AIFFLOR-run*')), key=os.path.getmtime)


class Actor(nn.Module):
    def __init__(self, ns=4, na=6, h1=16, h2=100):
        super().__init__()
        self.fc1 = nn.Linear(ns, h1); self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h1); self.fc4 = nn.Linear(h1, na)

    def forward(self, x):
        x = torch.relu(self.fc1(x)); x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x)); return self.fc4(x)


class Critic(nn.Module):
    def __init__(self, ns=4, na=6, h1=16, h2=100):
        super().__init__()
        self.fc1 = nn.Linear(ns, h1); self.fc4 = nn.Linear(h1, h2)
        self.fc2 = nn.Linear(h2 + na, 80); self.fc5 = nn.Linear(80, 50); self.fc3 = nn.Linear(50, 1)

    def forward(self, x, act):
        h = torch.relu(self.fc1(x)); h = self.fc4(h)
        h = torch.relu(self.fc2(torch.cat([h, act], 1)))
        h = torch.relu(self.fc5(h)); return self.fc3(h)


def load(cls, name):
    net = cls()
    net.load_state_dict(torch.load(os.path.join(D, name), map_location='cpu'))
    net.eval()
    return net


cols = ["step", "u", "pr", "SOC", "turbine_profit", "applied_action_H", "P_F", "r1"]
df = pd.read_csv(os.path.join(D, 'df_hsrl.csv'), usecols=cols)
A = np.vstack([np.fromstring(str(x).strip().lstrip("[").rstrip("]"), sep=" ") for x in df.applied_action_H])
df = df.drop(columns=["applied_action_H"]).reset_index(drop=True)
df["PG"] = df.turbine_profit * 60.0 / df.pr
df["a"] = A[:, :5].mean(1)

P_MAX = 3.7009                                  # refs_s1.csv（u=10 的生产物理最优）
A_STAR = np.array([0.34, 0.50, 0.24, 0.26, 1 / 3.0])

print('=' * 90)
print('A. S1 训练轨迹（%s，%d 步）' % (os.path.basename(D), len(df)))
print('   分 10 段：η = PG/P_max  与  5 维动作均值')
for k in range(10):
    s, e = k * len(df) // 10, (k + 1) * len(df) // 10
    print('   step %6d-%-6d  η %.3f  r1 %.2f  |  a = %s  |  P_F %.3f'
          % (df.step[s], df.step[e - 1], (df.PG[s:e] / P_MAX).mean(), df.r1[s:e].mean(),
             ' '.join('%.3f' % v for v in A[s:e, :5].mean(0)), df.P_F[s:e].mean()))
print('   a* = %s  ⇒ 最优 P = %.3f MW' % (' '.join('%.2f' % v for v in A_STAR), P_MAX))

# ---- critic 的 Q_p 沿每个维度 ----
actor = load(Actor, 'actor_6.pkl')
cp = load(Critic, 'critic_p_6.pkl')
idx = np.linspace(len(df) - 1, int(len(df) * 0.8), a.n).astype(int)
S = np.column_stack([df.pr.values / 1000.0, df.u.values / 10.0, df.SOC.values / 0.5, df.PG.values / 5.0])[idx]
st = torch.tensor(S, dtype=torch.float32)
with torch.no_grad():
    raw = actor(st).numpy()

print('\n' + '=' * 90)
print('B. critic 的 Q_p(a_j) 剖面（其余维固定在 actor 的原始输出）与 actor 的位置')
print('   %-5s %10s %10s %10s %10s' % ('dim', 'actor均值', 'a*', 'Q argmax', 'Q(argmax)-Q(actor)'))
cands = np.arange(0.02, 0.5001, 0.02)
for j in range(5):
    qs = []
    for c in cands:
        act = torch.tensor(raw, dtype=torch.float32).clone()
        act[:, j] = float(c)
        with torch.no_grad():
            qs.append(float(cp(st, act).mean()))
    qs = np.array(qs)
    k = int(np.argmax(qs))
    q_actor = float(np.interp(np.clip(raw[:, j].mean(), cands[0], cands[-1]), cands, qs))
    print('   %-5d %10.3f %10.2f %10.2f %10.4f' % (j, raw[:, j].mean(), A_STAR[j], cands[k], qs[k] - q_actor))

print('\n   若 "Q argmax" 明显偏向 a* 而 actor 没动 ⇒ actor 优化失败；')
print('   若 "Q argmax" ≈ actor 当前位置 ⇒ critic 没学到 a→P。')
