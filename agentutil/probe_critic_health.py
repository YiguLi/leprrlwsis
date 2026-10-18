#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""critic 健康检查：死单元 / Q 电平 / 动作坡度 / 是否退化成常数。

用法: python agentutil/probe_critic_health.py output/DIAG-S1 output/DIAG-S2 output/AIFFLOR-run90
判据：
  * fc1 活单元 = 0 ⇒ **状态路径全死**（Q 退化为 fc3.bias，既无状态也无动作依赖）
  * |∂Q/∂a| < 1e-3 且 Q std / |Q| < 1% ⇒ 动作通道被压成 0（actor 拿不到梯度）
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)


class Critic(nn.Module):
    def __init__(s, ns=4, na=6, h1=16, h2=100):
        super().__init__()
        s.fc1 = nn.Linear(ns, h1); s.fc4 = nn.Linear(h1, h2)
        s.fc2 = nn.Linear(h2 + na, 80); s.fc5 = nn.Linear(80, 50); s.fc3 = nn.Linear(50, 1)

    def forward(s, x, a):
        h = torch.relu(s.fc1(x)); h = s.fc4(h)
        h = torch.relu(s.fc2(torch.cat([h, a], 1)))
        h = torch.relu(s.fc5(h)); return s.fc3(h)


def check(run_dir, which='critic_p_6.pkl'):
    D = max(glob.glob(os.path.join(run_dir, 'AIFFLOR-run*')), key=os.path.getmtime, default=run_dir)
    p = os.path.join(D, which)
    if not os.path.exists(p):
        print('%-26s 缺 %s' % (os.path.basename(run_dir), which)); return
    sd = torch.load(p, map_location='cpu')
    df = pd.read_csv(os.path.join(D, 'df_hsrl.csv'),
                     usecols=['pr', 'u', 'SOC', 'turbine_profit', 'applied_action_H'])
    A = np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                   for x in df.applied_action_H])
    PG = (df.turbine_profit * 60 / df.pr).to_numpy()
    S = torch.tensor(np.column_stack([df.pr / 1000, df.u / 10, df.SOC / 0.5, PG / 5]).astype(np.float32))
    a0 = torch.tensor(A.astype(np.float32), requires_grad=True)

    z1 = S @ sd['fc1.weight'].T + sd['fc1.bias']
    r1 = torch.relu(z1); z4 = r1 @ sd['fc4.weight'].T + sd['fc4.bias']
    z2 = torch.cat([z4, a0], 1) @ sd['fc2.weight'].T + sd['fc2.bias']; r2 = torch.relu(z2)
    z5 = r2 @ sd['fc5.weight'].T + sd['fc5.bias']; r5 = torch.relu(z5); q = r5 @ sd['fc3.weight'].T + sd['fc3.bias']

    g, = torch.autograd.grad(q.mean(), a0)
    g = g.detach().numpy()[:, :5]
    qn = q.detach().numpy().ravel()
    print('%-26s fc1 活 %2d/%d | fc2 活 %2d/%d | fc5 活 %2d/%d | Q 均 %7.3f std %7.4f '
          '(std/|mean| %.3f%%) | fc3.bias %7.3f | |dQ/da| 均 %.2e'
          % (os.path.basename(run_dir), int((z1 > 0).any(0).sum()), z1.shape[1],
             int((z2 > 0).any(0).sum()), z2.shape[1], int((z5 > 0).any(0).sum()), z5.shape[1],
             qn.mean(), qn.std(), 100 * qn.std() / max(abs(qn.mean()), 1e-9),
             sd['fc3.bias'].item(), np.abs(g).mean()))


if __name__ == '__main__':
    for d in (sys.argv[1:] or ['output/DIAG-S1', 'output/DIAG-S2']):
        check(d)
