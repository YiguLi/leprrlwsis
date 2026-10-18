#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""critic 结构基准：在同一批数据上比"能不能表达 u-a-pr"，用监督回归作弊式地量。

任务：
  T1 物理回归：y = P(u,a)                      （纯映射能力）
  T2 S1 仿真  ：y = 2.8·(1 + 0.017·mod(a)) + N(0,0.3)  （小调制 + 大电平 + 高噪声）

结构：
  A 现行        : s 4→16 , [100+6]→80→50→1
  B 动作首层    : [10]→64→100→80→50→1
  C 双流(用户版): s 4→16→100 ∥ a 6→32→64 → [164]→130→80→50→1
  D C + LeakyReLU
  E C + 线性跳连（Q = w·[s,a] + MLP）

指标：参数量 / 验证 R² / 拟合后 |∂y/∂a|（在数据上） / 激活单元数。

用法: python agentutil/bench_critic_arch.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import aifflor.aifflor_turbine as AT                      # noqa: E402
from aifflor.aifflor_env import AifflorPhysics            # noqa: E402


def relu(kind):
    return nn.ReLU() if kind == 'relu' else nn.LeakyReLU(0.01)


class A(nn.Module):                     # 现行
    def __init__(s, act='relu'):
        super().__init__(); s.fc1 = nn.Linear(4, 16); s.fc4 = nn.Linear(16, 100)
        s.fc2 = nn.Linear(106, 80); s.fc5 = nn.Linear(80, 50); s.fc3 = nn.Linear(50, 1)
        s.r = relu(act)

    def forward(s, x, a):
        h = s.r(s.fc1(x)); h = s.fc4(h)
        h = s.r(s.fc2(torch.cat([h, a], 1))); h = s.r(s.fc5(h)); return s.fc3(h)


class B(nn.Module):                     # 动作首层
    def __init__(s, act='relu'):
        super().__init__(); s.fc1 = nn.Linear(10, 64); s.fc4 = nn.Linear(64, 100)
        s.fc2 = nn.Linear(100, 80); s.fc5 = nn.Linear(80, 50); s.fc3 = nn.Linear(50, 1)
        s.r = relu(act)

    def forward(s, x, a):
        h = s.r(s.fc1(torch.cat([x, a], 1))); h = s.fc4(h)
        h = s.r(s.fc2(h)); h = s.r(s.fc5(h)); return s.fc3(h)


class C(nn.Module):                     # 双流（用户版）
    def __init__(s, act='relu', skip=False):
        super().__init__()
        s.sfc1 = nn.Linear(4, 16); s.sfc2 = nn.Linear(16, 100)
        s.afc1 = nn.Linear(6, 32); s.afc2 = nn.Linear(32, 64)
        s.m1 = nn.Linear(164, 130); s.m2 = nn.Linear(130, 80)
        s.m3 = nn.Linear(80, 50); s.out = nn.Linear(50, 1)
        s.r = relu(act); s.skip = skip
        if skip:
            s.lin = nn.Linear(10, 1)

    def forward(s, x, a):
        hs = s.r(s.sfc2(s.r(s.sfc1(x))))
        ha = s.r(s.afc2(s.r(s.afc1(a))))
        h = s.r(s.m1(torch.cat([hs, ha], 1))); h = s.r(s.m2(h))
        h = s.r(s.m3(h)); q = s.out(h)
        if s.skip:
            q = q + s.lin(torch.cat([x, a], 1))
        return q


def parametrize(m):
    return sum(p.numel() for p in m.parameters())


def alive(m, X, A_, act):
    with torch.no_grad():
        if isinstance(m, C):
            hs = m.r(m.sfc2(m.r(m.sfc1(X)))); ha = m.r(m.afc2(m.r(m.afc1(A_))))
            h1 = m.r(m.m1(torch.cat([hs, ha], 1))); h2 = m.r(m.m2(h1)); h3 = m.r(m.m3(h2))
            return int((h1 > 0).any(0).sum()), h1.shape[1], int((h2 > 0).any(0).sum()), h2.shape[1]
        if isinstance(m, B):
            z = m.fc1(torch.cat([X, A_], 1)); h = m.r(z)
            return int((h > 0).any(0).sum()), h.shape[1], 0, 0
        z = m.fc1(X); h = m.r(z)
        return int((h > 0).any(0).sum()), h.shape[1], 0, 0


def grad_a(m, X, A_):
    a = A_.clone().requires_grad_(True)
    q = m(X, a).mean()
    g, = torch.autograd.grad(q, a)
    return float(g.abs().mean())


# ---------------- 数据 ----------------
torch.manual_seed(0); np.random.seed(0)
PH = AifflorPhysics(nt=5, calibrated=False, power_scale=AT.LEVEL_COEF)
N = 12000
u = np.random.uniform(4, 18, N); Araw = np.random.uniform(0.02, 0.48, (N, 5))
farm, _ = AT.eval_batch(PH.model, u, Araw)
P = np.asarray(farm, float) * PH.power_scale
A6 = np.hstack([Araw, np.zeros((N, 1))]).astype(np.float32)
X = np.column_stack([np.full(N, 580 / 1000.0), u / 10, np.full(N, 0.5), P / 5]).astype(np.float32)

gCp = 4 * Araw[:, 0] * (1 - Araw[:, 0]) ** 2                 # 只用 dim0 的 Cp 当调制
mod = (gCp - 0.2) / 0.5
Y1 = P.reshape(-1, 1).astype(np.float32)                      # T1
Y2 = (2.8 * (1 + 0.017 * mod)).reshape(-1, 1).astype(np.float32)   # T2 干净
n = int(N * 0.8)
tr = slice(0, n); va = slice(n, N)

CFG = [('A 现行', lambda: A('relu'), False),
       ('B 动作首层', lambda: B('relu'), False),
       ('C 双流(relu)', lambda: C('relu'), False),
       ('D 双流(leaky)', lambda: C('leaky'), False),
       ('E 双流(leaky)+skip', lambda: C('leaky', skip=True), True)]

for tname, Y, noise in (('T1 P(u,a)', Y1, 0.0), ('T2 S1 仿真', Y2, 0.30)):
    print('\n' + '=' * 100)
    print('%s（%d 样本，噪声 %.2f，训练 1500 步 Adam lr=1e-2）' % (tname, N, noise))
    print('  %-18s %7s %8s %8s %11s %10s %10s' % ('结构', '参数', 'R²(真值)', 'R²(干净)', '|dy/da|', '活单元1', '活单元2'))
    Xt = torch.tensor(X[tr]); At = torch.tensor(A6[tr])
    Xv = torch.tensor(X[va]); Av = torch.tensor(A6[va])
    Yt = torch.tensor(Y[tr]); Yv = torch.tensor(Y[va])
    for name, mk, _ in CFG:
        torch.manual_seed(1)
        m = mk(); opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for i in range(1500):
            tgt = Yt + noise * torch.randn_like(Yt) if noise > 0 else Yt
            opt.zero_grad(); loss = ((m(Xt, At) - tgt) ** 2).mean(); loss.backward(); opt.step()
        with torch.no_grad():
            pv = m(Xv, Av)
        r2_true = 1 - ((pv - Yv) ** 2).sum().item() / ((Yv - Yv.mean()) ** 2).sum().item()
        r2_clean = r2_true
        a1, n1, a2, n2 = alive(m, Xv, Av, 'relu')
        print('  %-18s %7d %8.4f %8.4f %11.2e %6d/%-3d %6d/%-3d'
              % (name, parametrize(m), r2_true, r2_clean, grad_a(m, Xv, Av), a1, n1, a2, n2))
