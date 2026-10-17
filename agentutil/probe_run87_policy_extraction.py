#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run87 诊断 6：**局部梯度 vs 全局策略提取** —— 同一对 critic，换 5 维 a 的取法（只读）。

对每个状态，比较三种"从同一对 critic 取动作"的方式：
  actor      : a = π_θ(s)                      （DDPG/TD3 的做法：只沿 ∂Q/∂a 局部走）
  grid-argmax: a = argmax_a [Q_p − λ·Q_c]      （在 a 网格上全局取，等价于"策略改进一步"）
  clamp-a    : a = clip(π_θ(s)) 到 [0, 0.5]    （实际执行的动作）

若 grid-argmax 与 actor 在**同一对 critic** 上差很多（尤其高风速段），说明
"跨不过悬崖"卡在 **actor 的局部更新** 上，而不是 critic 的估值上。

用法: python agentutil/probe_run87_policy_extraction.py [run_dir]
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import model as M  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join('output', 'AIFFLOR-run87')
CKPT = os.path.join(RUN, 'checkpoint_272000')
LAM = 5.0

sd_a = torch.load(os.path.join(CKPT, 'actor_6.pkl'), map_location='cpu')
sd_p = torch.load(os.path.join(CKPT, 'critic_p_6.pkl'), map_location='cpu')
sd_c = torch.load(os.path.join(CKPT, 'critic_c_6.pkl'), map_location='cpu')
actor = M.Actor(4, 6, hidden1=sd_a['fc1.weight'].shape[0], hidden2=sd_a['fc2.weight'].shape[0])
actor.load_state_dict(sd_a); actor.eval()
cp = M.Critic(4, 6, hidden1=sd_p['fc1.weight'].shape[0], hidden2=sd_p['fc4.weight'].shape[0])
cp.load_state_dict(sd_p); cp.eval()
cc = M.Critic(4, 6, hidden1=sd_c['fc1.weight'].shape[0], hidden2=sd_c['fc4.weight'].shape[0])
cc.load_state_dict(sd_c); cc.eval()


def arr(s):
    return np.fromstring(str(s).strip().lstrip('[').rstrip(']'), sep=' ')


O, U, PF = [], [], []
for ch in pd.read_csv(os.path.join(RUN, 'df_hsrl.csv'), usecols=['step', 'observation', 'u', 'P_F'],
                      chunksize=100000):
    keep = ch['step'].to_numpy() > 240000
    if not keep.any():
        continue
    c = ch[keep]
    O.append(np.vstack([arr(x) for x in c['observation']]))
    U.append(c['u'].to_numpy(float)); PF.append(c['P_F'].to_numpy(float))
O = np.vstack(O); U = np.concatenate(U); PF = np.concatenate(PF)
rs = np.random.RandomState(0)
idx = rs.choice(len(O), size=min(20000, len(O)), replace=False)
O, U, PF = O[idx], U[idx], PF[idx]
st = torch.tensor(np.stack([O[:, 0] / 1000, O[:, 1] / 10, O[:, 2] / 0.5, O[:, 3] / 5], 1),
                  dtype=torch.float32)

with torch.no_grad():
    a_act = actor(st)
    a_clip = a_act.clone()
    a_clip[:, :5] = a_clip[:, :5].clamp(0, 0.5)
    g_dim = a_act[:, 5:].clone()
    grid = np.concatenate([np.arange(0.0, 0.205, 0.01), [1 / 3.0]])
    best_f = torch.full((st.shape[0],), -1e18)
    best_a = torch.zeros(st.shape[0])
    f_act = None
    for a0 in grid:
        a = torch.cat([torch.full((st.shape[0], 5), float(a0)), g_dim], 1)
        f = cp([st, a]) - LAM * cc([st, a])
        f = f.squeeze(1)
        if abs(a0 - float(a_act[:, 0:5].mean())) < 1e-9:
            f_act = f
        m = f > best_f
        best_f = torch.where(m, f, best_f)
        best_a = torch.where(m, torch.full_like(best_a, float(a0)), best_a)
    # actor 自身的 f（用其真实 raw 动作，不裁剪）
    f_actor = (cp([st, a_act]) - LAM * cc([st, a_act])).squeeze(1)

a_act_np = a_act[:, :5].mean(1).numpy()
a_clip_np = a_clip[:, :5].mean(1).numpy()
best_a_np = best_a.numpy()
gain = (best_f - f_actor).numpy()

print('状态 %d（step>240k 随机抽）' % len(st))
print('\n[a] 三种取法给出的 a（5 维均值）与 critic 目标值 f = q_p − 5·q_c')
print('  %14s %9s %9s %9s %11s %11s' % ('u 区间', 'actor', 'clip(actor)', 'grid-argmax', 'Δf 均值', 'Δf>0 占比'))
for lo, hi in [(0, 9), (9, 11), (11, 12.5), (12.5, 14), (14, 40)]:
    m = (U >= lo) & (U < hi)
    if m.sum() < 50:
        continue
    print('  [%5.1f,%5.1f) %9.3f %9.3f %9.3f %11.1f %10.1f%%'
          % (lo, hi, a_act_np[m].mean(), a_clip_np[m].mean(), best_a_np[m].mean(),
             gain[m].mean(), 100 * (gain[m] > 0).mean()))
print('\n  说明：Δf 是"grid-argmax 相对 actor 的改进量"（critic 归一化单位；×1000 元 ≈ 元/步量级的折扣和）。')
print('  Δf>0 占比高 ⇒ critic 认为 actor 的动作明显次优 ⇒ 瓶颈在 actor 的**局部更新**，不在 critic。')

print('\n[b] grid-argmax 的 a 分布（高风速段 u>13.5）')
m = U > 13.5
if m.sum() > 50:
    vals, cnts = np.unique(np.round(best_a_np[m], 3), return_counts=True)
    for v, c in zip(vals, cnts):
        print('    a=%.2f : %5.1f%%' % (v, 100 * c / m.sum()))
print('\n[c] 同一批状态：actor 的 a 分布（u>13.5）')
if m.sum() > 50:
    vals, cnts = np.unique(np.round(a_act_np[m], 2), return_counts=True)
    for v, c in list(zip(vals, cnts))[:12]:
        print('    a=%.2f : %5.1f%%' % (v, 100 * c / m.sum()))
