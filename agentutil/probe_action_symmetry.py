#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""动作对称性探针（可复用诊断）：量化"前五维（5 台机组桨距）对价值的影响有多相似"。

用法:  python probe_action_symmetry.py [模型目录 ...]      # 缺省扫 output/TD3*、output/PER*

两个可测指标:
  A) 灵敏度谱: g_i = ∂Q_p/∂a_i 在状态网格上的协方差 Σ_g 的特征谱
     * 共同模占比 c = (1ᵀΣ_g1) / (5·tr(Σ_g)) ∈ [1/5, 1]
       c≈1 ⇒ 5 维影响几乎同向同幅（"规律相似"成立, 绑定噪声不亏）
       c≈0.2 ⇒ 5 维影响互相独立（绑定噪声会丢掉 ~80% 的探索方差）
  B) 已学策略的跨机组离散度: 每步 5 维输出的标准差（尾流转向是否被学到）
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块 (hsrl_*, myfunc, util, td3*/...)
sys.path.insert(0, _HERE)                              # agentutil 内互相 import

from model import Actor, Critic  # noqa: E402


def load(dirname, tag='6'):
    ap = os.path.join(dirname, 'actor_{}.pkl'.format(tag))
    cp = os.path.join(dirname, 'critic_p_{}.pkl'.format(tag))
    if not (os.path.isfile(ap) and os.path.isfile(cp)):
        return None
    sd = torch.load(ap, map_location='cpu')
    nb_states = next(v.shape[1] for k, v in sd.items() if v.dim() == 2)
    nb_actions = next(v.shape[0] for k, v in sd.items() if v.dim() == 2 and 'fc4' in k) \
        if any('fc4' in k for k in sd) else int(tag)
    # 隐藏层维度从 checkpoint 形状读取 (勿写死: 网络宽度会随配置变, 历史上出现过 16/100 与 64/128)
    h1, h2 = sd['fc1.weight'].shape[0], sd['fc2.weight'].shape[0]
    actor = Actor(nb_states, nb_actions, hidden1=h1, hidden2=h2, init_w=0.003)
    actor.load_state_dict(sd)
    csd = torch.load(cp, map_location='cpu')
    critic = Critic(nb_states, nb_actions, hidden1=csd['fc1.weight'].shape[0],
                    hidden2=csd['fc4.weight'].shape[0], init_w=0.003)
    critic.load_state_dict(csd)
    actor.eval(); critic.eval()
    return actor, critic, nb_states, nb_actions


def analyse(dirname):
    out = load(dirname)
    if out is None:
        print('  [skip] 缺少 actor_6/critic_p_6: {}'.format(dirname))
        return None
    actor, critic, nb_states, nb_actions = out
    rng = np.random.RandomState(0)
    N = 2000
    u = rng.uniform(4.0, 14.0, N)
    pr = rng.choice([300.0, 580.0, 980.0], N)
    E = rng.uniform(0.5, 5.5, N)
    PG = rng.uniform(0.0, 8.0, N)
    base = np.stack([pr / 1000.0, u / 10.0, E / 10.0, PG / 10.0], 1)
    if nb_states == 6:
        ang = 2 * np.pi * rng.uniform(0, 1440, N) / 1440.0
        S = np.concatenate([base, np.stack([np.sin(ang), np.cos(ang)], 1)], 1)
    else:
        S = base[:, :nb_states]
    S = torch.tensor(S, dtype=torch.float32)

    with torch.no_grad():
        a_full = actor(S)
    # ---- A) ∂Q_p/∂a 的灵敏度谱（在策略动作处求导；critic 需要完整动作维）----
    A = a_full.clone().detach().requires_grad_(True)
    q = critic([S, A]).sum()
    g_all = torch.autograd.grad(q, A)[0].detach().numpy()
    g = g_all[:, :5]                                         # 只看前五维(桨距)
    a = a_full.detach()[:, :5]
    Sig = np.cov(g.T)                                        # 5x5
    ev = np.linalg.eigvalsh(Sig)[::-1]
    ones = np.ones(5)
    c_common = float(ones @ Sig @ ones / (5 * np.trace(Sig)))
    # 相关系数矩阵（去尺度后的相似性）
    sd = np.sqrt(np.diag(Sig))
    Corr = Sig / np.outer(sd, sd)
    off = Corr[np.triu_indices(5, 1)]
    # ---- B) 策略的跨机组离散度 ----
    a_np = a.numpy()
    disp = a_np.std(axis=1)
    res = dict(dir=os.path.basename(dirname), nb_states=nb_states,
               c_common=c_common, ev=(ev / ev.sum()).round(3).tolist(),
               corr_mean=float(off.mean()), corr_min=float(off.min()),
               a_mean=float(a_np.mean()), a_disp_mean=float(disp.mean()),
               a_disp_p90=float(np.percentile(disp, 90)),
               g_abs_mean=float(np.abs(g).mean()), g_sd_mean=float(sd.mean()))
    return res, Corr


dirs = sys.argv[1:]
if not dirs:
    import glob
    dirs = sorted(glob.glob(os.path.join('output', 'TD3*'))) + \
        sorted(glob.glob(os.path.join('output', 'PER*')))
print('=' * 104)
print('前五维（5 台机组桨距）对价值的影响有多相似？')
print('  c_common = 共同模占灵敏度协方差的比例 (1/5=完全独立, 1=完全同向同幅)')
print('  a_disp   = 已学策略的跨机组动作标准差 (0=所有机组同桨距, 无尾流转向)')
print('=' * 104)
for d in dirs:
    r = analyse(d)
    if r is None:
        continue
    res, Corr = r
    print('{:<22s} c_common={:.3f}  λ谱={}  相关系数(均值/最小)={:.2f}/{:.2f}'
          .format(res['dir'], res['c_common'], res['ev'], res['corr_mean'], res['corr_min']))
    print('{:<22s} a均值={:.3f}  跨机组std: 均值={:.4f} p90={:.4f}  |∂Q/∂a|均值={:.3f}'
          .format('', res['a_mean'], res['a_disp_mean'], res['a_disp_p90'], res['g_abs_mean']))
    print('{:<22s} 灵敏度相关矩阵:\n{}'.format('', np.round(Corr, 2)))
    print('-' * 104)
