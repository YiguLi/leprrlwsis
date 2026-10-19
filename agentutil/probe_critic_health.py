#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""critic 健康检查（支持 v1 / v1af / v2 三种结构）。

用法: python agentutil/probe_critic_health.py output/DIAG-S1 output/DIAG-S3
判据：
  * 各激活层"整批恒负"的单元数（死单元）；fc1/state 支路全死 ⇒ 状态/动作通道断了
  * |∂Q/∂a| 均值：正常应 ~0.1~1（归一化 Q 单位/单位 a）；<1e-3 ⇒ 动作通道被压平
  * Q std / |Q|：<5% ⇒ Q 基本是常数（退化成 bias）
"""
import glob
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

import model  # noqa: E402


def infer_arch(sd, ns=4, na=6):
    if 's1.weight' in sd:
        return 'v2'
    if sd['fc1.weight'].shape[1] == ns + na:
        return 'v1af'
    return 'v1'


def layer_names(m):
    """按 forward 顺序给出激活层名（供打印）。"""
    if m.arch == 'v2':
        return ['s1', 's2', 'a1', 'a2', 'm1', 'm2', 'm3']
    if m.arch == 'v1af':
        return ['fc1', 'fc2', 'fc5']
    return ['fc1', 'fc2', 'fc5']


def check(run_dir, which='critic_p_6.pkl', tag='P'):
    D = max(glob.glob(os.path.join(run_dir, 'AIFFLOR-run*')), key=os.path.getmtime, default=run_dir)
    p = os.path.join(D, which)
    if not os.path.exists(p):
        print('%-24s 缺 %s' % (os.path.basename(run_dir.rstrip('/\\')), which)); return
    sd = torch.load(p, map_location='cpu')
    arch = infer_arch(sd)
    net = model.Critic(4, 6, hidden1=16, hidden2=100, arch=arch)
    net.load_state_dict(sd)
    net.eval()

    df = pd.read_csv(os.path.join(D, 'df_hsrl.csv'),
                     usecols=['pr', 'u', 'SOC', 'turbine_profit', 'applied_action_H'])
    A = np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                   for x in df.applied_action_H])
    PG = (df.turbine_profit * 60 / df.pr).to_numpy()
    S = torch.tensor(np.column_stack([df.pr / 1000, df.u / 10, df.SOC / 0.5, PG / 5]).astype(np.float32))
    a0 = torch.tensor(A.astype(np.float32), requires_grad=True)

    # 用 hook 抓每个激活层的输出 ⇒ 通用（v1/v1af/v2 都能算"死单元"）
    acts = []
    hs = [m.register_forward_hook(lambda mod, i, o: acts.append(o.detach())) for m in net.modules()
          if isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU))]
    q = net((S, a0))
    for h in hs:
        h.remove()
    alive = [(int((x > 0).any(0).sum()), x.shape[1]) for x in acts]
    g, = torch.autograd.grad(q.mean(), a0)
    g = g.detach().numpy()[:, :5]
    qn = q.detach().numpy().ravel()
    bias = float(sd['out.bias'] if 'out.bias' in sd else sd['fc3.bias'])
    print('%-22s [%s] arch=%-4s %s | Q 均 %8.3f std %8.4f (std/|Q| %5.1f%%) | out.bias %7.3f | |dQ/da| %.2e'
          % (os.path.basename(run_dir.rstrip('/\\')), tag, arch,
             ' '.join('%s %d/%d' % (n, a_, b_) for n, (a_, b_) in zip(layer_names(net), alive)),
             qn.mean(), qn.std(), 100 * qn.std() / max(abs(qn.mean()), 1e-9), bias, np.abs(g).mean()))


def check_both(run_dir):
    """profit 与 cost 两个通道各打一行（cost 通道的动作杠杆通常更大）。"""
    for which, tag in (('critic_p_6.pkl', 'P'), ('critic_c_6.pkl', 'C')):
        check(run_dir, which, tag)


if __name__ == '__main__':
    for d in (sys.argv[1:] or ['output/DIAG-S1', 'output/DIAG-S2']):
        check_both(d)
