#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真值对照：cost critic 的逐样本动作梯度 vs 物理真值 ∂c1/∂a。

用法: python agentutil/probe_cost_truth_grad.py [run_dir] [--n 800]

真值口径（与训练目标一致，逐**步**）：
    P_F(a)  = |P(a, u_t) − P_G(t−1)|
    c1(a)   = P_F + 20·max(0, P_F − 3)                      （c1-mode=uniform）
    ∂c1/∂a_j = (1 + 20·1{P_F>3}) · sign(P − P_G(t−1)) · ∂P/∂a_j
∂P/∂a_j 由中心差分（物理引擎）得到；critic 侧用逐样本 autograd（q.sum()）。

报告：
  * 逐维相关（Pearson/Spearman）与符号一致率
  * 越限样本(P_F>3) vs 非越限样本的 |梯度| 比值 —— 真值应 ≈ 21（折线），critic 实测多少
  * 隐含比例系数 K = mean|∂Q_c/∂a| / mean|∂c1/∂a|（只用来核对量纲）
"""
import argparse
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
import aifflor.aifflor_turbine as AT  # noqa: E402
from aifflor.aifflor_env import AifflorPhysics  # noqa: E402

EPS = 0.02
NT = 5


def find_run(d):
    cands = glob.glob(os.path.join(d, 'AIFFLOR-run*'))
    return max(cands, key=os.path.getmtime) if cands else d


def power_of(ph, u, acts):
    """acts: (N,5) 风机轴向因子, u: (N,) 风速 ⇒ P_G (MW, N)"""
    u = np.clip(np.asarray(u, dtype=float), AT.U_MIN_ENV, None)
    farm, _ = AT.eval_batch(ph.model, u, np.asarray(acts, dtype=float))
    return np.asarray(farm, dtype=float).ravel() * ph.power_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', nargs='?', default='output/DIAG-S3')
    ap.add_argument('--n', type=int, default=800, help='样本数（块首步）')
    ap.add_argument('--which', default='critic_c_6.pkl')
    a = ap.parse_args()

    D = find_run(a.run_dir)
    df = pd.read_csv(os.path.join(D, 'df_hsrl.csv'),
                     usecols=['step', 'u', 'pr', 'SOC', 'turbine_profit', 'P_F', 'c1',
                              'applied_action_H'])
    df['PG'] = df.turbine_profit * 60 / df.pr
    df['PG_prev'] = df.PG.shift(1)
    df['u_prev'] = df.u.shift(1)
    # 只取块首步（动作真正换挡的地方）、且前一步存在
    m = (df.step.values % 5 == 0) & df.PG_prev.notna().values
    sub = df[m].dropna().reset_index(drop=True).iloc[-a.n:].reset_index(drop=True)
    A_FULL = np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                        for x in sub.applied_action_H]).astype(float)
    A = A_FULL[:, :NT]
    u = sub.u.to_numpy(float)
    prev = sub.PG_prev.to_numpy(float)

    ph = AifflorPhysics(nt=NT, calibrated=False, power_scale=AT.LEVEL_COEF)
    P0 = power_of(ph, u, A)
    PF0 = np.abs(P0 - prev)
    dc1 = np.zeros_like(A)
    dP = np.zeros_like(A)
    for j in range(NT):
        Ap = A.copy(); Ap[:, j] += EPS
        Am = A.copy(); Am[:, j] -= EPS
        Pj = (power_of(ph, u, Ap) - power_of(ph, u, Am)) / (2 * EPS)
        dP[:, j] = Pj
        sgn = np.sign(P0 - prev)
        sgn[sgn == 0] = 1.0
        dc1[:, j] = (1.0 + 20.0 * (PF0 > 3.0)) * sgn * Pj

    S = torch.tensor(np.column_stack([sub.pr / 1000, sub.u / 10, sub.SOC / 0.5,
                                      prev / 5]).astype(np.float32))
    at = torch.tensor(A_FULL.astype(np.float32), requires_grad=True)  # 6 维（末维 = g）
    sd = torch.load(os.path.join(D, a.which), map_location='cpu')
    net = model.Critic(4, 6, hidden1=16, hidden2=100, arch='v2')
    net.load_state_dict(sd); net.eval()
    q = net((S, at))
    g, = torch.autograd.grad(q.sum(), at)
    gc = g.detach().numpy()[:, :NT]

    def corr(x, y):
        x = x.ravel(); y = y.ravel()
        if x.std() < 1e-12 or y.std() < 1e-12:
            return float('nan'), float('nan')
        pr = np.corrcoef(x, y)[0, 1]
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return pr, np.corrcoef(rx, ry)[0, 1]

    print('样本 %d 条（块首步）| P_F 分位 p50 %.2f p95 %.2f | P_F>3 占 %.1f%%'
          % (len(A), np.percentile(PF0, 50), np.percentile(PF0, 95), 100 * (PF0 > 3).mean()))
    pr_all, sp_all = corr(gc, dc1)
    print('\n[总体] critic ∂Q_c/∂a  vs  真值 ∂c1/∂a')
    print('  Pearson %.3f | Spearman %.3f | 符号一致率 %.1f%%'
          % (pr_all, sp_all, 100 * (np.sign(gc) == np.sign(dc1)).mean()))
    print('  量纲 K = mean|∂Q_c/∂a| / mean|∂c1/∂a| = %.3f'
          % (np.abs(gc).mean() / max(1e-12, np.abs(dc1).mean())))
    for j in range(NT):
        p, s = corr(gc[:, j], dc1[:, j])
        print('  dim%d: Pearson %+.3f | Spearman %+.3f | 符号一致 %.1f%%'
              % (j, p, s, 100 * (np.sign(gc[:, j]) == np.sign(dc1[:, j])).mean()))
    print('\n[折线] 越限(P_F>3) vs 未越限 的 |梯度| 比值（真值应 ≈ 21）')
    v = PF0 > 3.0
    if v.sum() > 10 and (~v).sum() > 10:
        print('  真值  : %.2f  (n=%d / %d)' % (np.abs(dc1[v]).mean() / max(1e-12, np.abs(dc1[~v]).mean()), v.sum(), (~v).sum()))
        print('  critic: %.2f' % (np.abs(gc[v]).mean() / max(1e-12, np.abs(gc[~v]).mean())))
        print('  越限样本上的相关: Pearson %.3f | 符号一致 %.1f%%'
              % (corr(gc[v], dc1[v])[0], 100 * (np.sign(gc[v]) == np.sign(dc1[v])).mean()))
    else:
        print('  越限样本不足（%d 条）' % v.sum())
    print('\n[纯出力] 与 ∂P/∂a 的相关（不看折线，只看"动作→出力"是否学对）')
    print('  Pearson %.3f | Spearman %.3f | 符号一致 %.1f%%'
          % (*corr(gc, dP), 100 * (np.sign(gc) == np.sign(dP)).mean()))


if __name__ == '__main__':
    main()
