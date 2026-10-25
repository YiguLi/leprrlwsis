#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 C：结构化解码（网络学 z≈ΔP，成本由解析层解码）——对照实验 A 的普通回归。

目标口径（冻结）：**块首步(t=1)** 的即时成本
    c1 = |ΔP| + 20·(|ΔP| − 3)⁺,   ΔP = P(a,u_1) − P_G_prev   （P_G_prev 在状态里）
⇒ 单步目标与"结构化解码"完全同构：一个光滑标量 z 就能精确解码出折点。

组别：
  C0  普通 MLP（AuxModel）+ 值损失
  C1  结构化（StructCostModel）+ 值损失
  C2  结构化 + 值 + 差分损失
  C3  结构化 + 值 + 差分 + z 直接监督（z ≈ ΔP，来自日志）

指标（独立测试日）：值 RMSE/MAE、幅值比、符号一致、Pearson、高风、大梯度、逐维。
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import model  # noqa: E402


def hl(x, y, d):
    return F.huber_loss(x, y, delta=d)


def run_arm(arm, data, delta='0.020', seeds=(0,), epochs=400, alpha=0.1, lr=1e-3, bs=256,
            feat=False, decode='hard'):
    S = data['state']; A6 = data['action6']; A = data['action']
    C = data['cost1_base']; DP = data['dp1_base']
    D = data['d1%s' % delta]
    split = data['split']; u0 = data['u0']
    tr = np.where(split == 0)[0]; te = np.where(split == 2)[0]
    delta = float(delta)
    out = []
    for sd in seeds:
        torch.manual_seed(2000 + sd)
        if arm in ('C0', 'C4'):
            net = model.AuxModel(4, 6, feat_power=feat)
        else:
            net = model.StructCostModel(4, 6, feat_power=feat, decode=decode)
            with torch.no_grad():                      # |z| 在 0 处次梯度为 0 ⇒ 给小偏置避免死梯度
                net.z_out.bias.data.fill_(1.0)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        St = torch.tensor(S); At = torch.tensor(A6); Ct = torch.tensor(C); DPt = torch.tensor(DP)
        Dt = torch.tensor(D)
        for ep in range(epochs):
            perm = np.random.RandomState(sd * 7777 + ep).permutation(tr)
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                pred = net((St[b], At[b])).squeeze(1)
                loss = hl(pred, Ct[b], 5.0)
                if arm in ('C2', 'C3', 'C4'):
                    Ab = At[b]; dl = 0.0
                    for j in range(5):
                        ap = Ab.clone(); am = Ab.clone()
                        ap[:, j] += delta; am[:, j] -= delta
                        ap[:, j] = ap[:, j].clamp(max=0.5)
                        am[:, j] = am[:, j].clamp(min=0.0)
                        h = net((St[b], ap)).squeeze(1) - net((St[b], am)).squeeze(1)
                        dl = dl + hl(h / (2 * delta), Dt[b, j], 0.5)
                    loss = loss + alpha * dl / 5.0
                if arm == 'C3':
                    loss = loss + alpha * hl(net.z((St[b], At[b])).squeeze(1), DPt[b], 1.0)
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net((St[te], At[te])).squeeze(1).numpy()
        G = np.zeros((len(te), 5), np.float32)
        for k, idx in enumerate(te):
            a1 = torch.tensor(A6[idx:idx + 1], requires_grad=True)
            q = net((St[idx:idx + 1], a1))
            g, = torch.autograd.grad(q.sum(), a1)
            G[k] = g.detach().numpy()[0, :5]
        out.append((pv, G))
    return out, dict(C=C[te], D=D[te], u0=u0[te])


def report(arm, res, meta, label=''):
    C, D, u0 = meta['C'], meta['D'], meta['u0']
    ok = (np.abs(D).sum(1) > 1e-6)
    R = []
    for pv, G in res:
        rmse = float(np.sqrt(np.mean((pv - C) ** 2)))
        mae = float(np.mean(np.abs(pv - C)))
        amp = float(np.abs(G[ok]).mean() / max(1e-9, np.abs(D[ok]).mean()))
        sign = 100 * float(np.mean(np.sign(G[ok]) == np.sign(D[ok])))
        pear = float(np.corrcoef(G[ok].ravel(), D[ok].ravel())[0, 1])
        hi = ok & (u0 >= 12)
        s_hi = 100 * float(np.mean(np.sign(G[hi]) == np.sign(D[hi]))) if hi.sum() > 5 else np.nan
        a_hi = float(np.abs(G[hi]).mean() / max(1e-9, np.abs(D[hi]).mean())) if hi.sum() > 5 else np.nan
        thr = np.median(np.abs(D[ok]))
        big = ok[:, None] & (np.abs(D) > thr)
        s_big = 100 * float(np.mean(np.sign(G[big]) == np.sign(D[big])))
        R.append((rmse, mae, amp, sign, pear, s_hi, a_hi, s_big))
    R = np.array(R, float)
    print('  %-3s %s| 值 RMSE %7.2f±%5.2f | MAE %6.2f | 幅值比 %5.3f | 符号 %5.1f%%±%4.1f | '
          'Pearson %+.3f | 高风: 符号 %5.1f%% 幅值 %5.3f | 大梯度: 符号 %5.1f%%'
          % (arm, label, R[:, 0].mean(), R[:, 0].std(), R[:, 1].mean(), R[:, 2].mean(),
             R[:, 3].mean(), R[:, 3].std(), R[:, 4].mean(), np.nanmean(R[:, 5]),
             np.nanmean(R[:, 6]), R[:, 7].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cf', default='output/counterfactual/cf_AIFFLOR-run3.npz')
    ap.add_argument('--delta', default='0.020')
    ap.add_argument('--arms', default='C0,C1,C2,C3')
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--decode', default='hard')
    a = ap.parse_args()
    z = np.load(a.cf, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    seeds = tuple(int(x) for x in a.seeds.split(','))
    print('[C] 数据 %s | 目标=块首步 c1 | 训练/测试块 %d/%d | δ=%s | α=%.3f | decode=%s'
          % (a.cf, (data['split'] == 0).sum(), (data['split'] == 2).sum(), a.delta, a.alpha, a.decode))
    for arm in a.arms.split(','):
        res, meta = run_arm(arm, data, delta=a.delta, seeds=seeds, epochs=a.epochs,
                            alpha=a.alpha, decode=a.decode)
        report(arm, res, meta)


if __name__ == '__main__':
    main()
