#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 A：A0/A1/A3(/A4) 四组对照 —— 配对反事实数据 + 差分损失能否恢复块级即时成本的动作偏导。

组别（指南 §4.4）：
  A0  普通 replay 数据, 仅值损失        （复现当前 aux_c 基线）
  A1  配对数据(基准分支), 仅值损失       （分离"数据"的作用）
  A2  普通模型输入, 值+差分损失          （本脚本不做：普通数据无差分标签）
  A3  配对数据, 值+差分损失             （完整方案）
  A4  配对 + P(a,u) 特征, 值+差分损失    （物理特征是否有增量）

指标在**独立测试日**上报告：值 RMSE/MAE、幅值比、符号一致、Pearson、高风段、大梯度子集、逐维。
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import model  # noqa: E402

SC = 100.0
HUBER = torch.nn.SmoothL1Loss(beta=1.0)


def load(fp):
    z = np.load(fp, allow_pickle=True)
    return {k: z[k] for k in z.files}


def build_pairs(S, A, D, delta, deff, oneside):
    """返回每维的正负动作 (N,5,5) 与有效 δ；单边样本的两支相同（用 δ_eff 归一）。"""
    ap = A[:, None, :].repeat(5, axis=1).copy()
    am = ap.copy()
    for j in range(5):
        ap[:, j, j] = A[:, j] + delta
        am[:, j, j] = A[:, j] - delta
        hi = ap[:, j, j] > 0.5
        lo = am[:, j, j] < 0.0
        ap[hi, j] = 0.5
        am[lo, j] = 0.0
    return ap, am


def train_and_eval(arm, data, delta='0.020', seeds=(0, 1, 2), epochs=400, alpha=1.0,
                   bs=256, lr=1e-3, feat=False, verbose=False):
    dk = delta
    delta = float(delta)
    S = data['state']; A = data['action']; A6 = data['action6']; C = data['cost_base'] / SC
    D = data['d%s' % dk]; DE = data['deff%s' % dk]; OS = data['oneside%s' % dk]
    split = data['split']; u0 = data['u0']
    tr = np.where(split == 0)[0]; va = np.where(split == 1)[0]; te = np.where(split == 2)[0]

    res = []
    for sd in seeds:
        torch.manual_seed(1000 + sd)
        net = model.AuxModel(4, 6, feat_power=feat)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        S_t = torch.tensor(S); A_t = torch.tensor(A6); C_t = torch.tensor(C)
        # 差分标签（按 δ 归一化到"每单位动作"）
        # 差分标签换算到"归一化成本单位/单位动作"，与值目标同量纲（关键：否则 α=1 会压垮值损失）
        Dn = torch.tensor(((D / np.maximum(DE, 1e-9)) / SC).astype(np.float32))
        for ep in range(epochs):
            perm = np.random.RandomState(sd * 10000 + ep).permutation(tr)
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = HUBER(net((S_t[b], A_t[b])).squeeze(1), C_t[b])
                if arm in ('A3', 'A4'):
                    Ab = A_t[b]
                    dl = 0.0
                    for j in range(5):
                        ap = Ab.clone(); am = Ab.clone()
                        ap[:, j] += delta; am[:, j] -= delta
                        ap[:, j] = ap[:, j].clamp(max=0.5)
                        am[:, j] = am[:, j].clamp(min=0.0)
                        h = net((S_t[b], ap)).squeeze(1) - net((S_t[b], am)).squeeze(1)
                        dl = dl + HUBER(h / (2 * delta), Dn[b, j])
                    loss = loss + alpha * dl / 5.0
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred_v = net((S_t[te], A_t[te])).squeeze(1).numpy() * SC
        # 测试集梯度：autograd（单样本循环以保证口径一致）
        G = np.zeros((len(te), 5), np.float32)
        for k, idx in enumerate(te):
            a1 = torch.tensor(A6[idx:idx + 1], requires_grad=True)
            q = net((S_t[idx:idx + 1], a1))
            g, = torch.autograd.grad(q.sum(), a1)
            G[k] = g.detach().numpy()[0, :5]
        res.append(dict(pred_v=pred_v, G=G))
    return res, dict(C=C * SC, D=D, u0=u0, te=te)


def report(arm, res, meta, delta):
    te = meta['te']; C = meta['C'][te]; D = meta['D'][te]; u0 = meta['u0'][te]
    ok = (np.abs(D).sum(1) > 1e-6)
    rows = []
    for r in res:
        G, pv = r['G'], r['pred_v']
        rmse = float(np.sqrt(np.mean((pv - C) ** 2)))
        mae = float(np.mean(np.abs(pv - C)))
        amp = float(np.abs(G).mean() / max(1e-9, np.abs(D[ok]).mean()))
        sign = 100 * float(np.mean(np.sign(G[ok]) == np.sign(D[ok])))
        pear = float(np.corrcoef(G[ok].ravel(), D[ok].ravel())[0, 1])
        hi = ok & (u0 >= 12)
        sign_hi = 100 * float(np.mean(np.sign(G[hi]) == np.sign(D[hi]))) if hi.sum() > 5 else float('nan')
        amp_hi = float(np.abs(G[hi]).mean() / max(1e-9, np.abs(D[hi]).mean())) if hi.sum() > 5 else float('nan')
        _thr = np.median(np.abs(D[ok]))
        big = ok[:, None] & (np.abs(D) > _thr)
        sign_big = 100 * float(np.mean(np.sign(G[big]) == np.sign(D[big])))
        rows.append((rmse, mae, amp, sign, pear, sign_hi, amp_hi, sign_big))
    rows = np.array(rows, float)
    print('  %-3s δ=%s | 值 RMSE %7.2f±%5.2f | MAE %6.2f | 幅值比 %5.3f | 符号 %5.1f%%±%4.1f | '
          'Pearson %+.3f | 高风: 符号 %5.1f%% 幅值 %5.3f | 大梯度: 符号 %5.1f%%'
          % (arm, delta, rows[:, 0].mean(), rows[:, 0].std(), rows[:, 1].mean(), rows[:, 2].mean(),
             rows[:, 3].mean(), rows[:, 3].std(), rows[:, 4].mean(),
             np.nanmean(rows[:, 5]), np.nanmean(rows[:, 6]), rows[:, 7].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cf', default='output/counterfactual/cf_AIFFLOR-run3.npz')
    ap.add_argument('--delta', default='0.020')
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--arms', default='A0,A1,A3,A4')
    a = ap.parse_args()
    data = load(a.cf)
    seeds = tuple(int(x) for x in a.seeds.split(','))
    print('[A] 数据 %s | 训练/验证/测试 块数 %d/%d/%d | δ=%s | α=%.2f'
          % (a.cf, (data['split'] == 0).sum(), (data['split'] == 1).sum(),
             (data['split'] == 2).sum(), a.delta, a.alpha))
    for arm in a.arms.split(','):
        res, meta = train_and_eval(arm, data, delta=a.delta, seeds=seeds,
                                   epochs=a.epochs, alpha=a.alpha, feat=(arm == 'A4'))
        report(arm, res, meta, a.delta)


if __name__ == '__main__':
    main()
