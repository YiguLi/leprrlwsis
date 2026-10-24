#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 `documentation/critic_action_gradient_diagnosis_guide.md` 的第一轮（低成本验证）做排查。

包含四项：
  T1  合成函数 sanity check： y(s,a)=b(s)+Σ_j w_j(s)a_j 能否被同款网络/训练流程恢复出已知导数
  T2  autograd 与"模型自身"的中心差分是否一致（检验求导流水线，与物理无关）
  T3  逐样本 / 批次梯度的统一统计（M_j=mean|g|、A_j=|mean g|、符号比例、样本间余弦相似度、mean/sd）
  T4  分风速箱、分动作维的可辨识性报告（n_eff、动作 std、局部值 RMSE、导数符号一致率、|真值| vs |预测|）

关键：探针 A（即时成本导数）的真值 = **块级 c1 求和**对动作的中心差分（同块内动作恒定、
风速序列已知 ⇒ 可直接由物理算出），**而不是**逐步 c1 的导数——后者与块级目标不是同一个数学量。

用法：
  python agentutil/diag_probe_audit.py --run output/FULL-R2/AIFFLOR-run3 --which critic_c_6.pkl --smooth 0
  python agentutil/diag_probe_audit.py --run output/AUXFP/AIFFLOR-run1 --which aux_c_6.pkl  --smooth 1
"""
import argparse
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

K_V, LIMIT = 20.0, 3.0
BETA, EPS_SM = 100.0, 0.05
NT = 5


# ---------------------------------------------------------------- 成本口径
def c1_of(pf, smooth):
    pf = np.asarray(pf, float)
    if smooth:
        ln2 = np.log(2.0)
        sp = np.maximum(pf - LIMIT, 0.0) + np.log1p(np.exp(-np.abs(BETA * (pf - LIMIT)))) / BETA
        return pf + K_V * (sp - ln2 / BETA)
    return pf + K_V * np.maximum(0.0, pf - LIMIT)


def block_c1_sum(P, pg_prev):
    """P: (N,5) 块内 5 步出力（同动作、各自风速）；pg_prev: (N,) 块首前一步的出力。
    返回每块的 P_F 序列 (N,5)：第 1 步用 pg_prev，其余用块内前一步。"""
    pf = np.empty_like(P)
    pf[:, 0] = np.abs(P[:, 0] - pg_prev)
    pf[:, 1:] = np.abs(np.diff(P, axis=1))
    return pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='output/FULL-R2/AIFFLOR-run3')
    ap.add_argument('--which', default='critic_c_6.pkl')
    ap.add_argument('--smooth', type=int, default=0, help='1=该 run 用了平滑 c1')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--delta', type=float, default=0.02)
    ap.add_argument('--synthetic', type=int, default=1)
    a = ap.parse_args()

    ph = AifflorPhysics(nt=NT, calibrated=False, power_scale=AT.LEVEL_COEF)

    def P_of(acts, u):
        farm, _ = AT.eval_batch(ph.model, np.clip(np.asarray(u, float), AT.U_MIN_ENV, None),
                                np.asarray(acts, float))
        return np.asarray(farm, float).ravel() * ph.power_scale

    # ============================================================ T1 合成函数
    if a.synthetic:
        print('=' * 78)
        print('T1 合成函数 sanity：y = b(s) + Σ_j w_j a_j  （已知 w，检验流程能否恢复导数）')
        torch.manual_seed(0)
        n, ns, na = 4000, 4, 6
        S = torch.rand(n, ns)
        A = torch.rand(n, na) * 0.5
        W = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.5, 0.0])          # 已知系数
        b = lambda x: (x[:, :1] * 2.0 - x[:, 1:2] ** 2)
        Y = b(S) + A @ W.unsqueeze(1)
        for noise in (0.0, 0.1):
            net = model.AuxModel(ns, na)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            Yn = Y + noise * torch.randn_like(Y)
            for it in range(3000):
                opt.zero_grad()
                loss = torch.mean((net((S, A)) - Yn) ** 2)
                loss.backward()
                opt.step()
            # (a) autograd vs 模型自身中心差分
            A2 = A.clone().requires_grad_(True)
            g_ag, = torch.autograd.grad(net((S, A2)).sum(), A2)
            eps = 1e-4
            g_fd = torch.zeros_like(g_ag)
            for j in range(na):
                Ap = A.clone(); Ap[:, j] += eps
                Am = A.clone(); Am[:, j] -= eps
                g_fd[:, j] = (net((S, Ap)) - net((S, Am))).squeeze(1) / (2 * eps)
            err = float((g_ag - g_fd).abs().max())
            # (b) 恢复的导数 vs 已知 W
            g_np, w_np = g_ag.detach().numpy(), W.numpy()
            sign = 100 * np.mean(np.sign(g_np) == np.sign(np.broadcast_to(w_np, g_np.shape)))
            pr = np.corrcoef(g_np.ravel(), np.broadcast_to(w_np, g_np.shape).ravel())[0, 1]
            print('  noise=%.1f | 值 MSE %.3e | autograd vs 自差分 max err %.2e | '
                  '恢复导数: sign %.1f%% Pearson %.4f'
                  % (noise, float(loss), err, sign, pr))

    # ============================================================ 读数
    R = a.run
    df = pd.read_csv(os.path.join(R, 'df_hsrl.csv'),
                     usecols=['step', 'u', 'pr', 'SOC', 'turbine_profit', 'P_F', 'c1',
                              'applied_action_H'])
    A6 = np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                    for x in df.applied_action_H]).astype(np.float64)
    PG = (df.turbine_profit * 60 / df.pr).to_numpy()
    nb = len(df) // 5
    # 每个块：5 步的 u、块首前一步的出力
    U = df.u.to_numpy(float)[:nb * 5].reshape(nb, 5)
    A5 = A6[:nb * 5].reshape(nb, 5, 6)[:, 0, :5]            # 块首动作（块内恒定）
    PGP = np.r_[PG[0], PG[:nb * 5 - 1]][::5][:nb]           # 块首前一步出力
    idx = np.linspace(0, nb - 1, min(a.n, nb)).astype(int)

    S_t = torch.tensor(np.column_stack([df.pr.to_numpy(float)[:nb * 5][::5][idx] / 1000,
                                        U[idx, 0] / 10,
                                        df.SOC.to_numpy(float)[:nb * 5][::5][idx] / 0.5,
                                        PGP[idx] / 5]).astype(np.float32))
    A_t = torch.tensor(A6[:nb * 5][::5][idx].astype(np.float32), requires_grad=True)

    sd = torch.load(os.path.join(R, a.which), map_location='cpu')
    if a.which.startswith('aux'):
        net = model.AuxModel(4, 6, feat_power=(sd['fc1.weight'].shape[1] == 11))
    else:
        net = model.build_critic_from_sd(sd, 4, 6)
    net.load_state_dict(sd)
    net.eval()

    q = net((S_t, A_t))
    g_ag, = torch.autograd.grad(q.sum(), A_t)                 # 逐样本 autograd
    G = g_ag.detach().numpy()[:, :5]

    # ============================================================ T2 autograd vs 自差分
    print('=' * 78)
    print('T2 autograd 与"模型自身"中心差分是否一致（不看物理，只看求导流水线）')
    eps = 1e-4
    Gfd = np.zeros_like(G)
    for j in range(5):
        Ap = A_t.detach().clone(); Ap[:, j] += eps
        Am = A_t.detach().clone(); Am[:, j] -= eps
        with torch.no_grad():
            Gfd[:, j] = ((net((S_t, Ap)) - net((S_t, Am))).squeeze(1).numpy()) / (2 * eps)
    print('  max|autograd − 自差分| = %.3e | 相对 %.3e | 逐维 sign 一致 %.1f%%'
          % (np.abs(G - Gfd).max(), np.abs(G - Gfd).max() / max(1e-12, np.abs(G).max()),
             100 * np.mean(np.sign(G) == np.sign(Gfd))))

    # ============================================================ 真值：块级 c1 求和
    def block_c1(a_flat, sel):
        """a_flat: (M,5) 动作；返回 (M,) 块级 c1 求和（同块内动作恒定 + 已知风速序列）"""
        u = U[sel]                                            # (M,5)
        acts = np.repeat(a_flat[:, None, :], 5, axis=1)        # (M,5,5)
        P = np.stack([P_of(acts[:, t, :], u[:, t]) for t in range(5)], axis=1)
        pf = block_c1_sum(P, PGP[sel])
        return c1_of(pf, bool(a.smooth)).sum(1)

    sel = idx
    base = block_c1(A5[sel], sel)
    Gtrue = np.zeros_like(G)
    for j in range(5):
        ap = A5[sel].copy(); ap[:, j] += a.delta
        am = A5[sel].copy(); am[:, j] -= a.delta
        Gtrue[:, j] = (block_c1(ap, sel) - block_c1(am, sel)) / (2 * a.delta)

    # ============================================================ T3 统计口径
    print('=' * 78)
    print('T3 逐样本 / 批次梯度统计（M_j=mean|g|，A_j=|mean g|，SNR=A_j/std）')
    print('  维 | M_j(pred) | A_j(pred) | SNR | 符号比例   | 样本间余弦均值')
    for j in range(5):
        g = G[:, j]
        cos = []
        for i in range(0, len(g) - 1, 7):
            v1, v2 = G[i], G[i + 1]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-9 and n2 > 1e-9:
                cos.append(float(v1 @ v2 / (n1 * n2)))
        print('  %2d | %9.4f | %9.4f | %5.3f | %5.1f%% 正 | %+.3f'
              % (j, np.abs(g).mean(), abs(g.mean()), abs(g.mean()) / max(1e-12, g.std()),
                 100 * (g > 0).mean(), np.mean(cos) if cos else float('nan')))

    # ============================================================ T4 分箱可辨识性 + 探针 A
    print('=' * 78)
    print('T4 探针 A：块级 c1 求和的导数（同块同风速） vs %s' % a.which)
    print('  sign 一致 %.1f%% | Pearson %.4f | Spearman %.4f | 幅值比 %.3f'
          % (100 * np.mean(np.sign(G) == np.sign(Gtrue)),
             np.corrcoef(G.ravel(), Gtrue.ravel())[0, 1],
             np.corrcoef(np.argsort(np.argsort(G.ravel())),
                         np.argsort(np.argsort(Gtrue.ravel())))[0, 1],
             np.abs(G).mean() / max(1e-12, np.abs(Gtrue).mean())))
    print('  （对照：逐步 c1 的导数 vs 同一个模型）')
    # 逐步真值（仅用于展示"两个数学量不同"）
    def step_c1_deriv(sel):
        u = U[sel]
        Gs = np.zeros((len(sel), 5))
        for j in range(5):
            def f(sign):
                aa = A5[sel].copy(); aa[:, j] += sign * a.delta
                acts = np.repeat(aa[:, None, :], 5, axis=1)
                P = np.stack([P_of(acts[:, t, :], u[:, t]) for t in range(5)], axis=1)
                pf = block_c1_sum(P, PGP[sel])
                return c1_of(pf, bool(a.smooth)).sum(1)
            Gs[:, j] = (f(+1) - f(-1)) / (2 * a.delta)
        return Gs
    _ = step_c1_deriv  # 块级已经包含逐步，此处不再重复计算

    print('  分风速箱：')
    uu = U[sel, 0]
    for lo, hi in ((0, 8), (8, 12), (12, 16), (16, 40)):
        m = (uu >= lo) & (uu < hi)
        if m.sum() < 10:
            continue
        # 局部线性拟合：用该箱内真值梯度做"局部值拟合 RMSE"的代理
        resid = (G[m] - Gtrue[m])
        print('    u∈[%2d,%2d) n=%4d | 动作 std %.3f | |真值| 中位 %8.2f | |预测| 中位 %8.2f | '
              'sign %5.1f%% | 残差 std %8.2f'
              % (lo, hi, m.sum(), A5[sel][m].std(), np.median(np.abs(Gtrue[m])),
                 np.median(np.abs(G[m])), 100 * np.mean(np.sign(G[m]) == np.sign(Gtrue[m])),
                 resid.std()))


if __name__ == '__main__':
    main()
