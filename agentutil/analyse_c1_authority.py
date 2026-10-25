#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyse_c1_authority.py —— "波动平抑"到底有多少是 H 动作能改变的？

用法（任意 CWD）：
    python agentutil/analyse_c1_authority.py [--run output/FULL-R2/AIFFLOR-run3] [--tail 0.15]

只读分析，不重训。回答三个问题：
  Q1 |ΔP_net| 的量级从哪来：风机项（受 H 动作 a 影响）与电池项（只受块总量 g 影响）
     各占多少？
  Q2 真实可控性：把本块动作整体缩放 a' = s·a（风机物理代理 P=c·u³·Σ4a(1−a)²），
     块成本 Σ_t c1 随 s 的真实斜率是多少？最优 s* 是多少？可压缩空间有多大？
  Q3 块级真值梯度 ∂Σc1/∂a_j 的"整场升 a"方向投影 mean/sd —— 即一阶梯度本身有多弱。

c1 口径：run3 实测 c1 = P_F + K_V·softplus_β(P_F−3, β=100) − 常数（|ΔP| 未平滑，
eps=0），因此这里直接用同一形状重算（不依赖当前 hc.SMOOTH 开关）。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import hsrl_cost as hc  # noqa: E402

K_V = float(hc.K_V)          # 20
BETA = 100.0                 # run3 的 c1_smooth_beta

USECOLS = ["step", "observation", "observation2", "applied_action_H", "g", "c1",
           "P_F", "u", "pr", "turbine_profit", "bess_profit", "SOC",
           "lam1", "lam1_phys", "d_block_H", "ema_c1_H", "q_cost_H"]


def _shift_sp(x):
    """softplus_β(x) − ln2/β（x=0 处取 0），向量化。"""
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(BETA * x))) / BETA - np.log(2.0) / BETA


def c1_of(P_F, hard=False):
    """run3 口径的波动当量 [MW·step]：|ΔP| + K_V·(kink)。hard=True 用硬折点做敏感性对照。"""
    P = np.abs(np.asarray(P_F, dtype=np.float64))
    return P + K_V * (np.maximum(P - 3.0, 0.0) if hard else _shift_sp(P - 3.0))


def _parse_vec(col, n):
    return np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                      for x in col]).astype(np.float64)[:, :n]


def load(run, tail):
    df = pd.read_csv(os.path.join(run, 'df_hsrl.csv'), usecols=USECOLS)
    obs = _parse_vec(df.observation, 4)
    obs2 = _parse_vec(df.observation2, 4)
    a = _parse_vec(df.applied_action_H, 6)
    df['u_'] = obs[:, 1]
    df['pr_'] = obs[:, 0]
    df['E'] = obs[:, 2]
    df['PG_prev'] = obs[:, 3]
    df['PG'] = obs2[:, 3]
    df['dP_obs'] = df['PG'] - df['PG_prev']
    for j in range(6):
        df['a%d' % j] = a[:, j]
    n0 = int(len(df) * (1 - tail))
    return df, max(0, n0 - (n0 % 5))


def validate_proxy(D):
    A = D[['a%d' % j for j in range(5)]].to_numpy()
    S = (4.0 * A * (1.0 - A) ** 2).sum(1)
    x = (D.u_.to_numpy() ** 3) * S
    y = D.turbine_profit.to_numpy() * 60.0 / D.pr_.to_numpy()
    k = float((x * y).sum() / (x * x).sum())
    yhat = k * x
    r2 = 1.0 - float(((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    mae = float(np.mean(np.abs(y - yhat)))
    print('[proxy] ptur ≈ %.6e·u³·Σ4a(1−a)²   R²=%.5f  MAE=%.3f MW (%.2f%% of mean|ptur|=%.3f)'
          % (k, r2, mae, 100 * mae / max(y.mean(), 1e-9), y.mean()))
    # 分风速检查（高风段越限多，最需要准）
    for lo, hi in [(0, 8), (8, 12), (12, 16), (16, 40)]:
        m = (D.u_.to_numpy() >= lo) & (D.u_.to_numpy() < hi)
        if m.sum() > 50:
            e = np.abs(y[m] - yhat[m]).mean()
            print('    u∈[%2d,%2d): n=%6d  MAE=%.3f MW  mean ptur=%.3f' % (lo, hi, m.sum(), e, y[m].mean()))
    return k


def decompose(D):
    pr = D.pr_.to_numpy()
    ptur = D.turbine_profit.to_numpy() * 60.0 / pr
    pbrule = ptur - D['PG'].to_numpy()
    dp = D.dP_obs.to_numpy()
    d_t = np.r_[np.nan, np.diff(ptur)]
    d_b = np.r_[np.nan, np.diff(pbrule)]
    m = np.isfinite(d_t) & np.isfinite(d_b)
    d_t, d_b, dp = d_t[m], d_b[m], dp[m]
    print('\n[Q1] 逐步 |ΔP_net| 分解（n=%d）' % len(dp))
    print('  mean|ΔP_net|=%.3f  mean|Δptur|=%.3f  mean|Δp_bess|=%.3f MW'
          % (np.mean(np.abs(dp)), np.mean(np.abs(d_t)), np.mean(np.abs(d_b))))
    print('  var 份额: 风机 %.1f%%  电池 %.1f%%  cov %.1f%%'
          % (100 * np.var(d_t) / np.var(dp), 100 * np.var(d_b) / np.var(dp),
             100 * 2 * np.cov(d_t, d_b)[0, 1] / np.var(dp)))
    print('  sign(ΔP_net): 升 %.1f%%  降 %.1f%%  零 %.1f%%  —— 单步动作梯度 ∝ sign(ΔP)·∂P/∂a，'
          '若"升/降"接近 50/50，则其总体均值≈0'
          % (100 * np.mean(dp > 1e-6), 100 * np.mean(dp < -1e-6),
             100 * np.mean(np.abs(dp) <= 1e-6)))
    return ptur, pbrule


def _blocks(D):
    nb = len(D) // 5
    D = D.iloc[:nb * 5]
    u = D.u_.to_numpy().reshape(nb, 5)
    pr = D.pr_.to_numpy().reshape(nb, 5)
    A = D[['a%d' % j for j in range(5)]].to_numpy().reshape(nb, 5, 5)
    ptur = D.turbine_profit.to_numpy().reshape(nb, 5) * 60.0 / pr
    PG = D['PG'].to_numpy().reshape(nb, 5)
    pbrule = ptur - PG
    PG_prev = D.PG_prev.to_numpy().reshape(nb, 5)[:, 0]
    c1 = D.c1.to_numpy().reshape(nb, 5)
    return dict(nb=nb, u=u, pr=pr, A=A, ptur=ptur, PG=PG, pbrule=pbrule,
                PG_prev=PG_prev, c1=c1)


def cost_curve(k, B, s_grid):
    def block_cost(a_new):
        g = 4.0 * a_new * (1.0 - a_new) ** 2
        p = k * (B['u'] ** 3) * g.sum(2) - B['pbrule']       # [nb,5] P_G
        prev = np.concatenate([B['PG_prev'][:, None], p[:, :-1]], axis=1)
        return c1_of(p - prev).sum(1)
    curves = np.stack([block_cost(s * B['A']) for s in s_grid], axis=1)
    return curves, block_cost(B['A'])


def report_curve(B, curves, base_hat, s_grid):
    tot = B['c1'].sum(1)
    print('\n[Q2] 反事实块成本（a → s·a，电池轨迹与风序列冻结）')
    print('  代理自洽性: mean|Σĉ1(a)−Σc1_obs| = %.3f vs mean Σc1_obs = %.3f  (%.1f%%)'
          % (np.mean(np.abs(base_hat - tot)), np.mean(tot),
             100 * np.mean(np.abs(base_hat - tot)) / np.mean(tot)))
    mu = curves.mean(0)
    print('  %6s %12s %12s %14s' % ('s', 'mean Σc1', 'median', 'd mean/ds'))
    for i, s in enumerate(s_grid):
        d = ((mu[i + 1] - mu[i - 1]) / (s_grid[i + 1] - s_grid[i - 1])
             if 0 < i < len(s_grid) - 1 else np.nan)
        print('  %6.2f %12.3f %12.3f %14s' % (s, mu[i], np.median(curves[:, i]),
                                              ('%+.3f' % d) if np.isfinite(d) else '-'))
    i1 = int(np.argmin(np.abs(np.asarray(s_grid) - 1.0)))
    slope = (mu[i1 + 1] - mu[i1 - 1]) / (s_grid[i1 + 1] - s_grid[i1 - 1])
    print('  ⇒ s=1 处 dE[Σc1]/ds = %+.3f MW·step/块；"整场多降载 10%%"的成本收益 = %.3f'
          % (slope, 0.1 * slope))
    best = np.asarray(s_grid)[np.argmin(curves, axis=1)]
    print('  ⇒ 逐块最优 s*: mean %.3f  median %.3f   |  mean Σc1(s=1)=%.3f → min %.3f (%.1f%% 可压缩)'
          % (best.mean(), np.median(best), mu[i1], mu.min(),
             100 * (1 - mu.min() / mu[i1])))
    # 首步（唯一"跟上一块"可控的那一步）占块成本的比例
    f = B['c1'][:, 0]
    print('  ⇒ 首步（可用 a 去"贴住"P_G_prev 的唯一一步）占块成本: mean %.1f%% (%.2f / %.2f)'
          % (100 * np.mean(np.abs(f) / np.maximum(np.abs(B['c1']).sum(1), 1e-9)),
             np.mean(np.abs(f)), np.mean(np.abs(B['c1']).sum(1))))
    return mu, slope


def truth_grad(B, k, delta=0.05):
    def cost(a_new):
        g = 4.0 * a_new * (1.0 - a_new) ** 2
        p = k * (B['u'] ** 3) * g.sum(2) - B['pbrule']
        prev = np.concatenate([B['PG_prev'][:, None], p[:, :-1]], axis=1)
        return c1_of(p - prev).sum(1)
    nb = B['nb']
    G = np.zeros((nb, 5))
    for j in range(5):
        Ap = B['A'].copy(); Am = B['A'].copy()
        Ap[:, :, j] += delta; Am[:, :, j] -= delta
        Ap[:, :, j] = np.clip(Ap[:, :, j], -0.05, 0.55)
        Am[:, :, j] = np.clip(Am[:, :, j], -0.05, 0.55)
        G[:, j] = (cost(Ap) - cost(Am)) / (2 * delta)
    print('\n[Q3] 块级真值 ∂Σc1/∂a_j（代理中心差分 δ=%.2f, n=%d 块）' % (delta, nb))
    print('  逐维: mean g_j / mean|g_j| / g_j>0 比例')
    for j in range(5):
        print('    dim%d %+9.3f %9.3f %8.1f%%' % (j, G[:, j].mean(), np.abs(G[:, j]).mean(),
                                                 100 * np.mean(G[:, j] > 0)))
    ssum = G.sum(1) / np.sqrt(5.0)            # 沿 (1,1,1,1,1) 的方向导数
    print('  沿"整场升 a"单位方向: mean %+8.3f  sd %8.3f  mean/sd %+6.3f  正号 %.1f%%'
          % (ssum.mean(), ssum.std(), ssum.mean() / max(ssum.std(), 1e-12),
             100 * np.mean(ssum > 0)))
    jump = (k * (B['u'][:, 0] ** 3) * (4 * B['A'][:, 0, :] * (1 - B['A'][:, 0, :]) ** 2).sum(1)) - B['PG_prev']
    print('  首步跳变 P(a,u0)−P_G_prev: mean|.|=%.3f MW;  -sign(∂c1/∂a) 与 sign(跳变) 一致率 %.1f%%'
          % (np.mean(np.abs(jump)), 100 * np.mean(np.sign(-G[:, 0]) == np.sign(jump))))
    # 高风段单独看（越限发生区）
    hi = B['u'][:, 0] >= 12
    if hi.sum() > 30:
        print('  高风段 (u0≥12, n=%d): 沿升 a 方向 mean/sd %+6.3f, 正号 %.1f%%'
              % (hi.sum(), ssum[hi].mean() / max(ssum[hi].std(), 1e-12),
                 100 * np.mean(ssum[hi] > 0)))
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join('output', 'FULL-R2', 'AIFFLOR-run3'))
    ap.add_argument('--tail', type=float, default=0.15)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df, n0 = load(args.run, args.tail)
    D = df.iloc[n0:]
    print('run=%s rows=%d tail 起点 step=%d' % (args.run, len(df), int(df.step.iloc[n0])))
    print('tail: mean c1=%.3f  mean P_F=%.3f  越限率=%.2f%%  Σc1/块=%.2f  d_block_H=%.4f  '
          'lam1_phys=%.1f  ema_c1_H=%.3f'
          % (D.c1.mean(), np.abs(D.dP_obs).mean(), 100 * np.mean(D.P_F > 3.0),
             5 * D.c1.mean(), D.d_block_H.iloc[-1], D.lam1_phys.iloc[-1], D.ema_c1_H.iloc[-1]))
    print('tail: mean a=%s   η≈(P_G/Pmax 见 hsrl_metrics)'
          % np.round(D[['a%d' % j for j in range(5)]].mean().to_numpy(), 3))

    k = validate_proxy(D)
    decompose(D)
    B = _blocks(D)
    s_grid = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    curves, base_hat = cost_curve(k, B, s_grid)
    report_curve(B, curves, base_hat, s_grid)
    truth_grad(B, k)

    if args.out:
        rows = [dict(block=i, c1_obs=B['c1'][i].sum(), c1_hat=base_hat[i],
                     **{'c1_s%.2f' % s: curves[i, j] for j, s in enumerate(s_grid)})
                for i in range(B['nb'])]
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print('\n[out] %s' % args.out)


if __name__ == '__main__':
    main()
