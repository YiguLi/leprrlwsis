#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_critic_blockgrad.py —— critic 的 ∂Q_c/∂a 与**块级**真值梯度对比（口径修正版）。

与旧探针 probe_cost_truth_grad.py 的两个关键区别
------------------------------------------------
1. **真值对象改成"块成本"**。旧探针拿"单步 c1 对 a 的偏导"当真值：
       ∂c1/∂a_j = (1+20·1{P_F>3}) · sign(P − P_G_prev) · ∂P/∂a_j
   其中 sign(·) 在样本间近似 50/50 ⇒ 这个"真值"本身的总体均值≈0
   （|x| 的一阶导在总体上是零均值的：E|X| 对尺度才有一阶导，对单步动作没有）。
   拿它去比 critic 只会得到"符号一致率 50%"这个**结构上必然**的结果，不能判 critic 失败。
   critic 回归的对象是 **5 步块成本 Σ_t c1**（nstep=5 的块级 Q_c），块内 4 个 |ΔP| 项
   随动作整体缩放 ⇒ 块级真值梯度有确定的一阶方向。必须用块级真值比。
2. **真值用 FLORIS**（与训练同一物理引擎，默认）；`--use_proxy 1` 时用无尾流解析代理
   P = k·u³·Σ4a(1−a)² 做对照，两者都打印以显示代理误差。

口径
----
    P_G'(t) = P(a', u_t) − pbrule_t            # pbrule_t 由实测反解后冻结（电池不随 a 变）
    Σc1(a') = Σ_{t=0..4} c1(|P_G'(t) − P_G'(t−1)|),   P_G'(−1) = 块首状态里的 P_G_prev
    c1(x)   = |x| + K_V·softplus_β(|x|−3), β=100   （run3 实测口径, eps=0）
    g_j     = [Σc1(a+δe_j) − Σc1(a−δe_j)] / 2δ      （δ=0.02, 越界截到 [0,0.5]）

critic 侧：Q_c 在归一化回归空间（s_c=100）⇒ 与真值同量纲需 ×100 后再比。
若 critic 学到了 E[块成本 | s,a]，则 |∂Q_c_raw/∂a| 应 ≈ |∂Σc1/∂a|，比值≈1。

用法
----
    python agentutil/probe_critic_blockgrad.py --run output/FULL-R2/AIFFLOR-run3 --n 400
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
from aifflor.aifflor_myfunc import Normalize  # noqa: E402

K_V, BETA = 20.0, 100.0
S_C, S_P = 100.0, 1000.0        # run3: --cost_scale 100 / --value_scale 1000
NT = 5


def c1_of(x):
    x = np.abs(np.asarray(x, dtype=np.float64))
    return x + K_V * (np.maximum(x - 3.0, 0.0)
                      + np.log1p(np.exp(-BETA * np.abs(x - 3.0))) / BETA - np.log(2.0) / BETA)


def parse_vec(col):
    return np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                      for x in col]).astype(np.float64)


def load_blocks(run, n):
    df = pd.read_csv(os.path.join(run, 'df_hsrl.csv'),
                     usecols=['step', 'observation', 'observation2', 'applied_action_H',
                              'turbine_profit', 'pr', 'u', 'c1', 'r1'])
    obs, obs2 = parse_vec(df.observation), parse_vec(df.observation2)
    N = len(df)
    nb = min(n, N // 5)
    rows = np.arange(N - nb * 5, N)
    B = dict(
        nb=nb,
        state0=obs[rows].reshape(nb, 5, 4)[:, 0, :],
        A=obs[rows].reshape(nb, 5, 4) * 0.0,      # 占位，下面填
        U=df.u.to_numpy()[rows].reshape(nb, 5),
        PR=df.pr.to_numpy()[rows].reshape(nb, 5),
        PG=obs2[rows, 3].reshape(nb, 5),          # P_G(t)
        PG_prev=obs[rows].reshape(nb, 5, 4)[:, 0, 3],
        ptur=df.turbine_profit.to_numpy()[rows].reshape(nb, 5) * 60.0
        / df.pr.to_numpy()[rows].reshape(nb, 5),
        C1=df.c1.to_numpy()[rows].reshape(nb, 5),
        R1=df.r1.to_numpy()[rows].reshape(nb, 5),
        A_full=parse_vec(df.applied_action_H)[rows].reshape(nb, 5, -1),
    )
    B['a0'] = B['A_full'][:, 0, :]
    B['pbrule'] = B['ptur'] - B['PG']             # 电池出力（正=充电）
    return B


# ------------------------------------------------------------------ 物理
_PH = {}


def _physics():
    import aifflor.aifflor_turbine as AT
    from aifflor.aifflor_env import AifflorPhysics
    if 'ph' not in _PH:
        _PH['ph'] = AifflorPhysics(nt=NT, calibrated=False, power_scale=AT.LEVEL_COEF)
        _PH['AT'] = AT
    return _PH['ph'], _PH['AT']


def power_floris(U, A):
    """U:(nb,5) A:(nb,5) ⇒ (nb,5) MW。A 在 5 步里相同（块内单动作）。"""
    ph, AT = _physics()
    Ab = np.repeat(A[:, None, :], U.shape[1], axis=1).reshape(-1, NT)
    farm, _ = AT.eval_batch(ph.model, np.clip(U.ravel(), AT.U_MIN_ENV, None), Ab)
    return (np.asarray(farm, dtype=float).ravel() * ph.power_scale).reshape(U.shape)


def block_cost(P_tur, B):
    PG = P_tur - B['pbrule']
    prev = np.concatenate([B['PG_prev'][:, None], PG[:, :-1]], axis=1)
    return c1_of(PG - prev).sum(1)


def truth_cost_fn(B, k_proxy, use_proxy):
    if use_proxy:
        def pw(a):
            s = (4.0 * a * (1.0 - a) ** 2).sum(1)
            return k_proxy * (B['U'] ** 3) * s[:, None]
        return pw
    return lambda a: power_floris(B['U'], a)


def central_grad(fn_cost, A0, delta):
    G = np.zeros_like(A0)
    for j in range(A0.shape[1]):
        Ap, Am = A0.copy(), A0.copy()
        Ap[:, j] += delta; Am[:, j] -= delta
        Ap[:, j] = np.clip(Ap[:, j], 0.0, 0.5)
        Am[:, j] = np.clip(Am[:, j], 0.0, 0.5)
        G[:, j] = (fn_cost(Ap) - fn_cost(Am)) / (2 * delta)
    return G


# ------------------------------------------------------------------ critic
def critic_out(B, run, which):
    sd = torch.load(os.path.join(run, which), map_location='cpu')
    net = model.build_critic_from_sd(sd, 4, 6)
    net.load_state_dict(sd); net.eval()
    S = torch.tensor(np.vstack([Normalize(o) for o in B['state0']]).astype(np.float32))
    at = torch.tensor(B['a0'].astype(np.float32), requires_grad=True)
    a_clip = at.clone()
    a_clip[:, :NT] = torch.clamp(at[:, :NT], 0.0, 0.5)
    q = net((S, a_clip))
    g, = torch.autograd.grad(q.sum(), at)
    return q.detach().numpy().ravel(), g.detach().numpy()[:, :NT]


def report(name, gc, G):
    ok = np.isfinite(G).all(1)
    gc, G = gc[ok], G[ok]
    big = np.abs(G) > np.percentile(np.abs(G), 50)
    print('  %-11s sign %5.1f%% (大梯度 %5.1f%%)  Pearson %+.3f  '
          'mean|∂Qc/∂a| / mean|真值| = %.3f'
          % (name, 100 * np.mean(np.sign(gc) == np.sign(G)),
             100 * np.mean(np.sign(gc[big]) == np.sign(G[big])),
             np.corrcoef(gc.ravel(), G.ravel())[0, 1],
             np.abs(gc).mean() / max(1e-12, np.abs(G).mean())))
    dg, dt = gc.sum(1) / np.sqrt(NT), G.sum(1) / np.sqrt(NT)
    print('              沿 +a 方向: critic mean %+8.2f sd %7.2f (mean/sd %+.2f, 正号 %.1f%%)'
          % (dg.mean(), dg.std(), dg.mean() / max(dg.std(), 1e-9), 100 * np.mean(dg > 0)))
    print('                          真值   mean %+8.2f sd %7.2f (mean/sd %+.2f, 正号 %.1f%%)'
          % (dt.mean(), dt.std(), dt.mean() / max(dt.std(), 1e-9), 100 * np.mean(dt > 0)))
    print('              逐维 [sign一致 / critic mean / 真值 mean / 真值 mean|.|]: '
          + ' | '.join('d%d %.0f%% %+.1f %+.1f/%.1f'
                       % (j, 100 * np.mean(np.sign(gc[:, j]) == np.sign(G[:, j])),
                          gc[:, j].mean(), G[:, j].mean(), np.abs(G[:, j]).mean())
                       for j in range(NT)))
    return gc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join('output', 'FULL-R2', 'AIFFLOR-run3'))
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--delta', type=float, default=0.02)
    ap.add_argument('--use_proxy', type=int, default=1,
                    help='1=解析代理（快，默认）；0=FLORIS（准，慢）')
    a = ap.parse_args()

    B = load_blocks(a.run, a.n)
    Ab = np.repeat(B['a0'][:, None, :NT], 5, axis=1)
    s = (4.0 * Ab * (1.0 - Ab) ** 2).sum(2)
    x = (B['U'] ** 3) * s
    k = float((x * B['ptur']).sum() / (x * x).sum())

    print('run=%s  块数 %d  观测 Σc1 均值 %.3f  代理 k=%.6e' % (a.run, B['nb'],
                                                          B['C1'].sum(1).mean(), k))
    print('实测: mean a=%s  mean u=%.1f  P_F>3 占 %.1f%%'
          % (np.round(B['a0'][:, :NT].mean(0), 3), B['U'].mean(),
             100 * np.mean(np.abs(B['PG'] - np.concatenate([B['PG_prev'][:, None], B['PG'][:, :-1]], 1)) > 3)))
    pw = truth_cost_fn(B, k, a.use_proxy)
    base = block_cost(pw(B['a0'][:, :NT]), B)
    print('代理/引擎自洽性: mean|Σĉ1 − Σc1_obs| = %.3f  vs  mean Σc1_obs = %.3f'
          % (np.mean(np.abs(base - B['C1'].sum(1))), np.mean(B['C1'].sum(1))))

    G = central_grad(lambda A: block_cost(pw(A), B), B['a0'][:, :NT], a.delta)
    print('\n[真值 ∂Σc1/∂a]（%s）' % ('代理' if a.use_proxy else 'FLORIS'))
    print('  沿 +a: mean %+.2f sd %.2f (mean/sd %+.2f) 正号 %.1f%%'
          % ((G.sum(1) / np.sqrt(NT)).mean(), (G.sum(1) / np.sqrt(NT)).std(),
             (G.sum(1) / np.sqrt(NT)).mean() / max((G.sum(1) / np.sqrt(NT)).std(), 1e-9),
             100 * np.mean(G.sum(1) > 0)))

    print('\n[critic ∂Q_c/∂a ×s_c  vs  真值 ∂Σc1/∂a]')
    for which in ('critic_c_6.pkl', 'critic_c2_6.pkl'):
        q, gc = critic_out(B, a.run, which)
        print(' %s: Q_c(raw) mean %.1f  (实测块成本均值 %.1f, 折扣后真值 ≈%.1f)'
              % (which, q.mean() * S_C, B['C1'].sum(1).mean(),
                 B['C1'].sum(1).mean() / (1 - 0.95 ** 5)))
        report(which.replace('.pkl', ''), gc * S_C, G)

    # ---- profit 通道与 λ 权衡（量级对照）----
    Grev = central_grad(lambda A: ((pw(A) - B['pbrule']) * B['PR'] / 60.0).sum(1),
                        B['a0'][:, :NT], a.delta)
    qp, gp = critic_out(B, a.run, 'critic_p_6.pkl')
    gp = gp * S_P
    qc, gcmax = critic_out(B, a.run, 'critic_c_6.pkl')
    print('\n[λ 权衡]（元/单位 a；actor 目标 = Q_p − λ_phys·Q_c；负值=该方向降成本占优）')
    dv_t = (Grev.sum(1) / np.sqrt(NT)).mean()
    dc_t = (G.sum(1) / np.sqrt(NT)).mean()
    dv_c = (gp.sum(1) / np.sqrt(NT)).mean()
    dc_c = (gcmax.sum(1) / np.sqrt(NT)).mean() * S_C
    print('  %-6s %26s %28s' % ('λ', '真值: Q_p − λ·Q_c', 'critic: Q_p − λ·Q_c'))
    for lam in (2.0, 10.0, 19.3, 50.0):
        print('  %5.1f  %+8.1f − %5.1f×%+7.1f = %+9.1f   | %+8.1f − %5.1f×%+7.1f = %+9.1f'
              % (lam, dv_t, lam, dc_t, dv_t - lam * dc_t, dv_c, lam, dc_c, dv_c - lam * dc_c))
    print('  逐维真值 vs critic（∂Q_p−λ∂Q_c, λ=50）:')
    for j in range(NT):
        print('    d%d  真值 %+9.1f   critic %+9.1f'
              % (j, Grev[:, j].mean() - 50.0 * G[:, j].mean(),
                 gp[:, j].mean() - 50.0 * gcmax[:, j].mean() * S_C))
    print('\n  实测策略 a = %s' % np.round(B['a0'][:, :NT].mean(0), 3))
    print('  ⇒ 若逐维"真值最优"是压到 0/0.5 边界, 而 critic 给出的逐维排序与之不符,'
          ' 演员就会压错维度。')


if __name__ == '__main__':
    main()
