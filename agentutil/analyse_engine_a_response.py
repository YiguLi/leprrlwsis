# -*- coding: utf-8 -*-
"""随机动作数据的进一步分析：**两引擎对 `a` 的最优响应是否一致**。

为什么需要它
------------
`probe_engine_random_actions.py` 实测出（LHS n=100）：
  比值 std 0.1388（a≡1/3 单点只有 0.0086）、`corr(比值, mean(a)) = +0.7014`、
  常数重标定后残差 **10.58%**（单点只有 0.465%）。

这暗示两引擎对 `a` 的**响应函数形状不同**。本脚本用同一批数据检验：
  ① 在控制风速后，`P_farm` 对 `mean(a)` 的响应曲线（两引擎分别拟合），最优点在哪；
  ② 把 `a` 全取同一水平（近均匀样本）时的比值，与同 `mean(a)` 但强差异化的比值对照；
  ③ 常数重标定残差能否靠"随风速/随动作"的修正消掉。

用法: F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_engine_a_response.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

JSON_DEFAULT = os.path.join('output', 'engine_random_actions.json')


def fit_response(u, P, am, order=2):
    """控制风速后 P 对 mean(a) 的多项式响应：P ≈ f(u³) + Σ b_k·am^k。"""
    X = np.c_[u ** 3, np.ones_like(u)]
    for k in range(1, order + 1):
        X = np.c_[X, am ** k]
    beta, *_ = np.linalg.lstsq(X, P, rcond=None)
    pred = X @ beta
    r2 = 1 - ((P - pred) ** 2).sum() / ((P - P.mean()) ** 2).sum()
    coef = beta[2:]                      # b_1..b_order（升幂）
    dcoef = np.array([k * coef[k - 1] for k in range(1, order + 1)])   # 导数系数（升幂）
    grid = np.linspace(0.0, 0.5, 2001)
    # P(am) 的 am 相关部分（去掉 u³/常数项，不影响 argmax）
    curve = sum(coef[k - 1] * grid ** k for k in range(1, order + 1))
    am_star = float(grid[int(np.argmax(curve))])
    return dict(beta=beta, r2=float(r2), dcoef=dcoef, am_star=am_star, order=order)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else JSON_DEFAULT
    d = json.load(open(path, encoding='utf-8'))
    u = np.asarray(d['wind'], dtype=float)
    A = np.asarray(d['actions'], dtype=float)
    w = np.asarray(d['p_wfsim'], dtype=float)
    f = np.asarray(d['p_floris'], dtype=float)
    cfg = d.get('config', {})
    am, asd = A.mean(axis=1), A.std(axis=1)
    m = np.isfinite(w) & np.isfinite(f) & (w > 1e-9)

    print('数据: %s  (mode=%s n=%d seed=%s)' % (path, cfg.get('mode'), cfg.get('n'),
                                                cfg.get('seed')))
    print('=' * 96)
    print('① 控制风速后 P_farm 对 mean(a) 的响应（二次拟合，样本 n=%d）' % m.sum())
    print('=' * 96)
    rw = fit_response(u[m], w[m], am[m])
    rf = fit_response(u[m], f[m], am[m])
    print('  %-10s %12s %12s %14s' % ('引擎', 'R²', '最优点 a*', '二次项系数'))
    print('  %-10s %12.6f %12.4f %14.6g' % ('WFSim', rw['r2'], rw['am_star'], rw['dcoef'][-1]))
    print('  %-10s %12.6f %12.4f %14.6g' % ('FLORIS', rf['r2'], rf['am_star'], rf['dcoef'][-1]))
    print('  ⇒ 两引擎的风场最优 a* 相差 %.4f' % abs(rw['am_star'] - rf['am_star']))
    print('     （本项目已测：WFSim 均匀最优 ≈0.20、FLORIS ≈0.40 ⇒ 预期差 ~0.2）')

    print('\n' + '=' * 96)
    print('② 同一 mean(a) 下，空间结构 std(a) 对 P 的偏效应（控制了 u³ 与 mean(a)）')
    print('=' * 96)
    for tag, P in (('WFSim', w), ('FLORIS', f)):
        X = np.c_[u[m] ** 3, np.ones(m.sum()), am[m], am[m] ** 2, asd[m]]
        beta, *_ = np.linalg.lstsq(X, P[m], rcond=None)
        pred = X @ beta
        r2 = 1 - ((P[m] - pred) ** 2).sum() / ((P[m] - P[m].mean()) ** 2).sum()
        print('  %-8s ∂P/∂std(a) = %+10.4f MW   (均值 P=%.3f MW)   R²=%.6f'
              % (tag, beta[4], P[m].mean(), r2))
    print('  读法：两引擎的 ∂P/∂std(a) 若**符号相反或量级差很多**，说明它们对"空间差异化"')
    print('        的偏好不同 —— 这正是 RL 在两边会学到不同动作结构的原因。')

    print('\n' + '=' * 96)
    print('③ 常数 vs 修正模型的重标定残差（相对 FLORIS 均值）')
    print('=' * 96)
    Pw, Pf = w[m], f[m]
    models = [
        ('常数 k',                     np.c_[Pw]),
        ('k·Pw + c',                   np.c_[Pw, np.ones_like(Pw)]),
        ('k·Pw + c·Pw·u',              np.c_[Pw, Pw * u[m]]),
        ('k·Pw + c·Pw·am',             np.c_[Pw, Pw * am[m]]),
        ('k·Pw + c·Pw·u + d·Pw·am',    np.c_[Pw, Pw * u[m], Pw * am[m]]),
        ('k·Pw + c·Pw·u + d·Pw·am + e·Pw·asd',
         np.c_[Pw, Pw * u[m], Pw * am[m], Pw * asd[m]]),
    ]
    print('  %-38s %10s %12s %12s' % ('模型', '参数数', '残差RMS', '最大|残差|'))
    for name, X in models:
        beta, *_ = np.linalg.lstsq(X, Pf, rcond=None)
        r = (Pf - X @ beta) / Pf.mean()
        print('  %-38s %10d %11.3f%% %11.3f%%'
              % (name, X.shape[1], 100 * np.sqrt((r ** 2).mean()), 100 * np.abs(r).max()))

    print('\n' + '=' * 96)
    print('④ 比值在 mean(a) 上的回归（判断"比值是否是 a 的函数"）')
    print('=' * 96)
    ratio = Pf / Pw
    for tag, X in (('仅常数', np.ones((m.sum(), 1))),
                   ('常数 + mean(a)', np.c_[np.ones(m.sum()), am[m]]),
                   ('常数 + mean(a) + u', np.c_[np.ones(m.sum()), am[m], u[m]]),
                   ('常数 + mean(a) + u + std(a)',
                    np.c_[np.ones(m.sum()), am[m], u[m], asd[m]])):
        X = np.atleast_2d(X.T).T if X.ndim == 1 else X
        beta, *_ = np.linalg.lstsq(X, ratio, rcond=None)
        r = ratio - X @ beta
        r2 = 1 - (r ** 2).sum() / ((ratio - ratio.mean()) ** 2).sum()
        print('  %-34s R²=%.4f  残差std=%.4f  系数=%s'
              % (tag, r2, r.std(), np.array2string(beta, precision=4)))
    print('  ⇒ 若"常数 + mean(a)"就能把 R² 抬到 ~0.5，说明比值主要是 mean(a) 的函数，')
    print('     而 mean(a) 正是策略的动作 —— 即重标定必须**随动作**变，不能用常数。')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
