#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 0 + A 的数据部分：公共随机数配对反事实数据集（块级即时成本）。

口径（冻结，写进元数据）：
  - 目标 = **块级**风力机侧 c1 求和： Σ_{t=1..5} c1(P_F_t)
      P_F_1 = |P(a,u_1) - P_G_prev|      （P_G_prev 取日志里块首前一步的出力）
      P_F_t = |P(a,u_t) - P(a,u_{t-1})|  （同块内动作恒定，风速序列为该块真实记录）
      c1(x) = x + 20·(x-3)⁺      （硬折点；与 --smooth 0 的 run 一致）
  - 动作 = 环境实际执行动作 applied_action_H 的前 5 维；块内恒定。
  - 状态 = [pr/1000, u_1/10, SOC/0.5, P_G_prev/5]（与 aux_c 基线一致）。
  - 划分 = 按**整天**（1440 步 = 1 天）划分，正负分支与基准分支永远同属一个集合。
  - δ = 0.005 / 0.02 / 0.05 三个尺度；靠近边界用单边差分并标记（不参与中心差分统计）。

产出： output/counterfactual/cf_<tag>.npz
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import aifflor.aifflor_turbine as AT  # noqa: E402
from aifflor.aifflor_env import AifflorPhysics  # noqa: E402

K_V, LIMIT, D = 20.0, 3.0, 5


def c1_of(pf):
    pf = np.asarray(pf, float)
    return pf + K_V * np.maximum(0.0, pf - LIMIT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='output/FULL-R2/AIFFLOR-run3')
    ap.add_argument('--n', type=int, default=4000, help='采样的块数')
    ap.add_argument('--deltas', default='0.005,0.02,0.05')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='output/counterfactual')
    ap.add_argument('--train_frac', type=float, default=0.7)
    ap.add_argument('--val_frac', type=float, default=0.15)
    a = ap.parse_args()

    ph = AifflorPhysics(nt=D, calibrated=False, power_scale=AT.LEVEL_COEF)

    def P_of(acts, u):
        farm, _ = AT.eval_batch(ph.model, np.clip(np.asarray(u, float), AT.U_MIN_ENV, None),
                                np.asarray(acts, float))
        return np.asarray(farm, float).ravel() * ph.power_scale

    print('[cf] 读取 %s' % a.run)
    df = pd.read_csv(os.path.join(a.run, 'df_hsrl.csv'),
                     usecols=['step', 'u', 'pr', 'SOC', 'turbine_profit', 'applied_action_H'])
    A6 = np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                    for x in df.applied_action_H]).astype(np.float64)
    PG = (df.turbine_profit * 60 / df.pr).to_numpy()
    nb = len(df) // 5
    U = df.u.to_numpy(float)[:nb * 5].reshape(nb, 5)
    A5 = A6[:nb * 5].reshape(nb, 5, 6)[:, 0, :5]
    PR = df.pr.to_numpy(float)[:nb * 5][::5]
    SOC = df.SOC.to_numpy(float)[:nb * 5][::5]
    PGP = np.r_[PG[0], PG[:nb * 5 - 1]][::5][:nb]
    DAY = (np.arange(nb) * 5) // 1440

    rng = np.random.RandomState(a.seed)
    sel = np.sort(rng.choice(nb, size=min(a.n, nb), replace=False))
    print('[cf] 采样 %d 块（总 %d）| 天数 %d' % (len(sel), nb, len(np.unique(DAY))))

    def block_cost(actions5, sel_):
        """actions5: (M,5)；返回 (M,) 块级 c1 求和"""
        u = U[sel_]
        acts = np.repeat(actions5[:, None, :], 5, axis=1)          # (M,5,5)
        P = np.stack([P_of(acts[:, t, :], u[:, t]) for t in range(5)], axis=1)
        pf = np.empty_like(P)
        pf[:, 0] = np.abs(P[:, 0] - PGP[sel_])
        pf[:, 1:] = np.abs(np.diff(P, axis=1))
        return c1_of(pf).sum(1)

    base = block_cost(A5[sel], sel)
    out = {'state': np.column_stack([PR[sel] / 1000, U[sel, 0] / 10, SOC[sel] / 0.5, PGP[sel] / 5]).astype(np.float32),
           'action': A5[sel].astype(np.float32),
           'action6': A6[:nb * 5].reshape(nb, 5, 6)[:, 0, :][sel].astype(np.float32),
           'cost_base': base.astype(np.float32),
           'day': DAY[sel].astype(np.int32),
           'u0': U[sel, 0].astype(np.float32),
           'step': (sel * 5).astype(np.int64)}

    for d in [float(x) for x in a.deltas.split(',')]:
        d_true = np.zeros((len(sel), 5), np.float32)
        d_eff = np.zeros((len(sel), 5), np.float32)
        one_side = np.zeros((len(sel), 5), np.int8)
        for j in range(5):
            ap_ = A5[sel].copy(); ap_[:, j] += d
            am_ = A5[sel].copy(); am_[:, j] -= d
            hi = ap_[:, j] > 0.5
            lo = am_[:, j] < 0.0
            ap_[hi, j] = 0.5
            am_[lo, j] = 0.0
            cp = block_cost(ap_, sel); cm = block_cost(am_, sel)
            eff = (ap_[:, j] - am_[:, j]) / 2.0
            good = (~hi) & (~lo)
            d_true[:, j] = np.where(good, (cp - cm) / np.maximum(2 * eff, 1e-9), (cp - cm) / np.maximum(eff, 1e-9))
            d_eff[:, j] = eff
            one_side[:, j] = (~good).astype(np.int8)
        out['d%.3f' % d] = d_true
        out['deff%.3f' % d] = d_eff
        out['oneside%.3f' % d] = one_side
        print('[cf] δ=%.3f | |d| 中位 %.2f | 单边样本占比 %.1f%% | 平均 δ_eff %.4f'
              % (d, np.median(np.abs(d_true)), 100 * one_side.mean(), d_eff.mean()))

    # 划分（按整天）
    days = np.unique(out['day'])
    n_tr = int(len(days) * a.train_frac)
    n_va = int(len(days) * a.val_frac)
    tr, va, te = days[:n_tr], days[n_tr:n_tr + n_va], days[n_tr + n_va:]
    split = np.where(np.isin(out['day'], tr), 0, np.where(np.isin(out['day'], va), 1, 2)).astype(np.int8)
    out['split'] = split
    out['meta'] = np.array(['target=block_c1_sum_turbine_only;state=[pr/1000,u0/10,SOC/0.5,PGprev/5];'
                            'split_by_day;deltas=%s' % a.deltas])
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.basename(os.path.normpath(a.run))
    fp = os.path.join(a.out, 'cf_%s.npz' % tag)
    np.savez_compressed(fp, **out)
    print('[cf] 训练/验证/测试 = %d/%d/%d 块 | 保存 %s (%.1f MB)'
          % ((split == 0).sum(), (split == 1).sum(), (split == 2).sum(), fp,
             os.path.getsize(fp) / 1e6))


if __name__ == '__main__':
    main()
