#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSRL run 打分（项目侧单一真源）。

`summarize()` 的口径与 `agentutil/diag_report.py` 里的实现**逐条一致**（公式照抄，
后者暂保留自己的一份，含 scalar 任务专用的 regret 分支），供两处使用：
  * `aifflor/aifflor_main.py` 训练结束时打印与 diag 同格式的汇总行；
  * `agentutil/diag_report.py` 追加 CSV 行。

指标（只用最后 `tail` 比例的数据）：
  eta        = P_G / P_max(u)                     出力最大化能力（1 = 达到该风速的物理最优）
  pattern_err= mean|a − a*(u)|                    与最优动作模式的偏离
  viol_pct   = 100·mean(P_F > 3)                  波动越限率
  a_std_step = std(Δa)                            动作逐块波动（策略抖动）
  r2_dua     = R²(P_G ~ u, a) − R²(P_G ~ u)       动作对出力的增量解释力
"""
import os

import numpy as np
import pandas as pd

_COLS = ["step", "u", "pr", "turbine_profit", "P_F", "c1", "applied_action_H",
         "original_action_H", "lam1"]


def _r2(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()


def summarize(run_dir, tail=0.2, refs_dir=os.path.join('output', 'diag'), tag=None,
              task='maxpower', usecols=None):
    """读 `<run_dir>/df_hsrl.csv` 并返回指标 dict（键名与 diag summary.csv 一致）。"""
    cols = usecols or _COLS
    df = pd.read_csv(os.path.join(run_dir, 'df_hsrl.csv'), usecols=cols)
    A = np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                   for x in df.applied_action_H])
    df = df.drop(columns=["applied_action_H", "original_action_H"], errors='ignore')
    df["PG"] = df.turbine_profit * 60.0 / df.pr
    df["a"] = A[:, :5].mean(1)

    n0 = int(len(df) * (1 - tail))
    D = df.iloc[n0:]
    Aa = A[n0:]

    refs_u = pd.read_csv(os.path.join(refs_dir, 'refs_u.csv'))
    gr = refs_u.u.to_numpy(float)
    pmax = np.interp(D.u.to_numpy(float), gr, refs_u.P_max.to_numpy(float))
    eta = D.PG.to_numpy(float) / np.maximum(pmax, 1e-9)
    astar = np.column_stack([np.interp(D.u.to_numpy(float), gr,
                                       refs_u['a%d' % j].to_numpy(float)) for j in range(5)])
    pat = np.abs(Aa[:, :5] - astar).mean()

    y = D.PG.to_numpy(float)
    r_u = _r2(y, [D.u.to_numpy(float)])
    r_ua = _r2(y, [D.u.to_numpy(float), D.a.to_numpy(float)])
    return dict(task=task, run=tag or os.path.basename(str(run_dir).rstrip('/\\')),
                eta=float(eta.mean()), eta_p5=float(np.percentile(eta, 5)),
                eta_min=float(eta.min()), pattern_err=float(pat),
                viol_pct=100.0 * float((D.P_F > 3.0).mean()),
                a_mean=float(D.a.mean()),
                a_std_step=float(np.diff(D.a.to_numpy(float)).std()),
                u_mean=float(D.u.mean()), PG_mean=float(D.PG.mean()),
                r2_u=float(r_u), r2_dua=float(r_ua - r_u),
                lam=float(D.lam1.median()), n=int(len(D)))


SUMMARY_HEADER = ('{:>14} {:>8} {:>7} {:>11} {:>9} {:>10} {:>8} {:>7}'
                  .format('run', 'task', 'eta', 'pattern_err', 'viol_pct',
                          'a_std_step', 'r2_dua', 'n'))


def format_row(m):
    """一行汇总（与 diag 的表格列序/宽度一致）。"""
    return ('{:>14} {:>8} {:>7.3f} {:>11.3f} {:>9.3f} {:>10.4f} {:>8.3f} {:>7d}'
            .format(m['run'], m['task'], m['eta'], m['pattern_err'], m['viol_pct'],
                    m['a_std_step'], m['r2_dua'], m['n']))
