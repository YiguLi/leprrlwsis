#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给诊断 run 打分：η（出力最大化能力）/ pattern 误差 / regret（实验 2）/ 越界率 / 动作 std。

用法:
  python agentutil/diag_report.py --run output/DIAG-S1 --task maxpower
  python agentutil/diag_report.py --run output/DIAG-E2 --task scalar --lam_phys 20
结果追加到 `output/diag/summary.csv`。
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

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--task', default='maxpower', choices=['maxpower', 'scalar'])
ap.add_argument('--refs', default=os.path.join('output', 'diag'))
ap.add_argument('--lam_phys', type=float, default=20.0)
ap.add_argument('--tail', type=float, default=0.2, help='只用最后这一比例的数据打分')
ap.add_argument('--tag', default=None)
a = ap.parse_args()

cols = ["step", "u", "pr", "turbine_profit", "P_F", "c1", "applied_action_H", "original_action_H", "lam1"]
df = pd.read_csv(os.path.join(a.run, 'df_hsrl.csv'), usecols=cols)
A = np.vstack([np.fromstring(str(x).strip().lstrip("[").rstrip("]"), sep=" ") for x in df.applied_action_H])
df = df.drop(columns=["applied_action_H", "original_action_H"]).reset_index(drop=True)
df["PG"] = df.turbine_profit * 60.0 / df.pr
df["a"] = A[:, :5].mean(1)

n0 = int(len(df) * (1 - a.tail))
D = df.iloc[n0:].reset_index(drop=True)
Aa = A[n0:]

refs_u = pd.read_csv(os.path.join(a.refs, 'refs_u.csv'))
gr = refs_u.u.to_numpy(float)
pmax = np.interp(D.u.to_numpy(float), gr, refs_u.P_max.to_numpy(float))
eta = D.PG.to_numpy(float) / np.maximum(pmax, 1e-9)
astar = np.column_stack([np.interp(D.u.to_numpy(float), gr, refs_u['a%d' % j].to_numpy(float)) for j in range(5)])
pat = np.abs(Aa[:, :5] - astar).mean()

# R² 增量（动作是否提供信息）
def r2(y, X):
    X1 = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ beta
    return 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()


y = D.PG.to_numpy(float)
r_u = r2(y, [D.u.to_numpy(float)])
r_ua = r2(y, [D.u.to_numpy(float), D.a.to_numpy(float)])
viol = 100.0 * (D.P_F > 3.0).mean()
std_a = float(np.diff(D.a.to_numpy(float)).std())
out = dict(task=a.task, run=a.tag or os.path.basename(a.run.rstrip('/\\')),
           eta=float(eta.mean()), eta_p5=float(np.percentile(eta, 5)),
           eta_min=float(eta.min()), pattern_err=float(pat), viol_pct=viol,
           a_mean=float(D.a.mean()), a_std_step=std_a, u_mean=float(D.u.mean()),
           PG_mean=float(D.PG.mean()), r2_u=float(r_u), r2_dua=r_ua - r_u,
           lam=float(D.lam1.median()), n=int(len(D)))

if a.task == 'scalar':
    head = pd.read_csv(os.path.join(a.refs, 'refs_e2_head.csv')).iloc[0]
    J_pol = float((D.pr * D.PG / 60.0).sum() - a.lam_phys * D.c1.sum())
    J_day = J_pol * 1440.0 / len(D)                       # 折算到"一天"
    out.update(J_policy=J_day, J_star=float(head.J_star),
               regret_pct=100.0 * (float(head.J_star) - J_day) / abs(float(head.J_star)),
               s_mean=float(D.a.mean() / 0.33))
    s_star = pd.read_csv(os.path.join(a.refs, 'refs_e2.csv')).s_star.to_numpy(float)
    out['s_star_mean'] = float(s_star.mean())

print('=' * 84)
for k, v in out.items():
    print('  %-14s %s' % (k, ('%.4f' % v) if isinstance(v, float) else v))
print('=' * 84)

os.makedirs(os.path.join('output', 'diag'), exist_ok=True)
f = os.path.join('output', 'diag', 'summary.csv')
pd.DataFrame([out]).to_csv(f, mode='a', header=not os.path.exists(f), index=False)
print('追加到 %s' % f)
