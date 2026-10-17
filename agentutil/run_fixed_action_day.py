#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""固定轴向因子动作 × FLORIS × 指定风速日 × **加噪池** —— 跑一整天（1440 步），出指标与图。

用户口径（2026-10）：
    action = [0.2, 0.2, 0.2, 0.2, 0.3]（5 台机轴向因子）+ g = 0（BESS 不动）
    风速池 = **FFC_speed_noise_1.csv**（noise 状态）
    日 = 训练固定序列的第 0 天（`FIXED_DAY_SEQUENCE[0]` = 122；`--day 1` 可换其它天）

为什么 g=0：H 的第 6 维是块总放电量；g=0 ⇒ 摊还执行器的零成本动作 pb=0 ⇒ **BESS 不参与**，
于是 P_F = |ΔP_tur|，测到的就是"这个固定动作本身造出多少波动"。

用法:
    python agentutil/run_fixed_action_day.py                       # 默认 [0.2×4, 0.3]，day=122，noise 池
    python agentutil/run_fixed_action_day.py --action 0.2 0.2 0.2 0.2 0.2
    python agentutil/run_fixed_action_day.py --day 1 --csv FFC_speed_1.csv
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
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.environ.get('TEMP', '/tmp'), 'mplcache'))

import matplotlib                      # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt        # noqa: E402

import aifflor.aifflor_turbine as AT               # noqa: E402
from aifflor.aifflor_env import AifflorPhysics, WindBatteryAIF   # noqa: E402

DAY = 1440
P_F_LIMIT, K_V = 3.0, 20.0             # run87 时代的 c1 口径（K_V=20）；当前代码是 100

ap = argparse.ArgumentParser()
ap.add_argument('--action', type=float, nargs=5, default=[0.2, 0.2, 0.2, 0.2, 0.3],
                help='5 台机的轴向因子（默认 0.2 0.2 0.2 0.2 0.3）')
ap.add_argument('--g', type=float, default=0.0, help='H 第 6 维：块总放电量 [MW·min]（默认 0）')
ap.add_argument('--day', type=int, default=122, help='风速日的列号（env 的 iloc[:, j]；默认 122）')
ap.add_argument('--csv', default='FFC_speed_noise_1.csv', help='风速池文件（默认加噪版）')
ap.add_argument('--steps', type=int, default=DAY)
ap.add_argument('--tag', default=None)
a = ap.parse_args()

ACT = np.asarray(a.action, dtype=float)
assert ACT.size == 5, ACT

physics = AifflorPhysics(nt=5, calibrated=False, power_scale=AT.LEVEL_COEF)
env = WindBatteryAIF(max_E=0.5, num_turbine=5, upload=False, engine=None, physics=physics)
# 明确指定风速池（env 默认已是无噪声版；这里按用户口径用加噪版）
env.wind_noise = pd.read_csv(a.csv)
seq = pd.Series(env.wind_noise.iloc[:, a.day].to_numpy(float))
env.episode_count = 0
obs = env.reset()
# reset() 会按 FIXED_DAY_SEQUENCE[0] 取天；若 --day 不是序列第 0 天，则强制覆盖
env.j = a.day
env.wind_day = seq
env.count_wind = 0
env.u = float(seq.iloc[0])
env.P_G_prev = float(env.PowerOutput(env.u, env.nominal()))
print('池=%s | 日 j=%d | 动作=%s | g=%.2f | 起始 u=%.3f'
      % (a.csv, env.j, ACT.tolist(), a.g, env.u))

rows = []
pb = 0.0                       # g=0 ⇒ 摊还执行器的零成本动作 pb=0（BESS 不动）
for t in range(a.steps):
    u_now = float(env.u)
    pw_tru = float(env.PowerOutput(u_now, ACT))
    obs2, = [None]
    out = env.step(np.append(ACT, a.g), np.array([pb]), pw_tru)
    pw_tru2, obs2_H, app_L, pen_L, done, done_L, info = out
    rows.append(dict(t=t, u=u_now, ptur=pw_tru, pr=float(obs2_H[0]), SOC=float(obs2_H[2]),
                     P_G=float(obs2_H[3]), P_F=float(info['P_F']), pb=float(app_L[0]),
                     soc_next=float(obs2_H[2])))
    pb = float(app_L[0])          # 下一拍保持"不动"：命令 0，实际 0
    pb = 0.0

D = pd.DataFrame(rows)
c1 = D.P_F + K_V * np.maximum(0.0, D.P_F - P_F_LIMIT)
D['c1'] = c1
blk = c1.to_numpy().reshape(-1, 5).sum(axis=1) if len(c1) % 5 == 0 else None
viol = (D.P_F > P_F_LIMIT + 1e-4).to_numpy()

print('\n' + '=' * 92)
print('结果：%d 步（1 天）| 动作 %s | 日 %d | %s' % (len(D), ACT.tolist(), env.j, a.csv))
print('=' * 92)
print('  u          : 均值 %.2f | min %.2f | max %.2f | σ(Δu) %.3f m/s'
      % (D.u.mean(), D.u.min(), D.u.max(), np.diff(D.u).std()))
print('  P_tur      : 均值 %.2f MW | min %.2f | max %.2f' % (D.ptur.mean(), D.ptur.min(), D.ptur.max()))
print('  P_G        : 均值 %.2f MW | min %.2f | max %.2f' % (D.P_G.mean(), D.P_G.min(), D.P_G.max()))
print('  P_F        : 均值 %.3f | p50 %.3f | p95 %.3f | p99 %.3f | max %.3f MW'
      % (D.P_F.mean(), *np.percentile(D.P_F, [50, 95, 99]), D.P_F.max()))
print('  违约 P_F>3 : %d 步 = %.2f%%（%d 次/天）' % (viol.sum(), 100 * viol.mean(), viol.sum()))
print('  c1 (K_V=%.0f): 和 %.1f MW·step | 块和 均值 %.2f | 最大 %.2f'
      % (K_V, c1.sum(), blk.mean() if blk is not None else np.nan,
         blk.max() if blk is not None else np.nan))
print('  收入       : %.2f 元/天（=Σ pr·P_G·dt；BESS 不动）' % float((D.pr * D.P_G / 60).sum()))
print('  SOC        : %.3f → %.3f（应为常数：BESS 不动）' % (D.SOC.iloc[0], D.soc_next.iloc[-1]))

tag = a.tag or ('day%d_act%s' % (env.j, '_'.join('%.1f' % v for v in ACT)))
os.makedirs('output/fixed_action', exist_ok=True)
csv_out = 'output/fixed_action/%s.csv' % tag
D.to_csv(csv_out, index=False)

# ---- 图 ----
fig, ax = plt.subplots(4, 1, figsize=(11, 10), dpi=140, sharex=True)
ax[0].plot(D.t, D.u, lw=1.0, color='#1f77b4')
ax[0].set_ylabel('$u$ [m/s]')
ax[0].set_title('%s | action=%s | g=%.1f | day j=%d | pool=%s'
                % (tag, ACT.tolist(), a.g, env.j, a.csv), fontsize=10)
ax[1].plot(D.t, D.ptur, lw=1.0, color='#2ca02c', label='P_tur')
ax[1].plot(D.t, D.P_G, lw=1.0, color='#ff7f0e', alpha=0.9, label='P_G (=P_tur, BESS idle)')
ax[1].set_ylabel('[MW]')
ax[1].legend(fontsize=8)
ax[2].plot(D.t, D.P_F, lw=1.0, color='#d62728')
ax[2].axhline(3.0, color='k', ls='--', lw=1.0, label='3 MW limit')
ax[2].set_ylabel('$P_F$ [MW]')
ax[2].legend(fontsize=8)
ax[2].set_title('P_F: mean %.2f | p95 %.2f | max %.2f | violations %d (%.2f%%)'
                % (D.P_F.mean(), np.percentile(D.P_F, 95), D.P_F.max(),
                   int(viol.sum()), 100 * viol.mean()), fontsize=9)
ax[3].plot(D.t, D.c1, lw=1.0, color='#9467bd')
ax[3].set_ylabel('$c_1$ [MW·step]')
ax[3].set_xlabel('minute of day')
ax[3].set_title('c1 (K_V=%.0f): sum %.1f | block-sum mean %.2f max %.2f'
                % (K_V, c1.sum(), blk.mean() if blk is not None else np.nan,
                   blk.max() if blk is not None else np.nan), fontsize=9)
for _a in ax:
    _a.grid(alpha=0.3, lw=0.5)
fig.tight_layout()
png_out = 'output/fixed_action/%s.png' % tag
fig.savefig(png_out)
plt.close(fig)
print('\n已落盘: %s\n        %s' % (csv_out, png_out))
