#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""事实核查：`FFC_speed_1.csv`（无噪声） vs `FFC_speed_noise_1.csv`（有噪声）——噪声幅度指标。

两份文件形状相同（1440 行 × 194 天列）；**同列 = 同一天**（逐列相关 0.57~0.97）。
本脚本给出：
  A. 残差 r = noisy − clean 的幅度（bias / σ / RMSE / MAE / 分位 / 相对幅度）；
  B. 噪声结构（是否白噪声、是否跨天独立、是否与风速水平相关、正态性）；
  C. **对 Δu 的放大**（这是 c1=|ΔP_G| 的直接驱动量）；
  D. 用 FLORIS 把 Δu 的放大换成"功率跳变"的放大，并对照 run87 的实际违约率；
  E. 图与汇总表（output/wind_days/）。

用法: python agentutil/check_ffc_noise.py
"""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402

DAY = 1440
RUN_DAYS = [122, 32, 162, 157, 104]        # env 的 iloc 位置（run87 前 5 个 episode）

clean = pd.read_csv('FFC_speed_1.csv', index_col=0)
noisy = pd.read_csv('FFC_speed_noise_1.csv', index_col=0)
assert clean.shape == noisy.shape, (clean.shape, noisy.shape)
C = clean.to_numpy(float)
N = noisy.to_numpy(float)
R = N - C
nday = C.shape[1]
print('两份文件: %d 分钟 × %d 天；NaN %d/%d' % (C.shape[0], nday, int(np.isnan(C).sum()), int(np.isnan(N).sum())))
print('全局: 风速 clean %.3f±%.3f | noisy %.3f±%.3f | 差 %.3f±%.3f m/s'
      % (C.mean(), C.std(), N.mean(), N.std(), R.mean(), R.std()))


def stats(v):
    v = np.asarray(v, float)
    return dict(mean=v.mean(), std=v.std(), rmse=np.sqrt(np.mean(v ** 2)),
                mae=np.mean(np.abs(v)), p50=np.percentile(np.abs(v), 50),
                p95=np.percentile(np.abs(v), 95), p99=np.percentile(np.abs(v), 99),
                max=np.abs(v).max())


print('\n' + '=' * 100)
print('A. 噪声幅度（残差 r = noisy − clean，单位 m/s）')
print('=' * 100)
print('  %-10s %8s %8s %8s %8s %8s %8s %8s %8s' %
      ('日(env j)', 'Bias', 'σ', 'RMSE', 'MAE', '|r|p95', '|r|p99', 'max|r|', 'RMSE/u'))
rows = []
for j in RUN_DAYS:
    col = j - 1                     # index_col 后列名 = "0".."193"
    r = R[:, col]
    s = stats(r)
    ubar = C[:, col].mean()
    rows.append(dict(day=j, **s, rmse_rel=s['rmse'] / ubar))
    print('  %-10d %8.4f %8.4f %8.4f %8.4f %8.3f %8.3f %8.3f %7.2f%%'
          % (j, s['mean'], s['std'], s['rmse'], s['mae'], s['p95'], s['p99'], s['max'],
             100 * s['rmse'] / ubar))
allr = stats(R.ravel())
print('  %-10s %8.4f %8.4f %8.4f %8.4f %8.3f %8.3f %8.3f'
      % ('全部194天', allr['mean'], allr['std'], allr['rmse'], allr['mae'],
         allr['p95'], allr['p99'], allr['max']))
print('  相对幅度: RMSE/mean(u) = %.2f%%   σ/mean(u) = %.2f%%   与 σ 比: E|r|/σ = %.3f（正态理论 0.798）'
      % (100 * allr['rmse'] / C.mean(), 100 * allr['std'] / C.mean(), allr['mae'] / allr['std']))

print('\n' + '=' * 100)
print('B. 结构检验')
print('=' * 100)
acf = [np.corrcoef(R[1:, :].ravel(), R[:-1, :].ravel())[0, 1],
       np.corrcoef(R[2:, :].ravel(), R[:-2, :].ravel())[0, 1],
       np.corrcoef(R[3:, :].ravel(), R[:-3, :].ravel())[0, 1]]
print('  残差自相关 ACF(lag1/2/3) = %.4f / %.4f / %.4f   （白噪声 ≈ 0）' % tuple(acf))
cross = [np.corrcoef(R[:, i], R[:, i + 1])[0, 1] for i in range(0, nday - 1, 7)]
print('  跨天残差相关（抽样 %d 对）: median %.4f, max|·| %.4f  （≈0 ⇒ 每天独立抽样）'
      % (len(cross), np.median(np.abs(cross)), np.max(np.abs(cross))))
sk = float(pd.Series(R.ravel()).skew())
ku = float(pd.Series(R.ravel()).kurt())
print('  偏度 %.4f | 峰度 %.4f（正态 0/0；均匀分布峰度 ≈ −1.2）' % (sk, ku))
# 异方差：按 clean 风速分箱看 σ(r)
bins = [0, 5, 8, 10, 12, 14, 30]
idx = np.digitize(C.ravel(), bins) - 1
print('  σ 与风速水平的关系（按 clean u 分箱）:')
for b in range(len(bins) - 1):
    m = idx == b
    if m.sum() < 100:
        continue
    print('    u∈[%2d,%2d): n=%7d | σ(r)=%.4f | mean(u)=%.2f | corr(u,r)=%+.3f'
          % (bins[b], bins[b + 1], m.sum(), R.ravel()[m].std(), C.ravel()[m].mean(),
             np.corrcoef(C.ravel()[m], R.ravel()[m])[0, 1]))
print('  噪声后最小值 = %.3f m/s（clean 最小 %.3f）' % (N.min(), C.min()))

print('\n' + '=' * 100)
print('C. 对 Δu（逐分钟变化）的放大 —— 这是 c1 = |ΔP_G| 的直接驱动量')
print('=' * 100)
dC = np.diff(C, axis=0)
dN = np.diff(N, axis=0)
for nm, d in (('clean', dC), ('noisy', dN)):
    ad = np.abs(d)
    print('  %-6s Δu: σ=%.4f | p50=%.4f p95=%.3f p99=%.3f max=%.3f | P(|Δu|>3)=%.4f%%'
          % (nm, d.std(), np.percentile(ad, 50), np.percentile(ad, 95),
             np.percentile(ad, 99), ad.max(), 100 * (ad > 3).mean()))
print('  ⇒ σ(Δu) 放大 %.2f×；p99(|Δu|) 放大 %.2f×'
      % (dN.std() / dC.std(), np.percentile(np.abs(dN), 99) / np.percentile(np.abs(dC), 99)))
print('  （若噪声独立同分布: σ(Δr) 应为 √2·σ(r) = %.4f；实测组合式: %.4f'
      % (np.sqrt(2) * R.std(), np.sqrt(max(dN.std() ** 2 - dC.std() ** 2, 0))))
print('  "分钟级相对粗糙度" σ(Δu)/σ(u): clean %.4f vs noisy %.4f（真实 1-min 风通常 0.1~0.3）'
      % (dC.std() / C.std(), dN.std() / N.std()))
print('  多尺度粗糙度（u(t+k)−u(t) 的 σ，单位 m/s）:')
for k in (1, 5, 15, 60):
    sc = float((C[k:] - C[:-k]).std())
    sn = float((N[k:] - N[:-k]).std())
    print('    k=%2d min: clean %.4f | noisy %.4f | noisy/clean %6.2f×'
          % (k, sc, sn, sn / max(sc, 1e-12)))

print('\n' + '=' * 100)
print('D. 换成"功率跳变"（FLORIS，均匀 a=0.20 的最大出力支）')
print('=' * 100)
import aifflor.aifflor_turbine as AT  # noqa: E402

model, _ = AT.build_model(verbose=False)
UG = np.arange(4.0, 20.01, 0.5)
A = np.array([0.20])
farm, _ = AT.eval_batch(model, UG, np.tile(A, (UG.size, 1)) * np.ones((1, AT.NT)))
P = farm
dPdu = np.gradient(P, UG)
print('  %6s %10s %12s' % ('u', 'P(u) MW', 'dP/du MW/(m/s)'))
for k in range(0, len(UG), 4):
    print('  %6.1f %10.2f %12.3f' % (UG[k], P[k], dPdu[k]))
# 用 run87 的时间序列直接估：noise 单独造成的 ΔP 分布
u_all = N.ravel()
slope = np.interp(u_all, UG, dPdu)
dR = np.zeros_like(R)
dR[1:] = np.diff(R, axis=0)
dP_noise = slope * dR.ravel()
print('\n  噪声单独造成的 ΔP（在 a=0.20 支上）: σ=%.3f MW | p95=%.3f | p99=%.3f | P(|ΔP|>3)=%.2f%%'
      % (dP_noise.std(), np.percentile(np.abs(dP_noise), 95),
         np.percentile(np.abs(dP_noise), 99), 100 * (np.abs(dP_noise) > 3).mean()))
for lo, hi in ((9, 11), (11, 12.5), (12.5, 14), (14, 30)):
    m = (u_all >= lo) & (u_all < hi)
    if m.sum() < 200:
        continue
    print('    u∈[%4.1f,%4.1f): n=%6d | σ(ΔP_noise)=%.2f MW | P(|ΔP|>3)=%.2f%%'
          % (lo, hi, m.sum(), dP_noise[m].std(), 100 * (np.abs(dP_noise[m]) > 3).mean()))
print('  对照 run87 实际违约率: u∈[11,12.5) 7.45%% | [12.5,14) 13.28%% | >14 24.38%%（全程 8.23%%）')

print('\n' + '=' * 100)
print('E. 落盘')
print('=' * 100)
df = pd.DataFrame(rows)
df.to_csv('output/wind_days/ffc_noise_metrics.csv', index=False)
print('  output/wind_days/ffc_noise_metrics.csv')

# ---- 图：day 162（env j=162 → index_col 列 "161"）：clean vs noisy + 残差 + Δu 分布 ----
col = 162 - 1
x = np.arange(DAY)
fig, ax = plt.subplots(3, 1, figsize=(11, 8), dpi=140, sharex=False)
ax[0].plot(x, C[:, col], lw=1.0, label='clean (FFC_speed_1)')
ax[0].plot(x, N[:, col], lw=0.8, alpha=0.75, label='noisy (FFC_speed_noise_1)')
ax[0].set_ylabel('$u$ [m/s]')
ax[0].set_title('FFC day col %d | clean vs noisy | residual σ = %.3f m/s'
                % (col, R[:, col].std()), fontsize=10)
ax[0].legend(fontsize=8, loc='upper left')
ax[0].grid(alpha=0.3, lw=0.5)
ax[1].plot(x, R[:, col], lw=0.8, color='#d62728')
ax[1].axhline(0, color='k', lw=0.6)
ax[1].set_ylabel('noise $r$ [m/s]')
ax[1].grid(alpha=0.3, lw=0.5)
ax[2].hist(np.diff(C[:, col]), bins=80, alpha=0.6, density=True, label='clean Δu')
ax[2].hist(np.diff(N[:, col]), bins=80, alpha=0.6, density=True, label='noisy Δu')
ax[2].set_xlabel('Δu over 1 min [m/s]')
ax[2].set_ylabel('density')
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3, lw=0.5)
fig.tight_layout()
p = 'output/wind_days/ffc_noise_day162.png'
fig.savefig(p)
plt.close(fig)
print('  ', p)
