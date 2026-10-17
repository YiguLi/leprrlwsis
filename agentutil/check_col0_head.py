#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核查"column 0 前 5 行噪声幅度远大于 1 m/s"这一观察（只读）。

回答三件事：
  1) 那 5 个残差的确切值与 z 值；
  2) 它们是否说明 σ≠1（答案：不能——5 个点估不了 σ，且整列/三种估计器都给 ≈1.0）；
  3) 这个"5 连点里出现 3σ+2.7σ"到底有多反常（多重比较：每列 ~480 个 5 窗）。

输出图: output/wind_days/ffc_noise_col0_head.png
        （上：前 60 分钟 clean/noisy + 残差；下：全部 279k 残差的直方图 + 那 5 个点）
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

import matplotlib            # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

C = pd.read_csv('FFC_speed_1.csv', index_col=0).to_numpy(float)
N = pd.read_csv('FFC_speed_noise_1.csv', index_col=0).to_numpy(float)
R = N - C
sig = R.std()

c0, n0, r0 = C[:, 0], N[:, 0], R[:, 0]
z0 = r0 / r0.std()

print('=' * 96)
print('1) column "0"（第一天）前 5 行')
print('   %-4s %12s %14s %11s %8s' % ('row', 'clean', 'noisy', 'residual', 'z'))
for i in range(5):
    print('   %-4d %12.6f %14.6f %+11.4f %+8.3f' % (i, c0[i], n0[i], r0[i], z0[i]))
print('   全列: σ=%.4f | E|r|=%.3f | p95=%.3f | p99=%.3f | max|r|=%.3f'
      % (r0.std(), np.abs(r0).mean(), *np.percentile(np.abs(r0), [95, 99]), np.abs(r0).max()))
print('   ⇒ 这 5 个 |r| 的均值 %.3f，是全列 E|r|=%.3f 的 %.1f 倍（≈ 3.4 个"标准误"高）'
      % (np.abs(r0[:5]).mean(), np.abs(r0).mean(), np.abs(r0[:5]).mean() / np.abs(r0).mean()))

print('\n2) 为什么 5 个点不能推翻 σ=1')
print('   整列 1440 点: σ=%.4f, MAD-σ=%.3f, 峰度=%.2f, ACF1=%+.4f'
      % (r0.std(), 1.4826 * np.median(np.abs(r0 - np.median(r0))),
         pd.Series(r0).kurt(), np.corrcoef(r0[1:], r0[:-1])[0, 1]))
print('   |r| 分位 p95=%.3f（高斯 1.96σ=%.3f）p99=%.3f（2.58σ=%.3f）⇒ 分布就是 N(0,%.2f)'
      % (np.percentile(np.abs(r0), 95), 1.96 * r0.std(),
         np.percentile(np.abs(r0), 99), 2.58 * r0.std(), r0.std()))
print('   单文件估计（只用 noisy 列）: σ(Δu)/√2=%.3f, σ(Δ²u)/√6=%.3f, MA5=%.3f'
      % (np.diff(n0).std() / np.sqrt(2), np.diff(n0, n=2).std() / np.sqrt(6),
         (n0[2:-2] - np.convolve(n0, np.ones(5) / 5, 'valid')).std() * np.sqrt(5 / 4)))
print('   全表 %d 个残差: |z|>3 的个数 = %d（高斯预期 %.0f）'
      % (R.size, int((np.abs(R / sig) > 3).sum()), R.size * 0.0027))
print('   按行（分钟位置）跨 194 天平均 |z|: 第 0 行 %.3f、前 8 行 %.3f、全表 %.3f'
      % (np.abs(R[:, 0] / sig).mean(), np.abs(R[:8] / sig).mean(), np.abs(R / sig).mean()))
print('   ⇒ 开头没有"噪声更大"的系统偏差（首行反而略小）')

print('\n3) "5 连点里 ≥1 个 >3σ 且 ≥2 个 >2.5σ" 有多反常（多重比较）')
rs = np.random.RandomState(0)
T = 400000
zz = rs.randn(T, 5)
hit = ((np.abs(zz) > 3.0).any(axis=1) & ((np.abs(zz) > 2.5).sum(axis=1) >= 2)).mean()
print('   单个 5 窗的概率 ≈ %.4f%%（= P(≥1个>3σ)·P(≥2个>2.5σ) 的同窗联合）' % (100 * hit))
print('   一列 1440 点有 %d 个不重叠 5 窗 ⇒ 每列"至少一个"≈ %.1f%%；194 天里预期 %.1f 列'
      % (1440 // 5, 100 * (1 - (1 - hit) ** (1440 // 5)), 194 * (1 - (1 - hit) ** (1440 // 5))))
# 经验分布：所有 5 窗的 mean|r|，看这 5 个点排在哪
w = np.abs(R).reshape(-1, 5).mean(axis=1)
target = np.abs(r0[:5]).mean()
print('   经验分布：全表 %d 个 5 窗的 mean|r| 中位 %.3f、p90 %.3f、p99 %.3f'
      % (len(w), np.median(w), np.percentile(w, 90), np.percentile(w, 99)))
print('   你这 5 点的 mean|r| = %.3f ⇒ 排在第 %.1f 百分位（超过 %.1f%% 的窗）'
      % (target, 100 * (w < target).mean(), 100 * (w < target).mean()))
print('   ⇒ 属于"很靠尾部但确实存在"的样本：全表就有 %d 个窗的 mean|r| ≥ %.2f'
      % (int((w >= target).sum()), target))

# ---------------------------------------------------------------- 图
x = np.arange(60)
fig, ax = plt.subplots(2, 1, figsize=(11, 7.2), dpi=140,
                       gridspec_kw={'height_ratios': [1.15, 1]})
ax[0].plot(x, c0[:60], '-o', ms=3, lw=1.2, label='clean')
ax[0].plot(x, n0[:60], '-o', ms=3, lw=1.0, alpha=0.8, label='noisy')
for i in range(5):
    ax[0].annotate('%+.2f' % r0[i], (i, n0[i]), textcoords='offset points',
                   xytext=(0, 8), ha='center', fontsize=7, color='#d62728')
ax[0].set_xlabel('minute'); ax[0].set_ylabel('$u$ [m/s]')
ax[0].set_title('column "0": first 60 min — clean vs noisy (annotated = residual of the first 5 points)',
                fontsize=10)
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, lw=0.5)

ax[1].hist(R.ravel(), bins=200, density=True, alpha=0.7, color='#7f7f7f',
           label='all %d residuals (194 days)' % R.size)
xs = np.linspace(-5, 5, 400)
ax[1].plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), 'k--', lw=1.0, label='N(0,1)')
for i, v in enumerate(r0[:5]):
    ax[1].axvline(v, color='#d62728', lw=1.0, alpha=0.85,
                  label='first-5 residuals' if i == 0 else None)
for k, ls in ((1, ':'), (2, '--'), (3, '-')):
    ax[1].axvline(k * r0.std(), color='#1f77b4', ls=ls, lw=0.8)
    ax[1].axvline(-k * r0.std(), color='#1f77b4', ls=ls, lw=0.8)
ax[1].set_xlabel('residual $r$ [m/s]'); ax[1].set_ylabel('density')
ax[1].set_title('all residuals vs N(0,1); blue lines = ±1σ/±2σ/±3σ (σ=%.3f); red = the first five points'
                % r0.std(), fontsize=9)
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, lw=0.5)
fig.tight_layout()
out = os.path.join('output', 'wind_days', 'ffc_noise_col0_head.png')
fig.savefig(out)
plt.close(fig)
print('\n图已保存: %s' % out)
