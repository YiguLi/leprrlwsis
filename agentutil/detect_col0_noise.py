#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单独检测两份 FFC 文件 **column "0"**（index_col=0 后的第一"天"列）的噪声水平。

⚠ 列名歧义，两个都算：
  * 原始 CSV 的**第 0 列**是行号 0..1439（两份文件完全相同，无噪声可谈）；
  * `index_col=0` 之后列名 `"0"` 的那一列 = **第一天**（= env 里的 `iloc[:, 1]`，j=1）。
本脚本检测后者，并给三种**互相独立**的噪声估计（不需要另一份文件）：

    σ̂₁ = std(Δu)/√2                   一阶差分（白噪声理论）
    σ̂₂ = std(Δ²u)/√6                  二阶差分
    σ̂₃ = std(u − MA₅(u))·√(5/4)       5 点滑动平均残差（白噪声理论）

若三者一致 ⇒ 噪声是白的；同时给出与另一份文件直接相减的"真值"残差做对照。

用法: python agentutil/detect_col0_noise.py
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

COL = '0'            # index_col=0 后的列名（= 第一天）


def lag1_acf(x):
    x = np.asarray(x, float)
    return float(np.corrcoef(x[1:], x[:-1])[0, 1])


def mad_sigma(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def noise_estimates(u):
    u = np.asarray(u, float)
    d1 = np.diff(u)
    d2 = np.diff(u, n=2)
    ma5 = np.convolve(u, np.ones(5) / 5, mode='valid')
    resid5 = u[2:len(u) - 2] - ma5
    return dict(s1=d1.std() / np.sqrt(2),
                s2=d2.std() / np.sqrt(6),
                s3=resid5.std() * np.sqrt(5 / 4))


clean = pd.read_csv('FFC_speed_1.csv', index_col=0)
noisy = pd.read_csv('FFC_speed_noise_1.csv', index_col=0)
raw_c = pd.read_csv('FFC_speed_1.csv')
raw_n = pd.read_csv('FFC_speed_noise_1.csv')

print('=' * 96)
print('0) 先排除歧义：原始 CSV 的**第 0 列**（行号）是否两文件相同？')
print('   raw col0 head:', raw_c.iloc[:5, 0].tolist(), '| identical:', bool((raw_c.iloc[:, 0].values == raw_n.iloc[:, 0].values).all()))
print('   ⇒ 行号列没有"噪声"概念；下面检测 index_col=0 后列名 "0" 的**第一天**（env 的 j=1）')

c = clean[COL].to_numpy(float)
n = noisy[COL].to_numpy(float)
r = n - c
print('\n' + '=' * 96)
print('1) 列 "0"（第一天）基本统计')
print('   %-8s %8s %8s %8s %8s %8s' % ('序列', 'mean', 'std', 'min', 'max', 'ACF1'))
for nm, v in (('clean', c), ('noisy', n), ('residual', r)):
    print('   %-8s %8.4f %8.4f %8.3f %8.3f %8.4f' % (nm, v.mean(), v.std(), v.min(), v.max(), lag1_acf(v)))

print('\n' + '=' * 96)
print('2) 三种**单文件**噪声估计（m/s）—— 不需要另一份文件')
print('   %-8s %10s %10s %10s %10s' % ('序列', 'σ̂₁=σ(Δu)/√2', 'σ̂₂=σ(Δ²u)/√6', 'σ̂₃=MA5残差', '三者极差'))
for nm, v in (('clean', c), ('noisy', n)):
    e = noise_estimates(v)
    print('   %-8s %13.4f %13.4f %13.4f %10.4f'
          % (nm, e['s1'], e['s2'], e['s3'], max(e.values()) - min(e.values())))

print('\n' + '=' * 96)
print('3) "真值"残差（noisy − clean）')
print('   σ = %.4f | RMSE = %.4f | MAE = %.4f | p95 = %.3f | p99 = %.3f | max = %.3f'
      % (r.std(), np.sqrt(np.mean(r ** 2)), np.mean(np.abs(r)),
         np.percentile(np.abs(r), 95), np.percentile(np.abs(r), 99), np.abs(r).max()))
print('   稳健 σ（MAD）= %.4f | 偏度 %.4f | 峰度 %.4f | ACF1 %.4f'
      % (mad_sigma(r), float(pd.Series(r).skew()), float(pd.Series(r).kurt()), lag1_acf(r)))
print('   与"白噪声"对照: σ(Δr)/√2 = %.4f（应 ≈ σ(r) = %.4f）'
      % (np.diff(r).std() / np.sqrt(2), r.std()))

print('\n' + '=' * 96)
print('4) 相对幅度与对 Δu 的影响（列 "0"）')
print('   RMSE/mean(u) = %.2f%% | RMSE/std(u_clean) = %.2f%%'
      % (100 * np.sqrt(np.mean(r ** 2)) / c.mean(), 100 * r.std() / c.std()))
print('   σ(Δu): clean %.4f → noisy %.4f （放大 %.2f×）| p99(|Δu|): %.3f → %.3f'
      % (np.diff(c).std(), np.diff(n).std(), np.diff(n).std() / np.diff(c).std(),
         np.percentile(np.abs(np.diff(c)), 99), np.percentile(np.abs(np.diff(n)), 99)))
print('   "相对粗糙度" σ(Δu)/σ(u): clean %.4f | noisy %.4f（纯白噪声理论值 √2=1.414/…=0.707）'
      % (np.diff(c).std() / c.std(), np.diff(n).std() / n.std()))

print('\n' + '=' * 96)
print('5) 对照：全部 194 天（**逐天算**再取中位；不能把 194 天首尾相接后一次算——'
      '天与天之间的人为跳变会污染差分）')
for nm, df in (('clean', clean), ('noisy', noisy)):
    per = [noise_estimates(df.iloc[:, j].to_numpy(float)) for j in range(df.shape[1])]
    m1 = np.median([e['s1'] for e in per])
    m2 = np.median([e['s2'] for e in per])
    m3 = np.median([e['s3'] for e in per])
    print('   %-6s 逐天中位: σ̂₁=%.4f | σ̂₂=%.4f | σ̂₃=%.4f  (min σ̂₁ %.4f, max %.4f)'
          % (nm, m1, m2, m3, min(e['s1'] for e in per), max(e['s1'] for e in per)))
print('   直接残差（逐点 noisy−clean）σ = %.4f'
      % (noisy.to_numpy(float) - clean.to_numpy(float)).std())
