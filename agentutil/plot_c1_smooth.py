#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""画 c1 的硬折点 vs sigmoid 平滑版（β=100, ε=0.05）以及两者的导数。

用法: python agentutil/plot_c1_smooth.py [--beta 100] [--eps 0.05] [--out output/diag/c1_smooth.png]
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

for _f in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans'):
    try:
        plt.rcParams['font.sans-serif'] = [_f]
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False


def make(beta, eps):
    ln2 = np.log(2.0)

    def pf_soft(dP):
        return np.sqrt(np.asarray(dP, float) ** 2 + eps ** 2) - eps

    def softplus(x):
        x = np.asarray(x, float)
        return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(beta * x))) / beta

    def c1_hard(pf):
        pf = np.asarray(pf, float)
        return pf + 20.0 * np.maximum(0.0, pf - 3.0)

    def c1_soft(pf):
        pf = np.asarray(pf, float)
        return pf + 20.0 * (softplus(pf - 3.0) - ln2 / beta)

    def dc1_hard(pf):
        pf = np.asarray(pf, float)
        return 1.0 + 20.0 * (pf >= 3.0)

    def dc1_soft(pf):
        pf = np.asarray(pf, float)
        return 1.0 + 20.0 / (1.0 + np.exp(-beta * (pf - 3.0)))

    def dPF(dP):
        dP = np.asarray(dP, float)
        return dP / np.sqrt(dP ** 2 + eps ** 2)

    return pf_soft, c1_hard, c1_soft, dc1_hard, dc1_soft, dPF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--beta', type=float, default=100.0)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--out', default=os.path.join('output', 'diag', 'c1_smooth.png'))
    a = ap.parse_args()
    pf_soft, c1h, c1s, dh, ds, dPF = make(a.beta, a.eps)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle('c1：硬折点 vs sigmoid 平滑（β=%g, ε=%g）' % (a.beta, a.eps), fontsize=14)

    # (a) 全范围
    pf = np.linspace(-6, 12, 4000)
    ax[0, 0].plot(pf, np.where(pf >= 0, c1h(pf), c1h(-pf)), lw=1.8, label='硬 $P_F+20(P_F-3)^+$')
    ax[0, 0].plot(pf, c1s(np.abs(pf)), '--', lw=1.8, label='软 $\\tilde P_F+20\\sigma_\\beta$')
    ax[0, 0].axvline(3.0, color='gray', ls=':', lw=0.9)
    ax[0, 0].set_xlabel('$\\Delta P$ / $P_F$  (MW)')
    ax[0, 0].set_ylabel('$c_1$  (MW·step)')
    ax[0, 0].set_title('(a) 全范围：两者几乎重合（偏差 ≤ %.3f）' % (a.eps + 20 * np.log(2) / a.beta))
    ax[0, 0].legend(fontsize=9)
    ax[0, 0].grid(alpha=0.3)

    # (b) 阈值 3 MW 附近放大
    pf = np.linspace(2.6, 3.4, 2000)
    ax[0, 1].plot(pf, c1h(pf), lw=2.2, label='硬（斜率 1→21 跳变）')
    ax[0, 1].plot(pf, c1s(pf), '--', lw=2.2, label='软（0.09 MW 内平滑过渡）')
    ax[0, 1].axvline(3.0, color='gray', ls=':', lw=0.9)
    ax[0, 1].axvspan(2.6, 3.0, color='C0', alpha=0.04)
    ax[0, 1].set_xlabel('$P_F$  (MW)')
    ax[0, 1].set_ylabel('$c_1$')
    ax[0, 1].set_title('(b) 阈值附近放大：3.000 MW 处两者取值相等')
    ax[0, 1].legend(fontsize=9)
    ax[0, 1].grid(alpha=0.3)

    # (c) 导数（阈值附近）
    pf = np.linspace(2.6, 3.4, 2000)
    ax[1, 0].step(pf, dh(pf), where='post', lw=2.2, label='硬：1 或 21（阶跃）')
    ax[1, 0].plot(pf, ds(pf), '--', lw=2.2, label='软：连续 S 形')
    ax[1, 0].axvline(3.0, color='gray', ls=':', lw=0.9)
    ax[1, 0].set_xlabel('$P_F$  (MW)')
    ax[1, 0].set_ylabel('$\\partial c_1/\\partial P_F$')
    ax[1, 0].set_title('(c) 导数：阶跃 → 连续（阈值处 1→21 平滑升）')
    ax[1, 0].legend(fontsize=9)
    ax[1, 0].grid(alpha=0.3)

    # (d) |ΔP| 折点附近：导数
    dP = np.linspace(-0.4, 0.4, 2000)
    ax[1, 1].step(dP, np.sign(dP), where='post', lw=2.2, label='硬：$\\mathrm{sign}(\\Delta P)$ 跳变')
    ax[1, 1].plot(dP, dPF(dP), '--', lw=2.2, label='软：$\\Delta P/\\sqrt{\\Delta P^2+\\varepsilon^2}$')
    ax[1, 1].axhline(0, color='gray', lw=0.6)
    ax[1, 1].axvline(0, color='gray', ls=':', lw=0.9)
    ax[1, 1].set_xlabel('$\\Delta P$  (MW)')
    ax[1, 1].set_ylabel('$\\partial P_F/\\partial \\Delta P$')
    ax[1, 1].set_title('(d) $|\\Delta P|$ 的折点：符号跳变 → 连续过零')
    ax[1, 1].legend(fontsize=9)
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out, dpi=150)
    print('已保存:', a.out)


if __name__ == '__main__':
    main()
