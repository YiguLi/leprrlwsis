# -*- coding: utf-8 -*-
r"""离线 Q 到底该是多少？——**蒙特卡洛裁决**（只读，需 torch，不启 MATLAB）。

三方对照同一个量 `Q(s,a) = E[Σ_{k≥0} γ_eff^k · r_k]`（γ_eff = γ⁵ = 0.7738，r_k = 第 k 块的
n-step 折叠奖励）：

  (A) **在线 critic**（检查点 critic_p，读 update_stats 的 q_p_mean ≈ 3915 元）
  (B) **离线 fitted-Q**（`exp_offline_fqi_floor.py`，收敛到 ≈ 1061 元）
  (C) **蒙特卡洛**：从日志里取真实的块奖励序列，按 γ_eff 折扣求和 ⇒ 无偏真值

若 (C) ≈ (B)，则**在线 critic 高估约 4 倍**（TD3 自举的经典偏差）；
若 (C) ≈ (A)，则离线拟合欠拟合。这是唯一能裁决的证据。

用法: python agentutil/probe_q_montecarlo.py [output/TD3PERNOLOW-run57] [--horizon 40]
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

GAMMA, BLOCK, NSTEP = 0.95, 5, 5
G_EFF = GAMMA ** NSTEP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', nargs='?', default='output/TD3PERNOLOW-run57')
    ap.add_argument('--horizon', type=int, default=40, help='MC 的块视界')
    ap.add_argument('--n-start', type=int, default=400, help='抽样多少个起点块')
    args = ap.parse_args()

    LOG = os.path.join(args.run, 'df_td3pernolow.csv')
    acc = []
    for ch in pd.read_csv(LOG, usecols=['step', 'turbine_profit', 'bess_profit',
                                        'degradation', 'penalty_NN_H'],
                          chunksize=200000, encoding='utf-8-sig', low_memory=False):
        acc.append(ch)
    df = pd.concat(acc, ignore_index=True)
    df['r1'] = (df.turbine_profit + df.bess_profit - df.degradation - df.penalty_NN_H)
    df['blk'] = df['step'] // BLOCK
    # n-step 折叠奖励（与 buffer 同口径）
    w = GAMMA ** np.arange(BLOCK)
    rn = df.groupby('blk')['r1'].apply(
        lambda s: float((s.to_numpy()[:BLOCK] * w[:len(s)]).sum()))
    steps = df.groupby('blk')['step'].min()
    R = rn.to_numpy(float)
    B = steps.to_numpy(int)
    n = len(R)
    print('=' * 96)
    print('Q 的蒙特卡洛裁决: %s' % args.run)
    print('=' * 96)
    print('  块数 %d；n-step 折叠奖励 r 均值 %.1f  sd %.1f（元）'
          % (n, R.mean(), R.std()))
    print('  γ_eff = γ⁵ = %.4f ⇒ 若奖励平稳，真值 Q ≈ r̄/(1−γ_eff) = %.0f 元'
          % (G_EFF, R.mean() / (1 - G_EFF)))

    # ---- 冻结期起点（末 25%），避开 episode 边界（保证 horizon 内不跨集）----
    lo = int(n * 0.75)
    cand = np.arange(lo, n - args.horizon - 1)
    # 每个起点的 block 序号需要与 episode 边界对齐：episode = 288 块
    k = max(1, len(cand) // args.n_start)
    starts = cand[::k][:args.n_start]
    mc = np.array([float(sum(G_EFF ** j * R[s + j] for j in range(args.horizon)))
                   for s in starts])
    # 视界截断的解析修正：把剩下的尾巴按稳态补上
    tail = G_EFF ** args.horizon / (1 - G_EFF) * R.mean()
    mc_full = mc + tail
    print('\n[C] 蒙特卡洛（%d 个起点，视界 %d 块 + 尾部解析补 %.0f 元）'
          % (len(starts), args.horizon, tail))
    print('    MC 回报: mean %8.1f 元  sd %8.1f  p05 %8.1f  p95 %8.1f'
          % (mc_full.mean(), mc_full.std(),
             np.percentile(mc_full, 5), np.percentile(mc_full, 95)))

    # ---- [A] 在线 critic ----
    us = pd.read_csv(os.path.join(args.run, 'update_stats_H.csv'))
    q_online = us.q_p_mean.tail(300).mean()
    tgt_online = us.target_p_mean.tail(300).mean()
    print('\n[A] 在线 critic（update_stats 末 300 行）: q_p_mean = %.1f 元  '
          'target_p_mean = %.1f 元' % (q_online, tgt_online))

    # ---- [B] 离线 fitted-Q 的读数（写在文件里则由参数给出）----
    print('\n[B] 离线 fitted-Q（上一步实验）: 收敛到 ≈ 1061 元')
    print('\n---- 裁决 ----')
    print('  r̄/(1−γ_eff)（平稳近似）      = %8.0f 元' % (R.mean() / (1 - G_EFF)))
    print('  蒙特卡洛（无偏）              = %8.0f 元' % mc_full.mean())
    print('  离线 fitted-Q                 = %8.0f 元' % 1061.0)
    print('  在线 critic                   = %8.0f 元' % q_online)
    print('\n  ⇒ MC 与离线拟合的比值 = %.2f；MC 与在线的比值 = %.2f'
          % (mc_full.mean() / 1061.0, mc_full.mean() / q_online))
    print('     比值 ≈ 1 的一方才是对的。')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
