# -*- coding: utf-8 -*-
"""warmup 行为范围探针（只读）。

⚠ **口径变更（2026-10）**：`WarmupSchedule` 的默认已改为**混合采样** `wt_sched='mix'`
（70% 先验结构 `WT_PRIOR=(0.2,0.2,0.2,0.2,1/3)` + 邻域扰动 `U(±0.03)`，30% 在
`[0.05,0.45]` 上逐维独立均匀）。下面描述的"逐维随机游走 + 11 个可达格点"只对
**旧口径 `wt_sched='walk'`**（以及本脚本所读的**历史 run**）成立 ⇒ 本脚本读历史 CSV
仍然有效，但**不要再**把它的格点叙事套用到新 run 上。

回答的问题：**warmup 到底给 critic 喂了多大的 (s,a) 覆盖？** 这决定"探索是否足够、
收敛策略是否可能最优"。上层 H 的 warmup 动作由 `hsrl_warmup.WarmupSchedule` 生成
（**以下为 walk 口径**）：

    wt ← wt + rng.choice([-step, +step], size=nb_turbine)   # 每维独立 ±0.05
    wt ← clip(wt, 0.05, 0.45)                               # 中性宽带
    a6  = sig_value = 0.0                                   # 恒定！

要点（本脚本逐条实测）：
  1. **格点偏移**：start=1/3 不是 step=0.05 的整数倍 ⇒ 可达集是 {1/3 + 0.05k}∩带，
     即 0.0833/0.1333/.../0.4333 这 9 个内部点 + 0.05/0.45 两个 clip 点
     ⇒ **永远踩不到 0.10/0.15/... 这类"整数格点"**。
  2. **饱和**：无约束随机游走 N 块后 SD = 0.05·√N。新口径 N≈205 ⇒ SD≈0.72 ≫ 带宽 0.4
     ⇒ 游走会**很快顶到上下界并长期贴在界上**（中段只在前几十块被访问）。
  3. **信号维零覆盖**：a6 在 warmup 期恒为 0 ⇒ Q 对第 6 维的依赖**完全没有 warmup 数据**，
     全靠策略期那点噪声（0.01~0.03）撑。
  4. **clip 诱导相关**：各维独立，但一旦贴界，5 维会出现"共同贴界"的伪相关。

用法::

    python agentutil/probe_warmup_coverage.py                     # 默认几个代表 run
    python agentutil/probe_warmup_coverage.py --arms ABPAVE2-BASE AB-H16N10
"""
import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

import hsrl_warmup as hwarm              # noqa: E402

LATTICE = np.array([hwarm.WT_LO] + [hwarm.WT_START + 0.05 * k for k in range(-5, 4)]
                   + [hwarm.WT_HI])      # 11 个可达点（含两个 clip 点）
BAND = (hwarm.WT_LO, hwarm.WT_HI)


def actions_of(df, n_dim=6):
    """把 CSV 的 original_action_H 列解析成 (n,6) 数组。"""
    return np.array([np.fromstring(str(x).strip('[]'), sep=' ') for x in
                     df.original_action_H.values])


def find_run(tag):
    ds = sorted(glob.glob(os.path.join('output', '%s-run*' % tag)))
    if not ds:
        return None

    def steps(d):
        f = os.path.join(d, 'update_stats_H.csv')
        try:
            return int(pd.read_csv(f).step.iloc[-1])
        except Exception:
            return -1
    return max(ds, key=steps)


def csv_of(run):
    for c in ('df_aifflorpave.csv', 'df_aifflor.csv'):
        f = os.path.join(run, c)
        if os.path.isfile(f):
            return pd.read_csv(f)
    return None


def report(tag, warmup_steps):
    run = find_run(tag)
    if not run:
        print('  (%s 无 run 目录)' % tag)
        return
    df = csv_of(run)
    if df is None:
        print('  (%s 无 CSV)' % tag)
        return
    w = df[df.step <= warmup_steps]
    wb = w[w.step % 5 == 0]                      # 只在块首采样（H 每块决策一次）
    A = actions_of(wb)
    print('=' * 92)
    print('%s   run=%s' % (tag, os.path.basename(run)))
    print('  warmup 步数=%d  块数=%d（H 决策数）  策略段起始=step %d'
          % (warmup_steps, len(wb), warmup_steps + 1))
    print('  %-8s %-8s %-8s %-8s %-8s %-8s %-9s %-9s' %
          ('dim', 'min', 'max', 'mean', 'std', 'distinct', '贴界%', '|Δ|均值'))
    for j in range(6):
        v = A[:, j]
        at = np.mean((np.abs(v - BAND[0]) < 1e-9) | (np.abs(v - BAND[1]) < 1e-9)) * 100
        d = np.abs(np.diff(v)).mean() if len(v) > 1 else 0.0
        print('  %-8s %-8.4f %-8.4f %-8.4f %-8.4f %-8d %-9.1f %-9.4f' %
              ('a%d' % (j + 1), v.min(), v.max(), v.mean(), v.std(),
               len(np.unique(np.round(v, 6))), at, d))
    v5 = A[:, 5]
    print('  第 6 维（信号）取值集合: %s' % np.unique(np.round(v5, 6))[:6])
    # 格点序号 k（a = 1/3 + 0.05k）的直方图 —— ⚠ CSV 只存 3 位小数，故必须容差比对
    k = np.round((A[:, :5] - hwarm.WT_START) / hwarm.WT_STEP).astype(int).ravel()
    ks, cnt = np.unique(k, return_counts=True)
    print('  格点序号 k 直方图（a=1/3+0.05k，5 维合并, 共 %d 样本）:' % k.size)
    print('    ' + '  '.join('k=%+d:%.1f%%' % (a_, 100.0 * b_ / k.size)
                             for a_, b_ in zip(ks, cnt)))
    mid = np.mean(np.abs(A[:, :5] - hwarm.WT_START) <= 0.10) * 100
    edge = np.mean(np.abs(A[:, :5] - hwarm.WT_START) >= 0.20) * 100
    print('    **中段(|a−1/3|≤0.10) 占 %.1f%%**  ｜  边缘(|a−1/3|≥0.20) 占 %.1f%%'
          % (mid, edge))
    print('    可达格点数 = %d（理论 %d）；实测访问到的不同取值（合并 5 维）= %d'
          % (len(ks), 11, len(np.unique(np.round(A[:, :5], 3)))))
    # 理论饱和对照：无约束游走 N 块后 SD = step·√N
    N = len(wb)
    print('    理论对照: 无约束游走 %d 块 ⇒ SD = %.3f（带宽 %.2f）⇒ 必然饱和'
          % (N, hwarm.WT_STEP * np.sqrt(N), hwarm.WT_HI - hwarm.WT_LO))
    # 每维"首次贴界"的块号（饱和速度）
    firsts = []
    for j in range(5):
        v = A[:, j]
        idx = np.where((np.abs(v - BAND[0]) < 1e-9) | (np.abs(v - BAND[1]) < 1e-9))[0]
        firsts.append(int(idx[0]) if len(idx) else -1)
    print('  各维首次贴界的块号: %s（块数上限 %d）' % (firsts, len(wb)))
    frac5 = np.mean(np.all(np.isclose(A[:, :5], BAND[0], atol=1e-9) |
                           np.isclose(A[:, :5], BAND[1], atol=1e-9), axis=1)) * 100
    print('  **5 维同时贴界**的块占比 = %.1f%%（伪相关来源）' % frac5)
    # 跨维相关
    C = np.corrcoef(A[:, :5].T)
    off = C[np.triu_indices(5, 1)]
    print('  跨维相关（5 维两两, 非对角）: 均值 %+.3f  范围 [%+.3f, %+.3f]'
          % (off.mean(), off.min(), off.max()))
    # 策略段对照：策略把动作放在哪
    p = df[df.step > warmup_steps]
    pb = p[p.step % 5 == 0]
    if len(pb):
        P = actions_of(pb)
        tail = P[-max(1, len(P) // 5):]
        print('  --- 策略段对照（末 20%% 块, %d 条）---' % len(tail))
        print('  %-6s %-8s %-8s %-8s %-8s' % ('dim', 'min', 'max', 'mean', 'std'))
        for j in range(6):
            v = tail[:, j]
            print('  %-6s %-8.4f %-8.4f %-8.4f %-8.4f' %
                  ('a%d' % (j + 1), v.min(), v.max(), v.mean(), v.std()))
        inb = np.mean((tail[:, :5] >= BAND[0]) & (tail[:, :5] <= BAND[1])) * 100
        print('  策略段 WT 落在 warmup 带 [%.2f,%.2f] 内的比例 = %.1f%%'
              % (BAND[0], BAND[1], inb))
        print('  策略段 WT |a−1/3| 的中位 = %.4f（warmup 起点）' %
              np.median(np.abs(tail[:, :5] - hwarm.WT_START)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='*',
                    default=['ABPAVE2-BASE', 'AB-H16N10', 'ABPAVE2-CURVONLY'])
    ap.add_argument('--bsize', type=int, default=256)
    a = ap.parse_args()
    print('WarmupSchedule 口径: start=%.4f step=%.3f band=[%.2f, %.2f] sig_value=%.1f'
          % (hwarm.WT_START, hwarm.WT_STEP, hwarm.WT_LO, hwarm.WT_HI, hwarm.SIG_VALUE))
    print('新口径 warmup = %.1f × bsize(%d) = %d 步 ⇒ H 决策 %d 次'
          % (hwarm.WARMUP_BSIZE_MULT, a.bsize, int(hwarm.WARMUP_BSIZE_MULT * a.bsize),
             int(hwarm.WARMUP_BSIZE_MULT * a.bsize) // 5))
    for tag, w in zip(a.arms, [int(hwarm.WARMUP_BSIZE_MULT * a.bsize)] * len(a.arms)):
        if tag == 'AB-H16N10':
            w = 200                     # 旧口径
        report(tag, w)
    return 0


if __name__ == '__main__':
    sys.exit(main())
