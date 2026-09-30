#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""analyse_yawprofit.py —— `yawprofit`（只调度 FLORIS 的 5-min 纯利润臂）的**学习稳定性诊断**。

输入 = 一个 run 目录（默认 output/ 下最新的 YAWPROFIT*）。
输出 = 五张紧凑表（不画图，避免 matplotlib 依赖）：

  [A] 块级奖励轨迹（每 20 块一桶）：mean/median/p10/p90 of 块 Σr1、块 Σprofit、块 Σ罚
      ⇒ "奖励电平是否单调/是否在振荡"
  [B] 越界诊断：`action_excess` 的块均值轨迹 + 罚/收益比  ⇒ "actor 是否漂出动作带"
  [C] 九例测试轨迹：profit / profit_nominal / gain / yaw_deg_mean  ⇒ "有没有比不转向更好"
  [D] critic 自洽：`q_p_mean` vs `target_p_mean`（比值 → 1 才算收敛）+ td_p_mean
  [E] 动作分布：每维原始动作的分位（看是否堆在 ±1 之外）

⚠ 表 [A]/[E] 的奖励与动作都**含探索噪声**（buffer 里存的就是带噪动作）。要区分
"策略自己走出带"与"噪声把贴边策略推出去"，用 `agentutil/probe_action_state_sensitivity.py`
（回放真实状态、不加噪声）。

用法
----
    python agentutil/analyse_yawprofit.py [output/YAWPROFIT-runN]
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))            # agentutil/
ROOT = os.path.dirname(_HERE)                                 # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

# 终端编码守卫（同 yawprofit_main / 契约测试）：GBK 控制台下 '⇒'/'−' 会打断输出
try:
    sys.stdout.reconfigure(errors='replace')
except Exception:                                            # pragma: no cover
    pass


def _parse_action(s):
    """CSV 里的动作列 → np.ndarray（兼容 str 与已是序列两种形态）。"""
    if isinstance(s, str):
        s = s.strip().strip('[]')
        return np.array(s.split(), dtype=float)
    return np.asarray(s, dtype=float)


def _latest_run(prefix='YAWPROFIT'):
    cands = [d for d in os.listdir('output')
             if d.startswith(prefix) and os.path.isdir(os.path.join('output', d))
             and os.path.exists(os.path.join('output', d, 'df_yawprofit.csv'))]
    if not cands:
        raise SystemExit('no run with df_yawprofit.csv under output/ matching ' + prefix)
    cands.sort(key=lambda d: int(d.split('-run')[-1]) if '-run' in d else -1)
    return os.path.join('output', cands[-1])


def main(run=None, bucket=20):
    run = run or _latest_run()
    print('=' * 100)
    print('run = {}'.format(run))
    df = pd.read_csv(os.path.join(run, 'df_yawprofit.csv'))
    print('逐步表 {} 行 (step {}..{})'.format(len(df), df.step.min(), df.step.max()))

    # 块级聚合（每 5 步一块）
    blk = df.groupby(df.step // 5).agg(
        step=('step', 'first'),
        r1=('r1', 'sum'),
        profit=('profit_step', 'sum'),
        pen=('penalty_NN_H', 'sum'),
        excess=('action_excess', 'mean'),
        yawdeg=('yaw_deg_mean', 'mean'),
        PF=('P_F', 'mean'),
        pr=('pr', 'mean'),
        u=('u', 'mean'),
    ).reset_index(drop=True)
    blk['block'] = np.arange(len(blk))

    def _buckets(sub, col):
        g = sub.groupby((sub.block // bucket) * bucket)
        return pd.DataFrame({
            'blk': g[col].mean().index,
            'mean': g[col].mean().values,
            'median': g[col].median().values,
            'p10': g[col].quantile(0.10).values,
            'p90': g[col].quantile(0.90).values,
        })

    print('\n[A] 块级奖励轨迹（每 {} 块一桶；单位=元/块(5 min)）'.format(bucket))
    print('{:>6} | {:>9} {:>9} {:>9} {:>9} | {:>9} {:>9} | {:>8} {:>8} | {:>7} {:>7}'.format(
        'block', 'r1_mean', 'r1_med', 'r1_p10', 'r1_p90',
        'prof_mean', 'pen_mean', 'excess', 'yawdeg', 'pr', 'u'))
    a = _buckets(blk, 'r1')
    for i in range(len(a)):
        b = blk[(blk.block // bucket) * bucket == a.blk[i]]
        print('{:>6.0f} | {:>9.1f} {:>9.1f} {:>9.1f} {:>9.1f} | {:>9.1f} {:>9.1f} | '
              '{:>8.3f} {:>8.2f} | {:>7.1f} {:>7.2f}'.format(
                  a.blk[i], a['mean'][i], a['median'][i], a['p10'][i], a['p90'][i],
                  b.profit.mean(), b.pen.mean(), b.excess.mean(), b.yawdeg.mean(),
                  b.pr.mean(), b.u.mean()))

    print('\n[B] 越界诊断（罚 = 100×excess，块内 5 步各记一次 ⇒ 块内合计 = 5×100×excess）')
    r = blk.pen.sum() / max(1e-9, blk.profit.sum())
    print('    全程: Σ罚/Σ收益 = {:.4f} ({:.2f}%) | excess 分位 '
          'p50={:.4f} p90={:.4f} max={:.4f} | 罚>0 的块占比 {:.1%}'.format(
              r, 100 * r, blk.excess.median(), blk.excess.quantile(0.9), blk.excess.max(),
              float((blk.pen > 1e-9).mean())))
    for i in range(len(a)):
        b = blk[(blk.block // bucket) * bucket == a.blk[i]]
        print('    blk {:>5.0f}: excess_mean={:.4f} 罚/收益={:>6.2%}'.format(
            a.blk[i], b.excess.mean(), b.pen.sum() / max(1e-9, b.profit.sum())))

    t9 = os.path.join(run, 'df_test9.csv')
    if os.path.exists(t9):
        d9 = pd.read_csv(t9)
        ov = d9[d9.wind_level == 'OVERALL']
        print('\n[C] 九例测试（45 步总量；profit_nominal = 同样风/电价下**不转向**的收益）')
        print('{:>7} {:>10} {:>10} {:>9} {:>9} {:>8} {:>8} {:>9}'.format(
            'step', 'profit', 'nominal', 'gain', 'gain%', 'pf_viol', 'yaw_deg', 'P_F_mean'))
        for _, x in ov.iterrows():
            g = x.gain
            print('{:>7.0f} {:>10.1f} {:>10.1f} {:>9.2f} {:>8.2%} {:>8.0f} {:>8.2f} {:>9.3f}'
                  .format(x.step, x.profit, x.profit_nominal, g, g / max(1e-9, x.profit_nominal),
                          x.pf_viol, x.yaw_deg_mean, x.P_F_mean))
        # 分档看 gain（哪一档在赚/在赔）
        case = d9[d9.wind_level != 'OVERALL']
        if len(case):
            print('\n[C2] 按 风×价 分档的平均 gain（元/5 步）:')
            piv = case.pivot_table(index='wind_level', columns='price_level',
                                   values='gain', aggfunc='mean')
            print(piv.round(1).to_string())

    upd = os.path.join(run, 'update_stats_H.csv')
    if os.path.exists(upd):
        du = pd.read_csv(upd)
        du['self_ratio'] = du.q_p_mean / du.target_p_mean.replace(0, np.nan)
        du['rel_td'] = du.td_p_mean / du.q_p_mean.abs().replace(0, np.nan)
        print('\n[D] critic 自洽（原始口径; self_ratio = q_p/target_p → 1 才算跟上目标）')
        print('{:>7} {:>10} {:>11} {:>10} {:>10} {:>9} {:>9}'.format(
            'step', 'q_p', 'target_p', 'self_ratio', 'td_p', 'rel_td', 'reward'))
        for _, x in du.iloc[::max(1, len(du) // 12)].iterrows():
            print('{:>7.0f} {:>10.1f} {:>11.1f} {:>10.3f} {:>10.1f} {:>9.3f} {:>9.1f}'.format(
                x.step, x.q_p_mean, x.target_p_mean, x.self_ratio, x.td_p_mean,
                x.rel_td, x.reward_mean))
        tail = du.tail(max(3, len(du) // 10))
        print('    末 10%: self_ratio {:.3f}±{:.3f} | td_p {:.1f} | 块收益(5 块 n-step 和) {:.1f}'
              .format(tail.self_ratio.mean(), tail.self_ratio.std(), tail.td_p_mean.mean(),
                      tail.reward_mean.mean()))

    print('\n[E] 动作分布（原始动作, 未夹; 5 维合并）')
    A = np.vstack(df.original_action_H.apply(_parse_action).values)
    print('    mean={:+.3f} sd={:.3f} | p01={:+.3f} p50={:+.3f} p99={:+.3f} | '
          '越上界占比={:.1%} 越下界占比={:.1%}'.format(
              A.mean(), A.std(), np.percentile(A, 1), np.percentile(A, 50),
              np.percentile(A, 99), float((A > 1).mean()), float((A < -1).mean())))
    Ac = np.vstack(df.applied_action_H.apply(_parse_action).values)
    print('    夹后 |γ|: mean={:.2f}° p90={:.2f}° 贴边(|γ|>24.5°)占比={:.1%}'.format(
        25 * np.abs(Ac).mean(), 25 * np.percentile(np.abs(Ac), 90), float((np.abs(Ac) > 0.98).mean())))
    for j in range(A.shape[1]):
        print('      dim{}: mean={:+.3f} sd={:.3f} 越界占比={:.1%}'.format(
            j, A[:, j].mean(), A[:, j].std(),
            float((np.abs(A[:, j]) > 1).mean())))


if __name__ == '__main__':
    run = sys.argv[1] if len(sys.argv) > 1 else None
    main(run)
