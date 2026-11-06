# -*- coding: utf-8 -*-
r"""「风机层」功率波动曲线 + 波动百分位（**不含 BESS**）—— 中等平均风速日的 greedy 配置。

配置（用户指定）：`a = [0.02, 0.02, 0.02, 0.02, 0.33]`（前四台限载到 0.02、末台推到单机 Cp 最优）
口径：与训练完全相同的引擎与调用签名
    `engine.WFSim_port(double([u]), double([0]), double([list(a)]), double([0]), False)`
    ⇒ `P_F = out / 1e6` [MW]（`env.py::PowerOutput`，`is_noise=False` ⇒ 只用风速序列、不加噪声）
风日：`FFC_speed_noise_1.csv` 的列（每天一列 1440 分钟）；中等平均风速 = 全 194 天里
      日均风速最接近**中位数**的那一天（同时打印前 5 近邻，便于替换）。

波动定义
--------
    P[t]  = PowerOutput(u[t], a)                     [MW]，逐步独立调用（引擎无状态）
    ΔP[t] = |P[t] − P[t−1]|  （t = 1..1439）          [MW]
输出：ΔP 的整十位百分位数 P10..P90 + P50/P90/P95/P99，并落盘逐分钟曲线。

用法
----
    python agentutil/exp_greedy_fluctuation.py [--day N] [--top 5] [--plot] [--smooth 0]
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

A_GREEDY = np.array([0.02, 0.02, 0.02, 0.02, 0.33])
WIND_CSV = 'FFC_speed_noise_1.csv'
OUT_CSV = 'output/greedy_fluctuation.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', type=int, default=None, help='指定风日列号；默认取中等风速日')
    ap.add_argument('--top', type=int, default=5, help='打印多少个候选日')
    ap.add_argument('--smooth', type=int, default=0,
                    help='>0 时对 P 做该宽度的滑动平均（仅供对照，默认不平滑）')
    ap.add_argument('--plot', action='store_true', help='另存 PNG 曲线')
    args = ap.parse_args()

    w = pd.read_csv(WIND_CSV)
    w = w.drop(columns=[c for c in w.columns if str(c).startswith('Unnamed')])
    print('=' * 100)
    print('风机层功率波动（不含 BESS）: a = %s' % np.round(A_GREEDY, 3))
    print('=' * 100)
    print('  风数据 %s: %d 分钟 × %d 天' % (WIND_CSV, w.shape[0], w.shape[1]))
    means = w.mean(axis=0)
    print('  日均风速: 全样本 mean %.3f  min %.3f  max %.3f  中位 %.3f m/s'
          % (means.mean(), means.min(), means.max(), means.median()))

    # ---- 选"中等平均风速"的那一天：日均最接近全样本中位数 ----
    med = float(means.median())
    order = (means - med).abs().sort_values().index
    if args.day is None:
        day_col = order[0]
        print('\n  候选日（按 |日均 − 中位 %.3f| 排序）:' % med)
        for c in order[:args.top]:
            print('    列 %-4s 日均 %6.3f m/s  (Δ %+.3f)' % (c, means[c], means[c] - med))
        print('  ⇒ 选用列 %s（日均 %.3f m/s）' % (day_col, means[day_col]))
    else:
        day_col = args.day if args.day in w.columns else order[0]
        print('\n  指定列 %s（日均 %.3f m/s）' % (day_col, means[day_col]))

    u = w[day_col].to_numpy(float)
    print('  当日风速: mean %.3f  min %.3f  max %.3f  sd %.3f m/s'
          % (u.mean(), u.min(), u.max(), u.std()))

    # ---- 引擎 ----
    print('\n  启动 MATLAB 引擎（WFSim_port）...')
    import matlab.engine
    eng = matlab.engine.start_matlab()
    print('  引擎就绪；开始扫 %d 分钟（每次一次 WFSim_port 调用）' % len(u))

    P = np.full(len(u), np.nan)
    bad = 0
    for t in range(len(u)):
        uu = float(u[t])
        if not (3.5 <= uu <= 25):
            P[t] = 0.0                     # 与 env.py::PowerOutput 的守卫一致
            continue
        try:
            r = eng.WFSim_port(matlab.double([uu]), matlab.double([0.0]),
                               matlab.double([list(A_GREEDY)]),
                               matlab.double([0.0]), False)
            P[t] = float(r['out']) / 1e6
        except Exception as e:                      # noqa: BLE001
            bad += 1
            if bad <= 3:
                print('    [warn] t=%d u=%.2f 调用失败: %s' % (t, uu, e))
        if (t + 1) % 200 == 0:
            print('    %4d/%d ... P 均值 %.4f MW' % (t + 1, len(u), np.nanmean(P)))
    print('  完成；失败 %d 次' % bad)

    if args.smooth > 1:
        P = pd.Series(P).rolling(args.smooth, min_periods=1).mean().to_numpy()

    dP = np.abs(np.diff(P))
    dP = dP[np.isfinite(dP)]
    print('\n' + '-' * 100)
    print('功率统计（a = %s，不含 BESS）' % np.round(A_GREEDY, 3))
    print('-' * 100)
    print('  P  [MW] : mean %.4f  sd %.4f  min %.4f  max %.4f'
          % (np.nanmean(P), np.nanstd(P), np.nanmin(P), np.nanmax(P)))
    print('  ΔP [MW] : mean %.4f  sd %.4f  min %.4f  max %.4f  （n=%d）'
          % (dP.mean(), dP.std(), dP.min(), dP.max(), len(dP)))
    print('\n  **ΔP 的整十位百分位数**（P10…P90）:')
    dec = np.percentile(dP, np.arange(10, 100, 10))
    for q, v in zip(range(10, 100, 10), dec):
        print('     P%-3d = %.4f MW' % (q, v))
    print('\n  补充: P50 %.4f  P75 %.4f  P90 %.4f  P95 %.4f  P99 %.4f MW'
          % tuple(np.percentile(dP, [50, 75, 90, 95, 99])))
    print('  超过 3 MW 的分钟数 %d / %d (%.2f%%)'
          % (int((dP > 3.0).sum()), len(dP), 100 * (dP > 3.0).mean()))
    print('\n  竖直直方图（ΔP，20 桶）:')
    cnt, edges = np.histogram(dP, bins=20)
    for c, e0, e1 in zip(cnt, edges[:-1], edges[1:]):
        print('    %6.3f~%6.3f | %-50s %d' % (e0, e1, '#' * int(50 * c / max(cnt.max(), 1)), c))

    df = pd.DataFrame({'minute': np.arange(len(u)), 'u': u, 'P_G_wt': P})
    df['dP'] = np.abs(df.P_G_wt.diff())
    df.to_csv(OUT_CSV, index=False)
    print('\n  逐分钟曲线已落盘: %s' % OUT_CSV)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            ax[0].plot(df.minute, df.P_G_wt, lw=1)
            ax[0].set_ylabel('P (MW)'); ax[0].set_title('greedy a=%s, day col %s (mean u %.2f m/s)'
                                                        % (list(A_GREEDY), day_col, u.mean()))
            ax[1].plot(df.minute, df.dP, lw=0.8, color='C3')
            for q, v in zip(range(10, 100, 10), dec):
                ax[1].axhline(v, color='gray', lw=0.4, alpha=0.5)
            ax[1].set_ylabel('|dP| (MW)'); ax[1].set_xlabel('minute')
            fig.tight_layout()
            fig.savefig('output/greedy_fluctuation.png', dpi=120)
            print('  图已落盘: output/greedy_fluctuation.png')
        except Exception as e:                      # noqa: BLE001
            print('  画图失败: %s' % e)
    try:
        eng.quit()
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
