# -*- coding: utf-8 -*-
r"""哪些风日能压到 10 MW BESS 的极限？（只读；真引擎，需 MATLAB）

背景：中等风速日（列 2）在 greedy 配置 `[0.02,0.02,0.02,0.02,0.33]` 下
      `P_max = 12.55 MW`、`ΔP_max = 6.81 MW`、`ΔP P99 = 3.99 MW` ⇒ 离 10 MW 的 BESS 还很远。
本脚本回答：**194 天里哪些天是真正的高波动日**、以及**哪一天才能把 ΔP 推到 13 MW 量级**。

两段
----
[A] 代理排序（快，无引擎）：`P ∝ u³`（已由 probe_a_farm_structure 与本次列 2 的
    `P_max` 12.553 vs 实测 12.5536 双证），用 `ΔP_proxy = |u³[t] − u³[t−1]|` 的 P90/P99/max
    给全部 194 天排序 ⇒ 挑出候选。
[B] 真引擎（WFSim_port）逐日跑：给候选日的 `P`、`ΔP` 的 P10..P90 与 P95/P99/max，
    以及 `ΔP > 10 MW` 的分钟数（BESS 功率上限对应的不可行区）。

用法
----
    python agentutil/exp_bess_limit_days.py --proxy            # 只做 [A]
    python agentutil/exp_bess_limit_days.py --days 116,173,92,107,192,100 --top 8
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

A_GREEDY = [0.02, 0.02, 0.02, 0.02, 0.33]
WIND_CSV = 'FFC_speed_noise_1.csv'
BESS_MW = 10.0


def load_wind():
    w = pd.read_csv(WIND_CSV)
    w = w.drop(columns=[c for c in w.columns if str(c).startswith('Unnamed')])
    return w


def proxy_rank(w, top):
    u = w.to_numpy(float)
    c3 = np.clip(u, 0, None) ** 3
    d = np.abs(np.diff(c3, axis=0))          # 代理 ΔP（只差常数 k）
    rows = []
    for j, col in enumerate(w.columns):
        rows.append(dict(col=col, mean_u=u[:, j].mean(), max_u=u[:, j].max(),
                         dP_p50=np.percentile(d[:, j], 50),
                         dP_p90=np.percentile(d[:, j], 90),
                         dP_p99=np.percentile(d[:, j], 99),
                         dP_max=d[:, j].max()))
    df = pd.DataFrame(rows)
    k = 0.0020926                            # 由列 2 标定（MW per u³）
    for c in ('dP_p50', 'dP_p90', 'dP_p99', 'dP_max'):
        df[c + '_MW'] = df[c] * k
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proxy', action='store_true', help='只做代理排序')
    ap.add_argument('--days', default=None, help='逗号分隔的列号；默认取代理 P99 前 N')
    ap.add_argument('--top', type=int, default=8)
    args = ap.parse_args()

    w = load_wind()
    pr = proxy_rank(w, args.top)
    print('=' * 108)
    print('BESS(%g MW) 压力日筛选 —— greedy a = %s' % (BESS_MW, A_GREEDY))
    print('=' * 108)
    print('  [A] 代理 ΔP（∝ |Δu³|，×0.0020926 MW/u³）排序，按 P99：')
    print('  %-5s %8s %8s %10s %10s %10s %10s' %
          ('col', 'mean_u', 'max_u', 'P50[MW]', 'P90[MW]', 'P99[MW]', 'max[MW]'))
    for _, r in pr.sort_values('dP_p99', ascending=False).head(args.top).iterrows():
        print('  %-5s %8.3f %8.3f %10.3f %10.3f %10.3f %10.3f'
              % (r.col, r.mean_u, r.max_u, r.dP_p50_MW, r.dP_p90_MW,
                 r.dP_p99_MW, r.dP_max_MW))
    print('\n  按代理 max 排序：')
    for _, r in pr.sort_values('dP_max', ascending=False).head(args.top).iterrows():
        print('  %-5s mean_u %6.3f max_u %7.3f | P99 %8.3f  max %8.3f MW'
              % (r.col, r.mean_u, r.max_u, r.dP_p99_MW, r.dP_max_MW))
    print('\n  全样本代理: P99 %.2f~%.2f MW   max %.2f~%.2f MW'
          % (pr.dP_p99_MW.min(), pr.dP_p99_MW.max(),
             pr.dP_max_MW.min(), pr.dP_max_MW.max()))
    print('  基准日列 2: P99 %.2f  max %.2f MW（真引擎实测 3.99 / 6.81）'
          % (pr[pr.col == '2'].dP_p99_MW.iloc[0], pr[pr.col == '2'].dP_max_MW.iloc[0]))
    if args.proxy:
        return 0

    days = (args.days.split(',') if args.days else
            list(pr.sort_values('dP_p99', ascending=False).col.head(args.top)))
    print('\n  [B] 真引擎逐日：%s' % days)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    print('  引擎就绪')

    rows, allres = [], {}
    for col in days:
        u = w[col].to_numpy(float)
        P = np.full(len(u), np.nan)
        for t in range(len(u)):
            uu = float(u[t])
            if not (3.5 <= uu <= 25):
                P[t] = 0.0
                continue
            r = eng.WFSim_port(matlab.double([uu]), matlab.double([0.0]),
                               matlab.double([A_GREEDY]), matlab.double([0.0]), False)
            P[t] = float(r['out']) / 1e6
        d = np.abs(np.diff(P))
        d = d[np.isfinite(d)]
        rec = dict(col=col, mean_u=u.mean(), max_u=u.max(),
                   P_mean=np.nanmean(P), P_max=np.nanmax(P),
                   dP_p50=np.percentile(d, 50), dP_p90=np.percentile(d, 90),
                   dP_p99=np.percentile(d, 99), dP_max=d.max(),
                   n_over_10=int((d > BESS_MW).sum()),
                   frac_over_10=100 * (d > BESS_MW).mean())
        rows.append(rec)
        allres[col] = (u, P, d)
        print('    列 %-5s P_mean %7.3f P_max %8.3f | ΔP P50 %6.3f P90 %6.3f '
              'P99 %7.3f max %8.3f | >10MW %d (%.2f%%)'
              % (col, rec['P_mean'], rec['P_max'], rec['dP_p50'], rec['dP_p90'],
                 rec['dP_p99'], rec['dP_max'], rec['n_over_10'], rec['frac_over_10']),
              flush=True)

    R = pd.DataFrame(rows)
    print('\n' + '-' * 108)
    print('汇总（真引擎，greedy 配置，不含 BESS）')
    print('-' * 108)
    print(R.to_string(index=False, float_format=lambda v: '%9.3f' % v))
    R.to_csv('output/bess_limit_days.csv', index=False)
    print('\n  已落盘 output/bess_limit_days.csv')
    best = R.loc[R.dP_max.idxmax()]
    print('\n  ⇒ 本次候选里 ΔP_max 最大的是列 %s: P_max %.3f MW, ΔP_max %.3f MW, '
          '>10MW 共 %d 分钟' % (best.col, best.P_max, best.dP_max, best.n_over_10))
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
