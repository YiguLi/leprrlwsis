# -*- coding: utf-8 -*-
"""检查某个 run 的健康度（**任意前缀**：TD3PER / AIFFLOR / SINGLEAIF / YAWFLOR …；默认 `output/TD3PER-run48`）。

CSV 名**不写死**：逐步日志表用 `runpaths.find_log_csv`（历史名 `df_td3per.csv` 优先，否则按
`original_action_H` + `turbine_profit` 两列识别 `df_<tag>.csv`），episode 汇总表按 `df_episode*`
通配。此前写死 `df_td3per.csv` / `df_episode_td3per.csv` ⇒ **对 aifflor run 会静默跳过 ①② 两节**
（只剩 test9 / update_stats），看着"健康"其实根本没查。

检查项
------
1. **非有限值**：任一关键列出现 NaN/Inf ⇒ 立即报
2. **episode 层面趋势**：profit / c1 / gap_pen / SOC 跟踪 是否改善或至少不恶化
3. **对偶变量 λ1/λ2**：是否收敛（不贴着上下界、不发散）
4. **动作健康度**：H 层前 5 维占用率的分布（是否长期饱和在 0/0.5 边界）
5. **波动惩罚**：P_F / pf_viol 是否被压住
6. **9 用例测试**：test9 的 profit / c1 / cvm 的跨用例一致性
7. **更新统计**：td_p/td_c 是否衰减（critic 是否在学）

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\check_run_health.py --run TD3PER-run48
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

from runpaths import find_log_csv  # noqa: E402  （逐步日志表：按列识别，不写死文件名）

FAIL, WARN = [], []


def rep(ok, msg, warn_only=False):
    if ok:
        print('    PASS  %s' % msg)
    else:
        print('    %s  %s' % ('WARN' if warn_only else 'FAIL', msg))
        (WARN if warn_only else FAIL).append(msg)


def nums(df, c):
    return pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float) \
        if c in df.columns else None


def finite_report(df, cols, tag):
    bad = {}
    for c in cols:
        if c not in df.columns:
            continue
        try:
            s = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float)
        except Exception:
            continue
        n = int((~np.isfinite(s)).sum())
        if n:
            bad[c] = n
    rep(not bad, '[%s] 无非有限值 (检查 %d 列)%s'
        % (tag, len([c for c in cols if c in df.columns]),
           '' if not bad else ' ⇒ 坏列: %s' % bad))
    return bad


def trend(x, name, unit='', warn_ratio=1.0):
    """前后半段对比：报方向与幅度。"""
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if x.size < 8:
        print('    ----  %s: 样本不足 (%d)' % (name, x.size))
        return None
    h = x.size // 2
    a, b = x[:h].mean(), x[h:].mean()
    d = b - a
    rel = d / abs(a) if abs(a) > 1e-12 else float('nan')
    print('    ----  %-22s 前半 %12.4f → 后半 %12.4f  (%+.2f%%%s)'
          % (name, a, b, 100 * rel, (' ' + unit) if unit else ''))
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='TD3PER-run48')
    ap.add_argument('--base', default='output')
    args = ap.parse_args()
    d = os.path.join(args.base, args.run)
    if not os.path.isdir(d):
        cand = [p for p in os.listdir(args.base)
                if os.path.isdir(os.path.join(args.base, p)) and args.run in p]
        if not cand:
            raise SystemExit('找不到 %s' % d)
        d = os.path.join(args.base, cand[0])
    print('=' * 104)
    print('run 健康检查: %s' % d)
    print('=' * 104)
    files = {f: os.path.join(d, f) for f in os.listdir(d) if f.endswith('.csv')}
    cks = sorted(int(f.split('_')[-1]) for f in os.listdir(d)
                 if f.startswith('checkpoint_'))
    print('  文件: %s' % ', '.join(sorted(os.path.basename(p) for p in files.values())))
    print('  checkpoint: %d 个 (最大 step = %s)'
          % (len(cks), max(cks) if cks else '—'))

    # ---------------- episode 层 ----------------
    ep_csv = files.get('df_episode_td3per.csv')
    if ep_csv is None:                 # aifflor / singleaif / yawflor 的 episode 表名不同
        for k in sorted(files):
            if k.startswith('df_episode'):
                ep_csv = files[k]
                break
    if ep_csv:
        df = pd.read_csv(ep_csv)
        print('\n① episode 层面（n=%d）' % len(df))
        finite_report(df, ['episode_reward', 'episode_cost', 'episode_cost2',
                           'episode_gap_pen', 'episode_profit', 'bess_profit',
                           'deg_charge', 'deg_discharge', 'wind_day'], 'episode')
        for c in ('episode_profit', 'bess_profit', 'episode_cost',
                  'episode_cost2', 'episode_gap_pen'):
            v = nums(df, c)
            if v is not None:
                trend(v, c, warn_ratio=1.0)
        sg = nums(df, 'episode_gap_pen')
        if sg is not None:
            rep(float(np.nanmax(np.abs(sg))) < 5.0,
                'episode_gap_pen 最大值 %.4f（判据 <5 ⇔ SOC 目标可行性罚受控）'
                % float(np.nanmax(np.abs(sg))), warn_only=True)
        wd = nums(df, 'wind_day')
        if wd is not None:
            print('    ----  wind_day: %d 个不同天，范围 %d~%d'
                  % (len(np.unique(wd)), int(np.nanmin(wd)), int(np.nanmax(wd))))

    # ---------------- 步层 ----------------
    step_csv = find_log_csv(d)         # df_td3per.csv 优先，否则按列识别 df_<tag>.csv
    if step_csv:
        df = pd.read_csv(step_csv)
        print('\n② 步层面（n=%d，列 %d）' % (len(df), df.shape[1]))
        key = ['r1', 'c1', 'c2', 'r2', 'gap_pen', 'soc', 'soc_target', 'lam1',
               'lam2', 'd1', 'd2', 'P_F', 'wind_speed', 'reward', 'cost']
        bad = finite_report(df, key, 'step')
        for c in ('r1', 'c1', 'P_F', 'gap_pen'):
            v = nums(df, c)
            if v is not None:
                trend(v, c)
        # λ
        for c in ('lam1', 'lam2'):
            v = nums(df, c)
            if v is None:
                continue
            v = v[np.isfinite(v)]
            if v.size:
                print('    ----  %-22s 末值 %10.4f  范围 %.4f~%.4f'
                      % (c, v[-1], v.min(), v.max()))
        rep(not bad, '步层面无非有限值')
        # SOC 跟踪
        s, t = nums(df, 'soc'), nums(df, 'soc_target')
        if s is not None and t is not None:
            err = np.abs(s - t)
            err = err[np.isfinite(err)]
            if err.size:
                print('    ----  |SOC − target| : 均值 %.4f  中位 %.4f  p95 %.4f  max %.4f'
                      % (err.mean(), np.median(err), np.percentile(err, 95), err.max()))
        # 动作占用率
        acols = [c for c in df.columns if c.startswith('action/a_H_clipped_')]
        if not acols:
            acols = [c for c in df.columns if c.startswith('a_H_clipped')]
        if acols:
            A = np.vstack([pd.to_numeric(df[c], errors='coerce').to_numpy(float)
                           for c in sorted(acols)]).T
            A = A[np.isfinite(A).all(axis=1)]
            if A.size:
                lo = float((A <= 1e-6).mean())
                hi = float((A >= 0.5 - 1e-6).mean())
                print('    ----  H 层动作: 形状 %s  低界(≈0)占比 %.2f%%  高界(≈0.5)占比 %.2f%%'
                      % (A.shape, 100 * lo, 100 * hi))
                rep(lo < 0.25 and hi < 0.25,
                    '动作未长期贴边界 (低界 %.1f%% / 高界 %.1f%%)' % (100 * lo, 100 * hi),
                    warn_only=True)

    # ---------------- test9 ----------------
    if 'df_test9.csv' in files:
        df = pd.read_csv(files['df_test9.csv'])
        print('\n③ 9 用例测试（n=%d）' % len(df))
        finite_report(df, ['r1', 'c1', 'r2', 'c2', 'gap_pen', 'cvm', 'pf_viol',
                           'profit', 'soc_start', 'soc_end'], 'test9')
        for c in ('profit', 'c1', 'cvm', 'pf_viol'):
            v = nums(df, c)
            if v is not None and np.isfinite(v).any():
                vv = v[np.isfinite(v)]
                print('    ----  %-10s 均值 %12.4f  范围 %.4f ~ %.4f'
                      % (c, vv.mean(), vv.min(), vv.max()))
        # 逐轮（每轮 = 9 用例 × 5 步 = 45 行）
        if len(df) % 45 == 0 and len(df) >= 90:
            k = len(df) // 45
            prof = [np.nanmean(nums(df.iloc[i * 45:(i + 1) * 45], 'profit'))
                    for i in range(k)]
            trend(prof, 'test9 profit/轮', '（应上升）')
            c1s = [np.nanmean(nums(df.iloc[i * 45:(i + 1) * 45], 'c1'))
                   for i in range(k)]
            trend(c1s, 'test9 c1/轮', '（应下降）')

    # ---------------- update_stats ----------------
    if 'update_stats_H.csv' in files:
        df = pd.read_csv(files['update_stats_H.csv'])
        print('\n④ H 层更新统计（n=%d）' % len(df))
        finite_report(df, list(df.columns), 'update')
        for c in ('td_p_mean', 'td_c_mean', 'q_p_mean', 'q_c_mean',
                  'target_p_mean', 'target_c_mean'):
            v = nums(df, c)
            if v is not None and np.isfinite(v).any():
                trend(v, c)

    print('\n' + '=' * 104)
    print('结论: %s' % ('**健康**' if not FAIL else '**发现 %d 项失败**' % len(FAIL)))
    for m in FAIL:
        print('   FAIL  %s' % m)
    for m in WARN:
        print('   WARN  %s' % m)
    print('=' * 104)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
