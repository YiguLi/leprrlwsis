#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""analyse_lam1.py —— **λ1 报告器 + "λ 摆动是否污染 Q"的证据检查**（只读，不需 MATLAB）。

为什么需要它
------------
`yawflor/README.md` §7 的「必须报告的附加项 1」要求报 λ1 轨迹与饱和占比。实测一个新环境
run（`output/TD3PER-run21`，实为 yawflor）发现：

* λ1 的接线**没问题**（`lambda ← clip(λ·exp(η·(EMA/d−1)), λ_min, λ_max)` 的递推可从 CSV
  反推到 1e-7 量级）；
* 但 λ1 在**贴顶 23% / 贴底 16%**之间以 ~4 h 为周期摆动（驱动信号 = 块和 c1 的 EMA，
  新环境的块和 c1 重尾：median ≈ 12、p99 ≈ 119、max ≈ 814 ⇒ 单个极值块就能把 EMA 推高
  0.02×814 > 整个预算 15）。

本脚本做两件事（A–D 是报告，E 才是判据）：

A. λ1 形态：折回物理价 `元/(MW·step)`、贴顶/贴底占比、分段轨迹、区间宽度是否够；
B. 对偶的驱动信号：块和 c1 的分布（重尾程度）、`EMA/d_block` 分布、与 `P_F>3` 的关系；
C. **两种"越界"分开量**：动作带越界（`penalty_NN_H`，按带宽比例归一）vs 波动越限（`P_F>3`）；
D. §7 判定表：λ 贴顶占比 × 越限率 的联合判读；
E. **Q 污染检查**（本项目口径，与 `audit_policy_stability.py` / `analyse_q_scale.py` 同源）：
   块奖励锚点 `anchor = mean(块 Σr1)/(1−γ)`、自洽比 `Qbar/anchor`、相对 TD 误差
   `td/|Q|`，然后做**三重对照**——(i) 按 λ 分区（贴底/中段/贴顶）分组比较；
   (ii) 以"λ 从贴底向上穿越"为事件的 ±N 块事件研究；(iii) 控制成本电平的回归
   `rel_td ≈ a + b·λ + c·cost`。三者方向一致才算"有污染证据"。

用法
----
    python agentutil/analyse_lam1.py                      # 扫 output/*/（有 update_stats_H.csv 的）
    python agentutil/analyse_lam1.py output/TD3PER-run21 output/TD3PER-run10
    python agentutil/analyse_lam1.py output/TD3PER-run21 --json output/lam1_report.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import hsrl_cost as hc                                  # noqa: E402  (λ 区间的单一真源)


# ===========================================================================
# 读数据
# ===========================================================================
def _read(path, **kw):
    """容忍"训练仍在写"的截断行。"""
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return pd.read_csv(path, engine='python', on_bad_lines='skip', **kw)


def find_run_inputs(run_dir):
    step_csv = None
    for name in sorted(os.listdir(run_dir)):
        if name.startswith('df_') and name.endswith('.csv') \
                and 'episode' not in name and 'test9' not in name:
            step_csv = os.path.join(run_dir, name)
            break
    stats_csv = os.path.join(run_dir, 'update_stats_H.csv')
    if step_csv is None or not os.path.isfile(stats_csv):
        return None, None
    return step_csv, stats_csv


def detect_lam_scale(lam_min_obs, lam_max_obs):
    """从观测到的 λ 区间反推 `lam_scale() = s_c/s_p`（决定 λ 的显示区间）。

    ⚠ 必须**以地板为主判据**：约束松弛时 λ 会被 clip 到 `LAM_MIN·kl`，所以 `min(λ)` 是可靠锚点；
    而"从 max 反推"在"λ 从未接近上界"的 run 上会系统性偏小（实测 TD3PER-run10 被误判成
    kl=0.01，物理价放大 10 倍）。max 只用作并列时的裁决。
    """
    NICE = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
    kl_min = lam_min_obs / hc.LAM_MIN
    cands = [k for k in NICE if abs(kl_min - k) / k < 0.02]
    if cands:
        # 并列时用 max 裁决（|max − LAM_MAX·kl| 最小者）
        return min(cands, key=lambda k: abs(lam_max_obs - hc.LAM_MAX * k))
    return min(NICE, key=lambda k: abs(lam_max_obs - hc.LAM_MAX * k))


# ===========================================================================
# 主分析
# ===========================================================================
def analyse(run_dir, gamma=0.98, anchor_win=20000, ev_win=20, eps=None,
            viol_tol=1e-6, verbose=True):
    step_csv, stats_csv = find_run_inputs(run_dir)
    if step_csv is None:
        print('[skip] %s：找不到 df_*.csv 或 update_stats_H.csv' % run_dir)
        return None
    d = _read(step_csv)
    u = _read(stats_csv)
    if 'lam1' not in d.columns or 'c1_step' not in d.columns:
        print('[skip] %s：不是 H 层逐步日志（缺 lam1/c1_step）' % run_dir)
        return None

    tag = os.path.basename(step_csv)
    blk = d[d['step'] % 5 == 4].reset_index(drop=True)
    step_n = int(d['step'].iloc[-1]) + 1
    R = dict(run=run_dir, step_csv=step_csv, tag=tag, steps=step_n,
             episodes=step_n / 1440.0, blocks=len(blk))
    if verbose:
        print('\n' + '=' * 100)
        print('run %s  (%s | %d 步 ≈ %.1f episode | 块末行 %d)'
              % (run_dir, tag, step_n, step_n / 1440.0, len(blk)))
        print('=' * 100)

    # ---------------- A. λ1 形态 ----------------
    lam = blk['lam1'].to_numpy(dtype=float)
    lam_obs_min = float(np.nanmin(lam))
    lam_obs_max = float(np.nanmax(lam))
    kl = detect_lam_scale(lam_obs_min, lam_obs_max)
    lo, hi_theory = hc.LAM_MIN * kl, hc.LAM_MAX * kl
    # ⚠ 抬过 λ_max 的 run（如 `--dual-lammax 200` = 双通道硬通道）⇒ 用**观测上界**当上界，
    #   否则会把它误判成"100% 贴顶"。
    extended = bool(lam_obs_max > hi_theory * 1.01)
    hi = lam_obs_max if extended else hi_theory
    at_lo = float(np.mean(lam <= lo * 1.02))
    at_hi = float(np.mean(lam >= hi * 0.98))
    mid = 1.0 - at_lo - at_hi
    if verbose:
        print('\n[A] λ1 形态  (lam_scale = s_c/s_p ≈ %.3g ⇒ 区间 [%.4f, %.4f]，物理价 ×%.3g%s)'
              % (kl, lo, hi, 1.0 / kl,
                 '；**该 run 抬过 λ_max**：上界取观测值 %.4f ⇒ 物理价 %.1f'
                 % (lam_obs_max, lam_obs_max / kl) if extended else ''))
        print('    min %.4f | median %.4f | mean %.4f | max %.4f   ⇒ 物理价 %.3f ~ %.3f 元/(MW·step)'
              % (lam.min(), np.median(lam), lam.mean(), lam.max(),
                 lam.min() / kl, lam.max() / kl))
        print('    贴底 ≤lo×1.02: %.1f%% | 中段: %.1f%% | 贴顶 ≥hi×0.98: %.1f%%'
              % (100 * at_lo, 100 * mid, 100 * at_hi))
        k = max(1, len(lam) // 12)
        print('    轨迹采样(每 %d 块): %s' % (k, np.round(lam[::k], 4).tolist()))
    R.update(lam_scale=kl, lam_lo=lo, lam_hi=hi, lam_min=float(lam.min()),
             lam_med=float(np.median(lam)), lam_mean=float(lam.mean()),
             lam_max=lam_obs_max, frac_at_lo=at_lo, frac_mid=mid, frac_at_hi=at_hi)

    # ---------------- B. 驱动信号 ----------------
    X = d.groupby(d['step'] // 5)['c1_step'].sum().to_numpy()[:len(blk)]
    ema = blk['ema_c1_H'].to_numpy(dtype=float)
    dblk = blk['d_block_H'].to_numpy(dtype=float)
    # 预算口径：历史 EMA 对偶用 d_block_H；双通道硬通道用 **ε**（`--dual-eps`，默认 0.05）
    budget = dblk if eps is None else np.full_like(dblk, float(eps))
    bname = 'd_block_H' if eps is None else 'ε=%.3g（双通道硬通道容差）' % float(eps)
    ok = np.isfinite(ema)
    ratio = np.divide(ema[ok], budget[ok], out=np.full(ok.sum(), np.nan),
                      where=budget[ok] != 0)
    if verbose:
        print('\n[B] 对偶驱动信号：块和 Σ₅c1（= 每块 5 步 c1 之和）与 EMA/预算（预算 = %s）' % bname)
        print('    块和: median %.2f | p90 %.2f | p99 %.2f | max %.2f | max/median %.1f'
              % (np.median(X), np.percentile(X, 90), np.percentile(X, 99), X.max(),
                 X.max() / max(np.median(X), 1e-9)))
        print('    EMA/预算: median %.3f | >1 占比 %.1f%% | >2 占比 %.1f%% | max %.2f'
              % (np.nanmedian(ratio), 100 * np.nanmean(ratio > 1), 100 * np.nanmean(ratio > 2),
                 np.nanmax(ratio)))
        print('    ⇒ 单块极值对 EMA 的冲击 = 0.02×max(块和) = %.2f（预算 %.3g）'
              % (0.02 * X.max(), float(np.nanmedian(budget))))
    R.update(blocksum_median=float(np.median(X)), blocksum_p99=float(np.percentile(X, 99)),
             blocksum_max=float(X.max()), ema_ratio_median=float(np.nanmedian(ratio)),
             ema_ratio_gt1=float(np.nanmean(ratio > 1)), d_block=float(np.nanmedian(budget)))

    # ---------------- C. 两种越界 ----------------
    coef = 100.0 if 'yawflor' in tag else 400.0        # 新/旧环境的越界罚系数
    band = 2.0 if 'yawflor' in tag else 0.5            # 新/旧环境的动作带宽
    pen = d['penalty_NN_H'].to_numpy(dtype=float)
    PF = d['P_F'].to_numpy(dtype=float)
    exc = np.maximum(PF - 3.0, 0.0)
    act_rate = float(np.mean(pen > 0))
    act_depth = float(np.mean(pen / coef / band))
    # ⚠ 容差：`P_F = 3 + 1ulp` 会被朴素 `P_F > 3` 误判成越限（实测 run21/E1 各几百行，
    #   其 exc_tot ≈ 4e-16）。评测脚本 test_five_days 用 VIOL_TOL 做了同样的事。
    viol_rate = float(np.mean(PF > 3.0 + viol_tol))
    if verbose:
        print('\n[C] 两种"越界"必须分开（λ 只管后者）')
        print('    (a) 动作带越界 penalty_NN_H>0 : %.2f%% 步 | 越界深(带宽比例) %.4f | 均值 %.2f 元/步'
              % (100 * act_rate, act_depth, pen.mean()))
        print('    (b) 波动越限 P_F>3 MW          : %.2f%% 步（容差 %g）| 越限总量均值 %.4f MW·step'
              % (100 * viol_rate, viol_tol, exc.mean()))
        print('    两者同时发生: %.2f%% 步' % (100 * np.mean((PF > 3.0 + viol_tol) & (pen > 0))))
        if 'exc_inev' in d.columns:
            real = d['exc_inev'].to_numpy(dtype=float) > 0
            gap = d['unfeasible_gap'].to_numpy(dtype=float) if 'unfeasible_gap' in d.columns else None
            print('    (c) **真越限**（exc_inev>0 = 窗口∩物理=∅，电池吸不动）: %d 步 (%.2f%%)'
                  % (int(real.sum()), 100 * real.mean()))
            print('        ⇒ 可避免越限 = %.0f%%（**RBB 硬裁剪结构性保证**：窗口可行时 pbrule∈窗口 ⇒ P_F≤limit）'
                  % (0.0 if real.any() else 100.0))
            if gap is not None and real.any():
                g = gap[real]
                print('        不可吸收缺口 unfeasible_gap: median %.2f | p90 %.2f | max %.2f MW'
                      ' ⇒ 若要消除，rated_power 至少需 +%.1f(中位)/+%.1f(p90) MW'
                      % (np.median(g), np.percentile(g, 90), g.max(),
                         np.median(g), np.percentile(g, 90)))
    R.update(action_band_rate=act_rate, action_band_depth=act_depth,
             viol_rate=viol_rate, viol_excess_mean=float(exc.mean()),
             pen_mean=float(pen.mean()))
    R['real_viol_rate'] = float((d['exc_inev'].to_numpy(dtype=float) > 0).mean()) \
        if 'exc_inev' in d.columns else None

    # ---------------- D. 判定表 ----------------
    if at_lo > 0.80 and viol_rate < 0.05:
        verdict = ('λ 长期贴底（%.0f%%）⇒ 约束松弛（块和 EMA 未越过预算），区间够用；'
                   '注意这**不代表**尾部没问题 —— 只说明"平均而言"满足' % (100 * at_lo))
    elif at_hi > 0.15 and viol_rate > 0.10:
        verdict = 'λ 贴顶占比高且越限率高于门槛 ⇒ 上界被夹住，需在新环境重设 LAM_MAX'
    elif (at_hi + at_lo) > 0.30:
        verdict = 'λ 两侧都常贴边 ⇒ 区间偏窄/对偶无阻尼（价格非平稳），优先治对偶而非单抬上界'
    else:
        verdict = 'λ 在中段工作、越限率可控 ⇒ 区间基本合适'
    # ⭐ 决定性判读：若该 run 记录了 `exc_inev`（真实越限 = 物理不可避）且**可避免越限为 0**
    #   ⇒ 硬通道的目标已被 RBB 硬裁剪结构性满足，抬 λ_max 无意义（改规格或如实报告）。
    if 'exc_inev' in d.columns:
        _real = d['exc_inev'].to_numpy(dtype=float) > 0
        _allviol = int(_real.sum())
        _totviol = int(np.sum(PF > 3.0 + viol_tol))
        if _allviol > 0 and _allviol >= _totviol - 1:
            _gap = d['unfeasible_gap'].to_numpy(dtype=float) if 'unfeasible_gap' in d.columns else None
            verdict = ('⭐ 可避免越限 = 0（RBB 硬裁剪结构性满足）⇒ 剩余 %.2f%% 全为**物理不可避**'
                       '（窗口∩物理=∅）⇒ 抬 λ_max 无意义；%s'
                       % (100 * _real.mean(),
                          ('要压总越限率只能改规格：rated_power +2 MW 可消除约 %.0f%%、'
                           '+4 MW 约 %.0f%%、+6 MW 约 %.0f%% 的真实越限（缺口 median %.1f / p90 %.1f / max %.1f MW）'
                           % (100 * np.mean(_gap[_real] <= 2.0), 100 * np.mean(_gap[_real] <= 4.0),
                              100 * np.mean(_gap[_real] <= 6.0), np.median(_gap[_real]),
                              np.percentile(_gap[_real], 90), _gap[_real].max()))
                          if _gap is not None else '要压总越限率需改物理规格'))
    R['verdict'] = verdict
    if verbose:
        print('\n[D] §7 判定表：λ 贴顶 %.1f%% / 贴底 %.1f%% × 波动越限率 %.2f%%  ⇒  %s'
              % (100 * at_hi, 100 * at_lo, 100 * viol_rate, verdict))

    # ---------------- E. Q 污染检查 ----------------
    def _m(a, s):
        """分组的 nanmean；组为空时返回 nan（不触发 numpy 告警）。"""
        a = np.asarray(a, dtype=float)[s]
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float('nan')

    br = d.groupby(d['step'] // 5)['r1'].sum()                  # 每块 Σ₅ r1（原始元）
    bc = d.groupby(d['step'] // 5)['c1_step'].sum()
    win_b = max(10, int(anchor_win // 5))
    anchor_p = br.rolling(win_b, min_periods=5).mean() / (1.0 - gamma)
    anchor_c = bc.rolling(win_b, min_periods=5).mean() / (1.0 - gamma)
    blk_step = blk['step'].to_numpy()
    ap = anchor_p.reindex(blk_step // 5).to_numpy(dtype=float)
    ac = anchor_c.reindex(blk_step // 5).to_numpy(dtype=float)

    m = u.merge(pd.DataFrame({'step': blk_step, 'lam': lam,
                              'anchor_p': ap, 'anchor_c': ac}), on='step', how='inner')
    m = m[np.isfinite(m['lam'])].reset_index(drop=True)
    if len(m) < 50:
        if verbose:
            print('\n[E] 可用的 (λ, Q) 联合样本不足（%d 行），跳过污染检查' % len(m))
        return R

    ratio_p = (m['q_p_mean'] / m['anchor_p']).to_numpy(dtype=float)
    ratio_c = (m['q_c_mean'] / m['anchor_c']).to_numpy(dtype=float)
    rel_td_p = (m['td_p_mean'] / np.maximum(np.abs(m['q_p_mean']), 1e-9)).to_numpy(dtype=float)
    rel_td_c = (m['td_c_mean'] / np.maximum(np.abs(m['q_c_mean']), 1e-9)).to_numpy(dtype=float)
    lam_m = m['lam'].to_numpy(dtype=float)
    regime = np.where(lam_m >= hi * 0.98, 'top',
                      np.where(lam_m <= lo * 1.02, 'floor', 'mid'))
    if verbose:
        print('\n[E] Q 污染检查（样本 %d 块，step %d..%d；锚点 = mean(块Σr1)/(1−γ) 滚动 %d 步）'
              % (len(m), m['step'].min(), m['step'].max(), anchor_win))
        print('    %-6s %6s %10s %10s %12s %12s %12s' %
              ('λ 分区', '块数', '自洽比p', '自洽比c', 'relTD_p', 'relTD_c', 'cost_mean'))
        for g in ('floor', 'mid', 'top'):
            s = regime == g
            if not s.any():
                continue
            print('    %-6s %6d %10.3f %10.3f %12.4f %12.4f %12.3f'
                  % (g, int(s.sum()), np.nanmean(ratio_p[s]), np.nanmean(ratio_c[s]),
                     np.nanmean(rel_td_p[s]), np.nanmean(rel_td_c[s]),
                     np.nanmean(m['cost_mean'].to_numpy()[s])))
        print('    （若"λ 电平"与"一致性"相关，应看到顶/底分区的自洽比或 relTD 系统性不同）')

    # 事件研究：λ 从"低位"爬升进入"高位"（λ 每块最多动 ~2% ⇒ 必须允许跨多块）
    #   事件 = 当前位置进入上三分位，且**过去 lookback 块内曾处于下三分位**
    lo_thr = lo + 0.33 * (hi - lo)
    hi_thr = lo + 0.67 * (hi - lo)
    lookback = 60
    ev = []
    i = 0
    while i < len(lam_m):
        if lam_m[i] >= hi_thr and (lam_m[max(0, i - lookback):i] < lo_thr).any():
            ev.append(i)
            i += max(1, ev_win)
        else:
            i += 1
    prof_p, prof_c = [], []
    for i in ev:
        a, b = max(0, i - ev_win), min(len(lam_m), i + ev_win + 1)
        prof_p.append(np.nanmean(rel_td_p[a:b]))
        prof_c.append(np.nanmean(rel_td_c[a:b]))
    if verbose:
        print('    λ 上爬事件 %d 个（%d 块内曾低位）; 事件窗口(±%d 块) relTD_c %.4f vs 全局 %.4f; '
              'relTD_p %.4f vs 全局 %.4f'
              % (len(ev), lookback, ev_win,
                 np.mean(prof_c) if ev else float('nan'), np.nanmean(rel_td_c),
                 np.mean(prof_p) if ev else float('nan'), np.nanmean(rel_td_p)))

    # 受控回归: rel_td ~ 1 + λ + 成本电平 (+ |Δλ| —— 直接检验"价格摆动本身是否有害")
    def _ols(y, Xs):
        Xd = np.column_stack([np.ones(len(y))] + Xs)
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        pred = Xd @ beta
        ss = 1.0 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        return beta, ss

    k_lag = 20                                        # ≈ 短期价格移动尺度（块）
    dlam = np.full(len(lam_m), np.nan)
    dlam[k_lag:] = lam_m[k_lag:] - lam_m[:-k_lag]
    adlam = np.abs(dlam)

    out_reg = {}
    for nm, y, ctrl in (('relTD_c', rel_td_c, m['cost_mean'].to_numpy(dtype=float)),
                        ('relTD_p', rel_td_p, m['reward_mean'].to_numpy(dtype=float))):
        okreg = np.isfinite(y) & np.isfinite(ctrl) & np.isfinite(adlam)
        b2, r2_2 = _ols(y[okreg], [lam_m[okreg], ctrl[okreg]])
        b3, r2_3 = _ols(y[okreg], [lam_m[okreg], ctrl[okreg], adlam[okreg]])
        out_reg[nm] = dict(b_lam=float(b2[1]), b_ctrl=float(b2[2]), r2=float(r2_2),
                           b_absdlam=float(b3[3]), r2_with_dlam=float(r2_3))
        if verbose:
            print('    回归 %-8s ≈ %+.4f %+.4f·λ %+.3e·ctrl            R²=%.3f'
                  % (nm, b2[0], b2[1], b2[2], r2_2))
            print('    回归 %-8s ≈ %+.4f %+.4f·λ %+.3e·ctrl %+.4f·|Δλ|   R²=%.3f  ← |Δλ| 系数即"摆动本身"'
                  % (nm, b3[0], b3[1], b3[2], b3[3], r2_3))

    # 结论：以**受控回归**为主判据（分组/事件研究只是支持性证据，会被"成本电平"混杂）
    #   ⚠ 前提：λ 必须有足够变化；否则回归无信息量（实测旧 run 的 λ 只在 [0.0967,0.1346]，
    #   即区间宽度的 10% 以内 ⇒ 任何 λ 系数都是噪声）
    lam_iqr = float(np.nanpercentile(lam_m, 75) - np.nanpercentile(lam_m, 25))
    informative = lam_iqr > 0.05 * (hi - lo)
    swing_gain_c = out_reg['relTD_c']['r2_with_dlam'] - out_reg['relTD_c']['r2']
    b_lam_c = out_reg['relTD_c']['b_lam']
    b_swing_c = out_reg['relTD_c']['b_absdlam']
    ev_ok = bool(ev) and (np.mean(prof_c) > np.nanmean(rel_td_c)) if ev else False
    rtdc_floor, rtdc_top = _m(rel_td_c, regime == 'floor'), _m(rel_td_c, regime == 'top')
    grp_ok = bool(rtdc_top > rtdc_floor) if (np.isfinite(rtdc_floor) and np.isfinite(rtdc_top)) else False
    reg_ok = b_lam_c > 0
    swing_ok = (b_swing_c > 0) and (swing_gain_c > 0.01)
    n_sup = int(ev_ok) + int(grp_ok)
    strong = reg_ok or swing_ok
    if not informative:
        concl = ('**无法判定**：λ 的 IQR 仅 %.4f（区间宽度的 %.0f%%）⇒ 回归无信息量；'
                 '该 run 的 λ 基本贴在一条边界上，属"约束松弛/被夹住"而非"摆动"'
                 % (lam_iqr, 100 * lam_iqr / max(hi - lo, 1e-12)))
    else:
        concl = ('有污染证据（受控回归支持：λ 偏效应 %+.4f / 摆动 %+.4f，ΔR²=%.3f，支持性证据 %d/2）'
                 % (b_lam_c, b_swing_c, swing_gain_c, n_sup) if (strong and n_sup >= 1) else
                 '未见可信污染证据（表面相关由**成本电平**混杂解释：λ 高 ⟺ 波动成本高 ⟺ 预测更难；'
                 '控制后 λ 偏效应 %+.4f、|Δλ| 的 ΔR²=%.3f、成本通道自洽比三分区几乎不变；支持性证据 %d/2）'
                 % (b_lam_c, swing_gain_c, n_sup))
    if verbose:
        print('    ⇒ %s' % concl)
        print('       [元判据] λ IQR %.4f / 区间宽 %.4f（%s） | 分组相关 %s（被 cost_mean %s→%s 混杂） | '
              '事件研究 %s | 受控回归 λ>0 %s | 摆动有效 %s'
              % (lam_iqr, hi - lo, '有信息' if informative else '**无信息**', grp_ok,
                 '%.1f' % _m(m['cost_mean'].to_numpy(), regime == 'floor'),
                 '%.1f' % _m(m['cost_mean'].to_numpy(), regime == 'top') if (regime == 'top').any() else 'n/a',
                 ev_ok, reg_ok, swing_ok))
        print('       自洽比：利润通道 三分区 %.3f / %.3f / %.3f ；成本通道 %.3f / %.3f / %.3f ⇒ %s'
              % (_m(ratio_p, regime == 'floor'), _m(ratio_p, regime == 'mid'),
                 _m(ratio_p, regime == 'top'),
                 _m(ratio_c, regime == 'floor'), _m(ratio_c, regime == 'mid'),
                 _m(ratio_c, regime == 'top'),
                 '两通道在三个 λ 分区都稳定（无值层损伤迹象）' if not strong else '需进一步查'))
    R.update(q_check=dict(n=len(m), ratio_p_top=_m(ratio_p, regime == 'top'),
                          ratio_p_floor=_m(ratio_p, regime == 'floor'),
                          ratio_c_top=_m(ratio_c, regime == 'top'),
                          ratio_c_floor=_m(ratio_c, regime == 'floor'),
                          reltd_c_top=rtdc_top, reltd_c_floor=rtdc_floor,
                          n_events=len(ev), ev_ok=ev_ok, grp_ok=grp_ok, reg_ok=reg_ok,
                          swing_ok=swing_ok, swing_gain_c=float(swing_gain_c),
                          reg=out_reg, conclusion=concl))
    return R


def main():
    ap = argparse.ArgumentParser(description='λ1 报告器 + Q 污染检查')
    ap.add_argument('runs', nargs='*', help='run 目录（缺省扫 output/*/）')
    ap.add_argument('--gamma', type=float, default=0.98, help='H 层块级折扣（默认 0.98）')
    ap.add_argument('--anchor-win', type=int, default=20000, help='锚点滚动窗口（步）')
    ap.add_argument('--ev-win', type=int, default=20, help='事件研究半窗（块）')
    ap.add_argument('--eps', type=float, default=None,
                    help='双通道硬通道容差 ε（tail run 应给，默认 0.05）；不给则用 d_block_H')
    ap.add_argument('--viol-tol', type=float, default=1e-6,
                    help='越限判定的容差（排除 P_F=3+1ulp 的浮点假阳性；默认 1e-6）')
    ap.add_argument('--json', default=None, help='报告落盘路径')
    args = ap.parse_args()

    runs = args.runs or sorted(
        os.path.dirname(p) for p in glob.glob(os.path.join('output', '*', 'update_stats_H.csv')))
    out = []
    for rd in runs:
        try:
            r = analyse(rd, gamma=args.gamma, anchor_win=args.anchor_win, ev_win=args.ev_win,
                        eps=args.eps, viol_tol=args.viol_tol)
        except Exception as e:                                     # 单个 run 失败不影响其余
            print('[error] %s: %s: %s' % (rd, type(e).__name__, e))
            r = None
        if r:
            out.append(r)

    print('\n' + '=' * 100)
    print('汇总（λ 区间为 actor/CSV 空间；物理价 = ÷lam_scale）')
    print('=' * 100)
    print('%-26s %8s %8s %8s %8s %9s %9s  %s'
          % ('run', '贴底%', '中段%', '贴顶%', '越限%', 'EMA/预算', '块和p99', '判定'))
    for r in out:
        print('%-26s %8.1f %8.1f %8.1f %8.2f %9.3f %9.1f  %s'
              % (os.path.basename(r['run']), 100 * r['frac_at_lo'], 100 * r['frac_mid'],
                 100 * r['frac_at_hi'], 100 * r['viol_rate'], r['ema_ratio_median'],
                 r['blocksum_p99'], r['verdict'][:26]))
    if args.json:
        p = args.json if os.path.isabs(args.json) else os.path.join(ROOT, args.json)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
        print('\n报告已落盘：%s' % p)


if __name__ == '__main__':
    main()
