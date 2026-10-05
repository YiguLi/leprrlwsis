# -*- coding: utf-8 -*-
"""随机风速日 + 随机行为：**aifflor(FLORIS) 与 WFSim(MATLAB) 各出一条出力曲线**。

做的事（一句话）
--------------
从训练风速池（`FFC_speed_noise_1.csv` 的 195 个"日"列）里随机抽**一个整天**（1440 min），
在同一天上生成**同一串随机动作**（逐台轴向因子 `a ∈ [0,0.5]`，可设保持块长），
然后把**逐位相同的 (u, a) 序列**分别喂给两个引擎，各得到一条 `P(t)` 曲线。

两个引擎的口径（**唯一差异是引擎**）
----------------------------------
                     aifflor 侧                                 WFSim 侧
  入口               WindBatteryAIF.PowerOutput                  env.WindBattery.PowerOutput
  物理后端           AifflorPhysics(calibrated=False,            MATLAB `WFSim_port.m`
                     power_scale=LEVEL_COEF≈0.59895)            （Make_Ax_b.m 的 dt=Inf 稳态解）
  动作               a ∈ [0,0.5]（逐台），再抬到 A_PHYS_LO=0.025 a ∈ [0,0.5]（逐台）
  围栏               3.5 ≤ u ≤ 25 否则 0.0（**两边同一个 env.py 口径**）  同
  布局/密度/切变     单排 5 台 5D、D=100、ρ=1.20、shear=0.12       同（meshing.m）

⚠ 两点必须随曲线一起声明：
  1. **aifflor 侧有 `A_PHYS_LO=0.025` 的抬升**（`a∈[0,0.025)` 按 0.025 计，占学习带 5%）；
     本脚本会统计被抬升的"机组·分钟"数并写进 JSON。
  2. 绝对电平本就不同（`LEVEL_COEF` 就是为此标定的）⇒ 看的是**形状/比值**，不是谁大谁小。

输出
----
  <out>/curve.csv    逐分钟：u、a0..a4、P_wfsim、P_aifflor、Δ、比值
  <out>/curve.json   元信息（抽中的日列/两个种子/块长/物理口径/计数器）+ 统计量
  <out>/curve.png    上下两栏：①两条曲线 + 风速；②比值 vs 风速

用法（**需要 MATLAB 写工作区外的权限**；只在 FLORIS 侧自检可加 --no-matlab）
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_engine_curve_random_day.py
    ... --day-seed 2026 --act-seed 7 --steps 1440 --block 1
    ... --block 5                 # 动作每 5 分钟才重抽（贴近 H 层决策周期）
    ... --no-matlab               # 只出 FLORIS 曲线（自检/无 MATLAB 时）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, 'output', '_mplcache'))

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

import env as ENV                                      # noqa: E402
from aifflor.aifflor_env import AifflorPhysics, WindBatteryAIF   # noqa: E402
from aifflor import aifflor_turbine as AT              # noqa: E402

WIND_CSV = 'FFC_speed_noise_1.csv'
NT = 5
MAX_E = 6.0
U_LO, U_HI = 3.5, 25.0
A_LO, A_HI = 0.0, 0.5


def pick_day(day_seed, steps):
    """按 test_five_days.sample_days 同一口径抽日（iloc 位置池 = 1..n_col-1）。"""
    df = pd.read_csv(WIND_CSV)
    pool = list(range(1, df.shape[1]))                 # 第 0 列是行号
    rng = np.random.RandomState(day_seed)
    col = int(rng.choice(pool, size=1, replace=False)[0])
    series = df.iloc[:steps, col].to_numpy(dtype=float)
    if series.size < steps:
        raise SystemExit('列 %d 只有 %d 行，不足 %d' % (col, series.size, steps))
    return col, str(df.columns[col]), series


def random_actions(act_seed, steps, block):
    """逐台 i.i.d. U[a_lo,a_hi]，可选每 block 分钟才重抽（保持 = 生产 H 层周期）。"""
    rng = np.random.RandomState(act_seed)
    raw = rng.uniform(A_LO, A_HI, size=(steps, NT))
    if block > 1:
        held = np.empty_like(raw)
        for i in range(steps):
            held[i] = raw[(i // block) * block]
        raw = held
    return raw


def run_floris(us, A, verbose=True):
    """aifflor 生产物理：WindBatteryAIF + AifflorPhysics（含围栏/抬升/守卫/预取）。"""
    phy = AifflorPhysics(nt=NT, calibrated=False, power_scale=AT.LEVEL_COEF)
    env_ai = WindBatteryAIF(MAX_E, NT, False, None, physics=phy)
    env_ai.wind_day = pd.Series(us)          # 让生产预取路径可用（数值与直算逐位相同）
    p = np.zeros(len(us), dtype=float)
    t0 = time.time()
    for i, u in enumerate(us):
        env_ai.count_wind = i
        env_ai.u = float(u)
        p[i] = float(env_ai.PowerOutput(float(u), A[i]))
        if verbose and (i + 1) % 200 == 0:
            print('  [aifflor] %4d/%d  (%.1f s)' % (i + 1, len(us), time.time() - t0),
                  flush=True)
    if verbose:
        print('  [aifflor] 完成 %.1f s | %s' % (time.time() - t0, env_ai.summary()), flush=True)
    return p, phy, env_ai


def run_wfsim(us, A, verbose=True):
    """WFSim 生产物理：env.WindBattery.PowerOutput → MATLAB WFSim_port。"""
    import matlab
    import matlab.engine
    if verbose:
        print('  [WFSim] 启动 MATLAB 引擎 ...', flush=True)
    t0 = time.time()
    eng = matlab.engine.start_matlab()
    if verbose:
        print('  [WFSim] 引擎就绪 (%.1f s)' % (time.time() - t0), flush=True)
    eng.cd(ROOT, nargout=0)
    env_ws = ENV.WindBattery(MAX_E, NT, False, eng)
    p = np.zeros(len(us), dtype=float)
    t0 = time.time()
    for i, u in enumerate(us):
        p[i] = float(env_ws.PowerOutput(float(u), A[i]))
        if verbose and (i + 1) % 200 == 0:
            print('  [WFSim] %4d/%d  (%.1f s)' % (i + 1, len(us), time.time() - t0),
                  flush=True)
    eng.quit()
    if verbose:
        print('  [WFSim] 完成 %.1f s' % (time.time() - t0), flush=True)
    return p


def stats(pw, pf, us, inside):
    d = pf - pw
    big = inside & (pw > 0.5)
    ratio = np.divide(pf, pw, out=np.full_like(pf, np.nan), where=pw > 0.5)
    rel = np.divide(d, pw, out=np.full_like(d, np.nan), where=np.abs(pw) > 1e-9)
    s = dict(
        n=int(len(us)), n_inside=int(inside.sum()), n_outside=int((~inside).sum()),
        mean_wfsim_mw=float(pw.mean()), mean_aifflor_mw=float(pf.mean()),
        std_wfsim_mw=float(pw.std()), std_aifflor_mw=float(pf.std()),
        energy_wfsim_mwh=float(pw.sum() / 60.0), energy_aifflor_mwh=float(pf.sum() / 60.0),
        energy_ratio=float(pf.sum() / max(pw.sum(), 1e-12)),
        mean_diff_mw=float(d[inside].mean()) if inside.any() else None,
        mean_rel_diff=float(np.nanmean(rel[inside])) if inside.any() else None,
        median_rel_diff=float(np.nanmedian(rel[inside])) if inside.any() else None,
        mape=float(np.nanmean(np.abs(rel[inside]))) if inside.any() else None,
        median_ape=float(np.nanmedian(np.abs(rel[inside]))) if inside.any() else None,
        ratio_min=float(np.nanmin(ratio[big])) if big.any() else None,
        ratio_median=float(np.nanmedian(ratio[big])) if big.any() else None,
        ratio_mean=float(np.nanmean(ratio[big])) if big.any() else None,
        ratio_max=float(np.nanmax(ratio[big])) if big.any() else None,
        dp_wfsim_mean_mw=float(np.abs(np.diff(pw[inside])).mean()) if inside.sum() > 1 else None,
        dp_aifflor_mean_mw=float(np.abs(np.diff(pf[inside])).mean()) if inside.sum() > 1 else None,
    )
    if big.sum() > 5:
        s['corr'] = float(np.corrcoef(pw[big], pf[big])[0, 1])
        s['k_best'] = float(np.dot(pw[big], pf[big]) / np.dot(pw[big], pw[big]))
        resid = pf[big] - s['k_best'] * pw[big]
        s['resid_rms_mw'] = float(np.sqrt((resid ** 2).mean()))
    return s


def make_plot(path, us, pw, pf, s, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.arange(len(us)) / 60.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=False,
                                   gridspec_kw=dict(height_ratios=[2, 1]))
    ax1.plot(t, pw, lw=1.0, color='#1f77b4', label='WFSim (MATLAB)')
    ax1.plot(t, pf, lw=1.0, color='#d62728', label='aifflor (FLORIS x LEVEL_COEF)')
    ax1.set_ylabel('farm power [MW]')
    ax1.set_xlabel('hour of day')
    ax1.grid(alpha=0.3)
    axt = ax1.twinx()
    axt.plot(t, us, lw=0.8, ls=':', color='0.45', label='wind u [m/s]')
    axt.set_ylabel('u [m/s]', color='0.45')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = axt.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9)
    ax1.set_title('random wind day (col {col}, seed {day_seed}) + random actions '
                  '(seed {act_seed}, block={block} min)\n'
                  'mean {mw:.3f} vs {mf:.3f} MW | energy ratio {er:.4f} | '
                  'corr {corr}'.format(col=cfg['wind_col'], day_seed=cfg['day_seed'],
                                       act_seed=cfg['act_seed'], block=cfg['block'],
                                       mw=s['mean_wfsim_mw'], mf=s['mean_aifflor_mw'],
                                       er=s['energy_ratio'],
                                       corr=('%.4f' % s['corr']) if 'corr' in s else 'n/a'),
                  fontsize=10)
    m = (us >= U_LO) & (us <= U_HI) & (pw > 0.5)
    r = pf[m] / pw[m]
    ax2.scatter(us[m], r, s=6, alpha=0.5, color='#2ca02c')
    ax2.axhline(1.0, color='0.4', lw=0.8, ls='--')
    if 'k_best' in s:
        ax2.axhline(s['k_best'], color='#d62728', lw=1.0,
                    label='k* = %.4f' % s['k_best'])
        ax2.legend(fontsize=9)
    ax2.set_xlabel('wind u [m/s]')
    ax2.set_ylabel('P_aifflor / P_wfsim')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='随机日+随机行为：aifflor vs WFSim 各出一条出力曲线')    ap.add_argument('--day-seed', type=int, default=2026, help='抽"哪一天"的随机种子')
    ap.add_argument('--act-seed', type=int, default=7, help='随机动作的种子')
    ap.add_argument('--steps', type=int, default=1440, help='分钟数（1440 = 完整一天）')
    ap.add_argument('--block', type=int, default=1, help='动作保持的分钟数（1=i.i.d.；5=H 层周期）')
    ap.add_argument('--out', default='output/engine_curve_random_day', help='输出目录')
    ap.add_argument('--no-matlab', action='store_true', help='只跑 aifflor(FLORIS) 侧')
    ap.add_argument('--no-plot', action='store_true', help='不画 PNG')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    print('=' * 96)
    print('  随机风速日 + 随机行为 → aifflor(FLORIS) / WFSim(MATLAB) 各一条出力曲线')
    print('=' * 96)

    col, colname, us = pick_day(args.day_seed, args.steps)
    A = random_actions(args.act_seed, args.steps, args.block)
    inside = (us >= U_LO) & (us <= U_HI)
    print('  抽中的日: FFC_speed_noise_1.csv 第 %d 列 (列名 %s)  n=%d  '
          'u∈[%.2f, %.2f] 均值 %.2f  (围栏 %.1f~%.1f 内 %d 条, 外 %d 条按 0 计)'
          % (col, colname, len(us), us.min(), us.max(), us.mean(), U_LO, U_HI,
             int(inside.sum()), int((~inside).sum())))
    print('  随机动作: seed=%d  block=%d min  a∈[%.2f,%.2f]^%d  '
          '实测范围 [%.4f, %.4f] 均值 %.4f'
          % (args.act_seed, args.block, A_LO, A_HI, NT, A.min(), A.max(), A.mean()))
    print('  物理口径: aifflor = AifflorPhysics(calibrated=False, power_scale=LEVEL_COEF=%.6f)'
          % AT.LEVEL_COEF)
    print('            （FLORIS %s；WFSim = MATLAB WFSim_port 稳态解）'
          % AT.build_model.__module__)

    t0 = time.time()
    pf, phy, env_ai = run_floris(us, A)
    print('  [aifflor] 曲线完成（%.1f s）' % (time.time() - t0))
    # ---- 诊断列：关掉 A_PHYS_LO 抬升（学习带 [0,0.5] 内原样送引擎），量化它对曲线的贡献。
    #      注意这是**诊断口径**、不是生产口径：生产口径永远带 0.025 抬升。
    _farm_nolift, _ = AT.eval_batch(phy.model, us, np.clip(A, A_LO, A_HI))
    pf_nolift = np.asarray(_farm_nolift, dtype=float) * AT.LEVEL_COEF
    print('  [aifflor] 诊断：不带 A_PHYS_LO 抬升的对照曲线已算完 '
          '（平均抬升 %+.4f MW，%d/%d 分钟受影响）'
          % ((pf - pf_nolift).mean(), int((np.abs(pf - pf_nolift) > 1e-9).sum()), len(us)))
    pw = None
    if not args.no_matlab:
        pw = run_wfsim(us, A)
    else:
        print('  [WFSim] 已按 --no-matlab 跳过')

    # ---- 统计 ----
    sumstats = {}
    if pw is not None:
        sumstats = stats(pw, pf, us, inside)

    # ---- 落盘 ----
    df = pd.DataFrame({'minute': np.arange(len(us)), 'u': us,
                       **{'a%d' % j: A[:, j] for j in range(NT)},
                       'P_aifflor_mw': pf,
                       'P_aifflor_nolift_mw': pf_nolift})
    if pw is not None:
        df['P_wfsim_mw'] = pw
        df['diff_mw'] = pf - pw
        df['ratio'] = np.divide(pf, pw, out=np.full_like(pf, np.nan), where=pw > 1e-9)
    df.to_csv(os.path.join(args.out, 'curve.csv'), index=False)

    lifted = int((A < AT.A_PHYS_LO).sum())
    meta = dict(
        wind_csv=WIND_CSV, wind_col=col, wind_col_name=colname,
        day_seed=args.day_seed, act_seed=args.act_seed, steps=args.steps,
        block_min=args.block, a_lo=A_LO, a_hi=A_HI, nt=NT,
        a_phys_lo=float(AT.A_PHYS_LO),
        n_turbine_minute_lifted=int(lifted),
        frac_turbine_minute_lifted=float(lifted / max(A.size, 1)),
        level_coef=float(AT.LEVEL_COEF),
        fence=[U_LO, U_HI],
        aifflor_side='WindBatteryAIF + AifflorPhysics(calibrated=False, power_scale=LEVEL_COEF)',
        wfsim_side='env.WindBattery -> MATLAB WFSim_port',
        aifflor_physics_summary=env_ai.summary(),
        aifflor_counters=dict(n_batch=phy.n_batch, n_direct=phy.n_direct,
                              n_hit=phy.n_hit, n_nonfinite=phy.n_nonfinite,
                              n_cache_bad=phy.n_cache_bad,
                              prefetch=env_ai.n_prefetch),
        summary=sumstats,
        lift_effect=dict(
            mean_gain_mw=float((pf - pf_nolift).mean()),
            max_gain_mw=float((pf - pf_nolift).max()),
            mean_gain_pct=float(100.0 * (pf - pf_nolift).mean() / max(pf.mean(), 1e-12)),
            n_minutes_changed=int((np.abs(pf - pf_nolift) > 1e-9).sum())),
        elapsed_s=round(time.time() - t_start, 1),
    )
    with open(os.path.join(args.out, 'curve.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    if not args.no_plot and pw is not None:      # --no-matlab 时不画对比图（会误导）
        make_plot(os.path.join(args.out, 'curve.png'), us, pw, pf, sumstats, meta)

    # ---- 打印 ----
    print('\n' + '=' * 96)
    print('  汇总' + ('' if pw is not None else '（仅 aifflor 侧）'))
    print('=' * 96)
    if pw is not None:
        print('  %-28s %14s %14s' % ('', 'WFSim', 'aifflor'))
        print('  %-28s %14.4f %14.4f' % ('全天电量 [MWh] (ΣP/60)',
                                         sumstats['energy_wfsim_mwh'],
                                         sumstats['energy_aifflor_mwh']))
        print('  %-28s %14.6f %14.6f' % ('平均出力 [MW]', sumstats['mean_wfsim_mw'],
                                         sumstats['mean_aifflor_mw']))
        print('  %-28s %14.6f %14.6f' % ('出力标准差 [MW]', sumstats['std_wfsim_mw'],
                                         sumstats['std_aifflor_mw']))
        print('  %-28s %14.6f %14.6f' % ('逐分钟 |ΔP| 均值 [MW]',
                                         sumstats['dp_wfsim_mean_mw'] or float('nan'),
                                         sumstats['dp_aifflor_mean_mw'] or float('nan')))
        print('  %-28s %14.4f' % ('电量比 (aifflor/WFSim)', sumstats['energy_ratio']))
        print('  %-28s %14.4f' % ('平均相对差 mean(Δ/P)', sumstats['mean_rel_diff']))
        print('  %-28s %14.4f' % ('中位相对差 median(Δ/P)', sumstats['median_rel_diff']))
        print('  %-28s %14.4f' % ('MAPE (围栏内)', sumstats['mape']))
        print('  %-28s %14.4f' % ('MedAPE (围栏内)', sumstats['median_ape']))
        if 'corr' in sumstats:
            print('  %-28s %14.6f' % ('corr(P_W, P_F)', sumstats['corr']))
            print('  %-28s %14.6f' % ('k* = argmin||P_F−k·P_W||', sumstats['k_best']))
            print('  %-28s %14.4f MW' % ('残差 RMS（抬平后）', sumstats['resid_rms_mw']))
        print('  %-28s %s' % ('比值 min/中位/均值/max',
                              '/'.join('%.4f' % sumstats[k] for k in
                                       ('ratio_min', 'ratio_median', 'ratio_mean',
                                        'ratio_max'))))
    print('  %-28s %d 个机组·分钟 (%.3f%%) 被抬到 A_PHYS_LO=%.3f'
          % ('aifflor 侧动作抬升', lifted, 100 * lifted / max(A.size, 1), AT.A_PHYS_LO))
    print('  %-28s 影响 %d/%d 分钟  平均 %+.4f MW (%+.3f%%)  最大 %+.4f MW'
          % ('  └ 抬升对曲线的作用', meta['lift_effect']['n_minutes_changed'], len(us),
             meta['lift_effect']['mean_gain_mw'], meta['lift_effect']['mean_gain_pct'],
             meta['lift_effect']['max_gain_mw']))
    print('  %-28s %s' % ('aifflor 计数器', meta['aifflor_counters']))
    print('\n  输出: %s  (curve.csv / curve.json%s)'
          % (args.out, ' / curve.png' if (not args.no_plot and pw is not None) else ''))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
