#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按「日均风速档 × 日内相对离散度」从风速池里挑 4 天，并打印 4 天风速图。

挑选口径（用户指令）
-------------------
    d1  低风速   低方差     5 ≤ 日均 < 10   m/s
    d2  低风速   高方差     5 ≤ 日均 < 10   m/s
    d3  中高风速 低方差     日均 ≥ 10       m/s
    d4  中高风速 高方差     日均 ≥ 10       m/s

「方差高低」用**全池相对值**判定，不用绝对 std（否则高风速日天然 std 大）：
    CV = std / mean            （日内 1440 点的总体标准差，ddof=0）
    CV*_low  = 全池 CV 的 P10（`--p-low`）
    CV*_high = 全池 CV 的 P90（`--p-high`）
⇒ "低方差" = 该带的 CV 贴近全池 P10 的日子；"高方差" = 贴近 P90 的日子。

⚠ 可达性截断（实测必须）：CV 与日均风速**强负相关** —— 低风速带的 CV 下限
实测为 0.155，而全池 P10 是 0.117 ⇒ **低风速带内不存在"全池 P10 级"的低方差日**。
因此目标一律先夹到该带**实际可达的 CV 区间** `[band_min, band_max]`：
    d1 低风速·低方差：全池 P10 0.117 → 截断到带内下限 0.155（"这个带里最平的日"）
    其它三格目标本身可达，不截断。
脚本会把"目标值 / 带内可达范围 / 是否截断"逐格打印，不静默。

带内怎么落到唯一一天（两段式，与"接近全池 P10/P90 的带内代表日"一致）
--------------------------------------------------------------------
    ① 主判据：CV 相对距离 |CV − CV*| / CV*，取带内最近的 `--short-frac`（默认 10%）
       作为候选邻域 ⇒ 保证选出的天确实"是这个带里的低/高方差日"；
    ② 次判据：在邻域内取 日均风速最接近**该带日均中位数**的一天（"代表日"，
       不追极端风速；`--mean-basis center` 可改为带内风速区间中点）；
       并列 → CV 距离更小 → iloc 更小（可复现）。

输出
----
    <out>/four_days.csv        四天清单（tag / iloc / 列名 / mean / std / CV / CV 全池分位）
    <out>/four_days.png        2×2 逐日风速图（1440 分钟）
    <out>/four_days_overlay.png  四条曲线叠加（原始 + 30 分钟滑动均值）
并在终端打印可直接粘进评测脚本的 `FROZEN_DAYS` 字面量。

⚠ 池子实际是 **194 列**（`FFC_speed_noise_1.csv` = 1440×195：iloc[:,0] 是行号，
其余 194 列是风速日；列标签编号到 '199' 但有 6 个跳号 20/21/22/31/32/64，
所以不是 198）。取列口径与 `env.py` 一致：`iloc[:, j]` 的**位置** j（1..194），
不是列标签。

用法
----
    python agentutil/select_4_wind_days.py                 # 选天 + 出图
    python agentutil/select_4_wind_days.py --dry-run       # 只打印，不出图/不写盘
    python agentutil/select_4_wind_days.py --p-low 20 --p-high 80 --short-frac 0.2
    python agentutil/select_4_wind_days.py --csv FFC_speed_1.csv   # 换无噪声基准池
"""
import argparse
import os
import sys

# ---------------------------------------------------------------------------
# 路径 bootstrap（AGENTS.md 约定：可被任意 CWD / 任意路径调用）
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块
sys.path.insert(0, _HERE)                              # agentutil 内互相 import

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors='replace')
    except Exception:
        pass

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

WIND_CSV = 'FFC_speed_noise_1.csv'
LOW_LO, LOW_HI = 5.0, 10.0          # 低风速带 [5, 10)
NB_POINTS = 1440

#: (tag, 展示名, 风速带, 方差档)
SPECS = (
    ('d1_low_lowcv', 'd1 低风速·低方差', 'low', 'low'),
    ('d2_low_highcv', 'd2 低风速·高方差', 'low', 'high'),
    ('d3_high_lowcv', 'd3 中高风速·低方差', 'high', 'low'),
    ('d4_high_highcv', 'd4 中高风速·高方差', 'high', 'high'),
)


# ===========================================================================
# 读池 / 统计
# ===========================================================================
def load_pool(csv_path):
    """返回 (df, stats)。

    df    : 原始读入（iloc[:, 0] 是行号，1.. 是风速日）
    stats : DataFrame，index = iloc 位置（1..N），列 = col/mean/std/cv
    """
    df = pd.read_csv(os.path.join(ROOT, csv_path))
    vals = df.iloc[:, 1:]
    n = vals.shape[1]
    means = vals.mean().to_numpy(dtype=float)
    # 日内总体标准差（ddof=0）：一天内部的离散度，不涉及跨天样本推断
    stds = vals.std(ddof=0).to_numpy(dtype=float)
    stats = pd.DataFrame({
        'iloc': np.arange(1, n + 1),
        'col': [str(c) for c in vals.columns],
        'mean': means,
        'std': stds,
    })
    stats['cv'] = stats['std'] / stats['mean']
    return df, stats


def band_of(stats, band):
    """按风速带筛出候选日。"""
    if band == 'low':
        return stats[(stats['mean'] >= LOW_LO) & (stats['mean'] < LOW_HI)]
    return stats[stats['mean'] >= LOW_HI]


def describe_pool(stats, p_low, p_high):
    cv = stats['cv']
    print('\n' + '=' * 92)
    print('  风速池：%d 天 × %d 分钟（列位置口径 = env.py 的 iloc[:, j]）'
          % (len(stats), NB_POINTS))
    print('=' * 92)
    print('  日均风速  %.2f .. %.2f m/s   (中位 %.2f)'
          % (stats['mean'].min(), stats['mean'].max(), stats['mean'].median()))
    print('  日内 std  %.2f .. %.2f m/s' % (stats['std'].min(), stats['std'].max()))
    print('  相对离散  CV = std/mean  %.4f .. %.4f   (中位 %.4f)'
          % (cv.min(), cv.max(), cv.median()))
    print('  全池分位: P%-4g %.4f | P50 %.4f | P%-4g %.4f   ← 低/高方差的目标水平'
          % (p_low, np.percentile(cv, p_low), cv.median(), p_high,
             np.percentile(cv, p_high)))
    for name, band in (('低风速', 'low'), ('中高风速', 'high')):
        b = band_of(stats, band)
        print('  %-6s 带: %3d 天 | 日均 %.2f..%.2f (中位 %.2f) | '
              'CV %.4f..%.4f (带内 P10 %.4f, P90 %.4f)'
              % (name, len(b), b['mean'].min(), b['mean'].max(), b['mean'].median(),
                 b['cv'].min(), b['cv'].max(),
                 np.percentile(b['cv'], p_low), np.percentile(b['cv'], p_high)))
    print('  ⇒ CV 随风速下降而上升：低风速带的 CV 下限 %.4f 已高于全池 P%-4g %.4f'
          % (band_of(stats, 'low')['cv'].min(), p_low,
             np.percentile(cv, p_low)))
    return float(np.percentile(cv, p_low)), float(np.percentile(cv, p_high))


def pick_one(band, cv_target, short_frac, mean_basis):
    """带内两段式选一天：CV 邻域 → 日均最接近带中位（或带中点）。"""
    d = band.copy()
    d['cv_dist'] = (d['cv'] - cv_target).abs() / max(cv_target, 1e-12)
    k = max(1, int(round(short_frac * len(d))))
    near = d.nsmallest(k, ['cv_dist', 'iloc'])
    if mean_basis == 'center':
        center = 0.5 * (band['mean'].min() + band['mean'].max())
    else:
        center = float(band['mean'].median())
    near = near.assign(mean_dist=(near['mean'] - center).abs())
    near = near.sort_values(['mean_dist', 'cv_dist', 'iloc'])
    return near.iloc[0], center, near


def select(stats, cv_lo, cv_hi, short_frac, mean_basis):
    """按 SPECS 挑 4 天；返回 [{...}]。

    目标 = 全池 P10 / P90，但先夹到该带**实际可达**的 CV 区间 ⇒ 不可达时退化为
    "带内最平/最抖的那一天"（`clamped=True`，会打印）。
    """
    cv_all = stats['cv'].to_numpy()
    picks = []
    for tag, label, band, var in SPECS:
        cand = band_of(stats, band)
        if len(cand) == 0:
            raise SystemExit('带 %r 内没有任何一天 —— 请放宽 --low-lo/--low-hi' % band)
        band_lo, band_hi = float(cand['cv'].min()), float(cand['cv'].max())
        pool_target = cv_lo if var == 'low' else cv_hi
        target = float(np.clip(pool_target, band_lo, band_hi))
        row, center, near = pick_one(cand, target, short_frac, mean_basis)
        picks.append(dict(
            tag=tag, label=label, band=band, var=var,
            iloc=int(row['iloc']), col=str(row['col']),
            mean=float(row['mean']), std=float(row['std']), cv=float(row['cv']),
            pool_target=pool_target, cv_target=target,
            clamped=bool(abs(target - pool_target) > 1e-12),
            band_cv_lo=band_lo, band_cv_hi=band_hi,
            cv_pct=float((cv_all <= row['cv']).mean() * 100.0),
            band_pct=float((cand['cv'] <= row['cv']).mean() * 100.0),
            mean_center=center, mean_basis=mean_basis,
            n_band=int(len(cand)), n_near=int(len(near)),
            shortlist=near.head(3).to_dict('records'),
        ))
    return picks


# ===========================================================================
# 打印 / 落盘
# ===========================================================================
def print_report(df, picks):
    print('\n' + '=' * 92)
    print('  选中的四天')
    print('=' * 92)
    print('  {:<14s} {:>6s} {:>5s} {:>8s} {:>7s} {:>8s} {:>9s} {:>9s}'
          .format('tag', 'iloc', '列名', '日均', 'std', 'CV', '带内分位', '全池分位'))
    print('  ' + '-' * 88)
    for p in picks:
        print('  {:<14s} {:>6d} {:>5s} {:>8.3f} {:>7.3f} {:>8.4f} {:>8.0f}% {:>8.0f}%'
              .format(p['tag'], p['iloc'], p['col'], p['mean'], p['std'], p['cv'],
                      p['band_pct'], p['cv_pct']))

    print('\n  判据明细（两段式：先 CV 邻域，再取最接近带风速中位的一天）')
    print('  ' + '-' * 88)
    for p in picks:
        tgt = '全池 P10' if p['var'] == 'low' else '全池 P90'
        print('  %-14s %s  带内 %d 天，CV 可达 [%.4f, %.4f]'
              % (p['tag'], p['label'], p['n_band'], p['band_cv_lo'], p['band_cv_hi']))
        print('  %-14s 目标 %s = %.4f%s → 采用 CV* = %.4f'
              % ('', tgt, p['pool_target'],
                 '（带内不可达，截断到下限/上限）' if p['clamped'] else '（可达）',
                 p['cv_target']))
        print('  %-14s CV 邻域 %d 天 → 取日均最接近 %s %.2f 的一天：'
              'iloc=%d 日均 %.3f std %.3f CV %.4f'
              % ('', p['n_near'],
                 '带内风速中点' if p['mean_basis'] == 'center' else '带内风速中位',
                 p['mean_center'], p['iloc'], p['mean'], p['std'], p['cv']))
        alt = ' | '.join('iloc %d(日均%.2f/CV%.4f)' % (r['iloc'], r['mean'], r['cv'])
                         for r in p['shortlist'])
        print('  %-14s 同邻域其它候选：%s' % ('', alt))

    print('\n  可直接粘贴到评测脚本：')
    print('  FROZEN_DAYS = {')
    for p in picks:
        print("      %-20r: %4d,   # 日均 %.3f m/s, std %.3f, CV %.4f (带内P%.0f/全池P%.0f), 列名 '%s'"
              % (p['tag'], p['iloc'], p['mean'], p['std'], p['cv'],
                 p['band_pct'], p['cv_pct'], p['col']))
    print('  }')
    print('  DAY_ILOCS   = [%s]' % ', '.join(str(p['iloc']) for p in picks))


def save_csv(out_dir, picks):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'four_days.csv')
    cols = ['tag', 'label', 'band', 'var', 'iloc', 'col', 'mean', 'std', 'cv',
            'band_pct', 'cv_pct', 'cv_target', 'pool_target', 'clamped',
            'band_cv_lo', 'band_cv_hi', 'mean_center', 'n_band', 'n_near',
            'min', 'max']
    rows = []
    for p in picks:
        r = dict(p)
        s = SERIES[p['iloc']]
        r['min'], r['max'] = float(s.min()), float(s.max())
        rows.append(r)
    pd.DataFrame(rows)[cols].to_csv(path, index=False, encoding='utf-8-sig')
    print('\n  四天清单 : %s' % path)
    return path


# ===========================================================================
# 画图
# ===========================================================================
#: iloc → 日内风速序列（save_csv / 画图共用；main 里填充）
SERIES = {}


def _setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ('Microsoft YaHei', 'SimHei', 'SimSun'):
        if cand in names:
            matplotlib.rcParams['font.sans-serif'] = [cand]
            break
    matplotlib.rcParams['axes.unicode_minus'] = False
    return plt


def plot_days(out_dir, picks, dpi):
    plt = _setup_matplotlib()
    t = np.arange(NB_POINTS) / 60.0                      # 小时
    colors = ('#1f77b4', '#d62728', '#2ca02c', '#ff7f0e')

    # ---- ① 2×2 逐日 ----
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6), sharex=True)
    # 四格共用同一纵轴，否则"抖不抖"会被各自的自动缩放夸大/抹平
    all_u = np.concatenate([SERIES[p['iloc']].to_numpy(dtype=float) for p in picks])
    ylo = max(0.0, np.floor(all_u.min() - 1.0))
    yhi = np.ceil(all_u.max() + 1.0)
    for ax, p, c in zip(axes.ravel(), picks, colors):
        u = SERIES[p['iloc']].to_numpy(dtype=float)
        ax.plot(t, u, color=c, lw=0.9, alpha=0.9)
        ax.axhline(p['mean'], color='k', ls='--', lw=0.9)
        ax.fill_between(t, p['mean'] - p['std'], p['mean'] + p['std'],
                        color=c, alpha=0.12, lw=0)
        ax.set_title('%s   |   iloc=%d (列 %s)\n日均 %.2f m/s   std %.2f   CV %.3f (带内P%.0f / 全池P%.0f)'
                     % (p['label'], p['iloc'], p['col'], p['mean'], p['std'],
                        p['cv'], p['band_pct'], p['cv_pct']), fontsize=10)
        ax.set_ylim(ylo, yhi)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_ylabel('风速 (m/s)')
        ax.text(0.99, 0.03, 'mean 虚线 / ±1σ 阴影', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='0.35')
    for ax in axes[-1]:
        ax.set_xlabel('日内时间 (h)')
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
    fig.suptitle('四天风速（%d 分钟/天，iloc 口径同 env.py）' % NB_POINTS, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = os.path.join(out_dir, 'four_days.png')
    fig.savefig(p1, dpi=dpi)
    plt.close(fig)
    print('  逐日图   : %s' % p1)

    # ---- ② 叠加对照（原始半透明 + 30 分钟滑动均值）----
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    for p, c in zip(picks, colors):
        u = SERIES[p['iloc']].to_numpy(dtype=float)
        sm = pd.Series(u).rolling(30, center=True, min_periods=1).mean().to_numpy()
        ax.plot(t, u, color=c, lw=0.5, alpha=0.22)
        ax.plot(t, sm, color=c, lw=1.8,
                label='%s  iloc=%d 日均%.2f σ%.2f CV%.3f' %
                      (p['label'], p['iloc'], p['mean'], p['std'], p['cv']))
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel('日内时间 (h)')
    ax.set_ylabel('风速 (m/s)')
    ax.set_title('四天风速叠加（粗线 = 30 分钟滑动均值；细线 = 原始）', fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    fig.tight_layout()
    p2 = os.path.join(out_dir, 'four_days_overlay.png')
    fig.savefig(p2, dpi=dpi)
    plt.close(fig)
    print('  叠加图   : %s' % p2)
    return p1, p2


# ===========================================================================
# 主流程
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='按日均风速档 × 相对离散度(CV) 从风速池挑 4 天并出图')
    p.add_argument('--csv', default=WIND_CSV, help='风速池（默认 %s）' % WIND_CSV)
    p.add_argument('--low-lo', type=float, default=LOW_LO, help='低风速带下界（默认 5）')
    p.add_argument('--low-hi', type=float, default=LOW_HI, help='低风速带上界（默认 10，含下不含上）')
    p.add_argument('--p-low', type=float, default=10.0, help='低方差目标分位（默认 P10）')
    p.add_argument('--p-high', type=float, default=90.0, help='高方差目标分位（默认 P90）')
    p.add_argument('--short-frac', type=float, default=0.10,
                   help='带内 CV 邻域比例（默认 0.10 = 带内最近 10%% 的天数作为候选）')
    p.add_argument('--mean-basis', choices=['median', 'center'], default='median',
                   help='邻域内挑"代表日"的风速基准：median=带内日均中位数（默认，'
                        '两臂风速可配对）/ center=带内日均区间中点')
    p.add_argument('--out', default=os.path.join('output', 'select_4_wind_days'),
                   help='输出目录（默认 output/select_4_wind_days）')
    p.add_argument('--dpi', type=int, default=130, help='图片 DPI（默认 130）')
    p.add_argument('--no-plot', action='store_true', help='只算/打印，不出图')
    p.add_argument('--dry-run', action='store_true', help='只打印，不出图也不写盘')
    return p.parse_args()


def main():
    args = parse_args()
    global LOW_LO, LOW_HI
    LOW_LO, LOW_HI = float(args.low_lo), float(args.low_hi)

    df, stats = load_pool(args.csv)
    cv_lo, cv_hi = describe_pool(stats, args.p_low, args.p_high)
    picks = select(stats, cv_lo, cv_hi, args.short_frac, args.mean_basis)
    print_report(df, picks)

    if args.dry_run:
        print('\n[dry-run] 不出图、不写盘。')
        return 0

    for p in picks:
        SERIES[p['iloc']] = df.iloc[:, p['iloc']].reset_index(drop=True)

    out_dir = os.path.abspath(args.out)
    save_csv(out_dir, picks)
    if not args.no_plot:
        plot_days(out_dir, picks, args.dpi)
    return 0


if __name__ == '__main__':
    sys.exit(main())
