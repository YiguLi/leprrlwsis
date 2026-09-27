# -*- coding: utf-8 -*-
"""对比多个 run 的 test9 评测 (output/TD3PER-*/df_test9.csv)。

为什么需要它
------------
df_test9.csv 的结构容易被误读：

  * 每次评测写 **10 行** = 9 个场景 (wind_level x price_level, 各 5 步 = 1 块)
    + 1 行 OVERALL。直接对整列求均值/极差会把 9 个量级完全不同的场景混在一起
    (极差可达 400%+)，毫无意义。
  * 只有 OVERALL 行是"整链"汇总，跨 run / 跨 step 对比必须只用 OVERALL。
  * profit_econ = profit **+** 580 * dsoc (580 = SOC_SETTLE_PR)。
    期末 SOC 比期初多 = 按 580 计价入账，少 = 扣账。于是"电量的借还"被单独结算：
    一个策略可以 raw profit 更高但 econ 更低（它把电池存量能量卖掉了）。
    看到 econ 差距时**必须先拆开这一项**，否则会把"放电深度不同"误判成"策略更差"。
    注意符号：dsoc>0 时 econ > profit，dsoc<0 时 econ < profit。
  * 9 个场景串成一条 9 块的链 (第 k 个场景的 soc_start = 第 k-1 个的 soc_end)，
    所以整条链只有 9 次决策 —— 动态范围极小：未训练策略 (step 480) 已经能拿到
    训练后 95%+ 的 profit_econ。用 test9 判断"策略好坏"信噪比很低，
    判 run 间优劣请优先用 test_five_days.py (1440 步/天, SOC 有时间回收)。

用法
----
    python agentutil/test9_compare.py run12 run13
    python agentutil/test9_compare.py --window 16320 27360 run12 run13
    python agentutil/test9_compare.py --tail 24 TD3PER-run12 output/TD3PER-run13

参数
----
位置参数: run 名 (自动补 output/TD3PER-<name>/df_test9.csv) 或直接给 csv 路径/目录。
--window LO HI : 同 step 窗口稳定性对比 (默认取各 run 公共 step 区间的后 1/3)。
--tail N       : 稳定性统计用最近 N 次评测 (默认 24)。
--settle PR    : SOC 结算价 (默认 580，须与 myfunc.SOC_SETTLE_PR 一致)。
--scenarios    : 额外打印末次评测的逐场景明细。
"""

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块
sys.path.insert(0, _HERE)                              # agentutil 内互相 import

try:
    import numpy as np
except ImportError:                                    # pragma: no cover
    sys.exit('需要 numpy')

# 与 myfunc.SOC_SETTLE_PR 保持一致；改了请用 --settle 覆盖
DEFAULT_SETTLE = 580.0
OVERALL = 'OVERALL'


def resolve(arg):
    """把 run 名 / 目录 / csv 路径统一解析成 (显示名, csv 路径)。"""
    cands = []
    if arg.endswith('.csv'):
        cands.append((os.path.basename(os.path.dirname(arg)) or arg, arg))
    else:
        d = arg if os.path.isdir(arg) and 'TD3PER-' in os.path.basename(arg.rstrip('/\\')) \
            else os.path.join('output', 'TD3PER-' + arg)
        cands.append((os.path.basename(d.rstrip('/\\')), os.path.join(d, 'df_test9.csv')))
    for name, p in cands:
        if os.path.exists(p):
            return name, p
    raise SystemExit('找不到 test9 结果: %s (试过 %s)' % (arg, ', '.join(p for _, p in cands)))


def load(path):
    """返回 (scenarios, overall)。

    scenarios[step] = [(wind, price, profit, profit_econ, dsoc, soc_err), ...]
    overall[step]   = (profit_econ, profit, dsoc, soc_err)
    """
    with open(path, newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        rows = list(rd)
    ix = {c: i for i, c in enumerate(head)}
    need = ('step', 'wind_level', 'price_level', 'profit', 'profit_econ', 'dsoc', 'soc_track_err')
    miss = [c for c in need if c not in ix]
    if miss:
        raise SystemExit('%s 缺列: %s' % (path, miss))

    def g(r, c):
        try:
            return float(r[ix[c]])
        except (ValueError, IndexError):
            return float('nan')

    sc, ov = {}, {}
    for r in rows:
        if not r or len(r) < len(head):
            continue
        try:
            st = int(float(r[ix['step']]))
        except (ValueError, IndexError):
            continue
        w, p = r[ix['wind_level']], r[ix['price_level']]
        if w == OVERALL:
            ov[st] = (g(r, 'profit_econ'), g(r, 'profit'), g(r, 'dsoc'), g(r, 'soc_track_err'),
                      g(r, 'r1'))
        else:
            sc.setdefault(st, []).append((w, p, g(r, 'profit'), g(r, 'profit_econ'),
                                          g(r, 'dsoc'), g(r, 'soc_track_err')))
    return sc, ov


def arr(ov, steps, k):
    return np.array([ov[s][k] for s in steps], dtype=float)


def spread(v):
    m = np.nanmean(v)
    if not np.isfinite(m) or abs(m) < 1e-12:
        return float('nan')
    return 100.0 * (np.nanmax(v) - np.nanmin(v)) / abs(m)


def describe(tag, ov, steps, settle, label=''):
    """打印一个 run 在给定 step 集合上的稳定性摘要。"""
    if not steps:
        print('  %-10s (无数据)' % tag)
        return None
    pe, pr, ds, se = (arr(ov, steps, k) for k in range(4))
    print('  %-10s n=%2d step %5d..%5d | profit_econ %8.1f sd %6.1f 极差 %5.2f%%'
          % (tag, len(steps), steps[0], steps[-1], np.nanmean(pe), np.nanstd(pe), spread(pe)))
    print('  %-10s %s  原始 profit %8.1f sd %5.1f | dsoc %+.3f±%.3f | soc_err %.2f±%.2f'
          % ('', label, np.nanmean(pr), np.nanstd(pr), np.nanmean(ds), np.nanstd(ds),
             np.nanmean(se), np.nanstd(se)))
    return dict(pe=np.nanmean(pe), pr=np.nanmean(pr), ds=np.nanmean(ds), se=np.nanmean(se),
                spread=spread(pe))


def main():
    ap = argparse.ArgumentParser(description='对比多个 run 的 test9 结果（只用 OVERALL 行）',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='run 名 / 模型目录 / df_test9.csv 路径')
    ap.add_argument('--window', nargs=2, type=int, default=None, metavar=('LO', 'HI'),
                    help='同 step 窗口对比（默认各 run 公共区间的后 1/3）')
    ap.add_argument('--tail', type=int, default=24, help='稳定性统计取最近 N 次评测（默认 24）')
    ap.add_argument('--settle', type=float, default=DEFAULT_SETTLE, help='SOC 结算价（默认 580）')
    ap.add_argument('--scenarios', action='store_true', help='打印末次评测的逐场景明细')
    args = ap.parse_args()

    data = []
    for a in args.runs:
        name, path = resolve(a)
        sc, ov = load(path)
        data.append((name, sc, ov))
        # 自洽校验: profit_econ 应当等于 profit + settle*dsoc
        err = max(abs(ov[s][0] - (ov[s][1] + args.settle * ov[s][2])) for s in ov)
        flag = '' if err < 0.5 else '  <== 与 --settle 不符, 请核对 SOC 结算价!'
        print('=== %s : %d 次评测, step %s..%s, 每次 %d 场景 + OVERALL   [结算自洽 max 偏差 %.3f]%s'
              % (name, len(ov), min(ov), max(ov), len(sc[max(sc)]), err, flag))

    # ---- 1) 同 step 对齐 ----
    common = sorted(set.intersection(*[set(ov) for _, _, ov in data]))
    if common:
        print('\n[1] 同 step 对齐 (末 6 个公共评测点)')
        hdr = '  %-7s' % 'step'
        for nm, _, _ in data:
            hdr += ' %12s' % nm
        print(hdr + '   (profit_econ | dsoc)')
        for st in common[-6:]:
            line = '  %-7d' % st
            for _, _, ov in data:
                line += ' %12.1f' % ov[st][0]
            line += '   '
            for _, _, ov in data:
                line += ' dsoc %+.3f' % ov[st][2]
            print(line)
    else:
        print('\n[1] 无公共 step（各 run 评测点不重合）')

    # ---- 2) 同窗口稳定性 ----
    print('\n[2] 同 step 窗口稳定性')
    if args.window:
        lo, hi = args.window
    elif common:
        lo = common[len(common) // 3]
        hi = common[-1]
    else:
        lo, hi = None, None
    if lo is not None:
        print('  窗口 %d..%d:' % (lo, hi))
        for nm, _, ov in data:
            steps = [s for s in sorted(ov) if lo <= s <= hi]
            describe(nm, ov, steps, args.settle, label='')
    for nm, _, ov in data:
        steps = sorted(ov)[-args.tail:]
        print('  最近 %d 次评测:' % len(steps))
        describe(nm, ov, steps, args.settle, label='')

    # ---- 3) SOC 结算拆解 ----
    print('\n[3] 末次评测: SOC 结算拆解  (profit_econ = profit + %.0f*dsoc)' % args.settle)
    print('  %-10s %8s %12s %12s %12s %9s' % ('run', 'step', 'raw profit', '%.0f*dsoc' % args.settle,
                                              'profit_econ', 'soc_err'))
    base = None
    for nm, _, ov in data:
        s = max(ov)
        pe, pr, ds, se = ov[s][:4]
        print('  %-10s %8d %12.1f %+12.1f %12.1f %9.2f' % (nm, s, pr, args.settle * ds, pe, se))
        if base is None:
            base = (nm, pe, pr, args.settle * ds)
        else:
            gap = base[1] - pe
            raw = base[2] - pr
            stl = base[3] - args.settle * ds
            print('  %-10s %8s 与 %s 的 econ 差 %+.1f = raw 差 %+.1f + 结算差 %+.1f'
                  % ('', '', base[0], gap, raw, stl))
            if abs(gap) > 1e-9:
                print('  %-10s %8s => 结算项解释了 econ 差的 %.0f%%'
                      % ('', '', 100.0 * stl / gap))

    print('\n[4] test9 动态范围 (step480 未训练 vs 最好) —— 范围越小，判别力越弱')
    for nm, _, ov in data:
        steps = sorted(ov)
        pe = arr(ov, steps, 0)
        print('  %-10s step480 %8.1f -> best %8.1f (%+.2f%%)  最差 %8.1f'
              % (nm, pe[0], np.nanmax(pe), 100.0 * (np.nanmax(pe) - pe[0]) / abs(pe[0]), np.nanmin(pe)))

    # ---- 5) 逐场景明细 ----
    if args.scenarios:
        print('\n[5] 末次评测逐场景 (profit_econ / dsoc)')
        for nm, sc, ov in data:
            s = max(ov)
            print('  --- %s @ step %d ---' % (nm, s))
            for w, p, pr, pe, ds, se in sorted(sc.get(s, []), key=lambda x: (x[0], x[1])):
                print('    %-6s %-6s profit_econ %9.1f  dsoc %+.3f  soc_err %.2f' % (w, p, pe, ds, se))


    # ---- 6) r1 与 profit 的分解 ----
    print('\n[6] r1 vs profit 分解 —— 二者是同一笔钱的不同包装:')
    print('    r1     = revenue - degradation - penalty_NN_H   (每步量, 见 td3per_myfunc.Cal_r1_continuous)')
    print('    profit = 45 步现金流总和 ;  profit_econ = profit + settle*dsoc')
    print('    => profit/45 - r1 = penalty_NN_H 的每步均值 >= 0  (动作越界罚, 不进钱包)')
    print('    含义: r1 抖而 profit/econ 稳 ==> 抖的是越界罚, 经济结果稳定。')
    print('    判经济用 profit/profit_econ; 判 RL 奖励用 r1; 不要混用。')
    print('  %-12s %8s %9s %9s %9s %11s %11s' % ('run', 'n', 'r1 cv%', 'profit cv%', 'econ cv%',
                                                '罚/步 均值', '罚/步 max'))
    for name, _, ov in data:
        steps = sorted(ov)
        if args.window:
            lo, hi = args.window
            steps = [s for s in steps if lo <= s <= hi]
        else:
            steps = steps[-args.tail:]
        if not steps:
            print('  %-12s (无数据)' % name)
            continue
        pe, pr, r1 = arr(ov, steps, 0), arr(ov, steps, 1), arr(ov, steps, 4)
        pen = pr / 45.0 - r1
        cv = lambda v: 100 * np.nanstd(v) / abs(np.nanmean(v)) if np.nanmean(v) else float('nan')
        bad = int(np.nansum(pen < -1e-6))
        print('  %-12s %8d %9.2f %9.2f %9.2f %11.3f %11.3f%s'
              % (name, len(steps), cv(r1), cv(pr), cv(pe), np.nanmean(pen), np.nanmax(pen),
                 '   <== 出现负残差, 恒等式不成立!' if bad else ''))


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
