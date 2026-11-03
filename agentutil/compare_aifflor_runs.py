#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""并排对比两个 aifflor run 的**测试期**指标（同 step 同场景 ⇒ 可配对）。

用法::

    python agentutil/compare_aifflor_runs.py \
        --a output/AIFFLOR-run16 --b output/AIFFLOR-V7-run20 \
        --label-a v6 --label-b v7

读两个文件（都来自 aifflor 主循环，CSV 格式相同）：
  * `df_test20_hsrl.csv`：每 480 步一次的 20 步测试协议，每次 5 个块；
    行内字段 wind_col/wind_start/price 决定场景 ⇒ 两个 run 在同 step 的场景
    **完全相同**（`hsrl_test_wind=fixed` 默认），所以可以逐 step 配对。
  * `df_hsrl.csv`：逐环境步日志（动作 / r1 / c1 / lam1 / SOC）。
  * `update_stats_H.csv`：critic 更新统计（q_c / td_c / reward / cost）。

只读、不需要 MATLAB/FLORIS。
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

TEST_METRICS = ('profit', 'profit_econ', 'c1', 'r1', 'viol', 'pf_viol', 'dsoc')


def read_csv(path):
    with open(path, encoding='utf-8', errors='replace', newline='') as f:
        return list(csv.DictReader(f))


def fnum(s, default=float('nan')):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def load_test(run):
    """step -> [5 行]"""
    out = defaultdict(list)
    for r in read_csv(os.path.join(run, 'df_test20_hsrl.csv')):
        out[int(fnum(r['step'], -1))].append(r)
    return dict(out)


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float('nan')


def sd(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2:
        return float('nan')
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_block(ta, tb, steps, keys=TEST_METRICS):
    """逐 step 配对（每个 step 先在 5 个块上取均值），返回 {key: 统计}。"""
    res = {}
    for k in keys:
        va, vb, d = [], [], []
        for s in steps:
            ma = mean([fnum(r[k]) for r in ta[s]])
            mb = mean([fnum(r[k]) for r in tb[s]])
            if math.isnan(ma) or math.isnan(mb):
                continue
            va.append(ma)
            vb.append(mb)
            d.append(mb - ma)
        n = len(d)
        if not n:
            res[k] = None
            continue
        res[k] = dict(n=n, a=mean(va), b=mean(vb), diff=mean(d),
                      se=sd(d) / math.sqrt(n) if n > 1 else float('nan'),
                      win=sum(1 for x in d if x > 0) / n,
                      sd_a=sd(va), sd_b=sd(vb))
    return res


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '%*s' % (12, '-')
    return '%*.{nd}f'.format(nd=nd) % (12, v)


def windows_of(steps):
    return [('early <40k', [s for s in steps if s < 40000]),
            ('mid 40–80k', [s for s in steps if 40000 <= s < 80000]),
            ('late ≥80k', [s for s in steps if s >= 80000])]


def print_paired(title, ta, tb, steps, la, lb):
    res = paired_block(ta, tb, steps)
    print('\n' + title)
    print('  %-12s %12s %12s %12s %12s %8s' % ('指标', la, lb, 'b−a', '±SE', 'b 胜率'))
    print('  ' + '-' * 76)
    for k in TEST_METRICS:
        r = res[k]
        if r is None:
            continue
        print('  %-12s %s %s %s %s %7.1f%%'
              % (k, fmt(r['a']), fmt(r['b']), fmt(r['diff'], 4),
                 fmt(r['se'], 4), 100 * r['win']))
    return res


def step_windows(run):
    """df_hsrl 的分段均值（只取非 warmup 的步：step>=7200）。"""
    rows = read_csv(os.path.join(run, 'df_hsrl.csv'))
    keys = ('r1', 'c1', 'lam1', 'SOC')
    out = {'n_rows': len(rows)}
    for lo, hi, tag in ((7200, 40000, 'warm→40k'), (40000, 80000, '40–80k'),
                        (80000, 130000, '80–130k'), (130000, 200000, '>130k')):
        sel = [r for r in rows if lo <= int(fnum(r['step'], -1)) < hi]
        if not sel:
            continue
        d = {k: mean([fnum(r.get(k)) for r in sel]) for k in keys}
        d['n'] = len(sel)
        out[tag] = d
    # 末 20k 步的逐维动作
    steps = [int(fnum(r['step'], -1)) for r in rows]
    if steps:
        cut = max(steps) - 20000
        sel = [r for r in rows if int(fnum(r['step'], -1)) >= cut]
        acts = []
        for r in sel[:2000]:
            txt = r.get('applied_action_H', '').strip().strip('[]')
            try:
                acts.append([float(x) for x in txt.split()])
            except ValueError:
                pass
        if acts:
            n = min(len(a) for a in acts)
            out['action_last20k'] = [mean([a[j] for a in acts]) for j in range(n)]
            out['action_last20k_n'] = len(acts)
    return out


def update_window(run, lo=80000, hi=200000):
    """update_stats_H 的末段均值。"""
    p = os.path.join(run, 'update_stats_H.csv')
    if not os.path.exists(p):
        return None
    rows = [r for r in read_csv(p)
            if lo <= int(fnum(r['step'], -1)) < hi and fnum(r.get('q_c_mean')) == fnum(r.get('q_c_mean'))]
    if not rows:
        return None
    keys = ('td_p_mean', 'td_c_mean', 'q_p_mean', 'q_c_mean', 'cost_mean', 'reward_mean')
    return {k: mean([fnum(r.get(k)) for r in rows]) for k in keys} | {'n': len(rows)}


def main():
    ap = argparse.ArgumentParser(description='aifflor 两 run 配对对比（只读）')
    ap.add_argument('--a', required=True, help='基线 run 目录（v6）')
    ap.add_argument('--b', required=True, help='新 run 目录（v7）')
    ap.add_argument('--label-a', default='a')
    ap.add_argument('--label-b', default='b')
    args = ap.parse_args()

    ta, tb = load_test(args.a), load_test(args.b)
    common = sorted(set(ta) & set(tb))
    print('=' * 84)
    print('aifflor 测试期配对对比 | %s=%s  vs  %s=%s'
          % (args.label_a, args.a, args.label_b, args.b))
    print('=' * 84)
    print('test20：%s %d 个测试点 [%d..%d] | %s %d 个 [%d..%d] | 共同 %d 个'
          % (args.label_a, len(ta), min(ta), max(ta),
             args.label_b, len(tb), min(tb), max(tb), len(common)))
    if not common:
        print('没有共同 step，无法配对')
        return 1
    scen_same = all(
        (ta[s][i]['wind_col'], ta[s][i]['wind_start'], ta[s][i]['price'])
        == (tb[s][i]['wind_col'], tb[s][i]['wind_start'], tb[s][i]['price'])
        for s in common for i in range(len(ta[s])))
    print('场景在共同 step 上逐块一致: %s' % scen_same)

    print_paired('A1. 全部共同测试点（%d × 5 块）' % len(common), ta, tb, common,
                 args.label_a, args.label_b)
    for tag, steps in windows_of(common):
        if steps:
            print_paired('A2. %s（%d 个测试点）' % (tag, len(steps)), ta, tb, steps,
                         args.label_a, args.label_b)

    print('\nB. 逐环境步（df_hsrl，非 warmup）')
    wa, wb = step_windows(args.a), step_windows(args.b)
    print('  %-12s %10s %10s %10s %10s' % ('窗口', 'r1', 'c1', 'lam1', 'SOC'))
    for tag in ('warm→40k', '40–80k', '80–130k', '>130k'):
        if tag in wa or tag in wb:
            for lab, w in ((args.label_a, wa), (args.label_b, wb)):
                d = w.get(tag)
                if not d:
                    continue
                print('  %-4s %-7s %10.3f %10.3f %10.3f %10.3f  (n=%d)'
                      % (lab, tag, d['r1'], d['c1'], d['lam1'], d['SOC'], d['n']))
    for lab, w in ((args.label_a, wa), (args.label_b, wb)):
        a = w.get('action_last20k')
        if a:
            print('  %-4s 末 20k 步动作均值: %s  (n=%d)'
                  % (lab, ' '.join('%.3f' % x for x in a), w['action_last20k_n']))

    print('\nC. critic 更新统计（update_stats_H，step ≥ 80k）')
    ua, ub = update_window(args.a), update_window(args.b)
    if ua and ub:
        print('  %-12s %12s %12s' % ('量', args.label_a, args.label_b))
        for k in ('td_p_mean', 'td_c_mean', 'q_p_mean', 'q_c_mean', 'cost_mean', 'reward_mean'):
            print('  %-12s %12.4f %12.4f' % (k, ua[k], ub[k]))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
