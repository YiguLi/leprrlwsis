#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""把多个 `train_c1.py` 的结果目录并排对比（成本精度 + 行为梯度）。

用法::

    python agentutil/critictest/compare_c1_arch.py \
        v6=agentutil/critictest/results/c1_obs_v6_log \
        v7=agentutil/critictest/results/c1_obs_v7_log

每个参数是 `名字=结果目录`（只给目录时名字 = 目录名）。目录里需要有
`metrics.json`（`train_c1.py` 末尾调 `evaluate_c1.evaluate_all` 生成）。

报告口径（与 README_c1.md 一致）：
  * 成本：test/real 的 R²、RMSE、MAE、log1p R²；
  * 折点检出：`C1>15` 的 AUC / F1；
  * 行为梯度 `∂C1/∂a`：pooled pearson / slope / sign_agree / 余弦 mean /
    余弦 >0.8 比例 / 余弦 <0 比例；
  * 折点分组（smooth = 未触发 PF>3，hinge = 触发）。
"""
import argparse
import json
import math
import os
import sys

WIDTH = 18


def load_run(spec):
    if '=' in spec:
        name, path = spec.split('=', 1)
    else:
        name, path = os.path.basename(os.path.normpath(spec)), spec
    with open(os.path.join(path, 'metrics.json'), encoding='utf-8') as f:
        m = json.load(f)
    return name, path, m


def get(m, *keys):
    """从 metrics.json 的嵌套 dict 里取一个数；缺失 ⇒ NaN。"""
    node = m
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return float('nan')
        node = node[k]
    try:
        return float(node)
    except (TypeError, ValueError):
        return float('nan')


def show(rows, title, specs):
    print('\n' + title)
    print('  %-12s' % '量' + ''.join('%*s' % (WIDTH, n) for n, _, _ in rows))
    print('  ' + '-' * (12 + WIDTH * len(rows)))
    for label, nd, keys in specs:
        line = '  %-12s' % label
        for name, path, m in rows:
            v = get(m, *keys)
            line += ('%*s' % (WIDTH, '-') if math.isnan(v)
                     else '%*.*f' % (WIDTH, nd, v))
        print(line)


def main():
    ap = argparse.ArgumentParser(description='对比多个 C1 critic 结果目录')
    ap.add_argument('runs', nargs='+', help='名字=目录 ...')
    a = ap.parse_args()
    rows = [load_run(s) for s in a.runs]

    print('=' * (12 + WIDTH * len(rows)))
    print('C1 critic 结构对比（%d 个 run）' % len(rows))
    print('=' * (12 + WIDTH * len(rows)))
    for name, path, m in rows:
        print('  %-10s arch=%-5s state=%s target=%s 参数=%d'
              % (name, m.get('arch', '?'), m.get('state_kind'),
                 m.get('target_kind'), m.get('n_params', 0)))
        print('  %-10s %s' % ('', os.path.abspath(path)))

    show(rows, '成本（split=test / real）', [
        ('test R²', 4, ('splits', 'test', 'r2')),
        ('test RMSE', 3, ('splits', 'test', 'rmse')),
        ('test MAE', 3, ('splits', 'test', 'mae')),
        ('test MAPE%', 2, ('splits', 'test', 'mape')),
        ('test log R²', 4, ('splits', 'test', 'log_r2')),
        ('real R²', 4, ('splits', 'real', 'r2')),
        ('real RMSE', 3, ('splits', 'real', 'rmse')),
    ])
    show(rows, '折点检出（C1>15，test）', [
        ('AUC', 4, ('detection', 'auc')),
        ('F1', 4, ('detection', 'f1')),
        ('precision', 4, ('detection', 'precision')),
        ('recall', 4, ('detection', 'recall')),
    ])
    show(rows, '行为梯度 ∂C1/∂a（test 400 点，δ=0.01）', [
        ('pearson r', 4, ('gradient', 'pooled', 'pearson')),
        ('slope', 4, ('gradient', 'pooled', 'slope')),
        ('sign agree', 4, ('gradient', 'pooled', 'sign_agree')),
        ('cos mean', 4, ('gradient', 'cosine', 'mean')),
        ('cos >0.8', 4, ('gradient', 'cosine', 'gt08')),
        ('cos <0', 4, ('gradient', 'cosine', 'neg')),
    ])
    for grp in ('smooth', 'hinge'):
        show(rows, '行为梯度 · %s 组' % grp, [
            ('n', 0, ('gradient', 'groups', grp, 'pooled', 'n')),
            ('pearson r', 4, ('gradient', 'groups', grp, 'pooled', 'pearson')),
            ('slope', 4, ('gradient', 'groups', grp, 'pooled', 'slope')),
            ('sign agree', 4,
             ('gradient', 'groups', grp, 'pooled', 'sign_agree')),
            ('cos mean', 4, ('gradient', 'groups', grp, 'cosine', 'mean')),
            ('cos <0', 4, ('gradient', 'groups', grp, 'cosine', 'neg')),
        ])
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
