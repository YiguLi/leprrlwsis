#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""五步波动任务对照：`f5/f5u3/f5n` 单模型、三者平均、`E[F5|u1]` 基线。

输出 test/real 的 R²/RMSE/MAE/MAPE，以及 test 上的配对 bootstrap：
  * 最佳单模型 vs `E[F5|u1]`（动作是否真的带来信息）；
  * `f5u3` vs `f5n`（目标参数化差异是否显著）；
  * 三模型平均 vs 最佳单模型（集成是否有收益）。

用法
----
    python agentutil/critictest/compare_fluct.py
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENTUTIL = os.path.dirname(_HERE)
ROOT = os.path.dirname(_AGENTUTIL)
os.chdir(ROOT)
for _p in (ROOT, _AGENTUTIL, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import common as C  # noqa: E402
import evaluate as E  # noqa: E402

ARMS = (('f5', 'f5_f5_16', 'f5'),
        ('f5u3', 'f5_f5u3_16', 'f5u3'),
        ('f5n', 'f5_f5n_16', 'f5n'))


def _boot(y, a, b, n_boot, seed=0):
    rng = np.random.RandomState(seed)
    n = y.size
    d = []
    for _ in range(int(n_boot)):
        idx = rng.randint(0, n, n)
        d.append(np.sqrt(np.mean((a[idx] - y[idx]) ** 2))
                 - np.sqrt(np.mean((b[idx] - y[idx]) ** 2)))
    d = np.asarray(d)
    return dict(mean=float(d.mean()),
                ci95=[float(np.percentile(d, 2.5)),
                      float(np.percentile(d, 97.5))])


def main():
    ap = argparse.ArgumentParser(description='五步波动任务方法对照')
    ap.add_argument('--data', default='agentutil/critictest/data/aifflor_ua_f5.npz')
    ap.add_argument('--n_boot', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    ds = np.load(args.data, allow_pickle=False)
    if 'f5' not in ds.files:
        raise SystemExit('这不是五步波动数据集（缺少 f5 列）')
    if ds['a'].ndim != 2 or ds['a'].shape[1] != C.NT:
        raise SystemExit('compare_fluct.py 只用于 5D 独立动作；'
                         '标量动作请直接用 evaluate.py 对比各 checkpoint')
    u = ds['u'].astype(np.float64)
    a = ds['a'].astype(np.float64)
    f5 = ds['f5'].astype(np.float64)
    split = ds['split']
    norm = C.window_fluct_norm(ds)

    preds = {}
    for tag, path, tgt in ARMS:
        q, _qt, meta = E.load_checkpoint(
            'agentutil/critictest/results/%s/critic_best.pt' % path)
        n = norm if tgt == 'f5n' else None
        preds[tag] = C.predict_power(q, u, a, target=tgt, feature='u10',
                                     window=5, norm=n)
    preds['ens3'] = np.mean([preds[t] for t in ('f5', 'f5u3', 'f5n')], axis=0)
    edges = np.arange(C.U_MIN, C.U_MAX + 0.5, 0.5)
    _, means = C.conditional_bin_mean(u[split == 0], f5[split == 0], edges)
    preds['base'] = C.bin_lookup(u, edges, means)

    out = dict(data=os.path.abspath(args.data), splits={}, bootstrap={})
    for name, code in (('test', 2), ('real', 3)):
        m = split == code
        print('\n[%s]' % name)
        out['splits'][name] = {}
        for tag in ('f5', 'f5u3', 'f5n', 'ens3', 'base'):
            r = C.regression_metrics(preds[tag][m], f5[m])
            out['splits'][name][tag] = r
            print('  %-6s n=%6d R²=%.6f RMSE=%.4f MAE=%.4f MAPE=%.3f%%'
                  % (tag, r['n'], r['r2'], r['rmse'], r['mae'], r['mape']))

    y = f5[split == 2]
    idx = np.flatnonzero(split == 2)
    print('\n[test paired bootstrap]')
    for A, Bn, label in (('f5u3', 'base', 'f5u3 vs E[F5|u1]'),
                         ('f5n', 'base', 'f5n vs E[F5|u1]'),
                         ('f5u3', 'f5n', 'f5u3 vs f5n'),
                         ('ens3', 'f5u3', 'ens3 vs f5u3')):
        d = _boot(y, preds[A][idx], preds[Bn][idx], args.n_boot, args.seed)
        out['bootstrap'][label] = d
        print('  RMSE(%s) − RMSE(%s) = %+.4f MW  95%%CI [%+.4f, %+.4f]'
              % (A, Bn, d['mean'], d['ci95'][0], d['ci95'][1]))

    if args.out:
        C.save_json(args.out, out)
        print('\n保存 %s' % os.path.abspath(args.out))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
