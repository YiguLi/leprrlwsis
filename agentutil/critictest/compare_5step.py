#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""五步任务的对照脚本：`sp5n`（直接 critic）vs `factorized`（单步 s × 风速统计）。

`factorized` 的口径（只作对照，critic 本身预测的是单步 s，不是五步和）：
    s(u1,a) = 1000·P_single(u1,a)/u1³        （用已训好的单步 `sp` critic）
    P5̂      = s(u1,a) · E[Σu³|u1] / 1000
其中 `E[Σu³|u1]` 由五步数据 train split 的窗口分箱估计，只依赖 u1。

同时给出 test 上的配对 bootstrap（RMSE 差、95% CI），判断差异是否显著。

用法
----
    python agentutil/critictest/compare_5step.py
    python agentutil/critictest/compare_5step.py \
        --single_ckpt agentutil/critictest/results/sp/critic_best.pt \
        --compare_ckpt agentutil/critictest/results/p5_sp5n_16/critic_best.pt
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


def _load_ds(path):
    ds = np.load(path, allow_pickle=False)
    return ds, json.loads(str(ds['meta']))


def _metrics(pred, true):
    return C.regression_metrics(pred, true)


def _fmt(tag, r):
    print('  %-12s n=%6d R²=%.6f RMSE=%.4f MAE=%.4f MAPE=%.3f%%'
          % (tag, r['n'], r['r2'], r['rmse'], r['mae'], r['mape']))


def main():
    ap = argparse.ArgumentParser(description='五步任务 sp5n vs factorized 对照')
    ap.add_argument('--data', default='agentutil/critictest/data/aifflor_ua_p5.npz')
    ap.add_argument('--single_ckpt',
                    default='agentutil/critictest/results/sp/critic_best.pt',
                    help='单步 sp critic（factorized 用）')
    ap.add_argument('--compare_ckpt',
                    default='agentutil/critictest/results/p5_sp5n_16/critic_best.pt',
                    help='五步直接 critic（默认 sp5n 16/100）')
    ap.add_argument('--n_boot', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='JSON 输出；默认不写')
    args = ap.parse_args()

    ds, dm = _load_ds(args.data)
    window = int(dm.get('window', 5))
    if window <= 1:
        raise SystemExit('compare_5step.py 只用于 window>1 的数据集')
    u = ds['u'].astype(np.float64)
    a = ds['a'].astype(np.float64)
    p5 = ds['p'].astype(np.float64)
    split = ds['split']
    u_seq = ds['u_seq'].astype(np.float64)

    edges = np.arange(C.U_MIN, C.U_MAX + 0.5, 0.5)
    _, means = C.conditional_bin_mean(u_seq[split == 0, 0],
                                      C.u3_sum(u_seq[split == 0]), edges)

    # ---- factorized：单步 sp critic + E[sum u^3 | u1] ----
    qs, _qt, ms = E.load_checkpoint(args.single_ckpt)
    # ---- sp5n：直接五步 critic ----
    q5, _qt5, m5 = E.load_checkpoint(args.compare_ckpt)
    if int(m5.get('window', 1)) != window or m5.get('target') != 'sp5n':
        raise SystemExit('compare_ckpt 必须是 target=sp5n、window=%d 的 checkpoint'
                         % window)
    norm = C.window_norm(ds, window=window)

    print('=' * 100)
    print('五步任务对照：window=%d | data=%s' % (window, os.path.abspath(args.data)))
    print('  factorized 单步 critic: %s' % os.path.abspath(args.single_ckpt))
    print('  sp5n 五步 critic      : %s' % os.path.abspath(args.compare_ckpt))
    print('=' * 100)

    preds = {'factorized': {}, 'sp5n': {}}
    for name, code in (('train', 0), ('test', 2), ('real', 3)):
        m = split == code
        if not np.any(m):
            continue
        ps = C.predict_power(qs, u[m], a[m], target='sp', feature='u10')
        s = C.S_SCALE * ps / np.power(u[m], 3)
        e = C.bin_lookup(u[m], edges, means)
        preds['factorized'][name] = s * e / 1000.0
        preds['sp5n'][name] = C.predict_power(
            q5, u[m], a[m], target='sp5n', feature='u10',
            window=window, norm=norm[m])
        print('\n[%s]' % name)
        _fmt('factorized', _metrics(preds['factorized'][name], p5[m]))
        _fmt('sp5n', _metrics(preds['sp5n'][name], p5[m]))

    # ---- test 配对 bootstrap ----
    y = p5[split == 2]
    rng = np.random.RandomState(args.seed)
    n = y.size
    d = []
    for _ in range(int(args.n_boot)):
        idx = rng.randint(0, n, n)
        e1 = preds['sp5n']['test'][idx] - y[idx]
        e2 = preds['factorized']['test'][idx] - y[idx]
        d.append(np.sqrt(np.mean(e1 ** 2)) - np.sqrt(np.mean(e2 ** 2)))
    d = np.asarray(d)
    print('\n[test paired bootstrap] RMSE(sp5n) − RMSE(factorized) = %.4f MW '
          '(95%% CI [%.4f, %.4f])'
          % (d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)))

    out = dict(window=window, data=os.path.abspath(args.data),
               single_ckpt=os.path.abspath(args.single_ckpt),
               compare_ckpt=os.path.abspath(args.compare_ckpt),
               splits={name: {'factorized': _metrics(preds['factorized'][name],
                                                     p5[split == code]),
                              'sp5n': _metrics(preds['sp5n'][name],
                                               p5[split == code])}
                       for name, code in (('test', 2), ('real', 3))},
               bootstrap_rmse_diff=dict(mean=float(d.mean()),
                                        ci95=[float(np.percentile(d, 2.5)),
                                              float(np.percentile(d, 97.5))],
                                        n_boot=int(args.n_boot)))
    if args.out:
        C.save_json(args.out, out)
        print('保存 %s' % os.path.abspath(args.out))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
