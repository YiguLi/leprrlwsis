#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""评价单步 PF critic：成本判别精度 + 行为梯度精度。

* 成本精度：R²/RMSE/MAE/MAPE、分 u_cur、分动作族、`PF>3 MW` 检出的
  AUC/Accuracy/Precision/Recall/F1；
* 基线：`E[PF|u_cur]` 与 `E[PF|u_prev,u_cur]`（分箱条件均值）；
* 梯度：autograd ∂PF/∂a vs aifflor 中心差分（逐样本余弦/斜率/符号）。
"""
import argparse
import csv
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
import torch  # noqa: E402

import common as C  # noqa: E402
import evaluate as E0  # noqa: E402
from train_pf import STATE_KINDS, TARGET_KINDS, build_state, predict

SPLIT_NAMES = ('train', 'val', 'test', 'real')
U_EDGES = (3.5, 5.0, 7.0, 9.0, 11.0, 13.0, 16.0, 20.0, 25.001)


def load_ckpt(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = ck['meta']
    q = C.make_critic(m['state_dim'], 1, m['hidden1'] if 'hidden1' in m else 16,
                      m['hidden2'] if 'hidden2' in m else 100,
                      seed=None, bias=m.get('bias0'))
    q.load_state_dict(ck['model'])
    q.eval()
    return q, m


def _bin2(up, uc, pf, edges):
    nb = len(edges) - 1
    iu = np.clip(np.digitize(up, edges) - 1, 0, nb - 1)
    iv = np.clip(np.digitize(uc, edges) - 1, 0, nb - 1)
    key = iu * nb + iv
    sums = np.bincount(key, weights=pf, minlength=nb * nb)
    cnts = np.bincount(key, minlength=nb * nb)
    means = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return key, means, nb


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n1 = int(labels.sum())
    n0 = int(labels.size - n1)
    if n0 == 0 or n1 == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(labels.size, dtype=float)
    ranks[order] = np.arange(1, labels.size + 1)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def _detect(pred, true, thr=3.0):
    lab = true > thr
    p = pred > thr
    tp = int(np.sum(p & lab)); fp = int(np.sum(p & ~lab))
    fn = int(np.sum(~p & lab)); tn = int(np.sum(~p & ~lab))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(threshold=thr, auc=_auc(lab, pred),
                accuracy=float((tp + tn) / max(lab.size, 1)),
                precision=float(prec), recall=float(rec), f1=float(f1),
                tp=tp, fp=fp, fn=fn, tn=tn)


def _cos_stats(gt, gp):
    cos = (gt * gp).sum(1) / (np.linalg.norm(gt, axis=1)
                              * np.linalg.norm(gp, axis=1) + 1e-12)
    nr = np.linalg.norm(gp, axis=1) / (np.linalg.norm(gt, axis=1) + 1e-12)
    return dict(mean=float(cos.mean()), median=float(np.median(cos)),
                gt05=float(np.mean(cos > 0.5)), gt08=float(np.mean(cos > 0.8)),
                neg=float(np.mean(cos < 0)),
                norm_ratio_median=float(np.median(nr)))


def gradient_check(model, ds, meta, n=400, delta=0.01, seed=123):
    up = ds['u_prev'].astype(np.float64)
    uc = ds['u_cur'].astype(np.float64)
    ae = ds['a_eff'].astype(np.float64)
    pf = ds['pf'].astype(np.float64)
    split = ds['split']
    state = build_state(ds, meta['state_kind'])
    cand = ((split == 2) & (ae[:, 0] >= 0.05) & (ae[:, 0] <= 0.45)
            & (pf >= 0.5))
    idx = np.flatnonzero(cand)
    if idx.size < n:
        idx = np.flatnonzero((split == 2) & (pf >= 0.5))
    rng = np.random.RandomState(seed)
    sel = rng.choice(idx, size=min(n, idx.size), replace=False)

    upg, ucg, aeg = up[sel], uc[sel], ae[sel]
    phys = C.build_physics(calibrated=False, power_scale=C.AT.LEVEL_COEF)
    gt = np.zeros(sel.size, dtype=np.float64)
    lo, hi = C.A_PHYS_LO, C.A_HI
    for _ in (0,):
        ap = np.clip(aeg + delta, lo, hi)
        am = np.clip(aeg - delta, lo, hi)
        ppp, _ = C.power_of(phys, upg, ap)
        pcp, _ = C.power_of(phys, ucg, ap)
        ppm, _ = C.power_of(phys, upg, am)
        pcm, _ = C.power_of(phys, ucg, am)
        gt = (np.abs(pcp - ppp) - np.abs(pcm - ppm)) / (ap[:, 0] - am[:, 0])

    x = torch.as_tensor(state[sel])
    A = torch.as_tensor(aeg.astype(np.float32)).requires_grad_(True)
    q = model([x, A])
    g = torch.autograd.grad(q.sum(), A)[0].detach().numpy().reshape(-1)
    if meta['target_kind'] == 'pf':
        gp = g * meta['pf_scale']
    else:
        gp = g * np.power(ucg, 3) / C.S_SCALE

    # autograd vs 网络自身差分
    nfd = min(20, sel.size)
    sd = 1e-3
    with torch.no_grad():
        ap = np.clip(aeg[:nfd] + sd, lo, hi)
        am = np.clip(aeg[:nfd] - sd, lo, hi)
        qp = model([torch.as_tensor(state[sel][:nfd]),
                    torch.as_tensor(ap.astype(np.float32))]).numpy().reshape(-1)
        qm = model([torch.as_tensor(state[sel][:nfd]),
                    torch.as_tensor(am.astype(np.float32))]).numpy().reshape(-1)
    if meta['target_kind'] == 'pf':
        fdp = (qp - qm) * meta['pf_scale'] / (ap[:, 0] - am[:, 0])
    else:
        fdp = ((qp - qm) * np.power(ucg[:nfd], 3) / C.S_SCALE
               / (ap[:, 0] - am[:, 0]))
    selfcheck = dict(median=float(np.median(np.abs(fdp - gp[:nfd]))),
                     p90=float(np.percentile(np.abs(fdp - gp[:nfd]), 90)),
                     delta=sd)

    cmp = E0._grad_cmp(gt, gp)
    return dict(n=int(sel.size), delta=float(delta), pooled=cmp,
                cosine=_cos_stats(gt[:, None], gp[:, None]),
                autograd_vs_netfd=selfcheck,
                plot=dict(gt=gt[:200].tolist(), gp=gp[:200].tolist()))


def evaluate_all(ckpt_path, data_path, out_dir, n_grad=400, delta=0.01,
                 make_plots=True):
    os.makedirs(out_dir, exist_ok=True)
    ds = np.load(data_path, allow_pickle=False)
    meta = json.loads(str(ds['meta']))
    q, ck = load_ckpt(ckpt_path)
    up = ds['u_prev'].astype(np.float64)
    uc = ds['u_cur'].astype(np.float64)
    a = ds['a'].astype(np.float64)
    ae = ds['a_eff'].astype(np.float64)
    pf = ds['pf'].astype(np.float64)
    split = ds['split']
    family = ds['family']
    state = build_state(ds, ck['state_kind'])
    pred = predict(q, state, a, uc, ck['target_kind'])

    print('\n' + '=' * 108)
    print('单步 PF critic 评价：%s' % os.path.abspath(ckpt_path))
    print('state=%s | target=%s | 参数 %s | 结构 1→16→100 ∥ 1→80→50→1'
          % (ck['state_kind'], ck['target_kind'], ck.get('n_params')))
    print('=' * 108)

    metrics = dict(checkpoint=os.path.abspath(ckpt_path),
                   data=os.path.abspath(data_path),
                   state_kind=ck['state_kind'], target_kind=ck['target_kind'],
                   n_params=ck.get('n_params'), splits={}, per_u_bin=[],
                   per_family={}, baseline={}, detection={}, gradient={})

    print('\n[1] 单步 PF（MW）')
    print('  %-6s %8s %12s %10s %10s %10s %10s' %
          ('split', 'n', 'R²', 'RMSE', 'MAE', 'MAPE%', 'max|e|'))
    for code, name in enumerate(SPLIT_NAMES):
        m = split == code
        if not np.any(m):
            continue
        r = C.regression_metrics(pred[m], pf[m])
        metrics['splits'][name] = r
        print('  %-6s %8d %12.6f %10.4f %10.4f %10.3f %10.4f'
              % (name, r['n'], r['r2'], r['rmse'], r['mae'], r['mape'],
                 r['max_abs']))

    # 分 u_cur
    mte = split == 2
    bins = C.per_bin_metrics(uc[mte], pred[mte], pf[mte], U_EDGES)
    metrics['per_u_bin'] = bins
    print('\n[2] test 按 u_cur 分箱')
    print('  %-12s %7s %10s %10s %10s' % ('u_cur', 'n', 'R²', 'RMSE', 'MAPE%'))
    for b in bins:
        print('  [%5.1f,%5.1f) %7d %10.5f %10.4f %10.3f'
              % (b['lo'], b['hi'], b['n'], b['r2'], b['rmse'], b['mape']))
    with open(os.path.join(out_dir, 'per_u_bin.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['lo', 'hi', 'n', 'r2', 'rmse', 'mae', 'mape', 'bias'])
        for b in bins:
            w.writerow([b['lo'], b['hi'], b['n'], b['r2'], b['rmse'],
                        b['mae'], b['mape'], b['bias']])

    # 分动作族
    mre = split == 3
    real_fams = meta.get('real_families', [])
    print('\n[3] real 按动作族')
    for j, fam in enumerate(real_fams):
        m = mre & (family == j)
        if not np.any(m):
            continue
        r = C.regression_metrics(pred[m], pf[m])
        r.update(family=fam)
        metrics['per_family'][fam] = r
        print('  %-10s n=%4d R²=%.5f RMSE=%.4f MAE=%.4f MAPE=%.2f%%'
              % (fam, r['n'], r['r2'], r['rmse'], r['mae'], r['mape']))

    # 基线
    edges = np.arange(C.U_MIN, C.U_MAX + 0.5, 0.5)
    mtr = split == 0
    _, means_u = C.conditional_bin_mean(uc[mtr], pf[mtr], edges)
    base_u = C.bin_lookup(uc, edges, means_u)
    key, means_2, nb = _bin2(up[mtr], uc[mtr], pf[mtr], edges)
    pred2 = np.full(pf.size, np.nan)
    k_all, _, _ = _bin2(up, uc, pf, edges)
    ok = np.isfinite(means_2[k_all])
    pred2[ok] = means_2[k_all[ok]]
    pred2[~ok] = base_u[~ok]
    print('\n[4] 条件均值基线')
    for name, code in (('test', 2), ('real', 3)):
        m = split == code
        ru = C.regression_metrics(base_u[m], pf[m])
        r2 = C.regression_metrics(pred2[m], pf[m])
        metrics['baseline'][name] = dict(by_ucur=ru, by_ucur_uprev=r2)
        print('  %-6s E[PF|u_cur]: R²=%.5f RMSE=%.4f | '
              'E[PF|u_prev,u_cur]: R²=%.5f RMSE=%.4f'
              % (name, ru['r2'], ru['rmse'], r2['r2'], r2['rmse']))

    # 成本检出（PF>3）
    print('\n[5] 成本检出 PF>3 MW（test）')
    det = _detect(pred[mte], pf[mte], 3.0)
    metrics['detection'] = det
    print('  AUC=%.4f Acc=%.4f Precision=%.4f Recall=%.4f F1=%.4f '
          '(TP=%d FP=%d FN=%d TN=%d)'
          % (det['auc'], det['accuracy'], det['precision'], det['recall'],
             det['f1'], det['tp'], det['fp'], det['fn'], det['tn']))

    # 梯度
    print('\n[6] 行为梯度 ∂PF/∂a（autograd vs aifflor 中心差分）')
    gc = gradient_check(q, ds, ck, n=n_grad, delta=delta)
    plot = gc.pop('plot')
    metrics['gradient'] = gc
    g = gc['pooled']
    print('  pooled r=%.4f slope=%.4f median_rel=%.4f sign=%.4f MAE=%.4f'
          % (g['pearson'], g['slope'], g['median_rel'], g['sign_agree'],
             g['mae']))
    cs = gc['cosine']
    print('  逐样本余弦 mean=%.3f median=%.3f | >0.8 %.1f%% | <0 %.1f%% | '
          '范数比中位=%.3f'
          % (cs['mean'], cs['median'], 100 * cs['gt08'], 100 * cs['neg'],
             cs['norm_ratio_median']))
    if make_plots:
        try:
            make_figures(out_dir, ds, pred, metrics, plot)
            print('\n[7] 图已写入 %s' % os.path.join(out_dir, 'plots'))
        except Exception as e:
            print('\n[7] 画图失败（主结果已保存）：%r' % (e,))

    C.save_json(os.path.join(out_dir, 'metrics.json'), metrics)
    summary = build_summary(metrics)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
    print('\n' + '=' * 108)
    print(summary)
    print('=' * 108)
    return metrics


def make_figures(out_dir, ds, pred, metrics, plot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9
    pdir = os.path.join(out_dir, 'plots')
    os.makedirs(pdir, exist_ok=True)
    split = ds['split']
    pf = ds['pf'].astype(np.float64)
    mte = split == 2
    true, pr = pf[mte], pred[mte]
    m = metrics['splits']['test']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, log in ((axes[0], False), (axes[1], True)):
        ax.scatter(true, pr, s=7, alpha=0.28, edgecolors='none')
        lo = min(true.min(), pr.min()); hi = max(true.max(), pr.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlabel('aifflor PF (MW)'); ax.set_ylabel('critic PF̂ (MW)')
        ax.set_title('test: R²=%.6f, RMSE=%.4f MW' % (m['r2'], m['rmse']))
        if log:
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_title(ax.get_title() + ' (log-log)')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, 'fit_test.png'), dpi=140)
    plt.close(fig)

    bins = metrics['per_u_bin']
    mids = [0.5 * (b['lo'] + b['hi']) for b in bins]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(mids, [max(1 - b['r2'], 1e-9) for b in bins], width=1.4,
                color='#3b7dd8')
    axes[0].set_yscale('log'); axes[0].set_ylabel('1 − R² (log)')
    axes[0].set_title('test by u_cur bin')
    axes[0].grid(alpha=0.3, axis='y', which='both')
    axes[1].bar(mids, [b['rmse'] for b in bins], width=1.4, color='#d8793b')
    axes[1].set_ylabel('RMSE (MW)'); axes[1].set_xlabel('u_cur (m/s)')
    axes[1].grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, 'per_u_bin.png'), dpi=140)
    plt.close(fig)

    hp = os.path.join(out_dir, 'history.csv')
    if os.path.exists(hp):
        rows = list(csv.DictReader(open(hp, encoding='utf-8')))
        if rows:
            st = [int(r['step']) for r in rows]
            vr = [float(r['val_rmse']) for r in rows]
            r2 = [float(r['val_r2']) for r in rows]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(st, vr, 'o-', ms=3)
            ax.set_xlabel('step'); ax.set_ylabel('val RMSE (MW)')
            ax.set_title('training curve'); ax.grid(alpha=0.3)
            ax2 = ax.twinx(); ax2.plot(st, r2, 'g:', lw=1)
            ax2.set_ylabel('val R²')
            fig.tight_layout()
            fig.savefig(os.path.join(pdir, 'train_curve.png'), dpi=140)
            plt.close(fig)

    if plot is not None:
        gt = np.asarray(plot['gt']); gp = np.asarray(plot['gp'])
        g = metrics['gradient']['pooled']
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(gt, gp, s=12, alpha=0.55, edgecolors='none')
        lo = min(gt.min(), gp.min()); hi = max(gt.max(), gp.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlabel('aifflor ∂PF/∂a (MW/unit)')
        ax.set_ylabel('critic ∂PF/∂a (MW/unit)')
        ax.set_title('r=%.3f slope=%.3f sign=%.3f'
                     % (g['pearson'], g['slope'], g['sign_agree']))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, 'gradient_parity.png'), dpi=140)
        plt.close(fig)


def build_summary(metrics):
    L = []
    L.append('单步 PF critic 结果')
    L.append('=' * 100)
    L.append('checkpoint : %s' % metrics['checkpoint'])
    L.append('state=%s | target=%s | 参数 %s'
             % (metrics['state_kind'], metrics['target_kind'],
                metrics.get('n_params')))
    L.append('')
    L.append('%-6s %8s %12s %10s %10s %10s' %
             ('split', 'n', 'R²', 'RMSE', 'MAE', 'MAPE%'))
    for name, r in metrics['splits'].items():
        L.append('%-6s %8d %12.6f %10.4f %10.4f %10.3f'
                 % (name, r['n'], r['r2'], r['rmse'], r['mae'], r['mape']))
    L.append('')
    b = metrics['baseline']
    L.append('基线（test）：E[PF|u_cur] R²=%.5f RMSE=%.4f | '
             'E[PF|u_prev,u_cur] R²=%.5f RMSE=%.4f'
             % (b['test']['by_ucur']['r2'], b['test']['by_ucur']['rmse'],
                b['test']['by_ucur_uprev']['r2'],
                b['test']['by_ucur_uprev']['rmse']))
    d = metrics['detection']
    L.append('成本检出 PF>3：AUC=%.4f Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f'
             % (d['auc'], d['accuracy'], d['precision'], d['recall'], d['f1']))
    L.append('')
    g = metrics['gradient']['pooled']
    L.append('行为梯度 ∂PF/∂a：r=%.4f slope=%.4f median_rel=%.4f sign=%.4f'
             % (g['pearson'], g['slope'], g['median_rel'], g['sign_agree']))
    cs = metrics['gradient']['cosine']
    L.append('  逐样本余弦 mean=%.3f median=%.3f | >0.8 %.1f%% | <0 %.1f%%'
             % (cs['mean'], cs['median'], 100 * cs['gt08'], 100 * cs['neg']))
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description='评价单步 PF critic')
    ap.add_argument('--ckpt', default='agentutil/critictest/results/pf_u2p/pf_best.pt')
    ap.add_argument('--data', default='agentutil/critictest/data/aifflor_pf_scalar.npz')
    ap.add_argument('--out', default=None)
    ap.add_argument('--n_grad', type=int, default=400)
    ap.add_argument('--delta', type=float, default=0.01)
    ap.add_argument('--no_plots', action='store_true')
    args = ap.parse_args()
    out = args.out or os.path.dirname(os.path.abspath(args.ckpt))
    evaluate_all(args.ckpt, args.data, out, n_grad=args.n_grad,
                 delta=args.delta, make_plots=not args.no_plots)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
