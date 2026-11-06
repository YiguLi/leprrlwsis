#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""评价五步块 C1 critic：成本精度 + 行为梯度（重点）。

C1 = Σ_t [PF_t + 20·max(0, PF_t−3)]，PF_t 为五步块内 5 个 |ΔP|。

输出：
  * 原始 C1 的 R²/RMSE/MAE/MAPE，log1p(C1) 的 R²/RMSE；
  * 分 u1、分动作族；
  * `C1>15`（d_block=3MW×5）检出 AUC/F1；
  * `∂C1/∂a`：autograd vs aifflor 中心差分，分“折点触发/未触发”两组。
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
from model import Critic
from train_c1 import C1_SCALE, build_state, c1_from_steps, inverse_of, predict

SPLIT_NAMES = ('train', 'val', 'test', 'real')
U_EDGES = (3.5, 5.0, 7.0, 9.0, 11.0, 13.0, 16.0, 20.0, 25.001)


def load_ckpt(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = ck['meta']
    arch = str(m.get('arch', 'v1'))
    if arch == 'v1':
        q = C.make_critic(m['state_dim'], 1, m.get('hidden1', 16),
                          m.get('hidden2', 100), seed=None, bias=m.get('bias0'))
    else:
        q = Critic(m['state_dim'], 1, hidden1=m.get('hidden1', 16),
                   hidden2=m.get('hidden2', 100), arch=arch)
    q.load_state_dict(ck['model'])
    q.eval()
    return q, m


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n1 = int(labels.sum()); n0 = int(labels.size - n1)
    if n0 == 0 or n1 == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(labels.size, dtype=float)
    ranks[order] = np.arange(1, labels.size + 1)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def _detect(pred, true, thr=15.0):
    lab = true > thr; p = pred > thr
    tp = int(np.sum(p & lab)); fp = int(np.sum(p & ~lab))
    fn = int(np.sum(~p & lab)); tn = int(np.sum(~p & ~lab))
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(threshold=thr, auc=_auc(lab, pred),
                accuracy=float((tp + tn) / max(lab.size, 1)),
                precision=float(prec), recall=float(rec), f1=float(f1),
                tp=tp, fp=fp, fn=fn, tn=tn)


def _cos_stats(gt, gp):
    gt = np.asarray(gt).reshape(-1, 1); gp = np.asarray(gp).reshape(-1, 1)
    cos = (gt * gp).sum(1) / (np.linalg.norm(gt, axis=1)
                              * np.linalg.norm(gp, axis=1) + 1e-12)
    nr = np.linalg.norm(gp, axis=1) / (np.linalg.norm(gt, axis=1) + 1e-12)
    return dict(mean=float(cos.mean()), median=float(np.median(cos)),
                gt05=float(np.mean(cos > 0.5)), gt08=float(np.mean(cos > 0.8)),
                neg=float(np.mean(cos < 0)),
                norm_ratio_median=float(np.median(nr)))


def _c1_of(ps, pp):
    pf = np.column_stack([np.abs(ps[:, 0] - pp),
                          np.abs(np.diff(ps, axis=1))])
    c1s = pf + 20.0 * np.maximum(0.0, pf - 3.0)
    return c1s.sum(1), pf


def gradient_check(model, ds, ck, n=400, delta=0.01, seed=123):
    u0 = ds['u0'].astype(np.float64)
    u_seq = ds['u_seq'].astype(np.float64)
    a = ds['a'].astype(np.float64)
    ae = ds['a_eff'].astype(np.float64)
    c1, pf_steps = c1_from_steps(ds['p_steps'], ds['p_prev'])
    split = ds['split']
    state = build_state(ds, ck['state_kind'])
    cand = ((split == 2) & (ae[:, 0] >= 0.05) & (ae[:, 0] <= 0.45)
            & (pf_steps.max(axis=1) >= 0.5))
    idx = np.flatnonzero(cand)
    if idx.size < n:
        idx = np.flatnonzero((split == 2) & (pf_steps.max(axis=1) >= 0.3))
    rng = np.random.RandomState(seed)
    sel = rng.choice(idx, size=min(n, idx.size), replace=False)

    u0g, usg, aeg = u0[sel], u_seq[sel], ae[sel]
    phys = C.build_physics(calibrated=False, power_scale=C.AT.LEVEL_COEF)
    lo, hi = C.A_PHYS_LO, C.A_HI
    ap = np.clip(aeg + delta, lo, hi)
    am = np.clip(aeg - delta, lo, hi)
    if ck['state_kind'] == 'obs':
        # 初始 obs 口径：P_G_prev 是实测过去量，固定；只对当前块功率求导
        p_prev_g = ds['p_prev'].astype(np.float64)[sel]
        psp, _ = C.power_steps(phys, usg, ap, window=5)
        psm, _ = C.power_steps(phys, usg, am, window=5)
        c1p, _ = _c1_of(psp, p_prev_g)
        c1m, _ = _c1_of(psm, p_prev_g)
    else:
        _f5p, _f4p, psp, ppp, _ = C.power_fluct(
            phys, u0g, usg, ap, window=5)
        _f5m, _f4m, psm, ppm, _ = C.power_fluct(
            phys, u0g, usg, am, window=5)
        c1p, _ = _c1_of(psp, ppp)
        c1m, _ = _c1_of(psm, ppm)
    gt = (c1p - c1m) / (ap[:, 0] - am[:, 0])

    x = torch.as_tensor(state[sel])
    A = torch.as_tensor(aeg.astype(np.float32)).requires_grad_(True)
    q = model([x, A])
    g = torch.autograd.grad(q.sum(), A)[0].detach().numpy().reshape(-1)
    if ck['target_kind'] == 'c1':
        gp = g * C1_SCALE
        pred_c1 = q.detach().numpy().reshape(-1) * C1_SCALE
    else:
        pred_c1 = np.expm1(np.clip(q.detach().numpy().reshape(-1), 0, 20))
        gp = g * (1.0 + pred_c1)

    # autograd vs 网络自身差分
    nfd = min(20, sel.size); sd = 1e-3
    with torch.no_grad():
        app = np.clip(aeg[:nfd] + sd, lo, hi)
        amm = np.clip(aeg[:nfd] - sd, lo, hi)
        qp = model([torch.as_tensor(state[sel][:nfd]),
                    torch.as_tensor(app.astype(np.float32))]).numpy().reshape(-1)
        qm = model([torch.as_tensor(state[sel][:nfd]),
                    torch.as_tensor(amm.astype(np.float32))]).numpy().reshape(-1)
    if ck['target_kind'] == 'c1':
        fdp = (qp - qm) * C1_SCALE / (app[:, 0] - amm[:, 0])
    else:
        c1p_net = np.expm1(np.clip(qp, 0, 20))
        c1m_net = np.expm1(np.clip(qm, 0, 20))
        fdp = (c1p_net - c1m_net) / (app[:, 0] - amm[:, 0])
    selfcheck = dict(median=float(np.median(np.abs(fdp - gp[:nfd]))),
                     p90=float(np.percentile(np.abs(fdp - gp[:nfd]), 90)),
                     delta=sd)

    hinge = (pf_steps[sel] > 3.0).any(axis=1)
    groups = {}
    for name, mask in (('all', np.ones(sel.size, bool)),
                       ('hinge', hinge), ('smooth', ~hinge)):
        if mask.sum() == 0:
            continue
        groups[name] = dict(n=int(mask.sum()),
                            pooled=E0._grad_cmp(gt[mask], gp[mask]),
                            cosine=_cos_stats(gt[mask], gp[mask]))
    return dict(n=int(sel.size), delta=float(delta),
                pooled=E0._grad_cmp(gt, gp),
                cosine=_cos_stats(gt, gp),
                groups=groups,
                autograd_vs_netfd=selfcheck,
                plot=dict(gt=gt[:200].tolist(), gp=gp[:200].tolist(),
                          hinge=hinge[:200].tolist()))


def evaluate_all(ckpt_path, data_path, out_dir, n_grad=400, delta=0.01,
                 make_plots=True):
    os.makedirs(out_dir, exist_ok=True)
    ds = np.load(data_path, allow_pickle=False)
    meta = json.loads(str(ds['meta']))
    q, ck = load_ckpt(ckpt_path)
    a = ds['a'].astype(np.float64)
    u1 = ds['u'].astype(np.float64)
    split = ds['split']
    family = ds['family']
    c1, pf_steps = c1_from_steps(ds['p_steps'], ds['p_prev'])
    state = build_state(ds, ck['state_kind'])
    pred = predict(q, state, a, ck['target_kind'])

    print('\n' + '=' * 108)
    print('五步 C1 critic 评价：%s' % os.path.abspath(ckpt_path))
    print('arch=%s | state=%s | target=%s | 参数 %s | C1 = Σ[PF+20(PF−3)+]'
          % (ck.get('arch', 'v1'), ck['state_kind'], ck['target_kind'],
             ck.get('n_params')))
    print('=' * 108)

    metrics = dict(checkpoint=os.path.abspath(ckpt_path),
                   data=os.path.abspath(data_path),
                   arch=ck.get('arch', 'v1'),
                   state_kind=ck['state_kind'], target_kind=ck['target_kind'],
                   n_params=ck.get('n_params'), splits={}, per_u_bin=[],
                   per_family={}, baseline={}, detection={}, gradient={})

    print('\n[1] 块总 C1（原始尺度，MW·step）')
    print('  %-6s %8s %12s %10s %10s %10s %12s' %
          ('split', 'n', 'R²', 'RMSE', 'MAE', 'MAPE%', 'log1p R²'))
    for code, name in enumerate(SPLIT_NAMES):
        m = split == code
        if not np.any(m):
            continue
        r = C.regression_metrics(pred[m], c1[m])
        rl = C.regression_metrics(np.log1p(pred[m]), np.log1p(c1[m]))
        r['log_r2'] = rl['r2']; r['log_rmse'] = rl['rmse']
        metrics['splits'][name] = r
        print('  %-6s %8d %12.6f %10.4f %10.4f %10.3f %12.6f'
              % (name, r['n'], r['r2'], r['rmse'], r['mae'], r['mape'],
                 r['log_r2']))

    mte = split == 2
    bins = C.per_bin_metrics(u1[mte], pred[mte], c1[mte], U_EDGES)
    metrics['per_u_bin'] = bins
    print('\n[2] test 按 u1 分箱')
    print('  %-12s %7s %10s %10s %10s' % ('u1', 'n', 'R²', 'RMSE', 'MAPE%'))
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

    mre = split == 3
    real_fams = meta.get('real_families', [])
    print('\n[3] real 按动作族')
    for j, fam in enumerate(real_fams):
        m = mre & (family == j)
        if not np.any(m):
            continue
        r = C.regression_metrics(pred[m], c1[m])
        r.update(family=fam)
        metrics['per_family'][fam] = r
        print('  %-10s n=%4d R²=%.5f RMSE=%.4f MAE=%.4f MAPE=%.2f%%'
              % (fam, r['n'], r['r2'], r['rmse'], r['mae'], r['mape']))

    # 基线 E[C1|u1]
    edges = np.arange(C.U_MIN, C.U_MAX + 0.5, 0.5)
    mtr = split == 0
    _, means = C.conditional_bin_mean(u1[mtr], c1[mtr], edges)
    base = C.bin_lookup(u1, edges, means)
    print('\n[4] 条件均值基线 E[C1|u1]')
    for name, code in (('test', 2), ('real', 3)):
        m = split == code
        r = C.regression_metrics(base[m], c1[m])
        metrics['baseline'][name] = r
        print('  %-6s R²=%.5f RMSE=%.4f MAE=%.4f'
              % (name, r['r2'], r['rmse'], r['mae']))

    # 折点/重尾分组成本精度
    print('\n[5] test 按“是否触发 PF>3 折点”分组')
    hinge = (pf_steps[mte] > 3.0).any(axis=1)
    for name, mask in (('smooth', ~hinge), ('hinge', hinge)):
        if mask.sum() == 0:
            continue
        r = C.regression_metrics(pred[mte][mask], c1[mte][mask])
        print('  %-6s n=%5d R²=%.6f RMSE=%.4f MAE=%.4f MAPE=%.2f%%'
              % (name, mask.sum(), r['r2'], r['rmse'], r['mae'], r['mape']))
        metrics.setdefault('hinge_groups', {})[name] = r

    # 成本检出 C1>15（d_block）
    print('\n[6] 成本检出 C1>15（d_block=3MW×5）')
    det = _detect(pred[mte], c1[mte], 15.0)
    metrics['detection'] = det
    print('  AUC=%.4f Acc=%.4f Precision=%.4f Recall=%.4f F1=%.4f '
          '(TP=%d FP=%d FN=%d TN=%d)'
          % (det['auc'], det['accuracy'], det['precision'], det['recall'],
             det['f1'], det['tp'], det['fp'], det['fn'], det['tn']))

    # 梯度
    print('\n[7] 行为梯度 ∂C1/∂a（autograd vs aifflor 中心差分）')
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
    for name in ('smooth', 'hinge'):
        if name not in gc['groups']:
            continue
        gg = gc['groups'][name]
        print('  [%s] n=%d r=%.4f slope=%.4f sign=%.4f 余弦mean=%.3f <0 %.1f%%'
              % (name, gg['n'], gg['pooled']['pearson'],
                 gg['pooled']['slope'], gg['pooled']['sign_agree'],
                 gg['cosine']['mean'], 100 * gg['cosine']['neg']))

    if make_plots:
        try:
            make_figures(out_dir, ds, pred, metrics, plot)
            print('\n[8] 图已写入 %s' % os.path.join(out_dir, 'plots'))
        except Exception as e:
            print('\n[8] 画图失败（主结果已保存）：%r' % (e,))

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
    pdir = os.path.join(out_dir, 'plots'); os.makedirs(pdir, exist_ok=True)
    split = ds['split']; c1, _ = c1_from_steps(ds['p_steps'], ds['p_prev'])
    mte = split == 2; true, pr = c1[mte], pred[mte]
    m = metrics['splits']['test']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, log in ((axes[0], False), (axes[1], True)):
        ax.scatter(true, pr, s=7, alpha=0.25, edgecolors='none')
        lo = min(true.min(), pr.min()); hi = max(true.max(), pr.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlabel('aifflor C1'); ax.set_ylabel('critic C1̂')
        ax.set_title('test R²=%.6f RMSE=%.4f' % (m['r2'], m['rmse']))
        if log:
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_title(ax.get_title() + ' (log-log)')
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(pdir, 'fit_test.png'), dpi=140)
    plt.close(fig)

    if plot is not None:
        gt = np.asarray(plot['gt']); gp = np.asarray(plot['gp'])
        hinge = np.asarray(plot['hinge'], dtype=bool)
        g = metrics['gradient']['pooled']
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(gt[~hinge], gp[~hinge], s=12, alpha=0.5, label='smooth')
        ax.scatter(gt[hinge], gp[hinge], s=16, alpha=0.8, label='hinge')
        lo = min(gt.min(), gp.min()); hi = max(gt.max(), gp.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlabel('aifflor ∂C1/∂a'); ax.set_ylabel('critic ∂C1/∂a')
        ax.set_title('r=%.3f slope=%.3f sign=%.3f'
                     % (g['pearson'], g['slope'], g['sign_agree']))
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, 'gradient_parity.png'), dpi=140)
        plt.close(fig)

    hp = os.path.join(out_dir, 'history.csv')
    if os.path.exists(hp):
        rows = list(csv.DictReader(open(hp, encoding='utf-8')))
        if rows:
            st = [int(r['step']) for r in rows]
            vr = [float(r['val_rmse']) for r in rows]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(st, vr, 'o-', ms=3)
            ax.set_xlabel('step'); ax.set_ylabel('val C1 RMSE')
            ax.set_title('training curve'); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(pdir, 'train_curve.png'), dpi=140)
            plt.close(fig)


def build_summary(metrics):
    L = []
    L.append('五步块 C1 critic 结果')
    L.append('=' * 100)
    L.append('checkpoint: %s' % metrics['checkpoint'])
    L.append('state=%s | target=%s | 参数 %s'
             % (metrics['state_kind'], metrics['target_kind'],
                metrics.get('n_params')))
    L.append('')
    L.append('%-6s %8s %12s %10s %10s %10s %12s' %
             ('split', 'n', 'R²', 'RMSE', 'MAE', 'MAPE%', 'log1p R²'))
    for name, r in metrics['splits'].items():
        L.append('%-6s %8d %12.6f %10.4f %10.4f %10.3f %12.6f'
                 % (name, r['n'], r['r2'], r['rmse'], r['mae'], r['mape'],
                    r.get('log_r2', float('nan'))))
    L.append('')
    b = metrics['baseline']
    L.append('基线 test：E[C1|u1] R²=%.5f RMSE=%.4f'
             % (b['test']['r2'], b['test']['rmse']))
    d = metrics['detection']
    L.append('检出 C1>15：AUC=%.4f Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f'
             % (d['auc'], d['accuracy'], d['precision'], d['recall'], d['f1']))
    L.append('')
    g = metrics['gradient']['pooled']; cs = metrics['gradient']['cosine']
    L.append('∂C1/∂a：r=%.4f slope=%.4f median_rel=%.4f sign=%.4f'
             % (g['pearson'], g['slope'], g['median_rel'], g['sign_agree']))
    L.append('  余弦 mean=%.3f median=%.3f | >0.8 %.1f%% | <0 %.1f%%'
             % (cs['mean'], cs['median'], 100 * cs['gt08'], 100 * cs['neg']))
    for name in ('smooth', 'hinge'):
        gg = metrics['gradient']['groups'].get(name)
        if gg:
            L.append('  %s: n=%d r=%.4f slope=%.4f sign=%.4f 余弦mean=%.3f'
                     % (name, gg['n'], gg['pooled']['pearson'],
                        gg['pooled']['slope'], gg['pooled']['sign_agree'],
                        gg['cosine']['mean']))
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description='评价五步块 C1 critic')
    ap.add_argument('--ckpt', default='agentutil/critictest/results/c1_useq6_log/c1_best.pt')
    ap.add_argument('--data', default='agentutil/critictest/data/aifflor_ua_f5_scalar.npz')
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
