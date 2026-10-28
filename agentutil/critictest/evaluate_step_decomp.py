#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""评价逐步奖励分解 critic（五步波动和）。

验收项（对应架构文档 §13）：
  1. 逐步波动预测误差（分 5 个阶段）；
  2. 块波动和 F5 的误差（组合后 QComposer）；
  3. `Σ step` 与 `+B` 的差异（终端头贡献）；
  4. 与 monolithic `f5n`、`E[F5|u1]` 基线对比；
  5. 动作梯度 ∂F5/∂a（逐维 + 逐样本余弦）；
  6. 图：拟合、分阶段、训练曲线、梯度、B 分布。
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
import step_decomp as SD  # noqa: E402
import evaluate as E0  # noqa: E402

SPLIT_NAMES = ('train', 'val', 'test', 'real')


def load_ckpt(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = ck['meta']
    cls = (SD.StepDecompCriticPER if m.get('arch', 'film') == 'per'
           else SD.StepDecompCritic)
    model = cls(state_dim=m['state_dim'], action_dim=m['action_dim'],
                f=m['f'], stage_dim=m['stage_dim'],
                gamma=m['gamma'], step_bias=m.get('step_bias', 0.0))
    model.load_state_dict(ck['model'])
    model.eval()
    target = cls(state_dim=m['state_dim'], action_dim=m['action_dim'],
                 f=m['f'], stage_dim=m['stage_dim'],
                 gamma=m['gamma'], step_bias=m.get('step_bias', 0.0))
    target.load_state_dict(ck['target'])
    target.eval()
    return model, target, m


def _cos_stats(gt, gp):
    cos = (gt * gp).sum(1) / (np.linalg.norm(gt, axis=1) * np.linalg.norm(gp, axis=1) + 1e-12)
    nr = np.linalg.norm(gp, axis=1) / (np.linalg.norm(gt, axis=1) + 1e-12)
    proj = (gt * gp).sum(1) / ((gt * gt).sum(1) + 1e-12)
    return dict(cos_mean=float(cos.mean()), cos_median=float(np.median(cos)),
                cos_p25=float(np.percentile(cos, 25)),
                cos_p75=float(np.percentile(cos, 75)),
                cos_gt05=float(np.mean(cos > 0.5)),
                cos_gt08=float(np.mean(cos > 0.8)),
                cos_neg=float(np.mean(cos < 0)),
                norm_ratio_mean=float(nr.mean()),
                norm_ratio_median=float(np.median(nr)),
                proj_mean=float(proj.mean()), proj_median=float(np.median(proj)))


def gradient_check(model, ds, meta, n=400, delta=0.01, seed=123):
    u = ds['u'].astype(np.float64)
    a_eff = ds['a_eff'].astype(np.float64)
    f5 = ds['f5'].astype(np.float64)
    split = ds['split']
    u_seq = ds['u_seq'].astype(np.float64)
    u0 = ds['u0'].astype(np.float64)
    cand = ((split == 2) & (a_eff.min(1) >= 0.05) & (a_eff.max(1) <= 0.45)
            & (f5 >= 0.3))
    idx = np.flatnonzero(cand)
    if idx.size < n:
        idx = np.flatnonzero(split == 2)
    rng = np.random.RandomState(seed)
    sel = rng.choice(idx, size=min(int(n), idx.size), replace=False)

    ug, ag, u0g, useq = u[sel], a_eff[sel], u0[sel], u_seq[sel]
    dm = meta.get('data_meta', {})
    pm = dm.get('physics', {}) if isinstance(dm, dict) else {}
    phys = C.build_physics(calibrated=bool(pm.get('calibrated', False)),
                           power_scale=float(pm.get('power_scale', C.AT.LEVEL_COEF)))

    gt = np.zeros((sel.size, C.NT), dtype=np.float64)
    lo = np.full_like(ag, C.A_PHYS_LO)
    for j in range(C.NT):
        ap = ag.copy(); am = ag.copy()
        ap[:, j] = np.clip(ag[:, j] + delta, lo[:, j], C.A_HI)
        am[:, j] = np.clip(ag[:, j] - delta, lo[:, j], C.A_HI)
        fp = C.power_fluct(phys, u0g, useq, ap, window=5)[0]
        fm = C.power_fluct(phys, u0g, useq, am, window=5)[0]
        gt[:, j] = (fp - fm) / (ap[:, j] - am[:, j])

    x = torch.as_tensor((ug / 10.0).reshape(-1, 1), dtype=torch.float32)
    A = torch.as_tensor(ag.astype(np.float32)).requires_grad_(True)
    q, _step, _term = model(x, A)
    gp = torch.autograd.grad(q.sum(), A)[0].detach().numpy().astype(np.float64)

    # autograd vs 网络自身差分（δ=1e-3）
    nfd = min(20, sel.size)
    self_d = 1e-3
    net_fd = np.zeros((nfd, C.NT), dtype=np.float64)
    for j in range(C.NT):
        ap = ag[:nfd].copy(); am = ag[:nfd].copy()
        ap[:, j] = np.clip(ag[:nfd, j] + self_d, lo[:nfd, j], C.A_HI)
        am[:, j] = np.clip(ag[:nfd, j] - self_d, lo[:nfd, j], C.A_HI)
        with torch.no_grad():
            qp, _, _ = model(torch.as_tensor((ug[:nfd] / 10.0).reshape(-1, 1),
                                             dtype=torch.float32),
                             torch.as_tensor(ap.astype(np.float32)))
            qm, _, _ = model(torch.as_tensor((ug[:nfd] / 10.0).reshape(-1, 1),
                                             dtype=torch.float32),
                             torch.as_tensor(am.astype(np.float32)))
        net_fd[:, j] = (qp.numpy() - qm.numpy()) / (ap[:, j] - am[:, j])
    _sd = np.abs(net_fd - gp[:nfd])
    selfcheck = dict(median=float(np.median(_sd)),
                     p90=float(np.percentile(_sd, 90)),
                     max=float(np.max(_sd)), delta=self_d)

    dims = {'dim%d' % j: E0._grad_cmp(gt[:, j], gp[:, j]) for j in range(C.NT)}
    pooled = E0._grad_cmp(gt.reshape(-1), gp.reshape(-1))
    return dict(n=int(sel.size), delta=float(delta), dims=dims, pooled=pooled,
                autograd_vs_netfd=selfcheck, cosine=_cos_stats(gt, gp),
                plot=dict(gt=gt[:200].tolist(), gp=gp[:200].tolist()))


def evaluate_all(ckpt_path, data_path, out_dir, n_grad=400, delta=0.01,
                 make_plots=True):
    os.makedirs(out_dir, exist_ok=True)
    ds = np.load(data_path, allow_pickle=False)
    if 'f5' not in ds.files:
        raise SystemExit('需要五步波动数据集')
    model, target, meta = load_ckpt(ckpt_path)
    f_steps, f5 = SD.load_step_targets(ds)
    x, a = SD.state_action(ds)
    split = ds['split']
    u = ds['u'].astype(np.float64)
    f5d = f5.astype(np.float64)

    print('\n' + '=' * 108)
    print('step-decomp 评价：%s' % os.path.abspath(ckpt_path))
    arch = meta.get('arch', 'film')
    print('f=5 | γ=%.3f | 参数 %s | arch=%s' %
          (meta['gamma'], meta.get('n_params', '?'), arch))
    if arch == 'per':
        print('结构: 1→16→100（无激活）→ cat(a) 105→80→50 → [h;e_t]→1 ×5 + h→1')
    else:
        print('结构: state64/action64 → FiLM → 192→256→256→128 → 逐步头/终端头')
    print('=' * 108)

    metrics = dict(checkpoint=os.path.abspath(ckpt_path),
                   data=os.path.abspath(data_path),
                   gamma=meta['gamma'], f=meta['f'],
                   n_params=meta.get('n_params'), splits={}, per_u_bin=[],
                   per_family={}, step_metrics={}, baseline={},
                   gradient={}, terminal={})

    # ---------------- 1. 各 split ----------------
    print('\n[1] 块波动和 F5（MW）')
    print('  %-6s %8s %12s %10s %10s %10s %10s' %
          ('split', 'n', 'R²', 'RMSE', 'MAE', 'MAPE%', 'B_abs'))
    preds = {}
    for code, name in enumerate(SPLIT_NAMES):
        m = split == code
        if not np.any(m):
            continue
        q, step, term = SD.predict(model, x[m], a[m])
        r = C.regression_metrics(q, f5d[m])
        metrics['splits'][name] = dict(power=r,
                                       step_sum=C.regression_metrics(step.sum(1), f5d[m]))
        preds[name] = dict(q=q, step=step, term=term)
        print('  %-6s %8d %12.6f %10.4f %10.4f %10.3f %10.4f'
              % (name, r['n'], r['r2'], r['rmse'], r['mae'], r['mape'],
                 float(np.mean(np.abs(term)))))
        if name in ('test', 'real'):
            metrics['terminal'][name] = dict(
                mean=float(np.mean(term)), mean_abs=float(np.mean(np.abs(term))),
                max_abs=float(np.max(np.abs(term))),
                std=float(np.std(term)))

    # ---------------- 2. 分阶段 step 误差 ----------------
    print('\n[2] 逐步波动预测（分 5 个阶段，test）')
    print('  %-5s %8s %10s %10s %10s %10s' %
          ('stage', 'n', 'R²', 'RMSE', 'MAE', 'MAPE%'))
    mte = split == 2
    for t in range(5):
        r = C.regression_metrics(preds['test']['step'][:, t], f_steps[mte, t])
        metrics['step_metrics']['stage%d' % t] = r
        print('  t=%-3d %8d %10.5f %10.4f %10.4f %10.3f'
              % (t, r['n'], r['r2'], r['rmse'], r['mae'], r['mape']))
    with open(os.path.join(out_dir, 'step_metrics.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['stage', 'n', 'r2', 'rmse', 'mae', 'mape', 'bias'])
        for t in range(5):
            r = metrics['step_metrics']['stage%d' % t]
            w.writerow([t, r['n'], r['r2'], r['rmse'], r['mae'], r['mape'],
                        r['bias']])

    # ---------------- 3. 基线 / monolithic 对照 ----------------
    dm = json.loads(str(ds['meta']))
    edges = np.arange(C.U_MIN, C.U_MAX + 0.5, 0.5)
    _, means = C.conditional_bin_mean(u[split == 0], f5d[split == 0], edges)
    base = C.bin_lookup(u, edges, means)
    for name, code in (('test', 2), ('real', 3)):
        m = split == code
        metrics['baseline'][name] = C.regression_metrics(base[m], f5d[m])
    print('\n[3] 参考基线 E[F5|u1]')
    for name in ('test', 'real'):
        r = metrics['baseline'][name]
        print('  %-6s R²=%.5f RMSE=%.4f MAPE=%.2f%%'
              % (name, r['r2'], r['rmse'], r['mape']))

    mono_path = 'agentutil/critictest/results/f5_f5n_16/critic_best.pt'
    if os.path.exists(mono_path):
        qm, _qt, mm = E0.load_checkpoint(mono_path)
        norm = C.window_fluct_norm(ds)
        pm = C.predict_power(qm, u, a, target='f5n', feature='u10',
                             window=5, norm=norm)
        metrics['monolithic_f5n'] = {}
        for name, code in (('test', 2), ('real', 3)):
            m = split == code
            metrics['monolithic_f5n'][name] = C.regression_metrics(pm[m], f5d[m])
        print('\n[3b] 对照 monolithic f5n（16/100）')
        for name in ('test', 'real'):
            r = metrics['monolithic_f5n'][name]
            print('  %-6s R²=%.5f RMSE=%.4f MAPE=%.2f%%'
                  % (name, r['r2'], r['rmse'], r['mape']))

    # ---------------- 4. 梯度 ----------------
    print('\n[4] 动作梯度 ∂F5/∂a（autograd vs aifflor 中心差分，%d 点，δ=%.3f）'
          % (n_grad, delta))
    gc = gradient_check(model, ds, meta, n=n_grad, delta=delta)
    grad_plot = gc.pop('plot', None)
    metrics['gradient'] = gc
    sc = gc['autograd_vs_netfd']
    print('  自检 autograd vs 网络自身差分（δ=%.0e）: median|Δ|=%.3e, p90=%.3e'
          % (sc['delta'], sc['median'], sc['p90']))
    print('  %-8s %10s %10s %12s %10s' %
          ('dim', 'pearson', 'slope', 'median_rel', 'sign'))
    for j in range(C.NT):
        g = gc['dims']['dim%d' % j]
        print('  dim%-5d %10.4f %10.4f %12.4f %10.4f'
              % (j, g['pearson'], g['slope'], g['median_rel'], g['sign_agree']))
    gp = gc['pooled']
    print('  pooled   %10.4f %10.4f %12.4f %10.4f'
          % (gp['pearson'], gp['slope'], gp['median_rel'], gp['sign_agree']))
    cs = gc['cosine']
    print('  逐样本余弦: mean %.3f median %.3f | >0.8 %.1f%% | <0 %.1f%% | '
          '范数比中位 %.3f'
          % (cs['cos_mean'], cs['cos_median'], 100 * cs['cos_gt08'],
             100 * cs['cos_neg'], cs['norm_ratio_median']))

    # ---------------- 5. 图 / 汇总 ----------------
    if make_plots:
        try:
            make_figures(out_dir, ds, preds, metrics, grad_plot)
            print('\n[5] 图已写入 %s' % os.path.join(out_dir, 'plots'))
        except Exception as e:
            print('\n[5] 画图失败（主结果已保存）：%r' % (e,))

    C.save_json(os.path.join(out_dir, 'metrics.json'), metrics)
    summary = build_summary(metrics)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
    print('\n' + '=' * 108)
    print(summary)
    print('=' * 108)
    return metrics


def make_figures(out_dir, ds, preds, metrics, grad_plot=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9
    pdir = os.path.join(out_dir, 'plots')
    os.makedirs(pdir, exist_ok=True)
    split = ds['split']
    f5 = ds['f5'].astype(np.float64)
    mte = split == 2
    true, pred = f5[mte], preds['test']['q']
    m = metrics['splits']['test']['power']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, log in ((axes[0], False), (axes[1], True)):
        ax.scatter(true, pred, s=7, alpha=0.28, edgecolors='none')
        lo = min(true.min(), pred.min()); hi = max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlabel('aifflor F5 (MW)'); ax.set_ylabel('critic F5̂ (MW)')
        ax.set_title('test: R²=%.6f, RMSE=%.4f MW' % (m['r2'], m['rmse']))
        if log:
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_title(ax.get_title() + ' (log-log)')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, 'fit_test.png'), dpi=140)
    plt.close(fig)

    # 分阶段
    stages = list(range(5))
    rmse = [metrics['step_metrics']['stage%d' % t]['rmse'] for t in stages]
    mae = [metrics['step_metrics']['stage%d' % t]['mae'] for t in stages]
    r2 = [metrics['step_metrics']['stage%d' % t]['r2'] for t in stages]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(stages, rmse, color='#d8793b', label='RMSE')
    axes[0].plot(stages, mae, 'ko-', ms=4, label='MAE')
    axes[0].set_xlabel('stage t'); axes[0].set_ylabel('MW')
    axes[0].set_title('per-step fluctuation error'); axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')
    axes[1].bar(stages, r2, color='#3b7dd8')
    axes[1].set_xlabel('stage t'); axes[1].set_ylabel('R²')
    axes[1].set_title('per-step R²')
    axes[1].grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, 'per_step.png'), dpi=140)
    plt.close(fig)

    # 训练曲线
    hp = os.path.join(out_dir, 'history.csv')
    if os.path.exists(hp):
        rows = list(csv.DictReader(open(hp, encoding='utf-8')))
        if rows:
            st = [int(r['step']) for r in rows]
            vr = [float(r['val_rmse']) for r in rows]
            sr = [float(r['val_step_rmse']) for r in rows]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(st, vr, 'o-', ms=3, label='val F5 RMSE')
            ax.plot(st, sr, 's--', ms=3, label='val step RMSE (mean)')
            ax.set_xlabel('step'); ax.set_ylabel('RMSE (MW)')
            ax.set_title('training curve'); ax.grid(alpha=0.3); ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(pdir, 'train_curve.png'), dpi=140)
            plt.close(fig)

    # 梯度
    if grad_plot is not None:
        gt = np.asarray(grad_plot['gt'], dtype=np.float64)
        gp = np.asarray(grad_plot['gp'], dtype=np.float64)
        gc = metrics['gradient']
        fig, axes = plt.subplots(1, C.NT, figsize=(4 * C.NT, 4))
        for j, ax in enumerate(axes):
            g = gc['dims']['dim%d' % j]
            ax.scatter(gt[:, j], gp[:, j], s=12, alpha=0.55, edgecolors='none')
            lo = min(gt[:, j].min(), gp[:, j].min())
            hi = max(gt[:, j].max(), gp[:, j].max())
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
            ax.set_title('dim%d  r=%.3f  slope=%.3f'
                         % (j, g['pearson'], g['slope']), fontsize=9)
            ax.set_xlabel('aifflor ∂F5/∂a (MW/unit)')
            if j == 0:
                ax.set_ylabel('critic ∂F5/∂a (MW/unit)')
            ax.grid(alpha=0.3)
        fig.suptitle('action-gradient parity on test (n=%d)' % gc['n'])
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, 'gradient_parity.png'), dpi=140)
        plt.close(fig)

    # 终端 B
    if 'test' in preds:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(preds['test']['term'], bins=60, color='#6a8caf')
        ax.axvline(0, color='k', lw=0.8)
        ax.set_xlabel('B̂ (MW)'); ax.set_ylabel('count')
        ax.set_title('terminal head output on test (target=0 in pure prediction)')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, 'terminal_hist.png'), dpi=140)
        plt.close(fig)


def build_summary(metrics):
    L = []
    L.append('step-decomp critic（五步功率波动和）')
    L.append('=' * 100)
    L.append('checkpoint : %s' % metrics['checkpoint'])
    L.append('f=%d | γ=%.3f | 参数 %s' % (metrics['f'], metrics['gamma'],
                                          metrics.get('n_params')))
    L.append('')
    L.append('%-6s %8s %12s %10s %10s %10s' %
             ('split', 'n', 'R²', 'RMSE', 'MAE', 'MAPE%'))
    for name, d in metrics['splits'].items():
        m = d['power']
        L.append('%-6s %8d %12.6f %10.4f %10.4f %10.3f'
                 % (name, m['n'], m['r2'], m['rmse'], m['mae'], m['mape']))
    L.append('')
    L.append('逐步波动（test）')
    for t in range(5):
        r = metrics['step_metrics']['stage%d' % t]
        L.append('  t=%d  R²=%.5f RMSE=%.4f MAE=%.4f MAPE=%.2f%%'
                 % (t, r['r2'], r['rmse'], r['mae'], r['mape']))
    L.append('')
    if metrics.get('monolithic_f5n'):
        L.append('对照 monolithic f5n：')
        for name in ('test', 'real'):
            if name in metrics['monolithic_f5n']:
                r = metrics['monolithic_f5n'][name]
                L.append('  %-6s R²=%.5f RMSE=%.4f MAPE=%.2f%%'
                         % (name, r['r2'], r['rmse'], r['mape']))
    L.append('基线 E[F5|u1]：')
    for name in ('test', 'real'):
        r = metrics['baseline'][name]
        L.append('  %-6s R²=%.5f RMSE=%.4f MAPE=%.2f%%'
                 % (name, r['r2'], r['rmse'], r['mape']))
    if metrics.get('terminal'):
        L.append('')
        L.append('终端头 B（纯预测目标=0）：')
        for name, d in metrics['terminal'].items():
            L.append('  %-6s mean=%+.4f mean|B|=%.4f max|B|=%.4f'
                     % (name, d['mean'], d['mean_abs'], d['max_abs']))
    L.append('')
    g = metrics['gradient']
    L.append('动作梯度 ∂F5/∂a（test，%d 点，δ=%.3f）' % (g['n'], g['delta']))
    L.append('  %-8s %10s %10s %12s %10s' %
             ('dim', 'pearson', 'slope', 'median_rel', 'sign'))
    for j in range(C.NT):
        q = g['dims']['dim%d' % j]
        L.append('  dim%-5d %10.4f %10.4f %12.4f %10.4f'
                 % (j, q['pearson'], q['slope'], q['median_rel'], q['sign_agree']))
    L.append('  pooled   %10.4f %10.4f %12.4f %10.4f'
             % (g['pooled']['pearson'], g['pooled']['slope'],
                g['pooled']['median_rel'], g['pooled']['sign_agree']))
    cs = g['cosine']
    L.append('  逐样本余弦: mean %.3f median %.3f | >0.8 %.1f%% | <0 %.1f%%'
             % (cs['cos_mean'], cs['cos_median'], 100 * cs['cos_gt08'],
                100 * cs['cos_neg']))
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description='评价 step-decomp critic')
    ap.add_argument('--ckpt',
                    default='agentutil/critictest/results/f5_stepdecomp/step_decomp_best.pt')
    ap.add_argument('--data', default='agentutil/critictest/data/aifflor_ua_f5.npz')
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
