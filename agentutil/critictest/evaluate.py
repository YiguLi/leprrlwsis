#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""评价 `critic-target` 对风场出力的拟合能力，并做动作梯度体检。

评价内容
--------
1. **出力拟合**：train / val / test / real 四个 split 的 R²、RMSE、MAE、MAPE、
   最大绝对误差、偏差；test 另按风速分箱报告（低风速段单独看，防止高风速
   大功率把 R² 撑高而掩盖低风速的相对误差）。
2. **target 一致性**：PER 的 target 网络与 online 网络在 test 上的差
   （监督单步任务里 target 只作 EMA 副本，不参与 loss）。
3. **动作梯度**：∂P/∂a_j（autograd） vs aifflor 中心差分真值，逐维报
   Pearson r、过原点斜率、中位相对误差、符号一致率。拟合得好但梯度不对，
   对控制/模型-based 用途是无效的，所以这一节不能省。
4. **解析代理对照**：`P ≈ k·u³·Σ4a(1−a)²`（k 在 train 上最小二乘），
   给"网络比简单物理代理多拟合了什么"一个参照。
5. 图：拟合散点、分箱误差、残差-风速、训练曲线、梯度对照、真实动作族。

用法
----
    python agentutil/critictest/evaluate.py \
        --ckpt agentutil/critictest/results/sp/critic_best.pt \
        --data agentutil/critictest/data/aifflor_ua_p.npz \
        --out  agentutil/critictest/results/sp
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

SPLIT_NAMES = ('train', 'val', 'test', 'real')
U_BIN_EDGES = (3.5, 5.0, 7.0, 9.0, 11.0, 13.0, 16.0, 20.0, 25.001)
GRAD_FLOOR = 0.05          # MW/unit：相对误差分母下限


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    meta = ck['meta']
    q = C.make_critic(meta['ns'], meta['na'], meta['hidden1'], meta['hidden2'],
                      meta.get('init_w', 3e-3))
    q.load_state_dict(ck['online'])
    q.eval()
    qt = C.make_critic(meta['ns'], meta['na'], meta['hidden1'], meta['hidden2'],
                       meta.get('init_w', 3e-3))
    qt.load_state_dict(ck['target'])
    qt.eval()
    return q, qt, meta


# ---------------------------------------------------------------------------
# 梯度体检
# ---------------------------------------------------------------------------
def _grad_cmp(gt, gp):
    """逐维比较真值梯度 gt 与网络梯度 gp（一维数组，单位 MW/unit a）。"""
    gt = np.asarray(gt, dtype=np.float64)
    gp = np.asarray(gp, dtype=np.float64)
    if gt.size < 3:
        return dict(n=int(gt.size), pearson=float('nan'), slope=float('nan'),
                    median_rel=float('nan'), sign_agree=float('nan'),
                    mae=float('nan'))
    r = float(np.corrcoef(gt, gp)[0, 1]) if np.std(gt) > 0 and np.std(gp) > 0 else float('nan')
    den = float(np.dot(gt, gt))
    slope = float(np.dot(gt, gp) / den) if den > 0 else float('nan')
    rel = np.abs(gp - gt) / np.maximum(np.abs(gt), GRAD_FLOOR)
    big = np.abs(gt) > 0.1
    sign = float(np.mean(np.sign(gp[big]) == np.sign(gt[big]))) if np.any(big) else float('nan')
    return dict(n=int(gt.size), pearson=r, slope=slope,
                median_rel=float(np.median(rel)), sign_agree=sign,
                mae=float(np.mean(np.abs(gp - gt))))


def gradient_check(q, ds, meta, target, feature, n_grad=400, delta=0.01,
                   seed=123, out_dir=None):
    """autograd ∂P/∂a vs aifflor 中心差分。"""
    u, a = ds['u'].astype(np.float64), ds['a'].astype(np.float64)
    a_eff, p = ds['a_eff'].astype(np.float64), ds['p'].astype(np.float64)
    split = ds['split']
    # 只挑：test 集、动作离边界远、出力不是近零（近零处 P→0 的相对误差没意义）
    cand = ((split == 2) & (a_eff.min(1) >= 0.05) & (a_eff.max(1) <= 0.45)
            & (p >= 0.3))
    idx = np.flatnonzero(cand)
    if idx.size < n_grad:
        idx = np.flatnonzero(split == 2)
    rng = np.random.RandomState(seed)
    sel = rng.choice(idx, size=min(int(n_grad), idx.size), replace=False)

    u_g, ae_g = u[sel], a_eff[sel]
    dm = meta.get('data_meta', {})
    pm = dm.get('physics', {}) if isinstance(dm, dict) else {}
    phys = C.build_physics(calibrated=bool(pm.get('calibrated', False)),
                           power_scale=float(pm.get('power_scale', C.AT.LEVEL_COEF)))

    gt = np.zeros((sel.size, C.NT), dtype=np.float64)
    lo = np.full_like(ae_g, C.A_PHYS_LO)
    for j in range(C.NT):
        # ⚠ 只扰动第 j 维（不能 ap = ae_g + delta，那是"五维同时动"的方向导数）
        ap = ae_g.copy()
        am = ae_g.copy()
        ap[:, j] = np.clip(ae_g[:, j] + delta, lo[:, j], C.A_HI)
        am[:, j] = np.clip(ae_g[:, j] - delta, lo[:, j], C.A_HI)
        pp, _ = C.power_of(phys, u_g, ap, chunk=4096)
        pm_, _ = C.power_of(phys, u_g, am, chunk=4096)
        gt[:, j] = (pp - pm_) / (ap[:, j] - am[:, j])

    x = torch.as_tensor(C.state_features(u_g, feature))
    A = torch.as_tensor(ae_g.astype(np.float32)).requires_grad_(True)
    out = q([x, A])
    g_t = torch.autograd.grad(out.sum(), A)[0].detach().numpy().astype(np.float64)
    if target == 'sp':
        gp = g_t * np.power(u_g, 3)[:, None] / C.S_SCALE
    elif target == 'p':
        gp = g_t * C.P_SCALE
    elif target == 'log':
        p_pred = C.predict_power(q, u_g, ae_g, target=target, feature=feature)
        gp = g_t * p_pred[:, None]
    else:
        raise ValueError(target)

    # autograd vs 网络自身的中心差分（证明梯度计算口径没写错）。
    # ⚠ 用 δ=1e-3 而不是外层 1e-2：ReLU 网络是分段线性，1e-2 的割线会跨多个
    #   折点，和"局部 autograd 导数"本来就不该逐点相等（那是折点效应，不是 bug）。
    net_fd = np.zeros((min(20, sel.size), C.NT), dtype=np.float64)
    nfd = sel.size and min(20, sel.size)
    self_d = 1e-3
    for j in range(C.NT):
        ap = ae_g[:nfd].copy()
        am = ae_g[:nfd].copy()
        ap[:, j] = np.clip(ae_g[:nfd, j] + self_d, lo[:nfd, j], C.A_HI)
        am[:, j] = np.clip(ae_g[:nfd, j] - self_d, lo[:nfd, j], C.A_HI)
        pp = C.predict_power(q, u_g[:nfd], ap, target=target, feature=feature)
        pm_ = C.predict_power(q, u_g[:nfd], am, target=target, feature=feature)
        net_fd[:, j] = (pp - pm_) / (ap[:, j] - am[:, j])
    if nfd:
        _sd = np.abs(net_fd - gp[:nfd])
        autograd_selfcheck = dict(median=float(np.median(_sd)),
                                  p90=float(np.percentile(_sd, 90)),
                                  max=float(np.max(_sd)), delta=self_d)
    else:
        autograd_selfcheck = dict(median=float('nan'), p90=float('nan'),
                                  max=float('nan'), delta=self_d)

    dims = {}
    for j in range(C.NT):
        dims['dim%d' % j] = _grad_cmp(gt[:, j], gp[:, j])
    pooled = _grad_cmp(gt.reshape(-1), gp.reshape(-1))
    keep = min(200, sel.size)
    res = dict(n=int(sel.size), delta=float(delta),
               autograd_vs_netfd=autograd_selfcheck,
               pooled=pooled, dims=dims,
               true_abs_median=float(np.median(np.abs(gt))),
               pred_abs_median=float(np.median(np.abs(gp))),
               true_abs_p90=float(np.percentile(np.abs(gt), 90)),
               pred_abs_p90=float(np.percentile(np.abs(gp), 90)),
               # 只给画图用；evaluate_all 会在写 JSON 前 pop 掉
               plot=dict(gt=gt[:keep].tolist(), gp=gp[:keep].tolist()))
    if out_dir:
        with open(os.path.join(out_dir, 'gradient.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['dim', 'n', 'pearson', 'slope', 'median_rel',
                        'sign_agree', 'mae'])
            for k, v in dims.items():
                w.writerow([k, v['n'], v['pearson'], v['slope'],
                            v['median_rel'], v['sign_agree'], v['mae']])
            w.writerow(['pooled', pooled['n'], pooled['pearson'], pooled['slope'],
                        pooled['median_rel'], pooled['sign_agree'], pooled['mae']])
    return res


# ---------------------------------------------------------------------------
# 主评价
# ---------------------------------------------------------------------------
def evaluate_all(ckpt_path, data_path, out_dir, n_grad=400, delta=0.01,
                 make_plots=True):
    os.makedirs(out_dir, exist_ok=True)
    ds = np.load(data_path, allow_pickle=False)
    dm = json.loads(str(ds['meta']))
    q, qt, meta = load_checkpoint(ckpt_path)
    target, feature = meta['target'], meta['feature']
    u, a, a_eff, p = (ds['u'].astype(np.float64), ds['a'].astype(np.float64),
                      ds['a_eff'].astype(np.float64), ds['p'].astype(np.float64))
    s_true = ds['s'].astype(np.float64)
    split = ds['split']
    family = ds['family']

    print('\n' + '=' * 108)
    print('critictest 评价：%s' % os.path.abspath(ckpt_path))
    print('target=%s | state_feat=%s | 结构 %d→%d→%d ∥ %d → 80 → 50 → 1 | 参数量 %s'
          % (target, feature, meta['ns'], meta['hidden1'], meta['hidden2'],
             C.NT, meta.get('n_params', '?')))
    print('=' * 108)

    metrics = dict(checkpoint=os.path.abspath(ckpt_path),
                   data=os.path.abspath(data_path),
                   target=target, feature=feature,
                   structure=dict(ns=meta['ns'], na=meta['na'],
                                  hidden1=meta['hidden1'],
                                  hidden2=meta['hidden2'],
                                  n_params=meta.get('n_params')),
                   splits={}, per_u_bin=[], per_family={},
                   analytic_baseline={}, online_target_gap={})

    # ---------------- 1. 各 split ----------------
    print('\n[1] 出力拟合（P，单位 MW）')
    print('  %-6s %8s %12s %10s %10s %10s %10s' %
          ('split', 'n', 'R²', 'RMSE', 'MAE', 'MAPE%', 'max|e|'))
    preds = {}
    for code, name in enumerate(SPLIT_NAMES):
        m = split == code
        if not np.any(m):
            continue
        pred = C.predict_power(q, u[m], a[m], target=target, feature=feature)
        tpred = C.predict_power(qt, u[m], a[m], target=target, feature=feature)
        # s 指标一律由**预测的 P** 反算（p/log 臂的原始输出不是 s，不能直接比）
        pred_s = C.S_SCALE * pred / np.power(u[m], 3)
        mP = C.regression_metrics(pred, p[m])
        mT = C.regression_metrics(tpred, p[m])
        mS = C.regression_metrics(pred_s, s_true[m], mape_floor=0.01)
        metrics['splits'][name] = dict(power=mP, target_power=mT, specific_power=mS)
        preds[name] = pred
        preds[name + '_target'] = tpred
        print('  %-6s %8d %12.6f %10.4f %10.4f %10.3f %10.3f'
              % (name, mP['n'], mP['r2'], mP['rmse'], mP['mae'],
                 mP['mape'], mP['max_abs']))
        if name in ('test', 'real'):
            gap = tpred - pred
            metrics['online_target_gap'][name] = dict(
                mean_abs=float(np.mean(np.abs(gap))),
                max_abs=float(np.max(np.abs(gap))),
                rmse=float(np.sqrt(np.mean(gap ** 2))),
            )
            print('        target 网络对照: R²=%.6f RMSE=%.4f | online-target 差 '
                  'mean|·|=%.5f max=%.5f'
                  % (mT['r2'], mT['rmse'],
                     metrics['online_target_gap'][name]['mean_abs'],
                     metrics['online_target_gap'][name]['max_abs']))

    # ---------------- 2. 风速分箱 ----------------
    mte = split == 2
    bins = C.per_bin_metrics(u[mte], preds['test'], p[mte], U_BIN_EDGES)
    metrics['per_u_bin'] = bins
    print('\n[2] test 按风速分箱')
    print('  %-12s %7s %10s %10s %10s' % ('u 区间', 'n', 'R²', 'RMSE', 'MAPE%'))
    for b in bins:
        print('  [%5.1f,%5.1f) %7d %10.5f %10.4f %10.3f'
              % (b['lo'], b['hi'], b['n'], b['r2'], b['rmse'], b['mape']))
    with open(os.path.join(out_dir, 'per_u_bin.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['lo', 'hi', 'n', 'r2', 'rmse', 'mae', 'mape', 'bias', 'mean_true'])
        for b in bins:
            w.writerow([b['lo'], b['hi'], b['n'], b['r2'], b['rmse'], b['mae'],
                        b['mape'], b['bias'], b['mean_true']])

    # ---------------- 3. 真实动作族 ----------------
    mre = split == 3
    print('\n[3] real 按动作族')
    print('  %-10s %7s %12s %10s %10s %10s' %
          ('family', 'n', 'R²', 'RMSE', 'MAPE%', 'mean P'))
    pred_real = preds.get('real')
    for j, fam in enumerate(C.REAL_FAMILIES):
        m = mre & (family == j)
        if not np.any(m):
            continue
        rec = C.regression_metrics(pred_real[family[mre] == j], p[m])
        rec.update(family=fam)
        metrics['per_family'][fam] = rec
        print('  %-10s %7d %12.6f %10.4f %10.3f %10.4f'
              % (fam, rec['n'], rec['r2'], rec['rmse'], rec['mape'], rec['mean_true']))
    with open(os.path.join(out_dir, 'per_family.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['family', 'n', 'r2', 'rmse', 'mae', 'mape', 'bias', 'mean_true'])
        for fam, rec in metrics['per_family'].items():
            w.writerow([fam, rec['n'], rec['r2'], rec['rmse'], rec['mae'],
                        rec['mape'], rec['bias'], rec['mean_true']])

    # ---------------- 4. 动作梯度 ----------------
    print('\n[4] 动作梯度 ∂P/∂a（autograd vs aifflor 中心差分，%d 点，δ=%.3f）'
          % (n_grad, delta))
    gc = gradient_check(q, ds, meta, target, feature, n_grad=n_grad,
                        delta=delta, out_dir=out_dir)
    grad_plot = gc.pop('plot', None)
    metrics['gradient'] = gc
    _sc = gc['autograd_vs_netfd']
    print('  自检 autograd vs 网络自身差分（δ=%.0e）: median|Δ|=%.3e, p90=%.3e, max=%.3e MW/unit'
          % (_sc['delta'], _sc['median'], _sc['p90'], _sc['max']))
    print('  %-8s %6s %10s %10s %12s %10s %10s' %
          ('dim', 'n', 'pearson', 'slope', 'median_rel', 'sign_agree', 'MAE'))
    for j in range(C.NT):
        g = gc['dims']['dim%d' % j]
        print('  dim%-5d %6d %10.4f %10.4f %12.4f %10.4f %10.4f'
              % (j, g['n'], g['pearson'], g['slope'], g['median_rel'],
                 g['sign_agree'], g['mae']))
    gp = gc['pooled']
    print('  pooled   %6d %10.4f %10.4f %12.4f %10.4f %10.4f'
          % (gp['n'], gp['pearson'], gp['slope'], gp['median_rel'],
             gp['sign_agree'], gp['mae']))
    print('  |g| 中位: 真值 %.3f vs 网络 %.3f | p90: %.3f vs %.3f'
          % (gc['true_abs_median'], gc['pred_abs_median'],
             gc['true_abs_p90'], gc['pred_abs_p90']))

    # ---------------- 5. 解析代理基线 ----------------
    mtr = split == 0
    k = C.fit_analytic_k(u[mtr], a_eff[mtr], p[mtr])
    proxy = k * C.analytic_proxy_x(u, a_eff)
    for name, m in (('train', mtr), ('test', mte), ('real', mre)):
        metrics['analytic_baseline'][name] = C.regression_metrics(proxy[m], p[m])
    print('\n[5] 解析代理基线  P≈k·u³·Σ4a(1−a)²（k=%.6f，train 最小二乘；忽略尾流）'
          % k)
    for name in ('train', 'test', 'real'):
        r = metrics['analytic_baseline'][name]
        print('  %-6s n=%6d R²=%.5f RMSE=%.4f MAPE=%.2f%%'
              % (name, r['n'], r['r2'], r['rmse'], r['mape']))
    metrics['analytic_baseline']['k'] = k

    # ---------------- 6. 图 / 汇总 ----------------
    if make_plots:
        try:
            make_figures(out_dir, ds, preds, metrics, meta, grad_plot=grad_plot)
            print('\n[6] 图已写入 %s' % os.path.join(out_dir, 'plots'))
        except Exception as e:  # 画图失败不能掩盖主结果
            print('\n[6] 画图失败（主结果已保存）：%r' % (e,))

    C.save_json(os.path.join(out_dir, 'metrics.json'), metrics)
    summary = build_summary(metrics, meta)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
    print('\n' + '=' * 108)
    print(summary)
    print('=' * 108)
    return metrics


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def make_figures(out_dir, ds, preds, metrics, meta, grad_plot=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9

    pdir = os.path.join(out_dir, 'plots')
    os.makedirs(pdir, exist_ok=True)
    u = ds['u'].astype(np.float64)
    p = ds['p'].astype(np.float64)
    split = ds['split']

    # (a) 拟合散点
    mte = split == 2
    true, pred = p[mte], preds['test']
    m = metrics['splits']['test']['power']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, log in ((axes[0], False), (axes[1], True)):
        ax.scatter(true, pred, s=7, alpha=0.28, edgecolors='none')
        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlabel('aifflor P (MW)')
        ax.set_ylabel('critic P̂ (MW)')
        ax.set_title('test: R²=%.6f, RMSE=%.4f MW' % (m['r2'], m['rmse']))
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(ax.get_title() + ' (log-log)')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, 'fit_test.png'), dpi=140)
    plt.close(fig)

    # (b) 分箱
    bins = metrics['per_u_bin']
    mids = [0.5 * (b['lo'] + b['hi']) for b in bins]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    # R² 都在 0.998~1.000，线性柱状图看不出差别 ⇒ 画 1−R²（对数，越低越好）
    one_minus = [max(1.0 - b['r2'], 1e-10) for b in bins]
    axes[0].bar(mids, one_minus, width=1.4, color='#3b7dd8')
    axes[0].set_yscale('log')
    axes[0].set_ylim(min(one_minus) * 0.7, max(one_minus) * 2.6)
    axes[0].set_ylabel('1 − R²  (log, lower is better)')
    axes[0].set_title('test by wind-speed bin (numbers = n)')
    for x, b, v in zip(mids, bins, one_minus):
        axes[0].text(x, v * 1.07, str(b['n']), ha='center', fontsize=7)
    axes[0].grid(alpha=0.3, axis='y', which='both')
    axes[1].bar(mids, [b['rmse'] for b in bins], width=1.4, color='#d8793b')
    axes[1].set_ylabel('RMSE (MW)')
    axes[1].set_xlabel('u (m/s)')
    axes[1].grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, 'per_u_bin.png'), dpi=140)
    plt.close(fig)

    # (c) 残差 vs u
    u_te = u[mte]
    residual = pred - true
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(u_te, residual, s=6, alpha=0.2, edgecolors='none')
    edges = list(U_BIN_EDGES)
    mu, mm, ms = [], [], []
    for i in range(len(edges) - 1):
        mm_ = (u_te >= edges[i]) & (u_te < edges[i + 1])
        if np.any(mm_):
            mu.append(0.5 * (edges[i] + edges[i + 1]))
            mm.append(float(np.mean(residual[mm_])))
            ms.append(float(np.std(residual[mm_])))
    ax.errorbar(mu, mm, yerr=ms, fmt='o-', color='crimson', lw=1.2,
                capsize=3, label='bin mean ± sd')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('u (m/s)')
    ax.set_ylabel('P̂ − P (MW)')
    ax.set_title('test residual vs wind speed')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, 'residual_vs_u.png'), dpi=140)
    plt.close(fig)

    # (d) 训练曲线
    hp = os.path.join(out_dir, 'history.csv')
    if os.path.exists(hp):
        import csv as _csv
        rows = list(_csv.DictReader(open(hp, encoding='utf-8')))
        if rows:
            st = [int(r['step']) for r in rows]
            vr = [float(r['val_rmse_mw']) for r in rows]
            tr = [float(r['target_val_rmse_mw']) for r in rows]
            r2 = [float(r['val_r2']) for r in rows]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(st, vr, 'o-', ms=3, label='online critic')
            ax.plot(st, tr, 's--', ms=3, label='target critic (EMA)')
            ax.set_xlabel('step')
            ax.set_ylabel('val RMSE (MW)')
            ax.set_title('training curve')
            ax.grid(alpha=0.3)
            ax2 = ax.twinx()
            ax2.plot(st, r2, 'g:', lw=1, label='val R²')
            ax2.set_ylabel('val R²')
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc='best')
            fig.tight_layout()
            fig.savefig(os.path.join(pdir, 'train_curve.png'), dpi=140)
            plt.close(fig)

    # (e) 梯度对照（真值 vs autograd）
    gc = metrics['gradient']
    if grad_plot is not None:
        gt = np.asarray(grad_plot['gt'], dtype=np.float64)
        gp = np.asarray(grad_plot['gp'], dtype=np.float64)
        fig, axes = plt.subplots(1, C.NT, figsize=(4 * C.NT, 4), sharex=False)
        for j, ax in enumerate(axes):
            g = gc['dims']['dim%d' % j]
            ax.scatter(gt[:, j], gp[:, j], s=12, alpha=0.55, edgecolors='none')
            lo = min(gt[:, j].min(), gp[:, j].min())
            hi = max(gt[:, j].max(), gp[:, j].max())
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
            ax.set_title('dim%d  r=%.3f  slope=%.3f'
                         % (j, g['pearson'], g['slope']), fontsize=9)
            ax.set_xlabel('aifflor ∂P/∂a (MW/unit)')
            if j == 0:
                ax.set_ylabel('critic ∂P/∂a (MW/unit)')
            ax.grid(alpha=0.3)
        fig.suptitle('action-gradient parity on test (n=%d)' % gc['n'])
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, 'gradient_parity.png'), dpi=140)
        plt.close(fig)

    # (f) 真实动作族
    fams = list(metrics['per_family'].keys())
    if fams:
        r2s = [metrics['per_family'][f]['r2'] for f in fams]
        rmses = [metrics['per_family'][f]['rmse'] for f in fams]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].bar(fams, [max(1.0 - r, 1e-10) for r in r2s], color='#3b7dd8')
        axes[0].set_yscale('log')
        axes[0].set_ylabel('1 − R²  (log, lower is better)')
        axes[0].set_title('real test by action family')
        axes[0].tick_params(axis='x', rotation=15)
        axes[1].bar(fams, rmses, color='#d8793b')
        axes[1].set_ylabel('RMSE (MW)')
        axes[1].tick_params(axis='x', rotation=15)
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, 'real_families.png'), dpi=140)
        plt.close(fig)

    # (g) online-target 差
    if 'test' in metrics['online_target_gap'] and 'test_target' in preds:
        gap = preds['test_target'] - preds['test']
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(gap, bins=80, color='#6a8caf', alpha=0.9)
        ax.axvline(0.0, color='k', lw=0.8)
        ax.set_xlabel('P̂_target − P̂_online (MW)')
        ax.set_ylabel('count')
        ax.set_title('online vs target critic on test  (mean|Δ|=%.5f MW)'
                     % metrics['online_target_gap']['test']['mean_abs'])
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, 'online_target.png'), dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 汇总文本
# ---------------------------------------------------------------------------
def build_summary(metrics, meta):
    L = []
    L.append('critictest 结果汇总')
    L.append('=' * 100)
    L.append('checkpoint : %s' % metrics['checkpoint'])
    L.append('target     : %s（sp = 1000·P/u³ 后乘回 u³；p = P/10；log = log P）'
             % metrics['target'])
    L.append('state_feat : %s' % metrics['feature'])
    L.append('structure  : %d → %d → %d ∥ %d → 80 → 50 → 1，参数量 %s'
             % (meta['ns'], meta['hidden1'], meta['hidden2'], C.NT,
                metrics['structure'].get('n_params')))
    L.append('')
    L.append('%-6s %8s %12s %10s %10s %10s' %
             ('split', 'n', 'R²', 'RMSE', 'MAE', 'MAPE%'))
    for name, d in metrics['splits'].items():
        m = d['power']
        L.append('%-6s %8d %12.6f %10.4f %10.4f %10.3f'
                 % (name, m['n'], m['r2'], m['rmse'], m['mae'], m['mape']))
    L.append('')
    if 'test' in metrics['online_target_gap']:
        g = metrics['online_target_gap']['test']
        L.append('online vs target（test，PER 的 τ 软更新副本）: '
                 'mean|ΔP|=%.5f MW, max=%.5f MW, RMSE=%.5f MW'
                 % (g['mean_abs'], g['max_abs'], g['rmse']))
    L.append('')
    L.append('动作梯度（test，%d 点，δ=%.3f；单位 MW/unit a）'
             % (metrics['gradient']['n'], metrics['gradient']['delta']))
    L.append('  %-8s %10s %10s %12s %10s' %
             ('dim', 'pearson', 'slope', 'median_rel', 'sign'))
    for j in range(C.NT):
        g = metrics['gradient']['dims']['dim%d' % j]
        L.append('  dim%-5d %10.4f %10.4f %12.4f %10.4f'
                 % (j, g['pearson'], g['slope'], g['median_rel'], g['sign_agree']))
    g = metrics['gradient']
    L.append('  pooled   %10.4f %10.4f %12.4f %10.4f'
             % (g['pooled']['pearson'], g['pooled']['slope'],
                g['pooled']['median_rel'], g['pooled']['sign_agree']))
    L.append('')
    L.append('解析代理基线（k=%.6f，只作参照，不含尾流）' % metrics['analytic_baseline']['k'])
    for name in ('train', 'test', 'real'):
        r = metrics['analytic_baseline'][name]
        L.append('  %-6s R²=%.5f RMSE=%.4f MAPE=%.2f%%'
                 % (name, r['r2'], r['rmse'], r['mape']))
    L.append('')
    L.append('结论口径：')
    L.append('  * R²/RMSE 在 **P（MW）** 上算；sp 臂的 s 指标另存 metrics.json。')
    L.append('  * 覆盖集是 u~U(3.5,25)、a~U(0,0.5)^5；real 用实测风速 + 5 种动作族。')
    L.append('  * 梯度斜率≈1 且 r 高才说明 critic 可用于"按 a 寻优"；'
             'R² 高但梯度差只能说明它拟合了 u³ 的趋势。')
    return '\n'.join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='评价 critic-target 对风场出力的拟合')
    ap.add_argument('--ckpt', default='agentutil/critictest/results/sp/critic_best.pt')
    ap.add_argument('--data', default='agentutil/critictest/data/aifflor_ua_p.npz')
    ap.add_argument('--out', default=None, help='默认 = checkpoint 所在目录')
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
