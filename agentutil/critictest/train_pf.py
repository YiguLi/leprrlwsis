#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""用 PER critic 结构判单步功率波动 `PF=|P(u_cur,a)−P(u_prev,a)|`。

* 网络：`Critic(ns,1,hidden1=16,hidden2=100,arch='v1')` + target（τ 软更新）；
* 状态：`u1`（仅 u_cur）/ `u2`（u_prev,u_cur）/ `u2p`（+P_prev）/
  `u2d`（+Δu,P_prev）；
* 目标：`pf`（PF/5）或 `pfu3`（1000·PF/u_cur³）；
* 单步 PF 在给定 `(u_prev,u_cur,a)` 时是确定性的，所以这是干净的成本判别任务。

用法
----
    python agentutil/critictest/train_pf.py --state u2p --target pf \
        --data agentutil/critictest/data/aifflor_pf_scalar.npz \
        --out  agentutil/critictest/results/pf_u2p
"""
import argparse
import csv
import json
import os
import sys
import time

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

PF_SCALE = 5.0
STATE_KINDS = ('u1', 'u2', 'u2p', 'u2d')
TARGET_KINDS = ('pf', 'pfu3')


def build_state(ds, kind):
    up = ds['u_prev'].astype(np.float64)
    uc = ds['u_cur'].astype(np.float64)
    pp = ds['p_prev'].astype(np.float64)
    if kind == 'u1':
        return (uc / 10.0).reshape(-1, 1).astype(np.float32)
    parts = [up / 10.0, uc / 10.0]
    if kind in ('u2p', 'u2d'):
        parts.append(pp / 5.0)
    if kind == 'u2d':
        parts.append((uc - up) / 10.0)
    return np.column_stack(parts).astype(np.float32)


def target_of(pf, u_cur, kind):
    pf = np.asarray(pf, dtype=np.float64)
    uc = np.asarray(u_cur, dtype=np.float64)
    if kind == 'pf':
        return (pf / PF_SCALE).astype(np.float32)
    if kind == 'pfu3':
        return (C.S_SCALE * pf / np.power(uc, 3)).astype(np.float32)
    raise ValueError(kind)


def inverse_of(out, u_cur, kind):
    out = np.asarray(out, dtype=np.float64)
    uc = np.asarray(u_cur, dtype=np.float64)
    if kind == 'pf':
        return out * PF_SCALE
    if kind == 'pfu3':
        return out * np.power(uc, 3) / C.S_SCALE
    raise ValueError(kind)


def predict(model, state, a, u_cur, kind, batch=8192):
    model.eval()
    out = np.empty(state.shape[0], dtype=np.float64)
    with torch.no_grad():
        for s in range(0, state.shape[0], int(batch)):
            e = min(s + int(batch), state.shape[0])
            q = model([torch.as_tensor(state[s:e]),
                       torch.as_tensor(a[s:e].astype(np.float32))])
            out[s:e] = q.cpu().numpy().reshape(-1)
    return inverse_of(out, u_cur, kind)


def main():
    ap = argparse.ArgumentParser(description='PER critic 单步 PF')
    ap.add_argument('--data', default='agentutil/critictest/data/aifflor_pf_scalar.npz')
    ap.add_argument('--out', default='agentutil/critictest/results/pf_u2p')
    ap.add_argument('--state', default='u2p', choices=STATE_KINDS)
    ap.add_argument('--target', default='pf', choices=TARGET_KINDS)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--bsize', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eta_min', type=float, default=1e-4)
    ap.add_argument('--hidden1', type=int, default=16)
    ap.add_argument('--hidden2', type=int, default=100)
    ap.add_argument('--tau', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=6000)
    ap.add_argument('--no_eval', action='store_true')
    args = ap.parse_args()

    C.set_seeds(args.seed)
    os.makedirs(args.out, exist_ok=True)
    ds = np.load(args.data, allow_pickle=False)
    meta_data = json.loads(str(ds['meta']))
    pf = ds['pf'].astype(np.float64)
    up = ds['u_prev'].astype(np.float64)
    uc = ds['u_cur'].astype(np.float64)
    a = ds['a'].astype(np.float64)
    split = ds['split']
    state = build_state(ds, args.state)
    y = target_of(pf, uc, args.target)

    m_tr, m_va = split == 0, split == 1
    xt = torch.as_tensor(state)
    at = torch.as_tensor(a.astype(np.float32))
    yt = torch.as_tensor(y).reshape(-1, 1)
    print('=' * 104)
    print('单步 PF critic  state=%s  target=%s' % (args.state, args.target))
    print('=' * 104)
    print('train %d / val %d / test %d / real %d | PF mean=%.4f sd=%.4f'
          % (m_tr.sum(), m_va.sum(), (split == 2).sum(), (split == 3).sum(),
             pf[m_tr].mean(), pf[m_tr].std()))

    ns = state.shape[1]
    bias0 = float(yt[m_tr].mean())
    q, qt = C.make_critic_pair(ns, 1, args.hidden1, args.hidden2,
                               seed=args.seed, bias=bias0)
    n_par = sum(p.numel() for p in q.parameters())
    print('state_dim=%d | 参数 %d | target 同构 | bias0=%.5f'
          % (ns, n_par, bias0))
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.steps), eta_min=args.eta_min)

    idx_tr = np.flatnonzero(m_tr)
    idx_va = np.flatnonzero(m_va)
    xva = state[m_va]
    a_va = a[m_va]
    uc_va = uc[m_va]
    pf_va = pf[m_va]

    def val_eval(model):
        pred = predict(model, xva, a_va, uc_va, args.target)
        return C.regression_metrics(pred, pf_va)

    hist = []
    best = dict(rmse=float('inf'), step=0)
    since = 0
    t0 = time.time()
    print('\n  %8s %10s %12s %12s %10s' % ('step', 'lr', 'train_mse', 'val_RMSE', 'val_R2'))
    for step in range(1, args.steps + 1):
        ii = torch.as_tensor(np.random.choice(idx_tr, args.bsize, replace=True))
        loss = torch.mean((q([xt[ii], at[ii]]) - yt[ii]) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=2.0)
        opt.step()
        sched.step()
        if args.tau > 0:
            C.soft_update(qt, q, args.tau)
        else:
            C.hard_update(qt, q)
        if step % args.eval_every == 0 or step == args.steps:
            r = val_eval(q)
            hist.append(dict(step=step, lr=opt.param_groups[0]['lr'],
                             train_mse=float(loss.item()),
                             val_rmse=r['rmse'], val_r2=r['r2']))
            print('  %8d %10.2e %12.6f %12.4f %10.6f'
                  % (step, opt.param_groups[0]['lr'], loss.item(),
                     r['rmse'], r['r2']))
            if r['rmse'] < best['rmse'] - 1e-4:
                best = dict(rmse=r['rmse'], step=step)
                since = 0
                torch.save({'model': q.state_dict(), 'target': qt.state_dict(),
                            'meta': dict(state_kind=args.state,
                                         target_kind=args.target,
                                         pf_scale=PF_SCALE, state_dim=ns,
                                         hidden1=args.hidden1,
                                         hidden2=args.hidden2,
                                         n_params=n_par, bias0=bias0,
                                         best_step=step,
                                         best_val_rmse=r['rmse'],
                                         data=os.path.abspath(args.data),
                                         data_meta=meta_data,
                                         train_args=vars(args))},
                           os.path.join(args.out, 'pf_best.pt'))
            else:
                since += args.eval_every
            if args.patience and since >= args.patience:
                print('  [early stop] 连续 %d 步未改善' % since)
                break

    secs = time.time() - t0
    print('\n训练结束：%d 步 / %.1fs（%.2f ms/step）| best step=%d val RMSE=%.4f MW'
          % (step, secs, 1000.0 * secs / max(step, 1), best['step'], best['rmse']))
    if hist:
        with open(os.path.join(args.out, 'history.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
            w.writeheader()
            w.writerows(hist)
    ckpt = os.path.join(args.out, 'pf_best.pt')
    if not os.path.exists(ckpt):
        torch.save({'model': q.state_dict(), 'target': qt.state_dict(),
                    'meta': dict(state_kind=args.state, target_kind=args.target,
                                 pf_scale=PF_SCALE, state_dim=ns,
                                 hidden1=args.hidden1, hidden2=args.hidden2,
                                 n_params=n_par, bias0=bias0, best_step=step,
                                 best_val_rmse=float('nan'),
                                 data=os.path.abspath(args.data),
                                 data_meta=meta_data, train_args=vars(args))},
                   ckpt)
    if not args.no_eval:
        import evaluate_pf as E
        E.evaluate_all(ckpt, args.data, args.out)
    print('\n产物目录：%s' % os.path.abspath(args.out))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
