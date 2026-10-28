#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""用五步块内风速训练**块总 c1** critic，重点检查行为梯度。

    C1 = Σ_{t=0..4} [ PF_t + 20·max(0, PF_t−3) ]
    PF_0 = |P(u1,a)−P(u0,a)|, PF_t = |P(u_{t+1},a)−P(u_t,a)|

状态：
  * `useq`  = u1..u5（块内五步，缺 u0）
  * `useq6` = u0..u5（含第一项所需的 u0）

目标：
  * `c1`    = C1/10（原始尺度）
  * `logc1` = log(1+C1)（压缩重尾，反变换 expm1）

用法
----
    python agentutil/critictest/train_c1.py --state useq6 --target logc1 \
        --data agentutil/critictest/data/aifflor_ua_f5_scalar.npz \
        --out  agentutil/critictest/results/c1_useq6_log
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

C1_SCALE = 10.0
STATE_KINDS = ('useq', 'useq6')
TARGET_KINDS = ('c1', 'logc1')


def c1_from_steps(p_steps, p_prev):
    """返回 (C1, PF_steps)；PF_steps 形状 (N,5)。"""
    ps = np.asarray(p_steps, dtype=np.float64)
    pp = np.asarray(p_prev, dtype=np.float64)
    pf = np.column_stack([np.abs(ps[:, 0] - pp),
                          np.abs(np.diff(ps, axis=1))])
    c1s = pf + 20.0 * np.maximum(0.0, pf - 3.0)
    return c1s.sum(1), pf


def target_of(c1, kind):
    c1 = np.asarray(c1, dtype=np.float64)
    if kind == 'c1':
        return (c1 / C1_SCALE).astype(np.float32)
    if kind == 'logc1':
        return np.log1p(c1).astype(np.float32)
    raise ValueError(kind)


def inverse_of(out, kind):
    out = np.asarray(out, dtype=np.float64)
    if kind == 'c1':
        return out * C1_SCALE
    if kind == 'logc1':
        return np.expm1(np.clip(out, 0.0, 20.0))
    raise ValueError(kind)


def predict(model, state, a, kind, batch=8192):
    model.eval()
    out = np.empty(state.shape[0], dtype=np.float64)
    with torch.no_grad():
        for s in range(0, state.shape[0], int(batch)):
            e = min(s + int(batch), state.shape[0])
            q = model([torch.as_tensor(state[s:e]),
                       torch.as_tensor(a[s:e].astype(np.float32))])
            out[s:e] = q.cpu().numpy().reshape(-1)
    return inverse_of(out, kind)


def main():
    ap = argparse.ArgumentParser(description='PER critic 五步块 C1')
    ap.add_argument('--data', default='agentutil/critictest/data/aifflor_ua_f5_scalar.npz')
    ap.add_argument('--out', default='agentutil/critictest/results/c1_useq6_log')
    ap.add_argument('--state', default='useq6', choices=STATE_KINDS)
    ap.add_argument('--target', default='logc1', choices=TARGET_KINDS)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--bsize', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eta_min', type=float, default=1e-4)
    ap.add_argument('--hidden1', type=int, default=16)
    ap.add_argument('--hidden2', type=int, default=100)
    ap.add_argument('--tau', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=8000)
    ap.add_argument('--no_eval', action='store_true')
    args = ap.parse_args()

    C.set_seeds(args.seed)
    os.makedirs(args.out, exist_ok=True)
    ds = np.load(args.data, allow_pickle=False)
    meta_data = json.loads(str(ds['meta']))
    split = ds['split']
    a = ds['a'].astype(np.float64)
    c1, pf_steps = c1_from_steps(ds['p_steps'], ds['p_prev'])
    state = C.build_state(ds, args.state)
    y = target_of(c1, args.target)

    m_tr, m_va = split == 0, split == 1
    xt = torch.as_tensor(state)
    at = torch.as_tensor(a.astype(np.float32))
    yt = torch.as_tensor(y).reshape(-1, 1)
    print('=' * 104)
    print('五步 C1 critic  state=%s  target=%s' % (args.state, args.target))
    print('=' * 104)
    print('train %d / val %d / test %d / real %d | C1 mean=%.3f sd=%.3f '
          'max=%.1f | PF>3 步占比 %.3f%%'
          % (m_tr.sum(), m_va.sum(), (split == 2).sum(), (split == 3).sum(),
             c1[m_tr].mean(), c1[m_tr].std(), c1[m_tr].max(),
             100.0 * (pf_steps[m_tr] > 3).mean()))

    ns = state.shape[1]
    bias0 = float(yt[m_tr].mean())
    q, qt = C.make_critic_pair(ns, 1, args.hidden1, args.hidden2,
                               seed=args.seed, bias=bias0)
    n_par = sum(p.numel() for p in q.parameters())
    print('state_dim=%d | 参数 %d | bias0=%.5f' % (ns, n_par, bias0))
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.steps), eta_min=args.eta_min)

    idx_tr = np.flatnonzero(m_tr)
    xva = state[m_va]
    a_va = a[m_va]
    c1_va = c1[m_va]

    def val_eval(model):
        pred = predict(model, xva, a_va, args.target)
        return C.regression_metrics(pred, c1_va)

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
        torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=5.0)
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
                meta = dict(state_kind=args.state, target_kind=args.target,
                            c1_scale=C1_SCALE, state_dim=ns,
                            hidden1=args.hidden1, hidden2=args.hidden2,
                            n_params=n_par, bias0=bias0, best_step=step,
                            best_val_rmse=r['rmse'],
                            data=os.path.abspath(args.data),
                            data_meta=meta_data, train_args=vars(args))
                torch.save({'model': q.state_dict(), 'target': qt.state_dict(),
                            'meta': meta},
                           os.path.join(args.out, 'c1_best.pt'))
            else:
                since += args.eval_every
            if args.patience and since >= args.patience:
                print('  [early stop] 连续 %d 步未改善' % since)
                break

    secs = time.time() - t0
    print('\n训练结束：%d 步 / %.1fs | best step=%d val RMSE=%.4f MW'
          % (step, secs, best['step'], best['rmse']))
    if hist:
        with open(os.path.join(args.out, 'history.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
            w.writeheader(); w.writerows(hist)
    ckpt = os.path.join(args.out, 'c1_best.pt')
    if not os.path.exists(ckpt):
        meta = dict(state_kind=args.state, target_kind=args.target,
                    c1_scale=C1_SCALE, state_dim=ns,
                    hidden1=args.hidden1, hidden2=args.hidden2,
                    n_params=n_par, bias0=bias0, best_step=step,
                    best_val_rmse=float('nan'),
                    data=os.path.abspath(args.data), data_meta=meta_data,
                    train_args=vars(args))
        torch.save({'model': q.state_dict(), 'target': qt.state_dict(),
                    'meta': meta}, ckpt)
    if not args.no_eval:
        import evaluate_c1 as E
        E.evaluate_all(ckpt, args.data, args.out)
    print('\n产物目录：%s' % os.path.abspath(args.out))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
