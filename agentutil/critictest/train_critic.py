#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""用 PER 结构的 critic + target 网络**监督拟合**风场出力 `P(u, a)`。

网络（逐层与 PER 的 `model.Critic(arch='v1')` 一致）
------------------------------------------------------
    state x ──► Linear(ns,16) ──► ReLU ──► Linear(16,100) ──(无激活，PER 原样)
                                          │
    action a ─────────────────────────────┴──► cat ──► Linear(105,80) ──► ReLU
                                                     ──► Linear(80,50) ──► ReLU
                                                     ──► Linear(50,1)
    target critic：同一结构，`hard_update` 同步初值，训练中每步 `soft_update(τ)`。

监督口径
--------
* 主目标 `s = 1000·P/u³`：aifflor 是准稳态 FLORIS，`P ∝ u³`（尾流只改变
  `u_eff/u`），所以把精确的 u³ 提到网络外，网络只学 5 维动作响应
  `s(u,a)`；预测时 `P̂ = ŝ·u³/1000`。
* 消融 `--target p`：直接回归 `P/10`；`--target log`：回归 `log P`。
* 单步确定性标签 ⇒ PER 里"用 target 网络算 bootstrap 目标"这一项自然消失
  （等价于 γ=0）。target 网络仍按 PER 的 τ 软更新维护，作为 EMA 副本做
  一致性/稳定性检查（`evaluate.py` 会报 online vs target 的差）。

用法
----
    python agentutil/critictest/train_critic.py \
        --data agentutil/critictest/data/aifflor_ua_p.npz \
        --out  agentutil/critictest/results/sp
    python agentutil/critictest/train_critic.py --target p --out .../results/p
"""
import argparse
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


def _split(ds):
    sp = ds['split']
    return (sp == 0, sp == 1, sp == 2, sp == 3)


def _as_tensors(u, a, s, feature, target):
    return (torch.as_tensor(C.state_features(u, feature)),
            torch.as_tensor(a.astype(np.float32)),
            torch.as_tensor(s.astype(np.float32)).reshape(-1, 1))


def load_checkpoint(path):
    """加载 `train_critic.py` 保存的 checkpoint。"""
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


def save_checkpoint(path, q, qt, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'online': q.state_dict(), 'target': qt.state_dict(), 'meta': meta}, path)


def main():
    ap = argparse.ArgumentParser(description='PER 结构 critic-target 监督拟合风场出力')
    ap.add_argument('--data', default='agentutil/critictest/data/aifflor_ua_p.npz')
    ap.add_argument('--out', default='agentutil/critictest/results/sp')
    ap.add_argument('--target', default='sp', choices=list(C.TARGET_KINDS),
                    help='sp=1000·P/u³（默认）| p=P/10 | log=log P')
    ap.add_argument('--state_feat', default='u10', choices=list(C.FEATURE_KINDS),
                    help='u10=[u/10]（默认）| u3=[u,u²,u³]/10 的消融')
    ap.add_argument('--hidden1', type=int, default=16, help='PER 默认 16')
    ap.add_argument('--hidden2', type=int, default=100, help='PER 默认 100')
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--bsize', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eta_min', type=float, default=1e-4,
                    help='余弦退火终点 lr；<=0 ⇒ 不挂 scheduler')
    ap.add_argument('--tau', type=float, default=1e-3, help='target 软更新系数（PER 口径）')
    ap.add_argument('--clip', type=float, default=2.0, help='critic 梯度裁剪（PER=2.0）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=8000,
                    help='val RMSE 连续多少步没改善就早停（0=不早停）')
    ap.add_argument('--min_delta', type=float, default=1e-4,
                    help='val RMSE 改善小于该值( MW )不算改善')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--no_eval', action='store_true', help='训练完不跑 evaluate.py')
    args = ap.parse_args()

    C.set_seeds(args.seed)
    C.set_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    print('=' * 104)
    print('critictest 训练：PER critic-target 监督拟合  target=%s  state_feat=%s'
          % (args.target, args.state_feat))
    print('=' * 104)
    ds = np.load(args.data, allow_pickle=False)
    meta_data = json.loads(str(ds['meta']))
    m_tr, m_va, m_te, m_re = _split(ds)
    u, a, p = ds['u'], ds['a'], ds['p']
    s_np = C.target_transform(p, u, args.target)
    print('数据 %s' % os.path.abspath(args.data))
    print('  train %d / val %d / test %d / real %d | P=[%.3f, %.3f] MW'
          % (m_tr.sum(), m_va.sum(), m_te.sum(), m_re.sum(), p.min(), p.max()))
    print('  目标 %s：train mean=%.5f  sd=%.5f'
          % (args.target, float(s_np[m_tr].mean()), float(s_np[m_tr].std())))

    Xtr, Atr, Str = _as_tensors(u[m_tr], a[m_tr], s_np[m_tr], args.state_feat, args.target)
    Xva, Ava, Sva = _as_tensors(u[m_va], a[m_va], s_np[m_va], args.state_feat, args.target)
    u_va, p_va = u[m_va], p[m_va]

    ns = Xtr.shape[1]
    bias0 = float(Str.mean())
    q, qt = C.make_critic_pair(ns, C.NT, args.hidden1, args.hidden2,
                               seed=args.seed, bias=bias0)
    n_par = sum(x.numel() for x in q.parameters())
    print('\n网络: state %d → %d → %d ∥ action %d → cat %d → 80 → 50 → 1'
          % (ns, args.hidden1, args.hidden2, C.NT, args.hidden2 + C.NT))
    print('      参数量 %d（online；target 同构）| 输出层 bias 初值 %.5f' % (n_par, bias0))

    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    sched = None
    if args.eta_min and args.eta_min > 0:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.steps), eta_min=args.eta_min)

    def val_eval(model, uu, aa, pp, feature, target, batch=8192):
        pred = C.predict_power(model, uu, aa, target=target, feature=feature, batch=batch)
        return C.regression_metrics(pred, pp), pred

    hist = []
    best = dict(rmse=float('inf'), step=0)
    since_best = 0
    t0 = time.time()
    print('\n  %8s %10s %12s %12s %12s %10s' %
          ('step', 'lr', 'train_mse', 'val_RMSE', 'val_R2', 'target_gap'))
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, Xtr.shape[0], (args.bsize,))
        pred = q([Xtr[idx], Atr[idx]])
        loss = torch.mean((pred - Str[idx]) ** 2)
        opt.zero_grad()
        loss.backward()
        if args.clip and args.clip > 0:
            torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=float(args.clip))
        opt.step()
        if sched is not None:
            sched.step()
        # ---- PER 的 target 软更新（每步；与 td3per 的 update_policy 同机制）----
        if args.tau and args.tau > 0:
            C.soft_update(qt, q, float(args.tau))
        else:
            C.hard_update(qt, q)

        if step % args.eval_every == 0 or step == args.steps:
            mva, _ = val_eval(q, u_va, a[m_va], p_va, args.state_feat, args.target)
            mvt, _ = val_eval(qt, u_va, a[m_va], p_va, args.state_feat, args.target)
            lr_now = opt.param_groups[0]['lr']
            gap = abs(mvt['rmse'] - mva['rmse'])
            hist.append(dict(step=step, lr=lr_now,
                             train_mse=float(loss.item()),
                             val_rmse_mw=mva['rmse'], val_r2=mva['r2'],
                             target_val_rmse_mw=mvt['rmse'],
                             target_gap_mw=gap))
            print('  %8d %10.2e %12.5f %12.4f %12.6f %10.4f'
                  % (step, lr_now, loss.item(), mva['rmse'], mva['r2'], gap))
            if mva['rmse'] < best['rmse'] - args.min_delta:
                best = dict(rmse=mva['rmse'], step=step)
                since_best = 0
                save_checkpoint(os.path.join(args.out, 'critic_best.pt'), q, qt,
                                meta=dict(
                                    target=args.target, feature=args.state_feat,
                                    ns=ns, na=C.NT,
                                    hidden1=args.hidden1, hidden2=args.hidden2,
                                    init_w=3e-3, bias0=bias0, n_params=n_par,
                                    best_step=step, best_val_rmse_mw=mva['rmse'],
                                    data=os.path.abspath(args.data),
                                    data_meta=meta_data,
                                    train_args=vars(args),
                                ))
            else:
                since_best += args.eval_every
            if args.patience and since_best >= args.patience:
                print('  [early stop] 连续 %d 步 val RMSE 未改善（best step=%d, %.4f MW）'
                      % (since_best, best['step'], best['rmse']))
                break

    train_secs = time.time() - t0
    print('\n训练结束：%d 步 / %.1fs（%.2f ms/step）| best step=%d val RMSE=%.4f MW'
          % (step, train_secs, 1000.0 * train_secs / max(step, 1),
             best['step'], best['rmse']))

    # 历史落盘
    if hist:
        import csv
        with open(os.path.join(args.out, 'history.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
            w.writeheader()
            w.writerows(hist)

    ckpt = os.path.join(args.out, 'critic_best.pt')
    if not os.path.exists(ckpt):
        # 极端情况（steps<eval_every 且从未 eval）也保存一次
        save_checkpoint(ckpt, q, qt, meta=dict(
            target=args.target, feature=args.state_feat, ns=ns, na=C.NT,
            hidden1=args.hidden1, hidden2=args.hidden2, init_w=3e-3,
            bias0=bias0, n_params=n_par, best_step=step,
            best_val_rmse_mw=float('nan'), data=os.path.abspath(args.data),
            data_meta=meta_data, train_args=vars(args)))

    if not args.no_eval:
        import evaluate as E
        E.evaluate_all(ckpt_path=ckpt, data_path=args.data, out_dir=args.out,
                       n_grad=400, delta=0.01, make_plots=True)
    print('\n产物目录：%s' % os.path.abspath(args.out))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
