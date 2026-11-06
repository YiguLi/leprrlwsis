#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""训练逐步奖励分解 critic（五步功率波动和）。

架构与损失见 `step_decomp.py` 的模块注释；本脚本负责：
  * 从 `aifflor_ua_f5.npz` 取 `p_steps/p_prev` 还原逐步波动目标；
  * 损失 `L = L_sum + λ_r L_r + λ_B L_B`
      - `L_sum`：组合后的 F5 对真实 F5（对应文档的 L_Q，纯监督无 TD）；
      - `L_r`  ：逐步波动 Huber（除以每步 σ_t，对应文档 §8.1）；
      - `L_B`  ：终端价值对 0（纯预测任务没有下一宏状态）；
  * target 网络按 τ 软更新（纯监督不参与 loss，只作一致性检查）。

用法
----
    python agentutil/critictest/train_step_decomp.py \
        --data agentutil/critictest/data/aifflor_ua_f5.npz \
        --out  agentutil/critictest/results/f5_stepdecomp
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
import torch.nn.functional as F  # noqa: E402

import common as C  # noqa: E402
import step_decomp as SD  # noqa: E402


def _split(ds):
    sp = ds['split']
    return sp == 0, sp == 1, sp == 2, sp == 3


def main():
    ap = argparse.ArgumentParser(description='逐步奖励分解 critic：五步波动和')
    ap.add_argument('--data', default='agentutil/critictest/data/aifflor_ua_f5.npz')
    ap.add_argument('--out', default='agentutil/critictest/results/f5_stepdecomp')
    ap.add_argument('--arch', default='film', choices=('film', 'per'),
                    help='film=文档的 FiLM/256 主体；per=用户给定的 PER 隐藏层')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--bsize', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eta_min', type=float, default=1e-4)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--lambda_r', type=float, default=1.0)
    ap.add_argument('--lambda_b', type=float, default=0.1)
    ap.add_argument('--huber_delta', type=float, default=1.0)
    ap.add_argument('--tau', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=6000)
    ap.add_argument('--min_delta', type=float, default=1e-4)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--no_eval', action='store_true')
    args = ap.parse_args()

    C.set_seeds(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    print('=' * 104)
    print('step-decomp critic：五步功率波动和  arch=%s  γ=%.3f  λ_r=%.3f  λ_B=%.3f'
          % (args.arch, args.gamma, args.lambda_r, args.lambda_b))
    print('=' * 104)
    ds = np.load(args.data, allow_pickle=False)
    if 'f5' not in ds.files:
        raise SystemExit('需要五步波动数据集（含 f5 列）')
    meta_data = json.loads(str(ds['meta']))
    f_steps, f5 = SD.load_step_targets(ds)
    m_tr, m_va, m_te, m_re = _split(ds)
    x, a = SD.state_action(ds)
    xt = torch.as_tensor(x)
    at = torch.as_tensor(a)
    st = torch.as_tensor(f_steps)
    yt = torch.as_tensor(f5).reshape(-1, 1)

    sigma = f_steps[m_tr].std(axis=0).astype(np.float32)
    sigma = np.maximum(sigma, 1e-3)
    step_bias = float(f_steps[m_tr].mean())
    sig_t = torch.as_tensor(sigma)
    print('数据 %s' % os.path.abspath(args.data))
    print('  train %d / val %d / test %d / real %d | F5 mean=%.3f sd=%.3f'
          % (m_tr.sum(), m_va.sum(), m_te.sum(), m_re.sum(),
             float(f5[m_tr].mean()), float(f5[m_tr].std())))
    print('  逐步波动 σ_t = %s | 步波动均值 %.3f'
          % (np.round(sigma, 4).tolist(), step_bias))

    cls = (SD.StepDecompCriticPER if args.arch == 'per'
           else SD.StepDecompCritic)
    model = cls(state_dim=1, action_dim=5, f=5,
                gamma=args.gamma, step_bias=step_bias)
    target = cls(state_dim=1, action_dim=5, f=5,
                 gamma=args.gamma, step_bias=step_bias)
    SD.hard_update(target, model)
    n_par = sum(p.numel() for p in model.parameters())
    print('  网络参数 %d（target 同构）' % n_par)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.steps), eta_min=args.eta_min)

    idx_tr = np.flatnonzero(m_tr)
    idx_va = np.flatnonzero(m_va)
    xva, ava = xt[idx_va], at[idx_va]
    yva = f5[idx_va]
    sva = f_steps[idx_va]

    def val_eval(m):
        q, step, term = SD.predict(m, xva.numpy(), ava.numpy())
        rq = C.regression_metrics(q, yva)
        step_rmse = float(np.sqrt(np.mean((step - sva) ** 2)))
        return rq, step_rmse, float(np.mean(np.abs(term))), q, step, term

    hist = []
    best = dict(rmse=float('inf'), step=0)
    since = 0
    t0 = time.time()
    print('\n  %8s %10s %10s %10s %10s %12s %10s %10s' %
          ('step', 'lr', 'L_sum', 'L_r', 'L_B', 'val_RMSE', 'val_R2', 'B_abs'))
    for step in range(1, args.steps + 1):
        idx = torch.as_tensor(np.random.choice(idx_tr, args.bsize, replace=True))
        xb, ab, sb, yb = xt[idx], at[idx], st[idx], yt[idx]
        q_pred, step_pred, term_pred = model(xb, ab)
        l_sum = F.huber_loss(q_pred.reshape(-1, 1) - yb, torch.zeros_like(yb),
                             delta=args.huber_delta, reduction='mean')
        l_r = F.huber_loss((step_pred - sb) / sig_t[None, :],
                           torch.zeros_like(sb), delta=args.huber_delta,
                           reduction='mean')
        l_b = F.huber_loss(term_pred.reshape(-1, 1), torch.zeros_like(yb),
                           delta=args.huber_delta, reduction='mean')
        loss = l_sum + args.lambda_r * l_r + args.lambda_b * l_b
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        opt.step()
        sched.step()
        if args.tau > 0:
            SD.soft_update(target, model, args.tau)
        else:
            SD.hard_update(target, model)

        if step % args.eval_every == 0 or step == args.steps:
            rq, step_rmse, b_abs, _, _, _ = val_eval(model)
            rq_t, _, _, _, _, _ = val_eval(target)
            hist.append(dict(step=step, lr=opt.param_groups[0]['lr'],
                             l_sum=float(l_sum.item()), l_r=float(l_r.item()),
                             l_b=float(l_b.item()), val_rmse=rq['rmse'],
                             val_r2=rq['r2'], val_step_rmse=step_rmse,
                             b_abs=b_abs, target_val_rmse=rq_t['rmse']))
            print('  %8d %10.2e %10.4f %10.4f %10.4f %12.4f %10.6f %10.4f'
                  % (step, opt.param_groups[0]['lr'], l_sum.item(), l_r.item(),
                     l_b.item(), rq['rmse'], rq['r2'], b_abs))
            if rq['rmse'] < best['rmse'] - args.min_delta:
                best = dict(rmse=rq['rmse'], step=step)
                since = 0
                meta = dict(arch=args.arch, f=5, gamma=args.gamma,
                            stage_dim=SD.STAGE_DIM,
                            state_dim=1, action_dim=5, step_bias=step_bias,
                            sigma=sigma.tolist(), n_params=n_par,
                            best_step=step, best_val_rmse=rq['rmse'],
                            data=os.path.abspath(args.data),
                            data_meta=meta_data,
                            train_args=vars(args))
                torch.save({'model': model.state_dict(),
                            'target': target.state_dict(),
                            'meta': meta},
                           os.path.join(args.out, 'step_decomp_best.pt'))
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
    ckpt = os.path.join(args.out, 'step_decomp_best.pt')
    if not os.path.exists(ckpt):
        torch.save({'model': model.state_dict(), 'target': target.state_dict(),
                    'meta': dict(arch=args.arch, f=5, gamma=args.gamma,
                                 stage_dim=SD.STAGE_DIM,
                                 state_dim=1, action_dim=5, step_bias=step_bias,
                                 sigma=sigma.tolist(), n_params=n_par,
                                 best_step=step, best_val_rmse=float('nan'),
                                 data=os.path.abspath(args.data),
                                 data_meta=meta_data, train_args=vars(args))},
                   ckpt)
    if not args.no_eval:
        import evaluate_step_decomp as E
        E.evaluate_all(ckpt, args.data, args.out)
    print('\n产物目录：%s' % os.path.abspath(args.out))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
