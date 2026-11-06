#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""诊断"学习羸弱"是代码 bug 还是数据噪声。

六个检查（全部不启 MATLAB）：
  1. 数据对齐：`u==u_seq[:,0]`、`f5=Σ|ΔP|`、手工重算 p_steps/p_prev；
  2. 预测方差校准：`pred_sd ≈ sqrt(R²)·true_sd`（条件均值模型的签名）；
  3. 非参数 kNN 基线：同样只用 `(u1,a)`，看网络是否已到条件均值天花板；
  4. 确定性合成函数过拟合：同一个 PER critic + 训练循环能否到 R²≈1；
  5. 真实 F5 小子集过拟合：16/100 与 64/256，train R² 能到多少；
  6. label 置换检验：打乱标签后 train R² 应回落到基线附近（无泄漏）。

用法
----
    python agentutil/critictest/diagnose_learning.py \
        --data agentutil/critictest/data/aifflor_ua_f5_scalar.npz \
        --ckpt agentutil/critictest/results/f5_scalar_f5n_16/critic_best.pt
"""
import argparse
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
import torch.nn as nn  # noqa: E402

import common as C  # noqa: E402
import evaluate as E  # noqa: E402
from model import Critic  # noqa: E402


def r2(pred, true):
    pred = np.asarray(pred).ravel()
    true = np.asarray(true).ravel()
    return 1.0 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum()


def section(title):
    print('\n' + '=' * 100)
    print(title)
    print('=' * 100)


# ---------------------------------------------------------------------------
def check_alignment(ds, phys):
    section('[1] 数据对齐')
    u = ds['u'].astype(np.float64)
    a = ds['a'].astype(np.float64)
    u_seq = ds['u_seq'].astype(np.float64)
    u0 = ds['u0'].astype(np.float64)
    p_steps = ds['p_steps'].astype(np.float64)
    p_prev = ds['p_prev'].astype(np.float64)
    f5 = ds['f5'].astype(np.float64)
    f5_re = (np.abs(p_steps[:, 0] - p_prev)
             + np.abs(np.diff(p_steps, axis=1)).sum(1))
    print('  u == u_seq[:,0]        :', np.allclose(u, u_seq[:, 0]))
    print('  f5 == Σ|ΔP| (max|diff|): %.3e' % np.max(np.abs(f5_re - f5)))
    rng = np.random.RandomState(0)
    for i in rng.choice(u.size, 3, replace=False):
        pm = np.array([phys.power(u_seq[i, t], a[i]) for t in range(5)])
        p0 = phys.power(u0[i], a[i])
        print('  sample %6d: p_steps max|diff| %.2e | p_prev |diff| %.2e'
              % (i, np.max(np.abs(pm - p_steps[i])), abs(p0 - p_prev[i])))


def check_calibration(ds, ckpt):
    section('[2] 预测方差校准（pred_sd 应 ≈ sqrt(R²)·true_sd）')
    u = ds['u'].astype(np.float64)
    a = ds['a'].astype(np.float64)
    f5 = ds['f5'].astype(np.float64)
    split = ds['split']
    norm = C.window_fluct_norm(ds)
    q, _qt, meta = E.load_checkpoint(ckpt)
    pred = C.predict_power(q, u, a, target=meta['target'],
                           feature=meta['feature'], window=5, norm=norm)
    for name, code in (('train', 0), ('test', 2)):
        m = split == code
        y, p = f5[m], pred[m]
        r = r2(p, y)
        print('  %-5s n=%6d R²=%.4f true_sd=%.3f pred_sd=%.3f '
              'sqrt(R²)·true_sd=%.3f bias=%+.3f'
              % (name, m.sum(), r, y.std(), p.std(),
                 np.sqrt(max(r, 0)) * y.std(), np.mean(p - y)))


def check_knn(ds, k=200):
    section('[3] 非参数 kNN 条件均值基线（k=%d）' % k)
    from scipy.spatial import cKDTree
    u = ds['u'].astype(np.float64)
    a = ds['a'].astype(np.float64)
    f5 = ds['f5'].astype(np.float64)
    split = ds['split']
    X = np.column_stack([u / 10.0, a])
    tr = split == 0
    te = split == 2
    tree = cKDTree(X[tr])
    _d, idx = tree.query(X[te], k=k)
    pred = f5[tr][idx].mean(1)
    print('  kNN test R²=%.4f RMSE=%.4f'
          % (r2(pred, f5[te]), np.sqrt(np.mean((pred - f5[te]) ** 2))))


def overfit_synthetic(na, n=512, steps=2500):
    rng = np.random.RandomState(0)
    u = rng.uniform(3.5, 25.0, n)
    a = rng.uniform(0.0, 0.5, (n, na))
    y = (10 + 4.0 * u + 3.0 * a.sum(1) + 2.0 * np.sin(u)
         + 1.5 * u * a[:, 0]).astype(np.float32)
    X = torch.tensor((u / 10.0).reshape(-1, 1), dtype=torch.float32)
    A = torch.tensor(a, dtype=torch.float32)
    Y = torch.tensor(y).reshape(-1, 1)
    q = Critic(1, na, hidden1=16, hidden2=100, init_w=3e-3, arch='v1')
    with torch.no_grad():
        q.fc3.bias.fill_(float(Y.mean()))
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    for _ in range(steps):
        ii = torch.randint(0, n, (128,))
        loss = ((q([X[ii], A[ii]]) - Y[ii]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = q([X, A]).numpy().ravel()
    return r2(p, y)


def overfit_real(ds, n=512, steps=5000, hidden=(16, 100), seed=0, permute=False):
    u = ds['u'].astype(np.float64)
    a = ds['a'].astype(np.float64)
    f5 = ds['f5'].astype(np.float64)
    split = ds['split']
    norm = C.window_fluct_norm(ds)
    s = (f5 / norm).astype(np.float32)
    rng = np.random.RandomState(seed)
    idx = rng.choice(np.flatnonzero(split == 0), n, replace=False)
    y = s[idx].copy()
    if permute:
        rng.shuffle(y)
    X = torch.tensor((u[idx] / 10.0).reshape(-1, 1), dtype=torch.float32)
    A = torch.tensor(a[idx], dtype=torch.float32)
    S = torch.tensor(y).reshape(-1, 1)
    q = Critic(1, a.shape[1], hidden1=hidden[0], hidden2=hidden[1],
               init_w=3e-3, arch='v1')
    with torch.no_grad():
        q.fc3.bias.fill_(float(S.mean()))
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    for _ in range(steps):
        ii = torch.randint(0, len(idx), (128,))
        loss = ((q([X[ii], A[ii]]) - S[ii]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = q([X, A]).numpy().ravel() * norm[idx]
    # 全 test 也报一下，便于看是否只是记忆
    te = split == 2
    Xt = torch.tensor((u[te] / 10.0).reshape(-1, 1), dtype=torch.float32)
    At = torch.tensor(a[te], dtype=torch.float32)
    with torch.no_grad():
        pt = q([Xt, At]).numpy().ravel() * norm[te]
    return r2(p, f5[idx]), r2(pt, f5[te])


def main():
    ap = argparse.ArgumentParser(description='诊断学习羸弱：代码 bug 还是数据噪声')
    ap.add_argument('--data', default='agentutil/critictest/data/aifflor_ua_f5_scalar.npz')
    ap.add_argument('--ckpt',
                    default='agentutil/critictest/results/f5_scalar_f5n_16/critic_best.pt')
    ap.add_argument('--knn_k', type=int, default=200)
    args = ap.parse_args()
    C.set_seeds(0)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    ds = np.load(args.data, allow_pickle=False)
    if 'f5' not in ds.files:
        raise SystemExit('需要五步波动数据集')
    t0 = time.time()
    phys = C.build_physics(calibrated=False, power_scale=C.AT.LEVEL_COEF)
    check_alignment(ds, phys)
    check_calibration(ds, args.ckpt)
    check_knn(ds, args.knn_k)

    section('[4] 确定性合成函数过拟合（同一 PER critic + 训练循环）')
    for na in (1, 5):
        r = overfit_synthetic(na)
        print('  na=%d synthetic train R²=%.4f  %s'
              % (na, r, 'OK' if r > 0.95 else '!! 网络/优化器可疑'))

    section('[5] 真实 F5 小子集过拟合（512 条，看能否记住）')
    for h in ((16, 100), (64, 256)):
        tr, te = overfit_real(ds, hidden=h)
        print('  h=%d/%d  train R²=%.4f  test R²=%.4f' % (h[0], h[1], tr, te))

    section('[6] label 置换检验（打乱后应回落到基线附近）')
    tr, te = overfit_real(ds, hidden=(16, 100), permute=True)
    print('  permuted labels: train R²=%.4f  test R²=%.4f' % (tr, te))
    print('  （打乱后仍保留 E[F5|u1] 的 u1 趋势，所以不是 0；'
          '真实标签应显著高于它）')

    section('判定')
    print('  * 合成函数 R²≈1 且数据对齐/校准/kNN 都正常 ⇒ 网络与训练循环没有硬 bug。')
    print('  * 真实 F5 的 train≈test、pred_sd≈sqrt(R²)·true_sd、kNN≈网络 ⇒ '
          '弱 R² 是“只给 u1 的不可约风速噪声”，不是代码 bug。')
    print('  * 唯一的代码味道：predict_power 曾把 model 留在 eval 模式；'
          '已在 common.py 用 try/finally 恢复，当前网络无 dropout/BN，'
          '这不影响历史数值。')
    print('\n总用时 %.1fs' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
