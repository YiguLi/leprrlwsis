#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch size 扫描：把"梯度估计更准"与"多学了几遍"分开。

问题
----
`--bsize` 翻倍时，若**更新次数不变**，则单位 env 时间消耗的数据也翻倍 ⇒
"遍历遍数 ∝ bsize"。我们之前测到的 32→64 使 val MSE −5.1% 就是**这两种效应混在一起**的
结果。本探针用两种模式把她们分开：

  * fixed_samples : **固定总样本数**（bsize 翻倍 ⇒ 步数减半）——
                    这一列只反映"梯度估计更准"，不含"多学一遍"。
  * fixed_steps   : 固定更新次数（bsize 翻倍 ⇒ 样本数翻倍）——
                    复现此前那种"混合"对比。

方法
----
* 数据: run12 (TD3PRRBB, 已停止, 文件终态) 的逐 step 日志, 聚合成 H 层**块级**转移
  (5 步一块): state = `Normalize(observation)` (4 维, 与训练同口径),
  action = `original_action_H` (6 维), target = 块内 r1 之和 (profit 通道)。
* 网络/优化器与项目一致: `model.Critic`(4,6,16,100), Adam(lr=0.01), clip_norm=2.0,
  目标归一化 r1/1000 (`hc.VALUE_SCALE`)。
* 指标: 留出集(20%)上的 val R² / val MSE(原始口径) / mean|dQ/da| / fc5 死 ReLU 比例。
* 3 个 seed 取中位, 避免单次实现误导。

注意: 这一步**不依赖 MATLAB**。
"""
import io
import csv as _csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
from torch.optim import Adam                                        # noqa: E402

from model import Critic                                            # noqa: E402
import hsrl_cost as hc                                              # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

CSV = 'output/TD3PRRBB-run12/df_td3prrbb.csv'
BSIZES = [64, 128, 256]
SEEDS = 3
BASE = 64 * 12000          # 固定样本预算 = 768k 样本
BLOCK = 5


def vec(s):
    return np.array([float(x) for x in s.strip().strip('[]').split()], dtype=np.float64)


def Normalize(o):          # 与 td3prrbb_myfunc.Normalize 一致
    return np.array([o[0] / 1000., o[1] / 10., o[2] / 10., o[3] / 10.])


print('=' * 96)
print('数据: 从 run12 重建 H 层块级转移')
print('=' * 96)
with io.open(CSV, encoding='utf-8', newline='') as fh:
    rows = list(_csv.reader(fh))
ncol, bad = len(rows[0]), sum(1 for r in rows[1:] if len(r) != len(rows[0]))
assert bad == 0, ('字段数不符的行 %d' % bad)
hdr = rows[0]
i_obs, i_act, i_r1, i_st = (hdr.index('observation'), hdr.index('original_action_H'),
                            hdr.index('r1'), hdr.index('step'))
S, A, R = [], [], []
cur_b, acc, s0, a0 = None, 0.0, None, None
for r in rows[1:]:
    b = int(float(r[i_st])) // BLOCK
    if b != cur_b:
        if cur_b is not None and s0 is not None:
            S.append(Normalize(s0)); A.append(a0); R.append(acc)
        cur_b, acc, s0, a0 = b, 0.0, None, None
    if s0 is None:
        s0, a0 = vec(r[i_obs]), vec(r[i_act])
    acc += float(r[i_r1])
if s0 is not None:
    S.append(Normalize(s0)); A.append(a0); R.append(acc)
S = np.asarray(S, dtype=np.float32)
A = np.asarray(A, dtype=np.float32)
R = np.asarray(R, dtype=np.float64)
print('  块数 = %d   state=%d   action=%d   块 r1: mean=%.2f sd=%.2f'
      % (len(S), S.shape[1], A.shape[1], R.mean(), R.std()))

rng = np.random.RandomState(0)
perm = rng.permutation(len(S))
n_tr = int(0.8 * len(S))
itr, ite = perm[:n_tr], perm[n_tr:]
St, At, Rt = (torch.tensor(S[itr]), torch.tensor(A[itr]), torch.tensor(R[itr]))
Se, Ae, Re = (torch.tensor(S[ite]), torch.tensor(A[ite]), torch.tensor(R[ite]))
SCALE = float(hc.VALUE_SCALE)


def run_arm(bsize, steps, seed):
    torch.manual_seed(seed)
    net = Critic(S.shape[1], A.shape[1], 16, 100)
    opt = Adam(net.parameters(), lr=0.01)
    tgt = (Rt / SCALE).view(-1, 1)
    n = len(St)
    for _ in range(steps):
        idx = torch.randint(0, n, (bsize,))
        loss = torch.mean((net([St[idx], At[idx]]) - tgt[idx]) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
        opt.step()
    with torch.no_grad():
        q = net([Se, Ae]).numpy().ravel() * SCALE
        # fc5 死 ReLU 比例 (手工复现 forward: fc1→relu→fc4→cat(a)→fc2→relu→fc5→relu→fc3)
        h1 = net.relu(net.fc1(Se))
        h4 = net.fc4(h1)
        h2 = net.relu(net.fc2(torch.cat([h4, Ae], 1)))
        h5 = net.relu(net.fc5(h2))
        dead5 = float((h5 == 0).float().mean())
    y = R[ite]
    r2 = 1 - ((y - q) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mse = float(np.mean((y - q) ** 2))
    Ae_g = Ae.clone().requires_grad_(True)
    g = torch.autograd.grad(net([Se, Ae_g]).sum(), Ae_g)[0].detach().numpy() * SCALE
    return r2, mse, float(np.abs(g).mean()), dead5


print()
print('=' * 96)
print('A. **固定总样本数** = %d 样本 (只反映"估计更准", 不含"多学一遍")' % BASE)
print('=' * 96)
print('  %-7s %-8s %-10s %-12s %-12s %-12s' % ('bsize', '步数', 'val R²', 'val MSE', 'mean|dQ/da|', 'fc5死ReLU'))
res_fs = {}
for bs in BSIZES:
    steps = BASE // bs
    out = np.array([run_arm(bs, steps, s) for s in range(SEEDS)])
    med = np.median(out, axis=0)
    res_fs[bs] = med
    print('  %-7d %-8d %-10.4f %-12.2f %-12.4f %-12.3f'
          % (bs, steps, med[0], med[1], med[2], med[3]))

print()
print('=' * 96)
print('B. 固定更新次数 = 12000 步 (复现"混合"对比: bsize 翻倍 ⇒ 样本翻倍)')
print('=' * 96)
print('  %-7s %-8s %-10s %-12s %-12s %-12s %s' % ('bsize', '样本数', 'val R²', 'val MSE', 'mean|dQ/da|', 'fc5死ReLU', '遍数'))
res_fx = {}
for bs in BSIZES:
    out = np.array([run_arm(bs, 12000, s) for s in range(SEEDS)])
    med = np.median(out, axis=0)
    res_fx[bs] = med
    ep = bs * 12000 / len(St)
    print('  %-7d %-8d %-10.4f %-12.2f %-12.4f %-12.3f %.0f'
          % (bs, bs * 12000, med[0], med[1], med[2], med[3], ep))

print()
print('=' * 96)
print('判读')
print('=' * 96)
r2_64 = res_fs[64][0]
for bs in BSIZES[1:]:
    d_fs = res_fs[bs][0] - r2_64
    d_fx = res_fx[bs][0] - res_fx[64][0]
    print('  bsize 64 -> %-4d : 固定样本 ΔR² = %+.4f | 固定步数 ΔR² = %+.4f' % (bs, d_fs, d_fx))
print()
print('  * 固定样本下 ΔR² ≈ 0  ⇒ 大 batch 的收益只是"多学了几遍", 不是"估计更准"')
print('    ⇒ 那就该用"更多更新次数"或"更大 rmsize"来拿, 而不是加 bsize(会同步放大遍历遍数)。')
print('  * 固定样本下 ΔR² > 0  ⇒ 大 batch 确实更准 ⇒ 值得加, 并同步放大 rmsize。')
print('  * fc5 死 ReLU 比例若随 bsize 明显下降 ⇒ 大 batch 还顺带缓解了死单元问题。')
