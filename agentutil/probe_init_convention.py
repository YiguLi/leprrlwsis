#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""初始化口径 A/B：`fanin_init` 到底该用 fan_in 还是 fan_out？

背景
----
`model.Critic` 被改成 5 层（fc1→fc4→fc2[cat a]→fc5→fc3）后，`init_weights`
仍只写 fc1/fc2/fc3。此前把这当成"fc4/fc5 漏初始化"的缺陷。但把每一层的实际
bound 打印出来后，结论相反：

    fanin_init(size) 里 `fanin = fanin or size[0]`，而 nn.Linear 的
    `weight.size()` = (out, in) ⇒ **它一直用的是 fan_out**；
    而 nn.Linear 的 PyTorch 默认是 U(±1/sqrt(fan_in))。

生产维度（--hidden1 16 / --hidden2 100，实测）：

    层          形状       实测 bound   = 1/sqrt(?)
    fc1.weight  (16, 4)    0.2468       1/sqrt(16)  = fan_out   ← 比 fan_in 窄 2×
    fc4.weight  (100, 16)  0.2499       1/sqrt(16)  = fan_in    ← 恰好正确
    fc2.weight  (80, 106)  0.1118       1/sqrt(80)  = fan_out   ← 比 fan_in 宽 15%
    fc5.weight  (50, 80)   0.1118       1/sqrt(80)  = fan_in    ← 恰好正确
    fc3.weight  (1, 50)    0.0029       init_w=3e-3 （有意的小输出层）

⇒ "没被初始化的" fc4/fc5 反而是全网络里唯二符合教科书 fan_in 口径的权重。
⇒ 按原方案把 `fanin_init` 套到 fc4/fc5 会把 fc4 从 0.25 收到 0.10（**窄 2.5×**）。

臂（arms）
---------
用**有监督代理目标**（在 run12 的冻结块级数据上拟合块 r1），在 **bsize 128**、
固定样本预算 768k 下比较 5 种权重口径。偏置全部不动（PyTorch 默认本就 = ±1/sqrt(fan_in)）：

    stock       : 完全不碰初始化（`Critic` 原生）—— **纯抽样噪声对照臂**
    current     : 现状口径的等价重抽。fc1/fc2 = fan_out, fc4/fc5 = PyTorch 默认, fc3 = ±3e-3
    true_fanin  : fc1/fc2/fc4/fc5 全部 U(±1/sqrt(fan_in)), fc3 = ±3e-3   ← "正确口径修法"
    old_scoped  : 现状 + 把 fanin_init(fan_out) 也套到 fc4/fc5          ← "原方案"
    fix_fc1     : 只把 fc1 改成 fan_in, 其余同 current                   ← 归因用消融

方法学（第 2 版，修正第 1 版的两个缺陷）
----------------------------------------
1. 第 1 版用全局 RNG 抽 minibatch，而各臂 `apply_init` 消耗的随机数个数不同
   ⇒ 各臂**看到的 minibatch 序列不同**，臂间差异与"换了条随机流"混在一起。
   本版改用**独立 generator**（`torch.Generator().manual_seed(10000+seed)`）
   抽 minibatch ⇒ 同一 seed 下所有臂看到**逐位相同的 minibatch 序列**（严格配对）。
2. 第 1 版只有 3 个 seed，且臂效应看起来与 seed 方差同量级。
   本版 8 个 seed，并给出 **stock 对照臂**作为"纯抽样噪声"基准：
   若各臂 ΔR² 与 stock 的 ΔR² 同量级，则说明初始化效应在噪声里。

判读要点：初始化差异主要影响**早期**，故在 500 / 1500 / 6000 步都留出去集打分；
若只在早期有差、6000 步抹平，则这个改动不值得付"重启 + 与历史 run 不可比"的代价。

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
BSIZE = 128
STEPS = 6000                      # 128 * 6000 = 768k 样本，与 bsize 探针同预算
SEEDS = 8
BLOCK = 5
CHECKPOINTS = (500, 1500, STEPS)
INIT_W = 3e-3
GEN_BASE = 10000                  # minibatch 专用 generator 的 seed 偏移

ARMS = ('stock', 'current', 'true_fanin', 'old_scoped', 'fix_fc1')


def vec(s):
    return np.array([float(x) for x in s.strip().strip('[]').split()], dtype=np.float64)


def Normalize(o):                 # 与 td3prrbb_myfunc.Normalize 一致 (4 维)
    return np.array([o[0] / 1000., o[1] / 10., o[2] / 10., o[3] / 10.])


# ---------------------------------------------------------------- 数据
print('=' * 100)
print('数据: 从 run12 重建 H 层块级转移')
print('=' * 100)
with io.open(CSV, encoding='utf-8', newline='') as fh:
    rows = list(_csv.reader(fh))
bad = sum(1 for r in rows[1:] if len(r) != len(rows[0]))
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
print('  blocks=%d  state=%d  action=%d  block r1: mean=%.2f sd=%.2f'
      % (len(S), S.shape[1], A.shape[1], R.mean(), R.std()))

rng = np.random.RandomState(0)
perm = rng.permutation(len(S))
n_tr = int(0.8 * len(S))
itr, ite = perm[:n_tr], perm[n_tr:]
St, At, Rt = (torch.tensor(S[itr]), torch.tensor(A[itr]), torch.tensor(R[itr]))
Se, Ae, Re = (torch.tensor(S[ite]), torch.tensor(A[ite]), torch.tensor(R[ite]))
SCALE = float(hc.VALUE_SCALE)
Y = R[ite]


# ---------------------------------------------------------------- 初始化口径
def apply_init(net, arm):
    """按 arm 覆写权重；偏置一律不动（PyTorch 默认即 ±1/sqrt(fan_in)）。

    bound 一律写成 1/sqrt(fan)：fan_out 用 w.shape[0]，fan_in 用 w.shape[1]
    （后者正是 `fanin_init` 本意，也是 nn.Linear 默认）。
    """
    if arm == 'stock':
        return                                    # 不碰：对照臂

    def uni(layer, fan_axis):
        w = layer.weight.data
        v = 1.0 / np.sqrt(float(w.shape[fan_axis]))
        w.uniform_(-v, v)

    net.fc3.weight.data.uniform_(-INIT_W, INIT_W)
    if arm == 'current':
        uni(net.fc1, 0); uni(net.fc2, 0)          # fan_out（现状，复刻 fanin_init）
    elif arm == 'true_fanin':
        uni(net.fc1, 1); uni(net.fc2, 1); uni(net.fc4, 1); uni(net.fc5, 1)
    elif arm == 'old_scoped':
        uni(net.fc1, 0); uni(net.fc2, 0); uni(net.fc4, 0); uni(net.fc5, 0)
    elif arm == 'fix_fc1':
        uni(net.fc1, 1); uni(net.fc2, 0)
    else:
        raise ValueError(arm)


def describe(net):
    return ' '.join('%s=%.4f' % (n.replace('.weight', ''), float(p.abs().max()))
                    for n, p in net.named_parameters() if n.endswith('.weight'))


print()
print('=' * 100)
print('各臂实际生效的权重上界 (seed=0，仅供核对口径)')
print('=' * 100)
for arm in ARMS:
    torch.manual_seed(0)
    net = Critic(S.shape[1], A.shape[1], 16, 100)
    apply_init(net, arm)
    print('  %-12s %s' % (arm, describe(net)))


# ---------------------------------------------------------------- 训练
def eval_metrics(net):
    with torch.no_grad():
        q = net([Se, Ae]).numpy().ravel() * SCALE
        h1 = net.relu(net.fc1(Se))
        h4 = net.fc4(h1)
        h2 = net.relu(net.fc2(torch.cat([h4, Ae], 1)))
        h5 = net.relu(net.fc5(h2))
        d1 = float((h1 == 0).float().mean())
        d2 = float((h2 == 0).float().mean())
        d5 = float((h5 == 0).float().mean())
    r2 = 1 - ((Y - q) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    mse = float(np.mean((Y - q) ** 2))
    Ae_g = Ae.clone().requires_grad_(True)
    g = torch.autograd.grad(net([Se, Ae_g]).sum(), Ae_g)[0].detach().numpy() * SCALE
    return dict(r2=float(r2), mse=mse, dq=float(np.abs(g).mean()),
                d1=d1, d2=d2, d5=d5)


def run_arm(arm, seed):
    torch.manual_seed(seed)
    net = Critic(S.shape[1], A.shape[1], 16, 100)
    apply_init(net, arm)
    opt = Adam(net.parameters(), lr=0.01)
    gen = torch.Generator().manual_seed(GEN_BASE + seed)   # ← 配对关键: 与臂无关
    tgt = (Rt / SCALE).view(-1, 1)
    n = len(St)
    out, done = {}, 0
    for cp in CHECKPOINTS:
        for _ in range(cp - done):
            idx = torch.randint(0, n, (BSIZE,), generator=gen)
            loss = torch.mean((net([St[idx], At[idx]]) - tgt[idx]) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            opt.step()
        done = cp
        out[cp] = eval_metrics(net)
    return out


print()
print('=' * 100)
print('A/B: bsize=%d, 固定样本预算 %d, %d 个 seed, minibatch 序列严格配对'
      % (BSIZE, BSIZE * STEPS, SEEDS))
print('=' * 100)
raw = {}
for arm in ARMS:
    raw[arm] = [run_arm(arm, s) for s in range(SEEDS)]
    print('  %-12s done' % arm)

METRICS = ('r2', 'mse', 'dq', 'd1', 'd2', 'd5')
med = {arm: {cp: {k: float(np.median([raw[arm][s][cp][k] for s in range(SEEDS)]))
                  for k in METRICS} for cp in CHECKPOINTS} for arm in ARMS}

for cp in CHECKPOINTS:
    print()
    print('  ---- step %d (val R2 / val MSE / mean|dQ/da| / dead-ReLU fc1,fc2,fc5) ----' % cp)
    print('  %-12s %-10s %-12s %-12s %-18s' % ('arm', 'val R2', 'val MSE', 'mean|dQ/da|',
                                               'dead fc1/fc2/fc5'))
    for arm in ARMS:
        m = med[arm][cp]
        print('  %-12s %-10.4f %-12.2f %-12.4f %.3f / %.3f / %.3f'
              % (arm, m['r2'], m['mse'], m['dq'], m['d1'], m['d2'], m['d5']))

print()
print('=' * 100)
print('判读: 相对 current 的**逐 seed 配对差** (mean ± sd, n=%d)' % SEEDS)
print('=' * 100)
for cp in CHECKPOINTS:
    print('  step %d:' % cp)
    for arm in ARMS:
        if arm == 'current':
            continue
        d_r2 = np.array([raw[arm][s][cp]['r2'] - raw['current'][s][cp]['r2']
                         for s in range(SEEDS)])
        d_mse = np.array([raw[arm][s][cp]['mse'] - raw['current'][s][cp]['mse']
                          for s in range(SEEDS)])
        n_win = int((d_r2 > 0).sum())
        print('    %-12s dR2 = %+.4f ± %.4f (胜 %d/%d)   dMSE = %+9.2f ± %8.2f'
              % (arm, d_r2.mean(), d_r2.std(ddof=1), n_win, SEEDS,
                 d_mse.mean(), d_mse.std(ddof=1)))
    print()

print('  * stock 那一行的 |dR2| 就是**纯抽样噪声下限**（它和 current 只差一次重抽）。')
print('  * 其余臂只有在"|dR2| 明显大于 stock"且"胜率明显偏离 50%"时才算真效应。')
