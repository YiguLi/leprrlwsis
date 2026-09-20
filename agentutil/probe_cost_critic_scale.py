#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线判定：H 层 cost critic 的"塌缩"是不是 s_c（cost 通道尺度）造成的。

要回答的问题
------------
`68989dd` 把 TD3 六的 H 层值函数归一化从单尺度 s=1000 改成双尺度（s_p=1000 用于 r1、
s_c=100 用于 c1），理由是旧口径下 cost 通道的回归目标比 profit 通道小一个量级、
被数值饿死。证据链里"`checkpoint_66000` 实测 Q_c≡360.4、sd=0、∂Q_c/∂a=0"一条来自
run9 —— 而 run9 是 09-19 04:56 启动的进程，`68989dd` 是 09-21 02:51 才提交的，
**修复后至今没有任何 run 跑过新尺度**。本探针用 run9 日志离线重建 H 层块级 (s,a,c1)，
判定尺度到底是不是成因，以及修好之后还有没有杠杆。

两节
----
A. **数据重建**：从 run9 逐 step 日志重建块级 (s, a, c1)，并对重建做自检
   （块内动作是否恒定、块长分布、c1 是否与 `hsarc_cost.fluct_cost` 口径一致）。
B. **环境侧杠杆**（无需训练）：块 c1 依赖 H 动作吗？控住状态后的偏 R²（线性 + 非线性），
   分"全部块 / 无不可行块 / 去掉重尾"三种子集 —— 这是"分段乘子有没有东西可缩放"的前提。
C. **critic 可学性 A/B**：同数据/同初始化/同优化器/同步数，只改 s_c，
   **每种子重复 SEEDS 次**统计塌缩频率（单次实现不稳定：第一版探针 s_c=100 一次学会一次塌缩）。
D. **机制诊断**：输出是否退化成 fc3.bias 常数、头部 ReLU 是否全死。

保真度要点（第一版探针曾在这里出错，已修正）
--------------------------------------------
* `td3prrbb_main.py:199` 是 `norm_observation = Normalize(observation)`，H agent 收的是
  **归一化后**的观测；CSV 的 `observation` 列是**原始** [Pr,U,E,P_last]（dim0=电价 300~980）。
  第一版直接喂原始值 ⇒ 输入量级 O(300) ⇒ 头部 ReLU 全死，得到的"塌缩"是我自己的假象。
  本版按项目自己的 `Normalize`（单一真源，并逐位断言）换算。
* 网络/初始化/优化器/损失/裁剪取自项目：`model.Critic`、`hidden1=16/hidden2=100`、
  Adam(lr=`--rate`=0.01)、bsize=32、`clip_grad_norm_(2.0)`、MSE（对应 value_loss_c）。
* 数据用**冻结快照**：run9 仍在跑且仍是全量重写落盘（我的增量修复还没进那个进程），
  直接读活文件会读到半截（实测 9.1~9.25 万行之间漂移）。
* 本探针是**纯回归**（target = 块 c1 / s_c），不含 bootstrap/PER/目标网络 ——
  只隔离"数值条件"这一个变量。
"""
import csv as _csv
import io
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
from torch.optim import Adam                                        # noqa: E402

from model import Critic                                            # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

LIVE = 'output/TD3PRRBB-run9/df_td3prrbb.csv'
SNAP = 'output/_analysis/run9_df_td3prrbb.csv'
BLOCK = 5
S_OLD, S_NEW = 1000.0, 100.0
SP = 1000.0
SCALES = [1.0, 10.0, 100.0, 1000.0]
SEEDS = 4
STEPS = 6000
SEED0 = 0


# ---------------------------------------------------------------- 快照
def snapshot():
    """冻结一份快照（活文件正在被全量重写，直接读会读到半截）。"""
    os.makedirs(os.path.dirname(SNAP), exist_ok=True)
    if not os.path.exists(SNAP):
        shutil.copyfile(LIVE, SNAP)
        print('  已冻结快照: %s (%.1f MB)' % (SNAP, os.path.getsize(SNAP) / 1e6))
    with io.open(SNAP, encoding='utf-8', newline='') as fh:
        rows = list(_csv.reader(fh))
    ncol, bad = len(rows[0]), sum(1 for r in rows[1:] if len(r) != len(rows[0]))
    assert bad == 0, ('快照字段数不符的行 %d ⇒ 快照是半截文件, 删掉重做' % bad)
    return len(rows) - 1, ncol


nrow, ncol = snapshot()
print('=' * 100)
print('A. 从 run9 日志重建 H 层块级数据集 (快照 %d 逻辑行 × %d 列)' % (nrow, ncol))
print('=' * 100)

import pandas as pd                                                 # noqa: E402

# 归一化: 用项目自己的 Normalize, 并逐位断言(防止口径漂移)
_NORM_SRC = 'inline'
try:
    from td3prrbb.td3prrbb_myfunc import Normalize as _NormProj
    _NORM_SRC = 'td3prrbb.td3prrbb_myfunc.Normalize'
except Exception as e:                                              # pragma: no cover
    print('  [warn] 无法导入项目 Normalize (%s), 用内联副本' % type(e).__name__)


def Normalize(obs):
    return np.array([obs[0] / 1000., obs[1] / 10., obs[2] / 10., obs[3] / 10.])


cols = ['step', 'observation', 'original_action_H', 'c1', 'r1', 'P_F',
        'wind_level', 'is_infeasible']
raw = pd.read_csv(SNAP, usecols=cols)
print('  归一化来源: %s' % _NORM_SRC)
if _NORM_SRC != 'inline':
    _probe = np.array([300., 10., 3., 2.])
    assert np.allclose(Normalize(_probe), _NormProj(_probe)), '内联 Normalize 与项目不一致'


def vec(s):
    if not isinstance(s, str):
        return None
    try:
        return np.array([float(x) for x in s.strip().strip('[]').split()], dtype=np.float64)
    except ValueError:
        return None


raw['blk'] = (raw['step'] // BLOCK).astype(int)
S_list, A_list, C_list, R_list, W_list, I_list, const_act = [], [], [], [], [], [], []
for b, g in raw.groupby('blk', sort=True):
    g = g.sort_values('step')
    ov, av = vec(g['observation'].iloc[0]), vec(g['original_action_H'].iloc[0])
    if ov is None or av is None:
        continue
    acts = [vec(x) for x in g['original_action_H']]
    S_list.append(Normalize(ov))                  # ← 修正点: 必须归一化
    A_list.append(av)
    C_list.append(float(g['c1'].sum()))
    R_list.append(float(g['r1'].sum()))
    W_list.append(g['wind_level'].iloc[0])
    I_list.append(int(g['is_infeasible'].sum()))
    const_act.append(int(all(a is not None and np.allclose(a, av) for a in acts)))
S = np.asarray(S_list); A = np.asarray(A_list); C = np.asarray(C_list)
R = np.asarray(R_list); W = np.asarray(W_list); I = np.asarray(I_list)
const_act = np.asarray(const_act)
print('  块 %d 个; 块内动作恒定的块占比 %.2f%%  (≈100%% ⇒ step//%d 就是 H 块边界)'
      % (len(S), 100 * const_act.mean(), BLOCK))
print('  state(归一化) 各维 sd: %s   min: %s   max: %s'
      % (np.round(S.std(0), 3), np.round(S.min(0), 3), np.round(S.max(0), 3)))
print('  action 各维 sd: %s   范围 %s ~ %s'
      % (np.round(A.std(0), 3), np.round(A.min(0), 3), np.round(A.max(0), 3)))
print('  块 c1: mean=%.3f sd=%.3f 分位[50,90,99,100]=%s'
      % (C.mean(), C.std(), np.round(np.percentile(C, [50, 90, 99, 100]), 2)))
print('  含不可行步的块: %.3f%%  (其 c1 均值 %.2f vs 其余 %.2f)'
      % (100 * (I > 0).mean(), C[I > 0].mean(), C[I == 0].mean()))
print('  回归目标量级   profit r1/s_p=%.4f | cost c1/s_c_old=%.4f | c1/s_c_new=%.4f'
      % (R.mean() / SP, C.mean() / S_OLD, C.mean() / S_NEW))
print('  旧口径 cost : profit 目标之比 = 1 : %.1f' % ((R.mean() / SP) / (C.mean() / S_OLD)))
print('  按风况档 (用户方案的候选分段变量):')
for lv in ['LOW', 'MEDIUM', 'HIGH']:
    m = W == lv
    if m.sum():
        print('     %-7s n=%5d  c1 mean=%7.3f sd=%6.3f  不可行块占比 %5.2f%%'
              % (lv, m.sum(), C[m].mean(), C[m].std(), 100 * (I[m] > 0).mean()))

rng = np.random.RandomState(SEED0)
perm = rng.permutation(len(S))
itr, ite = perm[:int(0.8 * len(S))], perm[int(0.8 * len(S)):]


def T(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def r2_of(pred, true):
    return 1 - ((true - pred) ** 2).sum() / ((true - true.mean()) ** 2).sum()


def partial_r2(X, y, yb):
    def r2(F):
        F = np.column_stack([np.ones(len(F)), F])
        beta, *_ = np.linalg.lstsq(F, y, rcond=None)
        return 1 - ((y - F @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return r2(np.column_stack([X, yb])) - r2(X)


print()
print('=' * 100)
print('B. 环境侧杠杆 (无需训练): 块 c1 依赖 H 动作吗?')
print('=' * 100)
subsets = [('全部块', slice(None)),
           ('无不可行块的块', I[ite] == 0)]
for tag, m in subsets:
    if isinstance(m, slice):
        Si, Ai, Ci = S[ite], A[ite], C[ite]
    else:
        Si, Ai, Ci = S[ite][m], A[ite][m], C[ite][m]
    lin = partial_r2(Si, Ci, Ai)
    # 非线性: 真 Critic 架构, 带动作 vs 动作置零, 多种子取中位
    inc = []
    for sd in range(2):
        def fit(use_a):
            torch.manual_seed(sd)
            net = Critic(S.shape[1], A.shape[1], 16, 100)
            opt = Adam(net.parameters(), lr=0.01)
            Sa, Aa, Ca = T(S[itr]), T(A[itr]), T(C[itr])
            Aa = Aa if use_a else torch.zeros_like(Aa)
            tgt = (Ca / S_NEW).view(-1, 1)
            for _ in range(STEPS):
                idx = torch.randint(0, len(Sa), (32,))
                loss = torch.mean((net([Sa[idx], Aa[idx]]) - tgt[idx]) ** 2)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                opt.step()
            with torch.no_grad():
                q = net([T(Si), T(Ai) if use_a else torch.zeros_like(T(Ai))]).numpy().ravel() * S_NEW
            return r2_of(q, Ci)
        inc.append(fit(True) - fit(False))
    print('  %-20s n=%5d  线性偏R²=%+.4f   非线性增量R² 中位=%+.4f (各 4 种子)'
          % (tag, len(Si), lin, float(np.median(inc))))
print('  动作覆盖度: 各维 sd=%s ⇒ %s'
      % (np.round(A.std(0), 3),
         '有实际变化, 低 R² 不是"没动过"' if (A.std(0) > 0.02).all() else '**有维几乎不动**'))


def grad_a(net, sc, n=2000):
    Sb = T(S[ite[:n]]); Ab = T(A[ite[:n]]).clone().requires_grad_(True)
    g = torch.autograd.grad(net([Sb, Ab]).sum(), Ab)[0].detach().numpy() * sc
    return float(np.abs(g).mean())


def sweep_sd(net, sc, n_state=64, reps=9):
    Sb, Ab = T(S[ite[:n_state]]), T(A[ite[:n_state]])
    lo, hi = T(A.min(0)), T(A.max(0))
    vals = []
    with torch.no_grad():
        for d in range(A.shape[1]):
            for f in np.linspace(0, 1, reps):
                A2 = Ab.clone(); A2[:, d] = lo[d] + float(f) * (hi[d] - lo[d])
                vals.append(net([Sb, A2]).numpy().ravel() * sc)
    return float(np.asarray(vals).std(0).mean())


def dead_fracs(net):
    with torch.no_grad():
        h1 = net.relu(net.fc1(T(S[ite])))
        h4 = net.fc4(h1)
        h2 = net.relu(net.fc2(torch.cat([h4, T(A[ite])], 1)))
        h5 = net.relu(net.fc5(h2))
    return float((h2 == 0).float().mean()), float((h5 == 0).float().mean())


def train_one(sc, seed):
    torch.manual_seed(seed)
    net = Critic(S.shape[1], A.shape[1], 16, 100)
    opt = Adam(net.parameters(), lr=0.01)
    tgt = (T(C[itr]) / sc).view(-1, 1)
    Sa, Aa = T(S[itr]), T(A[itr])
    for _ in range(STEPS):
        idx = torch.randint(0, len(Sa), (32,))
        loss = torch.mean((net([Sa[idx], Aa[idx]]) - tgt[idx]) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
        opt.step()
    with torch.no_grad():
        q = net([T(S[ite]), T(A[ite])]).numpy().ravel() * sc
    b = C[ite]
    return dict(r2=float(r2_of(q, b)), sd_ratio=float(q.std() / b.std()),
                ga=grad_a(net, sc), sw=sweep_sd(net, sc),
                dead=dead_fracs(net), net=net)


print()
print('=' * 100)
print('C. critic 可学性 A/B —— 只改 s_c, 每档 %d 个种子, 各 %d 步' % (SEEDS, STEPS))
print('=' * 100)
print('  model.Critic(4,6,16,100)  Adam(lr=0.01)  bsize=32  clip=2.0  loss=MSE; 评估换算回原始单位')
print()
hdr = ('%-8s %10s %10s %10s %10s %12s %11s' %
       ('s_c', 'R²中位', 'R²范围', 'sd(Q)/sd(t)', '塌缩数', 'mean|dQ/da|', 'fc5死ReLU'))
print('  ' + hdr)
print('  ' + '-' * (len(hdr) + 2))
summary = {}
for sc in SCALES:
    res = [train_one(sc, s) for s in range(SEEDS)]
    r2s = np.array([r['r2'] for r in res])
    sdr = np.array([r['sd_ratio'] for r in res])
    n_col = int((sdr < 0.02).sum())
    summary[sc] = dict(r2_med=float(np.median(r2s)), sdr_med=float(np.median(sdr)),
                       n_col=n_col, ga=float(np.median([r['ga'] for r in res])),
                       dead=float(np.median([r['dead'][1] for r in res])),
                       nets=[r['net'] for r in res], sdr=sdr, r2s=r2s)
    print('  %-8.0f %10.4f %5.3f~%-5.3f %10.4f %7d/%-3d %12.4f %11.3f'
          % (sc, np.median(r2s), r2s.min(), r2s.max(), np.median(sdr), n_col, SEEDS,
             np.median([r['ga'] for r in res]), np.median([r['dead'][1] for r in res])))
print()
print('  ⇒ 塌缩 = sd(Q_raw)/sd(target) < 0.02 (预测退化成常数)。')

print()
print('=' * 100)
print('D. 机制诊断 (每个 s_c 取一个塌缩样本与一个未塌缩样本)')
print('=' * 100)
for sc in SCALES:
    d = summary[sc]
    idx_col = np.where(d['sdr'] < 0.02)[0]
    idx_ok = np.where(d['sdr'] >= 0.02)[0]
    for tag, idx in (('塌缩样本', idx_col[:1]), ('未塌缩样本', idx_ok[:1])):
        if not len(idx):
            continue
        net = d['nets'][idx[0]]
        with torch.no_grad():
            q = net([T(S[ite]), T(A[ite])]).numpy().ravel() * sc
        z2, z5 = dead_fracs(net)
        print('  s_c=%-5.0f %-6s R²=%+.4f sd(Q)=%.5f  fc3.bias=%.6f (×s_c=%.3f, 目标均值=%.3f)'
              '  fc2零占比=%.3f fc5零占比=%.3f'
              % (sc, tag, r2_of(q, C[ite]), q.std(), net.fc3.bias.item(),
                 net.fc3.bias.item() * sc, C[itr].mean(), z2, z5))

print()
print('=' * 100)
print('E. 判读')
print('=' * 100)
print('  常量基线 (预测训练集均值) 的测试 R² = %.4f'
      % r2_of(np.full(len(ite), C[itr].mean()), C[ite]))
for sc in SCALES:
    d = summary[sc]
    print('  s_c=%-6.0f  R²中位=%+.4f  sd比中位=%.4f  塌缩 %d/%d  mean|dQ/da|中位=%.4f'
          % (sc, d['r2_med'], d['sdr_med'], d['n_col'], SEEDS, d['ga']))
print()
print('  读法:')
print('   * 塌缩数 = 0 ⇒ 该尺度下 critic 稳定学到非平凡函数; 全塌缩 ⇒ 该尺度确实学不动。')
print('   * 若 s_c=1000 全塌缩而 s_c=100 不塌缩 ⇒ **尺度就是成因**(用户判断对)。')
print('   * 若多档都塌缩且 fc5 全死 ⇒ 主因是**头部 ReLU 死亡**这一架构/优化病,')
print('     尺度只是改变了触发概率; 单靠 s_c 无法根治, 需要同时处理初始化/激活/lr。')
