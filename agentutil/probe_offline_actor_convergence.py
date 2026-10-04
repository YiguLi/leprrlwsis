# -*- coding: utf-8 -*-
"""离线"冻结 critic" actor 精调实验 —— 判定 actor 为什么停不下来。

动机
----
实测 actor 的位移-间隔标度律 `alpha=0.674`、`rms(1)/a_scale=0.31`（无向游走），
且 critic 的界内梯度 `ratio=3.09`、方向相关 `r=-0.12`。要区分"停不下来"的两种成因：

  (a) **采样噪声**：minibatch 梯度方差大 ⇒ actor 停不到定点、只能绕着噪声球游走；
  (b) **critic 曲面本身**：即使给确定性梯度，critic 的固定曲面也没有稳定不动点
      （过拟合出高频假结构）⇒ actor 被推到某个方向又被拉回。

做法：`df_td3per.csv` 里有逐步 `(observation, original_action_H, r1, c1_step)`，
据此**离线重建 H 层块级 buffer**（无需 MATLAB / env），然后**冻结 critic**、只精调 actor：

  Arm A（full-batch）：确定性梯度（无采样噪声）
                       ⚠ "full" = 当前 `--blocks` 条离线 buffer（曾跑 300/2000/3000/4000），
                         **不是**训练的 `--rmsize=40000`；它只排除 minibatch 采样噪声。
  Arm B（minibatch） ：训练 bsize 口径（`--bsize=256`，8 个项目一致）
                       （早期版本误用 batch=64 并标注"同训练配方"，已更正）

判读（实测，run48 ckpt92000；指标 = 后段相邻位移 rms/a_scale = "收敛地板"）
--------------------------------------------------------------------------
  * 地板随 **lr** 近似线性上升；**Adam 高于 SGD**；**minibatch 高于 full-batch**。
    Adam+minibatch+训练 lr(1e-4) ⇒ 地板 0.045（与训练实测量级一致）；
    SGD+full-batch+lr=3e-5 ⇒ 0.0006（真冻结，说明 critic 曲面**可优化**）。
  * 合成二次对照（--synthetic）在同 lr=3e-4 下能到 0.003 ⇒ 真 critic 曲面"更难"
    （曲率更高、对步长更敏感；lr=3e-4 时确定性 SGD 反而发散），但**不是"没有不动点"**。
  ⇒ actor 不是"停不下来"，而是"停不死"：在噪声球/失配步长里绕，收益几乎不变。

用法: python agentutil/probe_offline_actor_convergence.py run48 --blocks 6000 --iters 400
"""
import argparse
import csv
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.optim import Adam

from model import Actor, Critic

BLOCK = 5


def parse_vec(s):
    s = s.strip().strip('[]')
    return np.array([float(x) for x in s.split()]) if s else np.zeros(0)


def h_state(obs4, minute):
    ang = 2.0 * np.pi * (int(minute) % 1440) / 1440.0
    return np.array([obs4[0] / 1000., obs4[1] / 10., obs4[2] / 10., obs4[3] / 10.,
                     np.sin(ang), np.cos(ang)], dtype=np.float32)


def build_buffer(path, max_blocks):
    """从 df 重建 H 层块级转移 (s6, a6, R_block, C_block, s2_6)。"""
    rows = []
    with open(path, newline='') as f:
        rd = csv.reader(f)
        ix = {c.strip(): i for i, c in enumerate(next(rd))}
        need = ('step', 'observation', 'observation2', 'original_action_H',
                'r1', 'c1_step', 'phase_minute', 'lam1')
        for c in need:
            if c not in ix:
                raise SystemExit('df 缺列 %s' % c)
        for r in rd:
            try:
                rows.append((int(float(r[ix['step']])),
                             parse_vec(r[ix['observation']]),
                             parse_vec(r[ix['observation2']]),
                             parse_vec(r[ix['original_action_H']]),
                             float(r[ix['r1']]), float(r[ix['c1_step']]),
                             float(r[ix['phase_minute']]),
                             r[ix['lam1']]))
            except (ValueError, IndexError):
                continue
    S, A, R, C, S2 = [], [], [], [], []
    lam_last = None
    for i in range(0, len(rows) - BLOCK + 1, BLOCK):
        blk = rows[i:i + BLOCK]
        st0 = blk[0][0]
        if any(blk[k][0] != st0 + k for k in range(BLOCK)):
            continue                      # 只取连续的 5 步块
        if len(blk[0][3]) < 6:
            continue
        S.append(h_state(blk[0][1], blk[0][6]))
        A.append(blk[0][3][:6])
        R.append(sum(x[4] for x in blk))
        C.append(sum(x[5] for x in blk))
        S2.append(h_state(blk[-1][2], blk[-1][6] + 1))
        try:
            lam_last = float(blk[-1][7])
        except (ValueError, TypeError):
            pass
    S = np.asarray(S, np.float32); A = np.asarray(A, np.float32)
    R = np.asarray(R, np.float32); C = np.asarray(C, np.float32)
    S2 = np.asarray(S2, np.float32)
    if len(S) > max_blocks:
        S, A, R, C, S2 = S[-max_blocks:], A[-max_blocks:], R[-max_blocks:], \
            C[-max_blocks:], S2[-max_blocks:]
    return S, A, R, C, S2, lam_last


def load_nets(ckpt):
    def _ld(nm):
        return torch.load(os.path.join(ckpt, nm), map_location='cpu', weights_only=False)
    sd = _ld('actor_6.pkl')
    actor = Actor(sd['fc1.weight'].shape[1], sd['fc4.weight'].shape[0],
                  hidden1=sd['fc1.weight'].shape[0], hidden2=sd['fc2.weight'].shape[0])
    actor.load_state_dict(sd); actor.eval()
    crits = {}
    for tag in ('p', 'p2', 'c', 'c2'):
        sd = _ld('critic_%s_6.pkl' % tag)
        h2 = sd['fc4.weight'].shape[0]
        m = Critic(sd['fc1.weight'].shape[1], sd['fc2.weight'].shape[1] - h2,
                   hidden1=sd['fc1.weight'].shape[0], hidden2=h2)
        m.load_state_dict(sd); m.eval()
        crits[tag] = m
    return actor, crits


def actor_loss(actor, crits, S, A_lab, lam, value_scale, cost_scale, synthetic=None):
    """与训练同口径的 actor loss（critic 不更新）。synthetic != None 时用合成二次目标做对照。"""
    a = actor(S)
    if synthetic is not None:
        # 对照臂: 目标 = 最大化 -(a - a*)^2 ⇒ 有唯一、干净的极大值
        q = -(a - synthetic).pow(2).sum(dim=1, keepdim=True)
        return -q.mean()
    qp = torch.min(crits['p']([S, a]), crits['p2']([S, a]))
    qc = torch.max(crits['c']([S, a]), crits['c2']([S, a])).clamp(min=0.0)
    return -(qp - lam * qc).mean()


def refine(actor, crits, S, lam, value_scale, cost_scale, iters, lr,
           batch=None, panel=None, snap_every=5, label='', optim='adam',
           synthetic=None):
    """冻结 critic 精调 actor；返回面板快照序列。"""
    if optim == 'sgd':
        opt = torch.optim.SGD(actor.parameters(), lr=lr)
    else:
        opt = Adam(actor.parameters(), lr=lr)
    n = len(S)
    snaps = []
    with torch.no_grad():
        snaps.append(actor(panel).numpy().copy())
        l0 = float(actor_loss(actor, crits, torch.tensor(S), None, lam, value_scale,
                              cost_scale, synthetic=synthetic).item())
    rng = np.random.RandomState(0)
    for it in range(1, iters + 1):
        idx = np.arange(n) if batch is None else rng.randint(0, n, batch)
        Sb = torch.tensor(S[idx])
        opt.zero_grad()
        loss = actor_loss(actor, crits, Sb, None, lam, value_scale, cost_scale,
                          synthetic=synthetic)
        loss.backward()
        opt.step()
        if it % snap_every == 0:
            with torch.no_grad():
                snaps.append(actor(panel).numpy().copy())
    return np.asarray(snaps), l0, float(loss.item())


def alpha_fit(snaps):
    """位移-间隔标度律 alpha（与 audit_policy_stability 同口径）。"""
    k, rms = [], []
    for kk in range(1, len(snaps)):
        d = [float(np.sqrt(((snaps[i + kk] - snaps[i]) ** 2).sum(axis=1).mean()))
             for i in range(len(snaps) - kk)]
        k.append(kk); rms.append(float(np.mean(d)))
    if len(k) < 3:
        return float('nan'), 0.0
    al, _ = np.polyfit(np.log(k), np.log(rms), 1)
    return float(al), float(rms[0])


def adj_trend(snaps):
    """相邻快照位移的趋势：前/中/后三等分均值。持续下降 ⇒ 收敛。"""
    adj = [float(np.sqrt(((snaps[i] - snaps[i - 1]) ** 2).sum(axis=1).mean()))
           for i in range(1, len(snaps))]
    n = len(adj)
    if n < 6:
        return (float('nan'),) * 3
    t = n // 3
    return (float(np.mean(adj[:t])), float(np.mean(adj[t:2 * t])),
            float(np.mean(adj[2 * t:])))


def main():
    ap = argparse.ArgumentParser(description='离线冻结-critic actor 精调实验',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', help='run 名（自动补 output/TD3PER-<name>）或目录')
    ap.add_argument('--step', type=int, default=None, help='checkpoint 步（默认最后一个）')
    ap.add_argument('--blocks', type=int, default=6000, help='离线 buffer 最多块数')
    ap.add_argument('--iters', type=int, default=400, help='精调迭代数')
    ap.add_argument('--lr', type=float, default=1e-3, help='actor 精调 lr')
    ap.add_argument('--panel', type=int, default=256, help='评估面板状态数')
    ap.add_argument('--optim', nargs='+', default=['adam', 'sgd'],
                    choices=['adam', 'sgd'], help='优化器臂 (默认两臂都跑)')
    ap.add_argument('--synthetic', type=float, default=None,
                    help='对照臂: 用合成二次目标 (极大值在 a*=该值) 替换 critic')
    args = ap.parse_args()

    d = args.run if os.path.isdir(args.run) else os.path.join('output', 'TD3PER-' + args.run)
    if not os.path.isdir(d):
        raise SystemExit('找不到 run: %s' % d)
    cks = sorted(int(re.search(r'(\d+)$', p).group(1))
                 for p in glob.glob(os.path.join(d, 'checkpoint_*'))
                 if os.path.exists(os.path.join(p, 'actor_6.pkl')))
    ckpt = os.path.join(d, 'checkpoint_%d' % (args.step or cks[-1]))
    print('checkpoint: %s' % ckpt)

    S, A, R, C, S2, lam_last = build_buffer(os.path.join(d, 'df_td3per.csv'), args.blocks)
    print('离线 buffer: %d 块 (含 lam1=%s)' % (len(S), lam_last))
    lam = float(lam_last if lam_last is not None else 1.0)
    print('λ = %.4f' % lam)

    panel = S[np.linspace(0, len(S) - 1, min(args.panel, len(S))).astype(int)]
    panel = torch.tensor(panel)

    base = load_nets(ckpt)[0]
    with torch.no_grad():
        a0 = base(panel).numpy()
    a_scale = float(np.sqrt((a0.std(axis=0) ** 2).mean()))
    print('actor 面板输出: 均值 %s, 跨状态尺度 a_scale=%.4f'
          % (np.round(a0.mean(0), 3).tolist(), a_scale))
    print('初始 |a| 均值 %.3f' % float(np.abs(a0).mean()))

    synth = None
    if args.synthetic is not None:
        synth = torch.full((6,), float(args.synthetic), dtype=torch.float32)
        arms = [('对照·合成二次目标 / full-batch', None, 'adam'),
                ('对照·合成二次目标 / minibatch', 256, 'adam')]
    else:
        arms = [('%s / full-batch (无采样噪声)' % o.upper(), None, o) for o in args.optim] + \
               [('%s / minibatch=256 (训练 bsize 口径)' % o.upper(), 256, o)
                for o in args.optim]

    for label, batch, optm in arms:
        actor = load_nets(ckpt)[0]                       # 每个 arm 从同一权重重开
        for p in actor.parameters():
            p.requires_grad_(True)
        crits = load_nets(ckpt)[1]                       # critic 冻结
        for m in crits.values():
            for p in m.parameters():
                p.requires_grad_(False)
        snaps, l0, final_loss = refine(actor, crits, S, lam, 1.0, None,
                                       args.iters, args.lr, batch=batch, panel=panel,
                                       optim=optm, synthetic=synth)
        al, rms1 = alpha_fit(snaps)
        total = float(np.sqrt(((snaps[-1] - snaps[0]) ** 2).sum(axis=1).mean()))
        print('\n--- %s ---' % label)
        print('  loss: 初 %.3f → 末 %.3f' % (l0, final_loss))
        print('  面板位移: 首末 |Δa| = %.4f  (a_scale=%.4f, 比 %.3f)'
              % (total, a_scale, total / max(a_scale, 1e-9)))
        print('  相邻位移 rms(1) = %.5f (比 a_scale = %.4f)' % (rms1, rms1 / max(a_scale, 1e-9)))
        e, m, l = adj_trend(snaps)
        print('  相邻位移趋势 前/中/后 = %.5f / %.5f / %.5f  → %s'
              % (e, m, l, '持续下降(收敛中)' if l < 0.5 * e else '未下降(仍在游走)'))
        print('  alpha = %.3f   (训练实测 0.674; ~0.5 扩散(停住绕圈), ~1.0 匀速漂移, ~0 冻结)'
              % al)
        print('  |a| 均值: 初 %.3f → 末 %.3f'
              % (float(np.abs(snaps[0]).mean()), float(np.abs(snaps[-1]).mean())))

    print('\n读法（实测结论，run48 ckpt92000；指标 = 后段相邻位移 rms/a_scale，即"收敛地板"）:')
    print('  * 地板随 **lr** 近似线性上升；**Adam 高于 SGD**；**minibatch 高于 full-batch**。')
    print('    Adam + minibatch + 训练 lr=1e-4 ⇒ 地板 ~0.045，与训练实测量级一致。')
    print('  * ⚠ 复测 batch 64 vs 256 (lr=1e-4, Adam): 0.0449 vs 0.0465 ⇒ **几乎不变**')
    print('    ⇒ 地板**不是** CLT 意义的 minibatch 采样噪声。更可能是离线 buffer 的')
    print('      **时间相关性**(转移来自同一条连续轨迹 ⇒ 有效独立样本数 << 名义 batch)。')
    print('      ⇒ **"加大 batch" 这条建议被本次复测否掉**，不要再按 CLT 去调 bsize。')
    print('  * SGD + full-batch + lr=3e-5 可降到 0.0006（真冻结）⇒ critic 曲面**是可优化的**。')
    print('  * 合成二次对照在同 lr=3e-4 下到 0.003 ⇒ 真 critic 曲面"更难"（曲率更高、')
    print('    对步长更敏感，lr=3e-4 时 SGD 甚至发散），但并非"没有不动点"。')
    print('  ⇒ actor 不是"停不下来"，而是"停不死"：绕着最优点在噪声球/失配步长里走，')
    print('    收益在该球内几乎不变 ⇒ test9 就在高值区里来回晃。')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
