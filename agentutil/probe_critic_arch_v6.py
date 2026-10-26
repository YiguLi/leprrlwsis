#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_critic_arch_v6.py —— v6 critic 隐藏层设计审计（与 v2 对照）。

用法（任意 CWD）：
    python agentutil/probe_critic_arch_v6.py                 # A/B 接线+初始化审计，C/D v6 vs v2
    python agentutil/probe_critic_arch_v6.py --trace 1       # E 段：v6 的层健康随训练步数演化
    python agentutil/probe_critic_arch_v6.py --layers 1      # L 层

A. 接线：每种 arch 的层表/参数量；`--hidden1/--hidden2` 有没有真的接上；
   `infer_arch` 能否从 checkpoint 唯一还原。
B. 初始化尺度：`fanout_init`（v=1/sqrt(out_features)）逐层 Var(z)≈fan_in·v²/3。
C. 值-动作方差分解：同一个状态上扰动动作，动作能解释 Q 方差的多少（真值≈30%）。
D. 动作支路消融 + 逐样本/批均值梯度。
E. v6 的层健康（死单元比例、|h|、|W|、Q 统计）随训练步数演化。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import model  # noqa: E402
from aifflor.aifflor_myfunc import Normalize  # noqa: E402

ARCHS = ['v1', 'v1af', 'v2', 'v3', 'v4', 'v5', 'v6']


def parse_vec(col):
    return np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                      for x in col]).astype(np.float64)


def load_blocks(run, n, want_cols=('c1', 'r1')):
    """末尾 n 个**完整 5 步块**的块首 (S, A_full, c1_block, r1_block)。

    注意：run 的 df_hsrl.csv 行数不一定是 5 的整数倍（末尾可能截断），
    因此按 step//5 分组求和，只保留步数为 5 的完整块。
    """
    cols = ['step', 'observation', 'applied_action_H'] + list(want_cols)
    df = pd.read_csv(os.path.join(run, 'df_hsrl.csv'), usecols=cols)
    step = df.step.to_numpy()
    bt = step // 5
    sizes = pd.Series(bt).value_counts()
    complete = set(sizes[sizes == 5].index.tolist())
    mask = (step % 5 == 0) & np.isin(bt, list(complete))
    idx = np.where(mask)[0][-n:]
    obs = parse_vec(df.observation.iloc[idx])
    S = np.vstack([Normalize(o) for o in obs]).astype(np.float32)
    A = parse_vec(df.applied_action_H.iloc[idx]).astype(np.float32)
    out = [S, A]
    for c in want_cols:
        s = pd.Series(df[c].to_numpy()).groupby(bt).sum()
        out.append(s.loc[bt[idx]].to_numpy())
    return out


# ---------------------------------------------------------------- 前向（可消融）
def fwd(net, S, A, zero_ha=False, zero_hs=False, zero_a=False, want_h=False):
    if net.arch == 'v6':
        hs = net.relu(net.s1(S))
        ha = net.relu(net.a1(A))
        if zero_hs:
            hs = torch.zeros_like(hs)
        if zero_ha:
            ha = torch.zeros_like(ha)
        aa = torch.zeros_like(A) if zero_a else A
        m = torch.cat([hs, ha, aa], 1)
        h1 = net.relu(net.m1(m)); h2 = net.relu(net.m2(h1)); h3 = net.relu(net.m3(h2))
        q = net.out(h3)
    elif net.arch in ('v2', 'v3'):
        hs = net.relu(net.s2(net.relu(net.s1(S))))
        ha = net.relu(net.a2(net.relu(net.a1(A))))
        if zero_hs:
            hs = torch.zeros_like(hs)
        if zero_ha:
            ha = torch.zeros_like(ha)
        z = torch.cat([hs, ha], 1)
        h1 = net.relu(net.m1(z)) + z
        h2 = net.relu(net.m2(h1)); h3 = net.relu(net.m3(h2))
        q = net.out(h3)
    else:
        raise SystemExit('probe 只支持 v2/v3/v6，收到 %s' % net.arch)
    return (q, (hs, ha, h1, h2, h3)) if want_h else (q,)


def build(run, arch, h1, h2, layer=6, out_init='fanin'):
    f = os.path.join(run, 'critic_c_%d.pkl' % layer)
    sd = torch.load(f, map_location='cpu')
    ns = 4 if layer == 6 else (6 if int(sd['s1.weight'].shape[1]) == 6 else 5)
    na = 6 if layer == 6 else 1
    net = model.Critic(ns, na, hidden1=h1, hidden2=h2, arch=arch, out_init=out_init)
    net.load_state_dict(sd)
    net.eval()
    return net, sd


# ---------------------------------------------------------------- A. 接线
def audit_wiring(ns, na, tag):
    print('\n[A] ar%s (nb_states=%d, nb_actions=%d)' % (tag, ns, na))
    print('    %-5s %-7s %-9s %-22s %-12s %s'
          % ('arch', 'hidden', 'params', 'm1.weight', 'infer_arch', '备注'))
    for hid in [(16, 100), (64, 200), (256, 256)]:
        for arch in ARCHS:
            net = model.Critic(ns, na, hidden1=hid[0], hidden2=hid[1], arch=arch)
            sd = net.state_dict()
            n = sum(v.numel() for v in sd.values())
            m1 = sd['m1.weight'].shape if 'm1.weight' in sd else '-'
            rm1 = model.infer_arch(sd, ns, na)
            note = ''
            if rm1 != arch:
                note = '⚠推断成 %s' % rm1
            print('    %-5s %-7s %-9d %-22s %-12s %s'
                  % (arch, '%d/%d' % hid, n, str(m1), rm1, note))
    # hidden1/2 是否被 arch 使用
    print('    --hidden1/--hidden2 的实际接线:')
    for arch in ARCHS:
        n1 = sum(v.numel() for v in model.Critic(ns, na, hidden1=16, hidden2=100,
                                                arch=arch).state_dict().values())
        n2 = sum(v.numel() for v in model.Critic(ns, na, hidden1=64, hidden2=200,
                                                arch=arch).state_dict().values())
        print('      %-5s %7d vs %7d  %s' % (arch, n1, n2,
                                            '接上了' if n1 != n2 else '**完全不生效**'))


def audit_v6_layers():
    print('\n[A2] v6 逐层（H: 4→32 ∥ 6→32，cat(32+32+6)=70 → 256 → 256 → 128 → 1）')
    net = model.Critic(4, 6, hidden1=16, hidden2=100, arch='v6')
    rows = [('s1', 'state 4→32'), ('a1', 'action 6→32'), ('m1', 'merge 70→256'),
            ('m2', '256→256'), ('m3', '256→128'), ('out', '128→1')]
    tot = 0
    for k, desc in rows:
        w = getattr(net, k).weight
        p = w.numel() + getattr(net, k).bias.numel()
        tot += p
        print('    %-4s %-14s %-14s %7d' % (k, desc, str(tuple(w.shape)), p))
    merge = sum(getattr(net, k).weight.numel() + getattr(net, k).bias.numel()
                for k in ('m1', 'm2', 'm3'))
    print('    合计 %d；merge 三层 %d (%.1f%%)；状态编码 4→32 只有 %d 参数 (%.1f%%)'
          % (tot, merge, 100.0 * merge / tot, 160 + 32, 100.0 * 192 / tot))
    print('    对照 v2: s 4→16→100 ∥ a 6→32→64 → cat 164 →164(残差)→128→64→1 = 60 617')
    print('    ⚠ v6 状态支路只有 1 层（v2 是 2 层）；v6 无残差（v2/v4 的 m1 = relu(m1(z))+z）')


# ---------------------------------------------------------------- B. 初始化尺度
def audit_init_scale(ns=4, na=6):
    print('\n[B] fanout_init 逐层信号 Var(z) ≈ fan_in·v²/3（v = 1/sqrt(out_features)）')
    print('    %-4s %-12s %-8s %-12s %s' % ('层', '形状(out×in)', 'bound', 'Var(z)/Var(x)', '结论'))
    for k, (o, i) in [('s1', (32, ns)), ('a1', (32, na)), ('m1', (256, 70)),
                      ('m2', (256, 256)), ('m3', (128, 256))]:
        v = 1.0 / np.sqrt(o)
        g = i * v * v / 3.0
        print('    %-4s %-12s %-8.4f %-12.4f  ≈1/%.1f' % (k, '%d×%d' % (o, i), v, g, 1 / g))
    print('    ⇒ merge 栈 0.09 × 0.33 × 0.67 ≈ 1/50；而 v2 的 m1 有残差、不衰减。')
    print('      s1/a1 使用 fan_out ⇒ 32 个输出、fan_in 4/6 ⇒ 初始信号只有 1/24、1/16。')


# ---------------------------------------------------------------- C/D
def health(net, S, A, s_c, tag=''):
    with torch.no_grad():
        q, H = fwd(net, S, A, want_h=True)
    hs, ha, h1, h2, h3 = H
    def f(x):
        return '|h| %6.3f dead %5.1f%%' % (x.abs().mean(), 100 * (x == 0).float().mean())
    print('    %-8s Q(raw) mean %8.1f sd %7.1f | hs %s | ha %s'
          % (tag, q.mean().item() * s_c, q.std().item() * s_c, f(hs), f(ha)))
    print('    %-8s merge: m1 %s | m2 %s | m3 %s' % ('', f(h1), f(h2), f(h3)))
    print('    %-8s |W|: s1 %.3f a1 %.3f m1 %.3f m2 %.3f m3 %.3f out %.5f | out.bias %+.3f'
          % ('', net.s1.weight.abs().mean(), net.a1.weight.abs().mean(),
             net.m1.weight.abs().mean(), net.m2.weight.abs().mean(),
             net.m3.weight.abs().mean(), net.out.weight.abs().mean(),
             net.out.bias.item()))
    return q


def variance_split(net, S, A, s_c, sigma=0.07, reps=8, seed=0):
    rng = np.random.RandomState(seed)
    with torch.no_grad():
        base = fwd(net, S, A)[0]
        tot = float(base.var(unbiased=False)) * s_c * s_c
        within = []
        for _ in range(reps):
            Ax = A.clone()
            Ax[:, :5] = torch.clamp(Ax[:, :5] + torch.tensor(
                rng.randn(S.shape[0], 5) * sigma, dtype=torch.float32), 0.0, 0.5)
            q = fwd(net, S, Ax)[0]
            within.append(float((q - base).var(unbiased=False)) * s_c * s_c)
    share = float(np.mean(within)) / max(tot, 1e-12)
    print('    方差分解: sd(Q)=%.1f  扰动动作(±%.2f)可解释 %.1f%%（真值≈30%%）'
          % (np.sqrt(tot), sigma, 100 * share))
    return share


def ablation(net, S, A, s_c):
    with torch.no_grad():
        q0 = fwd(net, S, A)[0]
        q_noha = fwd(net, S, A, zero_ha=True)[0]
        q_nohs = fwd(net, S, A, zero_hs=True)[0]
        q_onlys = fwd(net, S, A, zero_ha=True, zero_a=True)[0] if net.arch == 'v6' else q_noha
        q_noa = fwd(net, S, A, zero_a=True)[0] if net.arch == 'v6' else q_noha
    m = lambda x: float(x.mean()) * s_c
    print('    消融: 全量 %8.1f | 去a支路 %8.1f | 去原始a直连 %8.1f | 只留状态 %8.1f | '
          '去状态支路 %8.1f' % (m(q0), m(q_noha), m(q_noa), m(q_onlys), m(q_nohs)))
    print('      ⇒ 动作整体贡献 |Q0−Q(only_s)| = %.1f  (= %.1f%% 的 sd(Q))'
          % (abs(m(q0) - m(q_onlys)),
             100 * abs(m(q0) - m(q_onlys)) / max(float(q0.std()) * s_c, 1e-9)))


def grad_stats(net, S, A, s_c):
    at = A.clone().requires_grad_(True)
    q = fwd(net, S, at)[0]
    g, = torch.autograd.grad(q.sum(), at)
    gp = g.detach()[:, :5] * s_c
    bm = gp.mean(0).abs().mean().item()
    print('    梯度: 逐样本 |∂Q/∂a| mean %8.2f median %8.2f | 批均值 %.2f | 比值 %.1f'
          % (gp.abs().mean(), gp.abs().median(), bm,
             gp.abs().mean() / max(bm, 1e-9)))
    d = gp.sum(1) / np.sqrt(5)
    print('          沿+a 方向: mean %+8.2f sd %8.2f (mean/sd %+.2f) 正号 %.1f%%'
          % (d.mean(), d.std(), d.mean() / max(d.std(), 1e-9),
             100.0 * float((d > 0).float().mean())))
    return gp


def fit_check(net, S, A, c1, s_c):
    with torch.no_grad():
        q = fwd(net, S, A)[0].numpy().ravel() * s_c
    print('    值: Q 均值 %.0f vs 实测块成本均值 %.2f（折扣后 ≈%.0f）| rank-corr %.3f'
          % (q.mean(), c1.mean(), c1.mean() / 0.05,
             np.corrcoef(np.argsort(np.argsort(q)), np.argsort(np.argsort(c1)))[0, 1]))


# ---------------------------------------------------------------- E. 随时间
def trace(run, steps, arch, h1, h2, n=1500):
    S, A, c1 = load_blocks(run, n, want_cols=('c1',))
    S_t, A_t = torch.tensor(S), torch.tensor(A)
    print('\n[E] v6 层健康随训练步数（同一批 %d 个块首状态）' % n)
    print('  %-8s %-6s %-22s %-22s %-22s %-9s %-9s' %
          ('step', '', 's1/a1', 'm1', 'm2', 'out', 'Q'))
    for st in steps:
        d = os.path.join(run, 'checkpoint_%d' % st)
        if not os.path.exists(os.path.join(d, 'critic_c_6.pkl')):
            continue
        net, _ = build(d, arch, h1, h2)
        with torch.no_grad():
            q, H = fwd(net, S_t, A_t, want_h=True)
        hs, ha, h1_, h2_, h3_ = H
        g = lambda x: (x.abs().mean().item(), 100 * (x == 0).float().mean().item())
        a1, d1 = g(hs); a2, d2 = g(ha); b1, e1 = g(h1_); b2, e2 = g(h2_); b3, e3 = g(h3_)
        print('  %-8d %-6s %.3f/%4.0f%% %.3f/%4.0f%% | %.3f/%4.0f%% %.3f/%4.0f%% | '
              '%.3f/%4.0f%%  bias %+6.2f | mean %7.1f sd %6.1f'
              % (st, '', a1, d1, a2, d2, b1, e1, b2, e2, b3, e3, net.out.bias.item(),
                 q.mean().item() * 100, q.std().item() * 100))
    # 每档再做一次梯度
    print('\n  %-8s %-26s %-26s' % ('step', '逐样本|∂Q/∂a| / 批均值', '沿+a mean/sd / 正号'))
    for st in steps:
        d = os.path.join(run, 'checkpoint_%d' % st)
        if not os.path.exists(os.path.join(d, 'critic_c_6.pkl')):
            continue
        net, _ = build(d, arch, h1, h2)
        at = A_t.clone().requires_grad_(True)
        q = fwd(net, S_t, at)[0]
        g, = torch.autograd.grad(q.sum(), at)
        gp = g.detach()[:, :5] * 100
        dd = gp.sum(1) / np.sqrt(5)
        print('  %-8d %10.1f / %-8.1f (x%.0f)      %+8.1f / %-8.1f (%.2f) %4.0f%%'
              % (st, gp.abs().mean(), gp.mean(0).abs().mean(),
                 gp.abs().mean() / max(gp.mean(0).abs().mean().item(), 1e-9),
                 dd.mean(), dd.std(), dd.mean() / max(dd.std(), 1e-9),
                 100 * float((dd > 0).float().mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v6_run', default=os.path.join('output', 'V6', 'AIFFLOR-run1'))
    ap.add_argument('--v2_run', default=os.path.join('output', 'FULL-R2', 'AIFFLOR-run3'))
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--hidden1', type=int, default=16)
    ap.add_argument('--hidden2', type=int, default=100)
    ap.add_argument('--trace', type=int, default=0)
    ap.add_argument('--layers', type=int, default=6, choices=[6, 1])
    a = ap.parse_args()

    audit_wiring(4, 6, 'H')
    audit_v6_layers()
    audit_init_scale()

    print('\n[C/D] checkpoint 实测（各自 run 末尾 %d 个块首状态）' % a.n)
    for tag, run, arch in [('v6', a.v6_run, 'v6'), ('v2', a.v2_run, 'v2')]:
        if not os.path.exists(os.path.join(run, 'critic_c_6.pkl')):
            print('  [%s] 缺 checkpoint，跳过' % tag); continue
        S, A, c1 = load_blocks(run, a.n, want_cols=('c1',))
        S_t, A_t = torch.tensor(S), torch.tensor(A)
        net, sd = build(run, arch, a.hidden1, a.hidden2)
        print('\n  == %s (critic_c_6.pkl, %d 参数) ==' % (tag, sum(v.numel() for v in sd.values())))
        fresh = model.Critic(4, 6, hidden1=a.hidden1, hidden2=a.hidden2, arch=arch,
                             out_init='fanin')
        print('    [未训练对照]'); health(fresh, S_t, A_t, 100.0, 'init')
        print('    [训练后]'); health(net, S_t, A_t, 100.0, 'trained')
        variance_split(net, S_t, A_t, 100.0)
        ablation(net, S_t, A_t, 100.0)
        grad_stats(net, S_t, A_t, 100.0)
        fit_check(net, S_t, A_t, c1, 100.0)

    print('\n[F] update_stats_H.csv 尾部 500 次（原始口径）')
    for tag, run in [('v6', a.v6_run), ('v2', a.v2_run)]:
        f = os.path.join(run, 'update_stats_H.csv')
        if not os.path.exists(f):
            print('  [%s] 无 update_stats_H.csv' % tag); continue
        d = pd.read_csv(f).tail(500).mean()
        print('  %-3s td_p %7.1f (%2.0f%%) | td_c %6.1f (%2.0f%%) | q_p %7.1f | '
              'q_c %6.1f | cost %6.1f | reward %7.1f'
              % (tag, d.td_p_mean, 100 * d.td_p_mean / max(d.q_p_mean, 1e-9),
                 d.td_c_mean, 100 * d.td_c_mean / max(d.q_c_mean, 1e-9),
                 d.q_p_mean, d.q_c_mean, d.cost_mean, d.reward_mean))

    if a.trace:
        trace(a.v6_run, [2000, 10000, 30000, 60000, 90000, 124000],
              'v6', a.hidden1, a.hidden2)


if __name__ == '__main__':
    main()
