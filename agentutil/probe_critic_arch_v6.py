#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_critic_arch_v6.py —— v6 critic 隐藏层审计（v6 / v6old / v2 通用）。

用法（任意 CWD）：
    python agentutil/probe_critic_arch_v6.py                 # 接线+初始化审计，再对比两个 run
    python agentutil/probe_critic_arch_v6.py --trace 1       # 段 E：层健康随训练步数演化
    python agentutil/probe_critic_arch_v6.py --v6_run output/V6/AIFFLOR-run1   # 历史 v6old

arch 由 checkpoint 自动推断（`model.infer_arch`），所以 **新 v6（96→128→256→256→128）
与历史 v6old（70→256→256→128）都能直接读**。

A. 接线：各 arch 层表/参数量；`--hidden1/--hidden2` 有没有真的接上；infer_arch 唯一性。
B. 初始化尺度：`fanout_init`（v=1/sqrt(out_features)）逐层 Var(z)≈fan_in·v²/3。
C. 值-动作方差分解：同一状态上扰动动作，动作能解释 Q 方差的多少（真值≈30%）。
D. 动作支路消融 + 逐样本/批均值梯度。
E. 层健康（死单元、|h|、|W|、Q）随训练步数演化。
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

ARCHS = ['v1', 'v1af', 'v2', 'v3', 'v4', 'v5', 'v6', 'v6old']


def parse_vec(col):
    return np.vstack([np.fromstring(str(x).strip().lstrip('[').rstrip(']'), sep=' ')
                      for x in col]).astype(np.float64)


def load_blocks(run, n, want_cols=('c1',)):
    """末尾 n 个完整 5 步块的块首 (S, A_full, *want_cols)。

    df_hsrl.csv 行数不一定是 5 的倍数（末尾截断）⇒ 按 step//5 分组、只留完整块。
    """
    cols = ['step', 'observation', 'applied_action_H'] + list(want_cols)
    df = pd.read_csv(os.path.join(run, 'df_hsrl.csv'), usecols=cols)
    step = df.step.to_numpy()
    bt = step // 5
    sizes = pd.Series(bt).value_counts()
    complete = set(sizes[sizes == 5].index.tolist())
    idx = np.where((step % 5 == 0) & np.isin(bt, list(complete)))[0][-n:]
    S = np.vstack([Normalize(o) for o in parse_vec(df.observation.iloc[idx])]).astype(np.float32)
    A = parse_vec(df.applied_action_H.iloc[idx]).astype(np.float32)
    out = [S, A]
    for c in want_cols:
        out.append(pd.Series(df[c].to_numpy()).groupby(bt).sum().loc[bt[idx]].to_numpy())
    return out


# ---------------------------------------------------------------- 前向（可消融）
def fwd(net, S, A, zero_ha=False, zero_hs=False, zero_a=False, want_h=False):
    r = net.relu
    if net.arch == 'v6':
        hs = r(net.s2(r(net.s1(S))))
        ha = r(net.a2(r(net.a1(A))))
        if zero_hs:
            hs = torch.zeros_like(hs)
        if zero_ha:
            ha = torch.zeros_like(ha)
        m = torch.cat([hs, ha], 1)                      # 96，无原始 a
        h1 = r(net.m1(m)); h2 = r(net.m2(h1)); h3 = r(net.m3(h2)); h4 = r(net.m4(h3))
        q = net.out(h4)
        H = (hs, ha, h1, h2, h3, h4)
    elif net.arch == 'v6old':
        hs = r(net.s1(S))
        ha = r(net.a1(A))
        if zero_hs:
            hs = torch.zeros_like(hs)
        if zero_ha:
            ha = torch.zeros_like(ha)
        aa = torch.zeros_like(A) if zero_a else A
        m = torch.cat([hs, ha, aa], 1)
        h1 = r(net.m1(m)); h2 = r(net.m2(h1)); h3 = r(net.m3(h2))
        q = net.out(h3)
        H = (hs, ha, h1, h2, h3, None)
    elif net.arch in ('v2', 'v3'):
        hs = r(net.s2(r(net.s1(S))))
        ha = r(net.a2(r(net.a1(A))))
        if zero_hs:
            hs = torch.zeros_like(hs)
        if zero_ha:
            ha = torch.zeros_like(ha)
        z = torch.cat([hs, ha], 1)
        h1 = r(net.m1(z)) + z
        h2 = r(net.m2(h1)); h3 = r(net.m3(h2))
        q = net.out(h3)
        H = (hs, ha, h1, h2, h3, None)
    else:
        raise SystemExit('probe 只支持 v2/v3/v6/v6old，收到 %s' % net.arch)
    return (q, H) if want_h else (q,)


def build(run, h1=16, h2=100, layer=6, out_init='fanin', arch=None):
    f = os.path.join(run, 'critic_c_%d.pkl' % layer)
    sd = torch.load(f, map_location='cpu')
    ns, na = (4, 6) if layer == 6 else (int(sd['s1.weight'].shape[1]), 1)
    arch = arch or model.infer_arch(sd, ns, na)
    net = model.Critic(ns, na, hidden1=h1, hidden2=h2, arch=arch, out_init=out_init)
    net.load_state_dict(sd)
    net.eval()
    return net, sd


# ---------------------------------------------------------------- A. 接线
def audit_wiring(ns, na, tag):
    print('\n[A] ar%s (nb_states=%d, nb_actions=%d)' % (tag, ns, na))
    print('    %-6s %-7s %-9s %-24s %-8s %s'
          % ('arch', 'hidden', 'params', 'm1.weight', 'infer', '备注'))
    for hid in [(16, 100), (64, 200)]:
        for arch in ARCHS:
            net = model.Critic(ns, na, hidden1=hid[0], hidden2=hid[1], arch=arch)
            sd = net.state_dict()
            n = sum(v.numel() for v in sd.values())
            m1 = sd['m1.weight'].shape if 'm1.weight' in sd else '-'
            rm1 = model.infer_arch(sd, ns, na)
            note = '' if rm1 == arch else '⚠推断成 %s' % rm1
            if arch == 'v6old':
                note = (note + ' ' if note else '') + '历史兼容，勿用于新训练'
            print('    %-6s %-7s %-9d %-24s %-8s %s'
                  % (arch, '%d/%d' % hid, n, str(m1), rm1, note))
    print('    --hidden1/--hidden2 是否生效:')
    for arch in ARCHS:
        n1 = sum(v.numel() for v in model.Critic(ns, na, hidden1=16, hidden2=100,
                                                arch=arch).state_dict().values())
        n2 = sum(v.numel() for v in model.Critic(ns, na, hidden1=64, hidden2=200,
                                                arch=arch).state_dict().values())
        print('      %-6s %7d vs %7d  %s' % (arch, n1, n2,
                                            '接上了' if n1 != n2 else '**完全不生效**'))


def audit_v6_layers():
    d = model.V6_DIMS
    print('\n[A2] 新 v6 逐层（V6_DIMS = %s）' % d)
    net = model.Critic(4, 6, hidden1=16, hidden2=100, arch='v6')
    rows = [('s1', 'state'), ('s2', 'state'), ('a1', 'action'), ('a2', 'action'),
            ('m1', 'merge'), ('m2', 'merge'), ('m3', 'merge'), ('m4', 'merge'),
            ('out', 'output')]
    tot = 0
    for k, part in rows:
        w = getattr(net, k).weight
        p = w.numel() + getattr(net, k).bias.numel()
        tot += p
        print('    %-4s %-7s %-10s %-14s %7d  (%.1f%%)'
              % (k, part, '%d→%d' % (w.shape[1], w.shape[0]), str(tuple(w.shape)),
                 p, 100.0 * p / 150081))
    sym = sum(getattr(net, k).weight.numel() + getattr(net, k).bias.numel()
              for k in ('s1', 's2', 'a1', 'a2'))
    merge = tot - sym - net.out.weight.numel() - net.out.bias.numel()
    print('    合计 %d；merge 四层 %d (%.1f%%)；两条支路 %d (%.1f%%)；out %d'
          % (tot, merge, 100.0 * merge / tot, sym, 100.0 * sym / tot,
             net.out.weight.numel() + net.out.bias.numel()))
    print('    L 层（na=1）merge 形状相同 ⇒ 也是 ~150k 参数（合并输入恒 96）')
    print('    对照 v2: 4→16→100 ∥ 6→32→64 → cat164 →164(残差)→128→64→1 = 60 617')
    print('    对照 v6old: 4→32 ∥ 6→32 → cat(32+32+6)=70 →256→256→128→1 = 117 377')


# ---------------------------------------------------------------- B. 初始化尺度
def audit_init_scale(d=None):
    d = d or model.V6_DIMS
    print('\n[B] fanout_init 逐层 Var(z) ≈ fan_in·v²/3（v = 1/sqrt(out_features)）— 新 v6')
    spec = [('s1', 4, d['s'][0]), ('s2', d['s'][0], d['s'][1]),
            ('a1', 6, d['a'][0]), ('a2', d['a'][0], d['a'][1]),
            ('m1', d['s'][1] + d['a'][1], d['merge'][0]),
            ('m2', d['merge'][0], d['merge'][1]), ('m3', d['merge'][1], d['merge'][2]),
            ('m4', d['merge'][2], d['merge'][3]), ('out', d['merge'][3], 1)]
    cum = 1.0
    print('    %-4s %-12s %-8s %-14s %s' % ('层', '形状(out×in)', 'bound', 'Var(z)/Var(x)', ''))
    for k, i, o in spec:
        v = 1.0 / np.sqrt(o)
        g = i * v * v / 3.0
        if k.startswith('m'):
            cum *= g
        print('    %-4s %-12s %-8.4f %-14.4f %s'
              % (k, '%d×%d' % (o, i), v, g, ('≈1/%.1f' % (1 / g)) if g < 1 else ''))
    print('    ⇒ merge 栈累计 Var 衰减 ≈ 1/%.0f（v2 靠 m1 的残差 +z 避开了这一段）' % (1 / cum))


# ---------------------------------------------------------------- C/D
def health(net, S, A, s_c, tag=''):
    with torch.no_grad():
        q, H = fwd(net, S, A, want_h=True)
    hs, ha, h1, h2, h3, h4 = H
    f = lambda x: '—' if x is None else '|h| %6.3f dead %5.1f%%' % (
        x.abs().mean(), 100 * (x == 0).float().mean())
    print('    %-8s Q(raw) mean %8.1f sd %7.1f | hs %s | ha %s'
          % (tag, q.mean().item() * s_c, q.std().item() * s_c, f(hs), f(ha)))
    print('    %-8s merge: m1 %s | m2 %s' % ('', f(h1), f(h2)))
    print('    %-8s        m3 %s | m4 %s' % ('', f(h3), f(h4)))
    print('    %-8s |W|: s1 %.3f s2 %.3f a1 %.3f a2 %.3f | m1 %.3f m2 %.3f m3 %.3f m4 %.3f'
          % ('', net.s1.weight.abs().mean(), net.s2.weight.abs().mean() if hasattr(net, 's2')
             else 0, net.a1.weight.abs().mean(),
             net.a2.weight.abs().mean() if hasattr(net, 'a2') else 0,
             net.m1.weight.abs().mean(), net.m2.weight.abs().mean(),
             net.m3.weight.abs().mean(),
             net.m4.weight.abs().mean() if hasattr(net, 'm4') else 0))
    print('    %-8s out |W| %.5f  out.bias %+.3f' % ('', net.out.weight.abs().mean(),
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
            within.append(float((fwd(net, S, Ax)[0] - base).var(unbiased=False)) * s_c * s_c)
    share = float(np.mean(within)) / max(tot, 1e-12)
    print('    方差分解: sd(Q)=%.1f  扰动动作(±%.2f)可解释 %.1f%%（真值≈30%%）'
          % (np.sqrt(tot), sigma, 100 * share))
    return share


def ablation(net, S, A, s_c):
    with torch.no_grad():
        q0 = fwd(net, S, A)[0]
        q_noha = fwd(net, S, A, zero_ha=True)[0]
        q_nohs = fwd(net, S, A, zero_hs=True)[0]
        q_onlys = fwd(net, S, A, zero_ha=True, zero_a=True)[0]
        q_onlya = fwd(net, S, A, zero_hs=True)[0]
    m = lambda x: float(x.mean()) * s_c
    print('    消融: 全量 %8.1f | 去a支路 %8.1f | 只留状态 %8.1f | 只留动作 %8.1f'
          % (m(q0), m(q_noha), m(q_onlys), m(q_onlya)))
    print('      ⇒ 动作整体均值贡献 |Q0−Q(only_s)| = %.1f (= %.1f%% 的 sd(Q))'
          % (abs(m(q0) - m(q_onlys)),
             100 * abs(m(q0) - m(q_onlys)) / max(float(q0.std()) * s_c, 1e-9)))


def grad_stats(net, S, A, s_c):
    at = A.clone().requires_grad_(True)
    q = fwd(net, S, at)[0]
    g, = torch.autograd.grad(q.sum(), at)
    gp = g.detach()[:, :5] * s_c
    bm = gp.mean(0).abs().mean().item()
    print('    梯度: 逐样本 |∂Q/∂a| mean %8.2f median %8.2f | 批均值 %.2f | 比值 %.1f'
          % (gp.abs().mean(), gp.abs().median(), bm, gp.abs().mean() / max(bm, 1e-9)))
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
def trace(run, steps, h1=16, h2=100, n=1500):
    S, A, c1 = load_blocks(run, n, want_cols=('c1',))
    S_t, A_t = torch.tensor(S), torch.tensor(A)
    print('\n[E] 层健康随训练步数（%s，同一批 %d 个块首状态）' % (os.path.basename(run), n))
    print('  %-8s %-30s %-30s %-10s' % ('step', 's 支路 / a 支路', 'm1 / m2', 'Q'))
    for st in steps:
        d = os.path.join(run, 'checkpoint_%d' % st)
        if not os.path.exists(os.path.join(d, 'critic_c_6.pkl')):
            continue
        net, _ = build(d, h1, h2)
        with torch.no_grad():
            q, H = fwd(net, S_t, A_t, want_h=True)
        hs, ha, h1_, h2_, h3_, h4_ = H
        g = lambda x: (x.abs().mean().item(), 100 * (x == 0).float().mean().item())
        s1v, s1d = g(hs); a1v, a1d = g(ha); m1v, m1d = g(h1_); m2v, m2d = g(h2_)
        m3v, m3d = g(h3_)
        print('  %-8d s %.3f/%3.0f%%  a %.3f/%3.0f%%      m1 %.3f/%3.0f%%  '
              'm2 %.3f/%3.0f%%  m3 %.3f/%3.0f%% | mean %6.1f sd %5.1f'
              % (st, s1v, s1d, a1v, a1d, m1v, m1d, m2v, m2d, m3v, m3d,
                 q.mean().item() * 100, q.std().item() * 100))
    print('\n  %-8s %-30s %s' % ('step', '逐样本 |∂Q/∂a| / 批均值', '沿+a mean/sd / 正号'))
    for st in steps:
        d = os.path.join(run, 'checkpoint_%d' % st)
        if not os.path.exists(os.path.join(d, 'critic_c_6.pkl')):
            continue
        net, _ = build(d, h1, h2)
        at = A_t.clone().requires_grad_(True)
        q = fwd(net, S_t, at)[0]
        g, = torch.autograd.grad(q.sum(), at)
        gp = g.detach()[:, :5] * 100
        dd = gp.sum(1) / np.sqrt(5)
        print('  %-8d %10.1f / %-8.1f (x%-3.0f)   %+8.1f / %-8.1f (%.2f) %4.0f%%'
              % (st, gp.abs().mean(), gp.mean(0).abs().mean(),
                 gp.abs().mean() / max(gp.mean(0).abs().mean().item(), 1e-9),
                 dd.mean(), dd.std(), dd.mean() / max(dd.std(), 1e-9),
                 100 * float((dd > 0).float().mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v6_run', default=os.path.join('output', 'V6', 'AIFFLOR-run1'),
                    help='v6/v6old 的 run 目录（历史 V6 跑的是 v6old）')
    ap.add_argument('--v2_run', default=os.path.join('output', 'FULL-R2', 'AIFFLOR-run3'))
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--hidden1', type=int, default=16)
    ap.add_argument('--hidden2', type=int, default=100)
    ap.add_argument('--trace', type=int, default=0)
    a = ap.parse_args()

    audit_wiring(4, 6, 'H')
    audit_v6_layers()
    audit_init_scale()

    print('\n[C/D] checkpoint 实测（各自 run 末尾 %d 个块首状态）' % a.n)
    for tag, run in [('new-run', a.v6_run), ('v2', a.v2_run)]:
        if not os.path.exists(os.path.join(run, 'critic_c_6.pkl')):
            print('  [%s] 缺 checkpoint，跳过' % tag); continue
        S, A, c1 = load_blocks(run, a.n, want_cols=('c1',))
        S_t, A_t = torch.tensor(S), torch.tensor(A)
        net, sd = build(run, a.hidden1, a.hidden2)
        ns, na = int(sd['s1.weight'].shape[1]), net.nb_actions
        print('\n  == %s (%s, %d 参数) ==' % (tag, net.arch, sum(v.numel() for v in sd.values())))
        print('    [未训练对照]')
        health(model.Critic(ns, na, hidden1=a.hidden1, hidden2=a.hidden2, arch=net.arch,
                            out_init='fanin'), S_t, A_t, 100.0, 'init')
        print('    [训练后]')
        health(net, S_t, A_t, 100.0, 'trained')
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
              a.hidden1, a.hidden2)


if __name__ == '__main__':
    main()
