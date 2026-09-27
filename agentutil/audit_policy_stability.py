# -*- coding: utf-8 -*-
"""跨 run 的 H 层**策略稳定性 / 值层自洽**同口径审计 (Common-Panel Audit)。

为什么需要"共同面板"
--------------------
单 run 审计时，面板通常取自该 run 自己的 `df_td3per.csv`（它访问过的状态）。
但不同 run 的状态分布不同，于是 `D_pi = E_s[|pi_b(s) - pi_a(s)|]` 的**绝对值不可跨 run 比**：
状态分布不同会系统性地改变这个期望。要做"runA 的策略比 runB 稳吗"这类判断，
必须让所有 run 在**同一组状态**上被查询。

本工具默认构造**均衡并集面板**: 每个 run 从自己的回放日志里等间隔抽 `--per-run`
条状态，拼成一个共同面板。这样面板覆盖所有 run 的访问区域，且各 run 权重相等。

口径 (与项目一致, 不 import 会启 MATLAB 的模块)
----------------------------------------------
* H 状态 = `Normalize_H(obs, minute, mode=1)` =
  `[obs0/1000, obs1/10, obs2/10, obs3/10, sin(2pi*m/1440), cos(...)]`  (6 维)
  与 `td3per/td3per_myfunc.py` 的 `Normalize` + `phase_features` 逐位一致；
  面板会用日志里的 `phase_sin/phase_cos` 列自校验 (`--check-phase`)。
* 动作边界 (CheckAction): 前 5 维 [0, 0.5]，第 6 维 signal [-0.5, 0.5]。
* 值层锚点 (Q 的"自洽水平"): `anchor = mean(块奖励) / (1-gamma) / s_p`，
  块奖励 = 每 5 步 (H 层一个决策块) 的 r1 之和；滚动窗口 `--anchor-win`。
  注意 `update_stats_H.csv` 里的 `reward_mean` **已按 s_p 归一**，所以直接用
  `Qbar / (rbar/(1-gamma)/s_p)` 就是无量纲自洽比。
* `gamma_eff` 校验: 对 `target_p_mean = a*reward_mean + g_eff*min_p_next_mean`
  做最小二乘 —— n=1 的 run 应得 `g_eff ~ gamma`，n=5 的 run 应得 `g_eff ~ gamma^5`。
  这是"某 run 到底跑的哪个 n"的数据证据 (见 `--check-gamma`)。

用法
----
    python agentutil/audit_policy_stability.py run9 run10 run13
    python agentutil/audit_policy_stability.py run9 run10 run12 run13 --steps 10000 16000 20000 26000
    python agentutil/audit_policy_stability.py run9 run10 run13 --panel-from run9 --check-gamma
"""

import argparse
import csv
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np
import torch

from model import Actor, Critic

BLOCK = 5                      # H 层一个决策块 = 5 个 env step
BOUNDS = [(-0.0, 0.5)] * 5 + [(-0.5, 0.5)]   # CheckAction: WT x5, signal x1


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------
def run_dir(name):
    if os.path.isdir(name) and os.path.exists(os.path.join(name, 'df_td3per.csv')):
        return name
    d = os.path.join('output', 'TD3PER-' + name)
    if os.path.isdir(d):
        return d
    raise SystemExit('找不到 run: %s' % name)


def ckpt_steps(d):
    out = []
    for p in glob.glob(os.path.join(d, 'checkpoint_*')):
        m = re.search(r'(\d+)$', p)
        if m and os.path.exists(os.path.join(p, 'actor_6.pkl')):
            out.append(int(m.group(1)))
    return sorted(out)


def read_log(d, every=1):
    """读 df_td3per.csv。

    返回 (rows, sampled):
      rows[step]     = (r1, lam1)            —— 全量, 供锚点/lam 滚动统计
      sampled        = [(obs4, phase_minute, phase_sin, phase_cos), ...] —— 每 every 行一条, 供构面板
    obs 只在需要时解析 (run9 日志 140MB, 逐行解析 4 个 float 太贵)。
    """
    p = os.path.join(d, 'df_td3per.csv')
    with open(p, newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        ix = {c.strip(): i for i, c in enumerate(head)}
        for c in ('step', 'observation', 'phase_minute', 'r1'):
            if c not in ix:
                raise SystemExit('%s 缺列 %s' % (p, c))
        i_st, i_ob, i_pm, i_r1 = ix['step'], ix['observation'], ix['phase_minute'], ix['r1']
        i_lam = ix.get('lam1')
        i_ps, i_pc = ix.get('phase_sin'), ix.get('phase_cos')

        rows, sampled = {}, []
        for n, r in enumerate(rd):
            if len(r) <= max(i_st, i_r1):
                continue
            try:
                st = int(float(r[i_st]))
                r1 = float(r[i_r1])
            except (ValueError, IndexError):
                continue
            lam = None
            if i_lam is not None:
                try:
                    lam = float(r[i_lam])
                except (ValueError, IndexError):
                    lam = None
            rows[st] = (r1, lam)
            if every > 1 and n % every:
                continue
            try:
                obs = np.array([float(x) for x in r[i_ob].strip().strip('[]').split()])
                if obs.size < 4:
                    continue
                pm = float(r[i_pm])
            except (ValueError, IndexError):
                continue
            ps = pc = None
            if i_ps is not None and i_pc is not None:
                try:
                    ps, pc = float(r[i_ps]), float(r[i_pc])
                except (ValueError, IndexError):
                    ps = pc = None
            sampled.append((obs[:4], pm, ps, pc))
    return rows, sampled


def h_state(obs, minute, mode=1):
    """与 td3per_myfunc.Normalize_H(mode=1) 逐位一致（不 import 该模块以免启 MATLAB）。"""
    base = [obs[0] / 1000.0, obs[1] / 10.0, obs[2] / 10.0, obs[3] / 10.0]
    if mode <= 0:
        return np.array(base)
    ang = 2.0 * np.pi * (int(minute) % 1440) / 1440.0
    return np.array(base + [np.sin(ang), np.cos(ang)])


def block_rewards(rows, win, st):
    """(st-win, st] 内每个块的奖励和（只需满足 5 步齐全）。rows[step] = (r1, lam)。"""
    out = []
    for b in range(((st - win) // BLOCK) * BLOCK, st, BLOCK):
        acc, ok = 0.0, True
        for k in range(BLOCK):
            rr = rows.get(b + k)
            if rr is None:
                ok = False
                break
            acc += rr[0]
        if ok:
            out.append(acc)
    return out


# ---------------------------------------------------------------------------
# 网络加载 / 评估
# ---------------------------------------------------------------------------
def load_actor(d, st):
    sd = torch.load(os.path.join(d, 'checkpoint_%d' % st, 'actor_6.pkl'),
                    map_location='cpu', weights_only=False)
    m = Actor(sd['fc1.weight'].shape[1], sd['fc4.weight'].shape[0],
              hidden1=sd['fc1.weight'].shape[0], hidden2=sd['fc2.weight'].shape[0])
    m.load_state_dict(sd)
    m.eval()
    return m


def load_critics(d, st):
    outs = []
    for nm in ('critic_p_6.pkl', 'critic_p2_6.pkl'):
        sd = torch.load(os.path.join(d, 'checkpoint_%d' % st, nm),
                        map_location='cpu', weights_only=False)
        h2 = sd['fc4.weight'].shape[0]
        m = Critic(sd['fc1.weight'].shape[1], sd['fc2.weight'].shape[1] - h2,
                   hidden1=sd['fc1.weight'].shape[0], hidden2=h2)
        m.load_state_dict(sd)
        m.eval()
        outs.append(m)
    return outs


def policy_actions(actor, X):
    with torch.no_grad():
        return actor(X).numpy()


def d_pi(a, b):
    """面板上的动作变化: mean_s max_k |a_k - b_k|（全 6 维）。"""
    return float(np.abs(a - b).max(axis=1).mean())


def d_pi5(a, b):
    return float(np.abs(a[:, :5] - b[:, :5]).max(axis=1).mean())


def out_of_bound(a):
    bad = np.zeros(len(a), dtype=bool)
    for k, (lo, hi) in enumerate(BOUNDS):
        bad |= (a[:, k] < lo - 1e-9) | (a[:, k] > hi + 1e-9)
    return float(bad.mean())


def check_gamma(d, tail=20000):
    """从 update_stats_H.csv 反解 target = a*reward + g_eff*min_next。"""
    p = os.path.join(d, 'update_stats_H.csv')
    if not os.path.exists(p):
        return None
    with open(p, newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        ix = {c.strip(): i for i, c in enumerate(head)}
        rows = list(rd)[-tail:]
    try:
        i_t, i_r, i_v = ix['target_p_mean'], ix['reward_mean'], ix['min_p_next_mean']
    except KeyError:
        return None

    def col(i):
        out = []
        for r in rows:
            try:
                out.append(float(r[i]))
            except (ValueError, IndexError):
                out.append(np.nan)
        return np.array(out)

    t, rw, v = col(i_t), col(i_r), col(i_v)
    m = np.isfinite(t) & np.isfinite(rw) & np.isfinite(v)
    if m.sum() < 100:
        return None
    t, rw, v = t[m], rw[m], v[m]
    A = np.column_stack([rw, v, np.ones_like(rw)])
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    r2 = 1 - ((t - A @ coef) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    return dict(a=float(coef[0]), g_eff=float(coef[1]), r2=float(r2), n=int(m.sum()))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='跨 run 共同面板的策略稳定性 / 值层审计',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='run 名（自动补 output/TD3PER-<name>）或目录')
    ap.add_argument('--panel-from', default='union',
                    help="面板来源: 'union'(默认, 各 run 均衡并集) 或某个 run 名")
    ap.add_argument('--per-run', type=int, default=1200, help='每个 run 贡献的面板状态数（默认 1200）')
    ap.add_argument('--every', type=int, default=1, help='读日志时的抽样步长（默认 1, 全量）')
    ap.add_argument('--steps', nargs='*', type=int, default=None,
                    help='要评估的 checkpoint 步（默认: 所有 run 的公共步）')
    ap.add_argument('--gamma', type=float, default=0.98, help='块级折扣（默认 0.98）')
    ap.add_argument('--s-p', type=float, default=1000.0, help='H 层 reward 归一尺度（默认 1000）')
    ap.add_argument('--anchor-win', type=int, default=20000, help='锚点滚动窗口（默认 20000 步）')
    ap.add_argument('--stat-win', type=int, default=4, help='[C] 摘要统计的窗口长度（末 N 个评估步，默认 4）')
    ap.add_argument('--lag-n', type=int, default=7, help='[F] 标度律用每个 run 自己最后 N 个连续 checkpoint（默认 7）')
    ap.add_argument('--anchor-min', type=float, default=1.0,
                    help='锚点稳健阈值: |锚点| < 该值时自洽比记为 nan（小分母相除会虚高，默认 1.0）')
    ap.add_argument('--check-gamma', action='store_true', help='反解各 run 的 gamma_eff')
    ap.add_argument('--check-phase', action='store_true', help='用日志的 phase_sin/cos 列校验状态口径')
    ap.add_argument('--baseline', nargs='*', default=None,
                    help='基线 run 标签（默认: 名字含 run9/run10 的）')
    ap.add_argument('--candidate', nargs='*', default=None,
                    help='候选 run 标签（默认: 其余全部）')
    args = ap.parse_args()

    # 显示标签 = 用户输入的写法; 目录单独解析
    labels = list(args.runs)
    dirs = {}
    for lab, a in zip(labels, args.runs):
        dirs[lab] = run_dir(a)
    runs = labels

    # ---- 读日志 ----
    logs, panels, phase_err = {}, {}, 0.0
    for nm in runs:
        rows, sampled = read_log(dirs[nm], every=args.every)
        logs[nm] = rows
        print('[%s] 日志 %d 行 (step %d..%d), checkpoint %d 个'
              % (nm, len(rows), min(rows), max(rows), len(ckpt_steps(dirs[nm]))))
        for s in sampled:
            if s[2] is not None and s[3] is not None:
                st_vec = h_state(s[0], s[1], 1)
                phase_err = max(phase_err, abs(st_vec[4] - s[2]), abs(st_vec[5] - s[3]))
        panels[nm] = sampled
    if args.check_phase:
        print('  状态口径自校验: |sin/cos 复算 - 日志列| 最大偏差 = %.3e %s'
              % (phase_err, 'OK' if phase_err < 1e-9 else '<== 口径不一致!'))

    # ---- 共同面板 ----
    if args.panel_from == 'union':
        picks = []
        for nm in runs:
            cand = [s for s in panels[nm] if len(s) >= 2]
            if not cand:
                continue
            idx = np.linspace(0, len(cand) - 1, min(args.per_run, len(cand))).astype(int)
            picks += [cand[i] for i in idx]
        src = 'union(%s)' % ','.join(runs)
    else:
        key = None
        for lab in runs:
            if args.panel_from == lab or os.path.basename(dirs[lab].rstrip('/\\')) == args.panel_from:
                key = lab
                break
        if key is None:
            raise SystemExit('--panel-from 指定的 run 不在本次列表里: %s' % args.panel_from)
        cand = [s for s in panels[key] if len(s) >= 2]
        idx = np.linspace(0, len(cand) - 1, min(args.per_run, len(cand))).astype(int)
        picks = [cand[i] for i in idx]
        src = key
    X = torch.tensor(np.stack([h_state(o, pm, 1) for o, pm, *_ in picks]).astype(np.float32))
    print('共同面板: %s, %d 个状态 (H state dim=%d)' % (src, len(X), X.shape[1]))

    # ---- 评估步 ----
    allsteps = {nm: ckpt_steps(dirs[nm]) for nm in runs}
    if args.steps:
        steps = [s for s in sorted(set(args.steps)) if all(s in allsteps[nm] for nm in runs)]
        miss = sorted(set(args.steps) - set(steps))
        if miss:
            print('  跳过(并非所有 run 都有该 checkpoint): %s' % miss)
    else:
        common = set.intersection(*[set(v) for v in allsteps.values()])
        steps = sorted(common)
    if not steps:
        raise SystemExit('没有公共 checkpoint 步可评（用 --steps 指定）')
    print('评估步: %s' % steps)

    # ---- 逐 run 逐步评估 ----
    R = {}
    for nm in runs:
        d = dirs[nm]
        acts, rec = {}, []
        first = None
        for st in steps:
            a_model = load_actor(d, st)
            a = policy_actions(a_model, X)
            c1, c2 = load_critics(d, st)
            with torch.no_grad():
                q = torch.min(c1([X, a_model(X)]), c2([X, a_model(X)])).squeeze(1).mean().item()
            xg = X.clone().requires_grad_(True)
            ag = a_model(xg)
            g = torch.autograd.grad(c1([xg, ag]).sum(), ag)[0].norm(dim=1).mean().item()
            br = block_rewards(logs[nm], args.anchor_win, st)
            anchor = (np.mean(br) / (1 - args.gamma) / args.s_p) if br else float('nan')
            ratio = (q / anchor) if anchor and abs(anchor) >= args.anchor_min else float('nan')
            lam = [v[1] for k, v in logs[nm].items() if v[1] is not None and st - args.anchor_win < k <= st]
            if first is None:
                first = a.copy()
            prev = acts.get(steps[steps.index(st) - 1]) if steps.index(st) > 0 else None
            rec.append(dict(st=st, d_first=d_pi(a, first), d_first5=d_pi5(a, first),
                            d_adj=(d_pi(a, prev) if prev is not None else float('nan')),
                            qbar=q, anchor=anchor, ratio=ratio,
                            ratio_raw=(q / anchor if anchor else float('nan')),
                            m=out_of_bound(a), grad=g, grad_rel=(g / abs(q) if q else float('nan')),
                            lam=(float(np.mean(lam)) if lam else float('nan'))))
            acts[st] = a
        # 固定锚变体: 全部用该 run 末点的锚点做分母 (分母不动, 只看 Qbar 是否漂移)
        anchor_last = rec[-1]['anchor']
        for r in rec:
            r['ratio_fix'] = (r['qbar'] / anchor_last
                              if abs(anchor_last) >= args.anchor_min else float('nan'))
        R[nm] = rec

    # ---- 输出 A: 策略稳定性 ----
    print('\n[A] 策略稳定性（共同面板; D_pi = mean_s max_k|a_k-a_k|, 全 6 维; 基准=首评步的策略）')
    print('  %-7s' % 'step' + ''.join(' | %-28s' % nm for nm in runs))
    print('  %-7s' % '' + ''.join(' | %-28s' % 'D_pi首   D_pi邻    M' for _ in runs))
    for i, st in enumerate(steps):
        line = '  %-7d' % st
        for nm in runs:
            r = R[nm][i]
            line += ' | %7.3f %8.3f %7.3f   ' % (r['d_first'], r['d_adj'], r['m'])
        print(line)

    # ---- 输出 B: 值层 ----
    print('\n[B] 值层自洽（Qbar / 锚点(rbar/(1-g)/(s_p)) / 自洽比 / |dQp/da| / lam1）')
    print('    自洽比后带 * = |锚点| < %.2f, 小分母相除不可信, 已在 [C] 中剔除' % args.anchor_min)
    for nm in runs:
        print('  --- %s ---' % nm)
        for r in R[nm]:
            flag = '' if r['ratio'] == r['ratio'] else ' *'
            print('    %-7d Qbar %7.3f  锚 %7.3f  比 %5.2f%s (固定锚 %5.2f)  |dQp/da| %6.3f (相对 %5.3f)  lam1 %6.4f'
                  % (r['st'], r['qbar'], r['anchor'], r['ratio'], flag, r['ratio_fix'],
                     r['grad'], r['grad_rel'], r['lam']))

    # ---- 输出 C: 改善摘要 ----
    def is_base(nm):
        return bool(re.search(r'run\s*9\b|run\s*10\b', nm))
    base = args.baseline if args.baseline else [nm for nm in runs if is_base(nm)]
    cand = args.candidate if args.candidate else [nm for nm in runs if nm not in base]
    if base and cand:
        win = min(args.stat_win, len(steps))
        keys = [('d_adj', 'D_pi邻', '低'), ('m', 'M越界', '低'),
                ('ratio', '自洽比(滚动锚)', '高'), ('ratio_fix', '自洽比(固定锚)', '高'),
                ('grad_rel', '|dQp/da|/Qbar', '低')]
        print('\n[C] 改善摘要（候选 vs 基线; "<" 表示候选更优）')
        print('  窗口 = 末 %d 个评估步 (step %d..%d)' % (win, steps[-win], steps[-1]))
        for k, label, good in keys:
            means = {}
            for nm in runs:
                raw = [R[nm][i][k] for i in range(len(steps) - win, len(steps))]
                v = [x for x in raw if x == x]
                means[nm] = (np.mean(v) if v else float('nan'),
                             max((abs(x) for x in v), default=float('nan')),
                             len(v), (np.std(v) if len(v) > 1 else float('nan')))
            print('    %-14s(%-3s)' % (label, good)
                  + ''.join(' %18s' % nm for nm in runs))
            print('    %-14s     ' % '均值'
                  + ''.join(' %18.3f' % means[nm][0] for nm in runs))
            print('    %-14s     ' % 'sd/最大|值|'
                  + ''.join(' %8.3f /%8.3f' % (means[nm][3], means[nm][1]) for nm in runs))
            print('    %-14s     ' % '有效点/窗口'
                  + ''.join(' %8d /%8d' % (means[nm][2], win) for nm in runs))
            for c in cand:
                for b in base:
                    if means[b][0] != means[b][0] or means[c][0] != means[c][0]:
                        continue
                    d = means[c][0] - means[b][0]
                    better = (d < 0) if good == '低' else (d > 0)
                    rel = 100.0 * d / abs(means[b][0]) if abs(means[b][0]) > 1e-12 else float('nan')
                    print('      %s vs %s: %+.3f (%+.0f%%)  -> %s'
                          % (c, b, d, rel, '候选更优 ✓' if better else '基线更优 ✗'))
        print('    注: |dQp/da|/Qbar 的"低"只是"更平缓"，本项目没有基准说它一定越好;'
              ' 该列若候选远高于基线，应单独追查 critic 的尺度/尖锐度。')

    # ---- 输出 D: gamma 证据 ----
    if args.check_gamma:
        print('\n[D] gamma_eff 反解（target_p = a*reward_mean + g_eff*min_p_next_mean）')
        print('    n=1 的 run 应得 g_eff ~ gamma=%.2f; n=5 应得 g_eff ~ gamma^5=%.4f'
              % (args.gamma, args.gamma ** 5))
        print('  %-10s %8s %10s %12s %10s' % ('run', 'n行', 'a', 'g_eff', 'R2'))
        for nm in runs:
            r = check_gamma(dirs[nm])
            if r:
                guess = 'n=1' if abs(r['g_eff'] - args.gamma) < abs(r['g_eff'] - args.gamma ** 5) else 'n>1'
                print('  %-10s %8d %10.5f %12.5f %10.5f   -> 更像 %s' % (nm, r['n'], r['a'], r['g_eff'], r['r2'], guess))

    # ---- 输出 E: 各 run 自身末点/最好 ----
    print('\n[E] 各 run 自身历史（不保证同 step）:')
    for nm in runs:
        d = dirs[nm]
        last = ckpt_steps(d)[-1]
        a = policy_actions(load_actor(d, last), X)
        a0 = policy_actions(load_actor(d, ckpt_steps(d)[0]), X)
        c1, c2 = load_critics(d, last)
        with torch.no_grad():
            q = torch.min(c1([X, load_actor(d, last)(X)]), c2([X, load_actor(d, last)(X)])).squeeze(1).mean().item()
        br = block_rewards(logs[nm], args.anchor_win, last)
        anchor = np.mean(br) / (1 - args.gamma) / args.s_p if br else float('nan')
        print('  %-8s 末 ckpt %6d: D_pi首 %6.3f  M %6.3f  Qbar %6.3f  锚 %6.3f  比 %5.2f  历史最靠后 D_pi首 见 [A]'
              % (nm, last, d_pi(a, a0), out_of_bound(a), q, anchor, q / anchor if anchor else float('nan')))

    # ---- 输出 F: 位移-间隔标度律 (策略到底停在定点, 还是还在匀速漂移?) ----
    print('\n[F] 策略是否还在漂移: 位移-间隔标度律 (每个 run 用自己最后 %d 个连续 checkpoint)' % args.lag_n)
    print('    rms(k) = sqrt(mean_s ||a_{i+k}(s) - a_i(s)||^2), 对同间隔的所有 i 取均值')
    print('    拟合 log rms = alpha * log k + c:')
    print('      alpha ~ 1.0  匀速漂移   -> 尚未收敛 (位移随间隔线性增长)')
    print('      alpha ~ 0.5  定点附近随机游走 -> 已收敛到稳定点 (位移按 sqrt(k) 扩散)')
    print('      alpha ~ 0.0  冻结       -> 完全不动')
    print('    a_scale = 面板上动作的标准差量级; rms(1)/a_scale 越小说明"checkpoint 间移动"相对')
    print('    "策略在不同状态间的差异"越可忽略 (经验: <0.1 基本可视为冻结)')
    print('  %-10s %5s %9s %9s %9s %10s  %s' % ('run', 'n', 'alpha', 'R2', 'rms(1)', 'rms1/a_sc', 'rms(k) k=1..'))
    for nm in runs:
        d = dirs[nm]
        ss = ckpt_steps(d)[-args.lag_n:]
        if len(ss) < 3:
            print('  %-10s (checkpoint 不足 3 个, 跳过)' % nm)
            continue
        acts = {s: policy_actions(load_actor(d, s), X) for s in ss}
        A_all = np.concatenate([acts[s] for s in ss], axis=0)
        a_scale = float(np.sqrt((A_all.std(axis=0) ** 2).mean()))
        ks, rms = [], []
        for k in range(1, len(ss)):
            ds = [float(np.sqrt(((acts[ss[i + k]] - acts[ss[i]]) ** 2).sum(axis=1).mean()))
                  for i in range(len(ss) - k)]
            ks.append(k)
            rms.append(float(np.mean(ds)))
        al, c0 = np.polyfit(np.log(ks), np.log(rms), 1)
        pred = al * np.log(ks) + c0
        r2 = 1 - ((np.log(rms) - pred) ** 2).sum() / ((np.log(rms) - np.mean(np.log(rms))) ** 2).sum()
        print('  %-10s %5d %9.3f %9.4f %9.4f %10.4f  %s'
              % (nm, len(ss), al, r2, rms[0], rms[0] / a_scale,
                 ' '.join('%.4f' % v for v in rms)))
        # 相邻 D_pi 的线性趋势 (末 N 个 checkpoint)
        adj = [d_pi(acts[ss[i]], acts[ss[i - 1]]) for i in range(1, len(ss))]
        sl = np.polyfit(np.arange(len(adj)), adj, 1)[0]
        verdict = ('仍在匀速漂移(未收敛)' if al > 0.8 else
                   '定点附近游走(已稳定)' if 0.35 <= al <= 0.8 else '冻结')
        print('             %s: 相邻 D_pi 趋势 %+.4f/ckpt, 序列 %s'
              % (verdict, sl, ' '.join('%.3f' % v for v in adj)))


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
