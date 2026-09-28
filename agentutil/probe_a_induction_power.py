# -*- coding: utf-8 -*-
"""从日志估 ∂(风场功率)/∂a_j —— 检验"轴向因子被推到 0.5 上界"是否与 env 物理一致。

为什么需要它
------------
`Cp(a) = 4a(1−a)²` 在 **a=1/3** 取最大 (Betz)，到 a=0.5 只有 0.5/0.5926 = **84%**
(单机功率掉 ~16%)；而 a>0.5 是湍流尾流态，机械上不允许 (env 自己也会
`np.clip(act_wind, 0, 0.5)`)。因此"Q 峰正好在 a_j=0.5 上"这个角点**不能只凭
critic 形状就判定为良性** —— 它必须先与 env 真实功率模型一致。

`probe_q_profile_boundary.py` 说的是 **critic 的偏好**(Q 峰/界外斜率)；
本脚本说的是 **env 的真实物理**(用日志里真实执行过的动作与真实功率做局部回归)。
两者矛盾 ⇒ 问题在奖励/critic 拟合，不在参数化、不在罚的大小。

口径
----
* 响应 `y = ptur` 由日志反解: `turbine_profit = w_wind·pr·ptur·time` (time=1/60 h)
  ⇒ `ptur = turbine_profit·60/(w_wind·pr)`（**同一行**的动作 ↔ 同一行的 `ptur`，
  因为循环里 `pw_tru = PowerOutput(obs[1], action_H)` 在当步开头算）。
  脚本会打印 `ptur` 与 `observation[3]` 同行/下一行的相关系数做**对齐自校验**。
* 控制变量: 按 **u 分箱**(默认 0.5 m/s)，箱内 OLS `y ~ 1 + a0..a4 + u`，
  再按样本量加权汇总各维斜率 ⇒ 在"同风速、其余四维线性调整后"的偏效应。
* 同时给出 `r1` / `c1_step` / `bess_profit` 的同类斜率: 若 `ptur` 的斜率<0 而
  `r1` 的斜率>0，说明该维被推上界是**储能/波动通道**驱动的，不是风功率驱动的。
* 只读日志，不 import `*_myfunc`，不启 MATLAB。

用法
----
    python agentutil/probe_a_induction_power.py run15
    python agentutil/probe_a_induction_power.py run15 --act raw --ubins 1.0
    python agentutil/probe_a_induction_power.py run13 run15
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

BOUNDS = [(0.0, 0.5)] * 5 + [(-0.5, 0.5)]


def run_dir(name):
    if os.path.isdir(name) and os.path.exists(os.path.join(name, 'df_td3per.csv')):
        return name
    d = os.path.join('output', 'TD3PER-' + name)
    if not os.path.isdir(d):
        raise SystemExit('找不到 run: %s' % name)
    return d


def vec_col(s):
    return np.stack(s.astype(str).str.strip().str.strip('[]')
                    .str.split(expand=True).astype(float).values)


def slopes(X, y, k):
    """OLS y ~ X; 返回 (beta_k, se_k, t_k)。X 已含截距列。"""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    s2 = float(res @ res) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(s2 * xtx_inv[k, k], 0.0)))
    return float(beta[k]), se, (float(beta[k]) / se if se > 0 else float('nan'))


def within_u(name, A, u, ptur, dec, minn, dims):
    """同 u 严格配对: u 取 dec 位小数分箱, 箱内 OLS 控制其余四维 ⇒ 消掉 u 的一切影响。

    比"分箱 + 线性控制 u"强得多: 后者在同箱内若 a_j 与 u 仍相关(策略对风速有阈值
    行为时很常见) 且 P∝u³, 泄漏足以伪造出 MW 级斜率。
    """
    key = np.round(u, dec)
    order = np.argsort(key, kind='stable')
    key_s, A_s, y_s = key[order], A[order], ptur[order]
    bnd = np.r_[0, np.flatnonzero(np.diff(key_s) != 0) + 1, len(key_s)]
    slopes = {j: [] for j in dims}
    ns = {j: [] for j in dims}
    ngrp = 0
    for i in range(len(bnd) - 1):
        s = slice(bnd[i], bnd[i + 1])
        n = bnd[i + 1] - bnd[i]
        if n < minn:
            continue
        Y, AA = y_s[s], A_s[s]
        if AA[:, 2].std() < 0.01:
            continue
        ngrp += 1
        X = np.c_[np.ones(n), AA[:, dims]]
        for k, j in enumerate(dims):
            bk, _se, _t = slopes_row(X, Y, 1 + k)
            slopes[j].append(bk)
            ns[j].append(n)
    print('\n  同 u 严格配对 (u 取 %d 位小数; 每组 n>=%d 且 a2 有变化; 组内 OLS 控制其余四维):' % (dec, minn))
    print('  %-5s %8s %14s %12s %12s' % ('dim', '组数', '样本加权斜率', '中位斜率', '正斜率占比'))
    for j in dims:
        if not slopes[j]:
            print('  a%-4d %8d %14s' % (j, 0, 'n/a'))
            continue
        v = np.asarray(slopes[j])
        w = np.asarray(ns[j], dtype=float)
        print('  a%-4d %8d %+14.3f %+12.3f %11.1f%%'
              % (j, len(v), float((v * w).sum() / w.sum()), float(np.median(v)),
                 100.0 * float((v > 0).mean())))


def slopes_row(X, y, k):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    s2 = float(res @ res) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(s2 * xtx_inv[k, k], 0.0)))
    return float(beta[k]), se, (float(beta[k]) / se if se > 0 else float('nan'))


def analyse(name, act_col, ubins, minn, dims, dec=2):
    d = run_dir(name)
    cols = ['step', 'observation', 'pr', 'u', 'w_wind', 'turbine_profit',
            'r1', 'c1_step', 'bess_profit', 'penalty_NN_H', act_col]
    df = pd.read_csv(os.path.join(d, 'df_td3per.csv'), usecols=cols)
    obs = vec_col(df['observation'])
    A = vec_col(df[act_col])
    pg_same = obs[:, 3]
    pg_next = np.r_[pg_same[1:], np.nan]
    ptur = df['turbine_profit'].values * 60.0 / np.maximum(df['w_wind'].values * df['pr'].values, 1e-9)
    m = np.isfinite(ptur) & (df['u'].values >= 3.5) & (df['u'].values <= 25)
    u = df['u'].values[m]
    A, ptur = A[m], ptur[m]
    r1, c1, bp = df['r1'].values[m], df['c1_step'].values[m], df['bess_profit'].values[m]
    print('\n' + '=' * 100)
    print('=== %s | %d 行 (u∈[3.5,25]) | 动作列 = %s ===' % (name, m.sum(), act_col))
    print('   对齐自校验: corr(ptur, obs[3] 同行) = %.4f | corr(ptur, obs[3] 下一行) = %.4f'
          % (np.corrcoef(ptur, pg_same[m])[0, 1],
             np.corrcoef(ptur[:-1], pg_next[m][:-1])[0, 1]))
    print('   ptur: 均值 %.3f MW | 各维动作均值 %s'
          % (ptur.mean(), np.array2string(A.mean(axis=0), precision=3)))
    edge = np.linspace(np.floor(u.min() / ubins) * ubins, np.ceil(u.max() / ubins) * ubins,
                       max(2, int(round((u.max() - u.min()) / ubins)) + 2))
    idx = np.digitize(u, edge)
    resps = [('ptur(真实风功率)', ptur), ('r1(上层奖励)', r1), ('c1_step(波动成本)', c1),
             ('bess_profit(储能收益)', bp)]
    acc = {nm: np.zeros((len(dims), 3)) for nm, _ in resps}     # 加权 beta / se / 命中箱数
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() < minn:
            continue
        X = np.c_[np.ones(sel.sum()), A[sel][:, dims], u[sel]]
        for nm, y in resps:
            for k, j in enumerate(dims):
                bk, sek, _tk = slopes(X, y[sel], 1 + k)
                acc[nm][k] += np.array([bk * sel.sum(), sek * sel.sum(), sel.sum()])
    print('\n  偏效应 ∂y/∂a_j  (按 u 分箱 %.2f m/s, 箱内 OLS 控制其余四维 + u; 样本加权):' % ubins)
    print('  %-20s %s' % ('响应', ''.join('   a%d (t)          ' % j for j in dims)))
    for nm, _y in resps:
        cell = ''
        for k in range(len(dims)):
            wsum = acc[nm][k, 2]
            if wsum <= 0:
                cell += '   n/a            '
                continue
            bk = acc[nm][k, 0] / wsum
            sek = acc[nm][k, 1] / wsum
            cell += ' %+8.3f (%+5.1f)' % (bk, (bk / sek if sek > 0 else float('nan')))
        print('  %-20s %s' % (nm, cell))
    # 目标对比: a2 高/低两组的 ptur 与其余维是否平衡
    j = 2
    hi = A[:, j] >= 0.49
    lo = (A[:, j] >= 0.42) & (A[:, j] <= 0.46)
    print('\n  dim%d 目标对比 (同 u 分箱内配对前先看平衡性):' % j)
    print('   %-6s %6s %8s %9s   %s' % ('组', 'n', 'ptur', 'u均值', '其余四维均值 a0/a1/a3/a4'))
    for tag, sel in (('a2>=.49', hi), ('.42-.46', lo)):
        if sel.sum() == 0:
            continue
        oth = [k for k in dims if k != j]
        print('   %-6s %6d %8.3f %9.3f   %s'
              % (tag, sel.sum(), ptur[sel].mean(), u[sel].mean(),
                 '/'.join('%.3f' % A[sel][:, k].mean() for k in oth)))
    print('  读法: ptur 的斜率 < 0 而 Q 峰在 0.5 ⇒ critic 偏好与 env 物理**矛盾** (问题在奖励/拟合,')
    print('        不在罚的大小/参数化); ptur 斜率 > 0 ⇒ 该维确实越往上越有利, 角点是真的。')
    within_u(name, A, u, ptur, dec, max(8, minn // 10), dims)


def main():
    ap = argparse.ArgumentParser(description='∂(风场功率)/∂a_j 日志回归 (物理 vs critic 偏好)',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='run 名（自动补 output/TD3PER-<name>）或目录')
    ap.add_argument('--act', choices=['applied', 'raw'], default='applied',
                    help='用 applied_action_H(env 真正看到的, 默认) 还是 original_action_H')
    ap.add_argument('--ubins', type=float, default=0.5, help='u 分箱宽度 (默认 0.5 m/s)')
    ap.add_argument('--minn', type=int, default=150, help='每箱最少样本数 (默认 150)')
    ap.add_argument('--dims', nargs='*', type=int, default=list(range(5)), help='要看的维')
    ap.add_argument('--dec', type=int, default=2, help='同 u 配对的 u 小数位 (默认 2 = 0.01 m/s)')
    args = ap.parse_args()
    col = 'applied_action_H' if args.act == 'applied' else 'original_action_H'
    for nm in args.runs:
        analyse(nm, col, args.ubins, args.minn, args.dims, args.dec)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
