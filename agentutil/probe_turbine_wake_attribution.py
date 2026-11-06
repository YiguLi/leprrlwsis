# -*- coding: utf-8 -*-
"""探针：**逐台尾流归因** —— 为什么"中间机组"的动作效应解释不清。

动机（run16 的实测）
--------------------
actor 自己的梯度 `G_i = dQ_p/da_i − λ·dQ_c/da_i` 在五维上极不均匀：两端（dim1/dim5）
被推到 0 且梯度很大，中间（dim2/3/4）梯度小、且 critic 的符号与静态真值相反。
根因要在**物理**里找：单排 5 台、5D 间距、yaw≡0 ⇒

    P_i ∝ Cp(a_i)·u_eff,i³，  u_eff,i = u·(1 − 上游尾流亏损)
    ∂P_total/∂a_j = ∂P_j/∂a_j  (自身)
                  + Σ_{i>j} ∂P_i/∂a_j   ← 只有**下游**受影响（j 的 CT↑ ⇒ 下游更少风）

* **最后一台（T5）**：无下游 ⇒ 只有自身项 ⇒ 干净的单调通道；
* **第一台（T1）**：自身项最大（吃自由流），且影响全部 4 台下游 ⇒ 自身与下游两项**反号**；
* **中间（T2–T4）**：自身项被深尾流压小（u_eff 低），下游耦合项也被"叠加饱和"削弱
  ⇒ **两项近乎抵消** ⇒ 净灵敏度小、且符号随 (u, 动作) 变化 ⇒ 数据里难辨识。

本脚本用**真实 FLORIS 单点求值**（不推进 episode、不起环境）给出：
  [1] 尾流剖面 `u_eff,i/u`、逐台出力份额（u=10、a=1/3）
  [2] **交互矩阵** `M[i,j] = ∂P_i/∂a_j`（真实动作点）⇒ 看"自身项 vs 下游项"如何抵消
  [3] **逐维真实成本梯度** `∂c1_block/∂a_i`（真实块：5 个真实风速 + 真实动作）
      ⇒ 均值、**跨块符号翻转率**、`|·|` 中位 —— 直接回答"中间维度是不是没法解释"
  [4] 与"策略实际动作 / pooled-OLS"对照

用法
----
```powershell
python agentutil/probe_turbine_wake_attribution.py --run output/AIFFLOR-run16 --blocks 24
```

⚠ 只做 **hundreds 次单点** FLORIS 求值（一次 `eval_batch` 批量完成），秒级；
**不**推进 episode、**不**起 env、**不**训练。
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import hsrl_cost as hc                                    # noqa: E402
from aifflor import aifflor_turbine as AT                  # noqa: E402
from env import FIXED_DAY_SEQUENCE, WIND_CSV               # noqa: E402

NT = 5
LIM = 3.0


def cp(a):
    a = np.clip(np.asarray(a, float), 0.0, 0.5)
    return 4.0 * a * (1.0 - a) ** 2


def phi(pf):
    """c1 的每步口径（`hc.c1_apply`，uniform 模式 = P_F + K_V·(P_F−3)⁺）。"""
    pf = np.asarray(pf, float)
    return np.asarray([hc.c1_apply(0.0, float(v)) for v in np.ravel(pf)]).reshape(pf.shape)


def vec(x, n=4):
    t = str(x).strip().strip('[](),').replace(',', ' ')
    v = np.array([float(s) for s in t.split()], float)
    if v.size < n:
        v = np.concatenate([v, np.full(n - v.size, np.nan)])
    return v[:n]


def main():
    ap = argparse.ArgumentParser(
        description='逐台尾流归因（FLORIS 单点，不跑 episode）',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--run', default='output/AIFFLOR-run16')
    ap.add_argument('--blocks', default=24, type=int, help='抽样多少个真实块首状态')
    ap.add_argument('--seed', default=0, type=int)
    ap.add_argument('--delta', default=5e-3, type=float, help='中心差分步长')
    ap.add_argument('--wake', default='gauss', help='FLORIS 尾流模型（与生产同：gauss）')
    ap.add_argument('--out', default='', help='可选：逐维梯度 CSV')
    a = ap.parse_args()

    model, info = AT.build_model(nt=NT, calibrated=False, verbose=False,
                                 wake_model=a.wake)
    hc.set_mode('uniform')          # c1 口径与生产一致（P_F + K_V·(P_F−3)⁺）
    hc.set_smooth(0, 100.0, 0.05)
    SC = float(AT.LEVEL_COEF)
    print('=' * 84)
    print('逐台尾流归因  | run=%s  FLORIS=%s  wake=%s  LEVEL_COEF=%.6f'
          % (a.run, info.get('floris_version', '?'), a.wake, SC))
    print('=' * 84)

    # ---------------- [1] 尾流剖面 ----------------
    u0 = 10.0
    aa = np.full(NT, 1.0 / 3.0)
    farm, turb = AT.eval_batch(model, [u0], aa.reshape(1, -1))
    P = np.asarray(turb[0], float) * SC
    ue = (P / np.maximum(cp(aa), 1e-12)) ** (1.0 / 3.0)
    ue = ue / ue[0] * u0                       # T1 吃自由流 ⇒ u_eff,1 = u
    print('\n[1] 尾流剖面（u=%.0f m/s，a=1/3 均匀）' % u0)
    print('    台号   u_eff [m/s]  u_eff/u   出力 [MW]   份额')
    for i in range(NT):
        print('     T%d      %7.3f    %6.3f    %8.4f   %5.1f%%'
              % (i + 1, ue[i], ue[i] / u0, P[i], 100 * P[i] / P.sum()))
    print('     —— 单排 5D：T1 吃自由流，往下逐台衰减到 %.0f%%（合并尾流）'
          % (100 * ue[-1] / u0))

    # ---------------- 抽真实块 ----------------
    d = _load_blocks(a.run, a.blocks, a.seed)
    if not d:
        raise SystemExit('CSV 里没有可用块')
    print('\n抽样 %d 个真实块首（块内 5 步风速已按 step 还原）' % len(d))

    # ---------------- [2] 交互矩阵（在真实动作上，u 取块首那一步） ----------------
    A0 = np.mean([r['a'] for r in d], axis=0)
    A0[:NT] = np.clip(A0[:NT], 0.0, 0.5)
    A0p = np.maximum(A0[:NT], AT.A_PHYS_LO)
    h = a.delta
    acts = [A0p]
    for j in range(NT):
        for s in (+1, -1):
            v = A0p.copy()
            v[j] = min(max(v[j] + s * h, AT.A_PHYS_LO), 0.5)
            acts.append(v)
    acts = np.asarray(acts)
    us_rep = np.full(len(acts), u0)
    farm2, turb2 = AT.eval_batch(model, us_rep, acts)
    T = np.asarray(turb2, float) * SC             # (11, nt)
    M = np.zeros((NT, NT))
    for j in range(NT):
        plus = T[1 + 2 * j]
        minus = T[2 + 2 * j]
        M[:, j] = (plus - minus) / (acts[1 + 2 * j][j] - acts[2 + 2 * j][j])
    print('\n[2] 交互矩阵 M[i,j] = ∂P_i/∂a_j  [MW / 单位 a]（真实动作均值 %s）'
          % np.round(A0p, 3))
    print('        ' + ''.join('      j=%d ' % (j + 1) for j in range(NT)))
    for i in range(NT):
        print('   i=%d  ' % (i + 1) + ''.join('%9.3f ' % M[i, j] for j in range(NT)))
    dp = M.sum(axis=0)
    own = np.diag(M)
    print('\n   逐维净灵敏度 ∂P_total/∂a_j = 自身项 + 下游耦合项:')
    print('     dim  自身     下游耦合      净       |耦合|/|自身|')
    for j in range(NT):
        coup = dp[j] - own[j]
        print('      %d  %8.3f  %10.3f  %8.3f     %5.2f'
              % (j + 1, own[j], coup, dp[j], abs(coup) / max(1e-9, abs(own[j]))))
    print('    —— T5 的下游耦合恒为 0（最后一台）；中间台的耦合把自身项吃掉大半')

    # ---------------- [3] 逐维真实成本梯度 ----------------
    print('\n[3] 逐维真实 ∂c1_block/∂a_i（FLORIS + env 的 c1 口径；中心差分 δ=%g）'
          % a.delta)
    G = _true_grad(d, model, SC, a.delta)
    names = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5']
    print('     dim   |∂c1/∂a| 中位     均值    符号为正的块占比   当前动作均值')
    for j in range(NT):
        g = G[:, j]
        ok = np.isfinite(g)
        print('     %s   %12.3f  %9.3f       %5.1f%%        %5.3f'
              % (names[j], np.median(np.abs(g[ok])), np.mean(g[ok]),
                 100 * np.mean(g[ok] > 0), A0[j]))
    print('    —— "符号为正的占比"越接近 50%，说明该维的真实效应**正负参半**'
          '（依赖块内风向涨落方向）⇒ 数据里不可辨识')
    print('    —— 对照 pooled-OLS（run16 branch 数据）: '
          '[+21.2, +5.4, +7.4, -3.1, +10.5]')

    # ---------------- [3b] 符号由什么决定？两个候选拆分 ----------------
    trend = np.asarray([r['us'][-1] - r['us'][0] for r in d])
    du = np.asarray([np.diff(r['us']) for r in d])
    mono = np.all(du > 0, axis=1) | np.all(du < 0, axis=1)
    up, dn = trend > 0.15, trend < -0.15
    print('\n[3b] 符号由什么决定？（两个候选拆分，看哪个能把符号分开）')
    print('  (a) 按"块内风趋势"（actor 看不到、--critic_wind5 1 时 critic 看得到）:')
    print('     上升 %d 块 / 下降 %d 块' % (up.sum(), dn.sum()))
    print('     dim   上升 sign(+)   下降 sign(+)   ⇒ 随趋势翻转？')
    for j in range(NT):
        fu = 100 * np.mean(G[up, j] > 0) if up.any() else float('nan')
        fd = 100 * np.mean(G[dn, j] > 0) if dn.any() else float('nan')
        print('     dim%d     %5.1f%%          %5.1f%%        %s'
              % (j + 1, fu, fd,
                 ('**是**' if abs(fu - fd) > 50 else ('是(弱)' if abs(fu - fd) > 25 else '否'))))
    print('  (b) 按"块内风是否单调"（非单调 ⇒ ΔP 在块内变号 ⇒ 逐维贡献自我抵消）:')
    print('     单调 %d 块 / 非单调 %d 块' % (mono.sum(), (~mono).sum()))
    print('     dim   单调 sign(+)   非单调 sign(+)  ⇒ 单调块更稳定？')
    for j in range(NT):
        fm = 100 * np.mean(G[mono, j] > 0) if mono.any() else float('nan')
        fn = 100 * np.mean(G[~mono, j] > 0) if (~mono).any() else float('nan')
        print('     dim%d     %5.1f%%          %5.1f%%        %s'
              % (j + 1, fm, fn,
                 ('**是**' if abs(fm - 50) > abs(fn - 50) + 10 else '否')))

    if a.out:
        import pandas as pd
        pd.DataFrame(G, columns=names).to_csv(a.out, index=False)
        print('\n逐维梯度已写: %s' % a.out)


def _load_blocks(run, n_blocks, seed):
    import pandas as pd
    csv = os.path.join(run, 'df_hsrl.csv')
    d = pd.read_csv(csv, usecols=['step', 'observation', 'observation2',
                                  'original_action_H', 'P_F'])
    d = d[(d.step % 5) == 0].reset_index(drop=True)
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(d), size=min(n_blocks, len(d)), replace=False))
    day = pd.read_csv(WIND_CSV)
    nd = len(day)

    def wind_after(st):
        ep = st // 1440
        j = int(FIXED_DAY_SEQUENCE[(ep // 3) % len(FIXED_DAY_SEQUENCE)])
        cw = st % 1440
        return np.asarray([day.iloc[(cw + 1 + i) % nd, j] for i in range(4)], float)

    out = []
    for i in idx:
        r = d.iloc[i]
        obs = vec(r.observation, 4)
        act = vec(r.original_action_H, 6)
        if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(act))):
            continue
        st = int(r.step)
        us = np.concatenate([[obs[1]], wind_after(st)])
        if np.any(us < 3.6):
            continue
        out.append(dict(step=st, obs=obs, a=act, us=us, PG0=float(obs[3])))
    return out


def _true_grad(blocks, model, SC, h):
    """对每个块，用 FLORIS 算真实 ∂c1_block/∂a_i（中心差分，5 个风速一起算）。"""
    nt = NT
    G = np.full((len(blocks), nt), np.nan)
    # 一次性打包：每个块 × (1+2nt) 个动作 × 5 个风速
    rows_u, rows_a, blocks_of_row = [], [], []
    for bi, b in enumerate(blocks):
        a0 = np.minimum(np.maximum(b['a'][:nt], AT.A_PHYS_LO), 0.5)
        variants = [a0]
        for j in range(nt):
            for s in (+1, -1):
                v = a0.copy()
                v[j] = min(max(v[j] + s * h, AT.A_PHYS_LO), 0.5)
                variants.append(v)
        for k, v in enumerate(variants):
            for t in range(5):
                rows_u.append(b['us'][t])
                rows_a.append(v)
                blocks_of_row.append((bi, k))
    farm, turb = AT.eval_batch(model, np.asarray(rows_u), np.asarray(rows_a))
    P = np.asarray(farm, float) * SC                    # (nrow,)
    # 重排成 (nblock, 11, 5)
    nb, nv = len(blocks), 1 + 2 * nt
    P = P.reshape(nb, nv, 5)
    for bi, b in enumerate(blocks):
        c1 = np.zeros(nv)
        prev0 = b['PG0']
        for k in range(nv):
            prev = prev0
            s = 0.0
            for t in range(5):
                pf = abs(P[bi, k, t] - prev)
                s += float(phi(np.array([pf]))[0])
                prev = P[bi, k, t]
            c1[k] = s
        for j in range(nt):
            ip, im = 1 + 2 * j, 2 + 2 * j
            a0 = np.minimum(np.maximum(b['a'][:nt], AT.A_PHYS_LO), 0.5)
            dp = min(max(a0[j] + h, AT.A_PHYS_LO), 0.5) - a0[j]
            dm = min(max(a0[j] - h, AT.A_PHYS_LO), 0.5) - a0[j]
            G[bi, j] = (c1[ip] - c1[im]) / max(1e-12, dp - dm)
    return G


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
