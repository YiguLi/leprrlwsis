# -*- coding: utf-8 -*-
"""探针：在线 `cost` critic 的 `∂Q_c/∂a` 是否**可辨识**（计划 §8.5 / §7）。

**不推进任何仿真**：只读 `df_hsrl.csv` + checkpoint（`.pkl`），做三件事

1. 用 checkpoint 的 `critic_c / critic_c2` 求 `∂Q_c/∂a`（torch 自动微分，逐行）；
2. 用**物理解析代理**或 FLORIS 单点求值求 `∂c1/∂a`（**同一动作点、同一状态**）；
3. 报告计划 §7 的验收量：sign 一致率、Pearson、折点梯度比、K_c、动作增量 R²、覆盖。

物理 `c1` 的口径（**逐位对齐训练**）
------------------------------------
`aifflor_hsrl.cal_rew_rb_hsrl`：

```
c1 = c1_apply( C1_RISK_COEF·(10·(P_F−3)⁺ + 0.1·P_F), P_F )
P_F = |P_G(t) − P_G(t−1)|,        P_G(t) = P_tur(a) − pb(t)
```

对**当前步动作** `a` 求导时 `P_G(t−1)` 与 `pb(t)` 都是常数 ⇒

```
∂c1/∂a_i = K_PF · sign(P_G(t) − P_G(t−1)) · ∂P_tur/∂a_i
K_PF = 0.1·C1_RISK_COEF  （界内）/ (0.1+10)·C1_RISK_COEF （界外, P_F>3）
```

⇒ 本探针用中心差分算 `∂P_tur/∂a_i`，再用上式拼出 `∂c1/∂a`（**不截断折点**）。

`P_tur(a)` 的两种口径（`--physics`）
-----------------------------------
* `proxy`（**默认，纯静态**）：`P_tur = c · u³ · Σ_i 4a_i(1−a_i)²`，标量 `c` 用本 run 的
  CSV **最小二乘拟合**（`P_tur` 由 `P_G + pb_applied` 反推，`--c_fixed` 可覆盖）。
  这与 `model.py` 的 `feat_power` 同形，只保留"电平"一个自由度 ⇒ **形状**（逐台差异）
  与 FLORIS 不完全一致，**sign/Pearson 可用，折点比的绝对值要看 `--physics floris`**。
* `floris`：调 `aifflor_turbine.farm_power` 做**单点**求值（`n×5×2` 次，不起环境、
  不推进 episode）。这是**唯一**会碰 FLORIS 的开关，默认关闭。

用法
----
```powershell
# 纯静态（推荐先跑这个）
python agentutil/probe_cost_grad_hsrl_wind5.py `
    --run output/AIFFLOR-run14 --ckpt checkpoint_20000 `
    --critic_arch v6 --critic_wind5 1 --phase_feat 1 --c1_mode uniform --c1_smooth 0

# 精确物理（会做 n×5×2 次 FLORIS 单点求值；**先问用户**再跑）
python agentutil/probe_cost_grad_hsrl_wind5.py --run output/AIFFLOR-run14 --physics floris
```

输出：一张指标表 + 可选 `--out <csv>`（逐行梯度，便于离线画图）。
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import pandas as pd                                              # noqa: E402
import torch                                                     # noqa: E402

import hsrl_cost as hc                                           # noqa: E402
from aifflor import aifflor_hsrl as hsrl                         # noqa: E402
from aifflor.aifflor_myfunc import (                             # noqa: E402
    Normalize_H, C1_RISK_COEF, PHASE_DIM, phase_daily_table,
)
from model import Critic                                         # noqa: E402

NA = 6            # 动作维（5 风机 + 1 目标 g）
NT = 5            # 风机台数


# =============================================================================
# 1. 读 CSV → (obs, a, phase_minute, c1, P_F, P_G, pb, u, step)
# =============================================================================
def _vec(x, n):
    """把 CSV 里的 `'[300. 10. 0.25 1.5]'` 解析成长度 n 的向量。"""
    if isinstance(x, (list, tuple, np.ndarray)):
        v = np.asarray(x, dtype=float).reshape(-1)
    elif isinstance(x, str):
        t = x.strip().strip('[],()')
        t = t.replace(',', ' ').replace('nan', 'nan')
        v = np.array([float(s) for s in t.split() if s not in ('', 'nan')],
                     dtype=float) if t else np.zeros(0)
    else:
        v = np.asarray([x], dtype=float)
    if v.size < n:
        v = np.concatenate([v, np.full(n - v.size, np.nan)])
    return v[:n]


def load_rows(csv_path, n=128, seed=0, drop_warmup_frac=0.0):
    df = pd.read_csv(csv_path)
    need = ('observation', 'original_action_H')
    for c in need:
        if c not in df.columns:
            raise SystemExit('CSV 缺列 %r（本探针需要 df_hsrl.csv 的逐 step 记录）' % c)
    if drop_warmup_frac > 0:
        df = df.iloc[int(len(df) * drop_warmup_frac):].reset_index(drop=True)
    rng = np.random.RandomState(seed)
    k = min(int(n), len(df))
    idx = np.sort(rng.choice(len(df), size=k, replace=False))
    rows = []
    for i in idx:
        r = df.iloc[int(i)]
        obs = _vec(r['observation'], 4)
        a = _vec(r['original_action_H'], NA)
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(a)):
            continue
        pm = float(r['phase_minute']) if 'phase_minute' in df.columns else 0.0
        rows.append({
            'step': int(r['step']) if 'step' in df.columns else -1,
            'obs': obs, 'a': a, 'phase_minute': pm,
            'c1': float(r['c1']) if 'c1' in df.columns else np.nan,
            'P_F': float(r['P_F']) if 'P_F' in df.columns else np.nan,
            'P_G': float(_vec(r['observation2'], 4)[3]) if 'observation2' in df.columns else np.nan,
            'pb': float(r['pb_applied']) if 'pb_applied' in df.columns else 0.0,
            'u': float(obs[1]),
        })
    return rows, df


# =============================================================================
# 2. 物理侧：P_tur(a) 与 ∂c1/∂a
# =============================================================================
def fit_proxy_c(df):
    """用本 run 的 CSV 拟合 `P_tur ≈ c·u³·Σ 4a(1−a)²` 的标量 c（最小二乘）。

    `P_tur` 由 `P_G + pb_applied` 反推（`P_G = P_tur − pb`）—— 与 `env.step` 同式。
    返回 (c, R²)。
    """
    if not all(c in df.columns for c in ('observation2', 'original_action_H', 'u', 'pb_applied')):
        return float('nan'), float('nan')
    y, x = [], []
    for i in range(len(df)):
        r = df.iloc[i]
        a = _vec(r['original_action_H'], NA)[:NT]
        u = float(r['u']) if 'u' in df.columns else np.nan
        pg = float(_vec(r['observation2'], 4)[3])
        pb = float(r['pb_applied'])
        if not (np.isfinite(u) and np.isfinite(pg) and np.isfinite(pb)):
            continue
        f = float(np.sum(4.0 * a * (1.0 - a) ** 2)) * float(u) ** 3
        y.append(pg + pb)
        x.append(f)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 10 or float(np.sum(x * x)) <= 0:
        return float('nan'), float('nan')
    c = float(np.sum(x * y) / np.sum(x * x))
    pred = c * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return c, r2


class ProxyPhysics(object):
    """`P_tur = c·u³·Σ 4a(1−a)²`（**纯数学**，不碰 FLORIS）。"""

    name = 'proxy'

    def __init__(self, c):
        self.c = float(c)

    def power(self, u, a):
        a = np.clip(np.asarray(a, dtype=float).reshape(-1)[:NT], 0.0, 0.5)
        return float(self.c) * float(u) ** 3 * float(np.sum(4.0 * a * (1.0 - a) ** 2))

    def dpower_da(self, u, a, h=1e-3):
        """中心差分（`a` 必须在 (0, 0.5) 内部；边界行会被调用方剔除）。"""
        a = np.asarray(a, dtype=float).reshape(-1)[:NT]
        g = np.zeros(NT, dtype=float)
        for i in range(NT):
            ap = a.copy()
            am = a.copy()
            ap[i] += h
            am[i] -= h
            g[i] = (self.power(u, ap) - self.power(u, am)) / (2.0 * h)
        return g


class FlorisPhysics(object):
    """单点 FLORIS 求值（`aifflor_turbine.farm_power`）。**会调 FLORIS**，默认不用。"""

    name = 'floris'

    def __init__(self, calibrated=False):
        from aifflor import aifflor_turbine as AT
        self.AT = AT
        self.model, _ = AT.build_model(nt=NT, calibrated=bool(calibrated), verbose=False)
        self.scale = float(getattr(AT, 'LEVEL_COEF', 1.0))

    def power(self, u, a):
        a = np.clip(np.asarray(a, dtype=float).reshape(-1)[:NT], 0.0, 0.5)
        if not (3.5 <= float(u) <= 25.0):
            return 0.0
        return float(self.AT.farm_power(self.model, float(u), a)) * self.scale

    def dpower_da(self, u, a, h=1e-3):
        a = np.asarray(a, dtype=float).reshape(-1)[:NT]
        g = np.zeros(NT, dtype=float)
        for i in range(NT):
            ap = a.copy()
            am = a.copy()
            ap[i] += h
            am[i] -= h
            g[i] = (self.power(u, ap) - self.power(u, am)) / (2.0 * h)
        return g


# =============================================================================
# 3b. 折点放大比（计划 §7 的"折点梯度比 ≥10"）
# =============================================================================
def _critic_q(critics, s_vec, a6):
    """单点 Q_c（twin 均值），numpy in → float out。"""
    s_t = torch.tensor(np.asarray(s_vec, dtype=np.float32).reshape(1, -1))
    a_t = torch.tensor(np.asarray(a6, dtype=np.float32).reshape(1, -1))
    with torch.no_grad():
        return float(0.5 * (critics[0]([s_t, a_t]) + critics[1]([s_t, a_t])).reshape(-1)[0])


def fold_amplification(critics, phys, rows, S, A6, sc, band=(1.0, 5.0),
                       delta=5e-3, n_scan=121, s_lo=-0.25, s_hi=0.25):
    """把 5 维动作**整体平移** `s`（`a_i ← clip(a_i+s)`），让 `P_F` 跨过 3 MW 折点，

    然后比较折点**两侧**的斜率：
        ratio = |f(s*+δ) − f(s*)| / |f(s*) − f(s*−δ)|

    * `f = c1`（物理，`--physics` 口径）⇒ **真值**（uniform 模式下解析值 = 21）；
    * `f = Q_c·s_c`（critic）⇒ 计划要求 ≥ 10。

    `s*` 由 `P_F(a+s) = 3` 二分求出（`PF` 在 `s` 上单调：整场同步升载 ⇒ 出力单调）。
    找不到跨点（该 state 用 ±0.25 的平移也跨不过 3 MW）的行直接跳过。
    """
    res = {'phys': [], 'critic': [], 'n': 0}
    for i, r in enumerate(rows):
        pf0 = float(r['P_F']) if np.isfinite(r['P_F']) else -1.0
        if not (band[0] <= pf0 <= band[1]):
            continue
        u = float(r['obs'][1])
        a = np.clip(np.asarray(r['a'], dtype=float).reshape(-1)[:NT], 0.0, 0.5)
        P_prev = float(r['obs'][3])
        pb = float(r['pb'])
        s_vec = S[i]
        g = float(A6[i, NT]) if A6.shape[1] > NT else 0.0

        def pf_of(s):
            return abs(phys.power(u, np.clip(a + s, 0.0, 0.5)) - pb - P_prev)

        grid = np.linspace(s_lo, s_hi, int(n_scan))
        vals = np.asarray([pf_of(s) - 3.0 for s in grid])
        idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
        if idx.size == 0:
            continue
        j = int(idx[0])
        lo, hi = float(grid[j]), float(grid[j + 1])
        for _ in range(40):                       # 二分找 P_F = 3
            mid = 0.5 * (lo + hi)
            if (pf_of(mid) - 3.0) * (pf_of(lo) - 3.0) > 0:
                lo = mid
            else:
                hi = mid
        s_star = 0.5 * (lo + hi)

        def c1_of(s):
            return c1_scalar(pf_of(s))

        def q_of(s):
            a6 = np.concatenate([np.clip(a + s, 0.0, 0.5), [g]])
            return _critic_q(critics, s_vec, a6) * float(sc)

        # ⚠ 方向：`P_tur(a)` 不是 `a` 的单调函数（`Cp` 在 `a=1/3` 取峰），
        #   所以"升载侧"不一定是 `s+δ`。用 P_F 的大小判方向，别默认符号。
        pf_up, pf_dn = pf_of(s_star + delta), pf_of(s_star - delta)
        if pf_up >= pf_dn:
            s_above, s_below = s_star + delta, s_star - delta
        else:
            s_above, s_below = s_star - delta, s_star + delta
        # 两侧都不能撞到 [0,0.5] 的 clip（撞了就不是"斜率"了）
        for _s in (s_above, s_below):
            if np.any((a + _s) <= 0.0) or np.any((a + _s) >= 0.5):
                break
        else:
            d_up_p = abs(c1_of(s_above) - c1_of(s_star))
            d_dn_p = abs(c1_of(s_star) - c1_of(s_below))
            d_up_q = abs(q_of(s_above) - q_of(s_star))
            d_dn_q = abs(q_of(s_star) - q_of(s_below))
            if d_dn_p > 1e-12 and d_dn_q > 1e-12:
                res['phys'].append(d_up_p / d_dn_p)
                res['critic'].append(d_up_q / d_dn_q)
                res['n'] += 1
    return res


def c1_scalar(P_F):
    """与 `cal_rew_rb_hsrl` **逐位同式**的 c1（`hc.c1_apply` 会按模式改写）。"""
    legacy = (C1_RISK_COEF * (10.0 * (P_F - 3.0) + 0.1 * P_F) if P_F > 3.0 + 1e-4
              else C1_RISK_COEF * (0.1 * P_F))
    return float(hc.c1_apply(legacy, P_F))


def dc1_da(phys, row, h_pf=1e-3):
    """`∂c1/∂a`（5 维）—— **数值微分 c1 本身**（对折点/平滑都自动正确）。

    `c1` 只通过 `P_F = |P_G(t) − P_G(t−1)|` 依赖动作，且 `P_G(t) = P_tur(a) − pb`：

        ∂c1/∂a_i = [∂c1/∂P_G(t)] · ∂P_tur/∂a_i
        ∂c1/∂P_G(t) ≈ [c1(|P_now+h−P_prev|) − c1(|P_now−h−P_prev|)] / (2h)

    ⇒ 在 `P_F ≈ 3` 的折点上取到的是**左/右导数的平均**（对 sign 一致性无害，
       对"折点梯度比"是有意的保守取值）。
    """
    u = float(row['obs'][1])
    a = np.clip(np.asarray(row['a'], dtype=float).reshape(-1)[:NT], 0.0, 0.5)
    P_prev = float(row['obs'][3])           # P_G(t−1)
    pb = float(row['pb'])
    P_now = phys.power(u, a) - pb           # P_G(t)
    dP = phys.dpower_da(u, a)
    dc1_dP = (c1_scalar(abs(P_now + h_pf - P_prev))
              - c1_scalar(abs(P_now - h_pf - P_prev))) / (2.0 * h_pf)
    return dc1_dP * dP, abs(P_now - P_prev), dP


# =============================================================================
# 3. critic 侧：∂Q_c/∂a
# =============================================================================
def build_state_vector(row, critic_wind5=False, phase_feat=0, phase_table=None,
                       wind5=None):
    """`[Normalize_H(obs, phase)] (+ [u2..u5]/10)` —— 与训练时 memory 里存的口径一致。"""
    s = Normalize_H(np.asarray(row['obs'], dtype=float), float(row['phase_minute']),
                    int(phase_feat), phase_table)
    s = np.asarray(s, dtype=float).reshape(-1)
    if critic_wind5:
        u1 = float(row['obs'][1])
        ups = (np.full(4, u1) if wind5 is None
               else np.asarray(wind5, dtype=float).reshape(-1)[:4])
        if ups.size < 4:
            ups = np.full(4, u1)
        s = np.concatenate([s, ups / 10.0])
    return s


def load_critics(run_dir, ckpt, nb_actions=NA, critic_nb_states=4,
                 arch='v6', act='relu', out_init='3e3', feat_power=False,
                 auto_ns=True):
    """加载 twin cost critics。

    `auto_ns=True`（默认）时**自动推断** critic state 的维数（= 训练时
    `4 + PHASE_DIM[phase_feat] + 4·wind5`）：对候选维数逐个 `load_state_dict`，
    成功即采用 ⇒ 用户不必记住当时开了哪些开关。返回值第三项是推断出的维数。
    """
    d = os.path.join(run_dir, ckpt) if ckpt else run_dir
    cfg = dict(arch=arch, act=act, out_init=out_init, feat_power=bool(feat_power))
    names = ['critic_c_%d.pkl' % nb_actions, 'critic_c2_%d.pkl' % nb_actions]
    paths = [os.path.join(d, n) for n in names]
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit('找不到 %s（--run/--ckpt 指向训练输出目录）' % p)
    states = []
    for p in paths:
        try:
            states.append(torch.load(p, map_location='cpu', weights_only=True))
        except TypeError:                       # torch < 1.13
            states.append(torch.load(p, map_location='cpu'))

    cands = [int(critic_nb_states)]
    if auto_ns:
        cands = [4 + PHASE_DIM[p] + 4 * w for p in (0, 1, 2) for w in (0, 1)]
        cands = sorted(set(cands + [int(critic_nb_states)]))
    nets = None
    for ns in cands:
        try:
            trial = []
            for sd in states:
                net = Critic(int(ns), int(nb_actions), **cfg)
                net.load_state_dict(sd)
                net.eval()
                trial.append(net)
            nets = trial
            break
        except RuntimeError:
            continue
    if nets is None:
        raise SystemExit('critic 权重与任何候选 state 维数都不匹配（arch=%s）；'
                         '请显式给 --critic_wind5/--phase_feat' % arch)
    return nets, int(ns)


def spec_from_ns(ns):
    """`4 + PHASE_DIM[p] + 4·w` 反解 (phase_feat, wind5)。"""
    for p in (0, 1, 2):
        for w in (0, 1):
            if 4 + PHASE_DIM[p] + 4 * w == int(ns):
                return p, w
    return None, None


def dq_da(critics, S, A6):
    """`∂Q_c/∂a`（twin 均值），逐行。返回 (n,5) —— 只对动作前 5 维求导。

    `A6` 是**完整 6 维**动作（第 6 维 = 块总放电量 g）；g 作为常数输入（不求导），
    前 5 维才是分支方案要辨识的方向。
    """
    s_t = torch.tensor(np.asarray(S, dtype=np.float32))
    a6 = np.asarray(A6, dtype=np.float32)
    a5 = torch.tensor(a6[:, :NT], requires_grad=True)
    a_full = torch.cat([a5, torch.tensor(a6[:, NT:NT + 1])], dim=1)
    with torch.enable_grad():
        q = 0.5 * (critics[0]([s_t, a_full]) + critics[1]([s_t, a_full]))
        g = torch.autograd.grad(q.sum(), a5)[0].detach().numpy()
        qv = q.detach().numpy().reshape(-1)
    return g, qv


# =============================================================================
# 4. 动作增量 R²（state-only vs state+action）
# =============================================================================
def ridge_r2(X, y, lam=1e-3):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < max(20, X.shape[1] + 5):
        return float('nan'), int(len(y))
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    w = np.linalg.solve(A, Xb.T @ y)
    pred = Xb @ w
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return (1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')), int(len(y))


def action_increment(df, c1_mode='uniform', smooth=False):
    """计划 §7 的"动作增量 R²"：`state+action` 相对 `state-only` 的 R² 增量。"""
    hc.set_mode(str(c1_mode))
    hc.set_smooth(int(smooth), 100.0, 0.05)
    st, act, y = [], [], []
    for i in range(len(df)):
        r = df.iloc[i]
        obs = _vec(r['observation'], 4)
        a = _vec(r['original_action_H'], NA)
        if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(a))):
            continue
        pm = float(r['phase_minute']) if 'phase_minute' in df.columns else 0.0
        base = Normalize_H(obs, pm, 0, None) if 'phase_minute' in df.columns \
            else Normalize_H(obs, 0.0, 0, None)
        u = float(obs[1])
        feats = np.concatenate([base, [u ** 2, u ** 3, obs[3]]])
        st.append(feats)
        act.append(np.asarray(a[:NT], dtype=float))
        y.append(float(r['c1']))
    st = np.asarray(st)
    act = np.asarray(act)
    y = np.asarray(y)
    r2_s, n1 = ridge_r2(st, y)
    r2_f, n2 = ridge_r2(np.concatenate([st, act], axis=1), y)
    return dict(r2_state=r2_s, r2_full=r2_f, delta_r2=(r2_f - r2_s), n=min(n1, n2))


# =============================================================================
# 5. 主流程
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description='aifflor 在线 cost critic 的 ∂Q_c/∂a 可辨识性探针（纯静态，不跑仿真）',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--run', default='', help='训练输出目录（含 df_hsrl.csv 与 .pkl）')
    ap.add_argument('--csv', default='', help='默认 <run>/df_hsrl.csv')
    ap.add_argument('--ckpt', default='', help='子目录（如 checkpoint_20000）；空 = run 根')
    ap.add_argument('--n', default=128, type=int, help='抽多少行做梯度对比')
    ap.add_argument('--seed', default=0, type=int)
    ap.add_argument('--drop_warmup_frac', default=0.0, type=float,
                    help='丢掉 CSV 前这一比例的行（跳过 warmup 段）')
    ap.add_argument('--critic_arch', default='v6',
                    choices=['v1', 'v1af', 'v2', 'v3', 'v4', 'v5', 'v6'])
    ap.add_argument('--critic_act', default='relu', choices=['relu', 'leaky'])
    ap.add_argument('--critic_out_init', default='3e3', choices=['3e3', 'fanin'])
    ap.add_argument('--critic_feat_power', default=0, type=int)
    ap.add_argument('--critic_wind5', default=0, type=int,
                    help='1 = critic state 多 4 维 (u2..u5)/10')
    ap.add_argument('--phase_feat', default=0, type=int, choices=[0, 1, 2])
    ap.add_argument('--c1_mode', default='uniform', choices=['uniform', 'legacy'])
    ap.add_argument('--c1_smooth', default=0, type=int)
    ap.add_argument('--cost_scale', default=1.0, type=float,
                    help='训练时的 --cost_scale（cost 塔把 c1 除以它）。默认 1.0。'
                         '⚠ 只有 K_c / 幅度比依赖它；sign / Pearson / 折点放大比不受影响')
    ap.add_argument('--physics', default='proxy', choices=['proxy', 'floris'],
                    help='proxy = 纯数学（默认）；floris = 单点 FLORIS 求值（会调 FLORIS）')
    ap.add_argument('--c_fixed', default=None, type=float,
                    help='proxy 的电平系数（默认从本 run 的 CSV 拟合）')
    ap.add_argument('--wind5_mode', default='rep', choices=['rep', 'day'],
                    help='--critic_wind5 1 时 u2..u5 的来源：rep = 全部取 u1（近似）；'
                         'day = 从 FFC_speed_noise_1.csv 按 step 还原**真实**未来 4 步')
    ap.add_argument('--day_steps', default=1440, type=int)
    ap.add_argument('--day_repeat', default=3, type=int)
    ap.add_argument('--out', default='', help='可选：逐行梯度 CSV 输出路径')
    args = ap.parse_args()

    if not args.run and not args.csv:
        ap.error('至少给 --run 或 --csv')
    csv_path = args.csv or os.path.join(args.run, 'df_hsrl.csv')
    if not os.path.isfile(csv_path):
        raise SystemExit('CSV 不存在: %s' % csv_path)

    hc.set_mode(str(args.c1_mode))
    hc.set_smooth(int(args.c1_smooth), 100.0, 0.05)
    sc = float(args.cost_scale)

    print('=' * 78)
    print('aifflor cost-critic 梯度探针（静态）')
    print('  CSV   : %s' % csv_path)
    print('  run   : %s | ckpt=%s' % (args.run or '(仅 CSV)', args.ckpt or '(根)'))
    print('  critic: arch=%s act=%s out_init=%s wind5=%d feat_power=%d | phase=%d'
          % (args.critic_arch, args.critic_act, args.critic_out_init,
             int(args.critic_wind5), int(args.critic_feat_power), int(args.phase_feat)))
    print('  c1    : mode=%s smooth=%d | cost_scale s_c=%.4g' % (args.c1_mode, args.c1_smooth, sc))
    print('=' * 78)

    rows, df = load_rows(csv_path, n=args.n, seed=args.seed,
                         drop_warmup_frac=args.drop_warmup_frac)
    if not rows:
        raise SystemExit('CSV 里没有可用行')

    # ---- 物理侧 ----
    if args.physics == 'floris':
        print('[!] --physics floris：将做 %d 次 FLORIS **单点**求值（不起环境/不推进 episode）'
              % (len(rows) * NT * 2))
        phys = FlorisPhysics(calibrated=False)
        c_proxy, r2_proxy = float('nan'), float('nan')
    else:
        if args.c_fixed is not None:
            c_proxy, r2_proxy = float(args.c_fixed), float('nan')
        else:
            c_proxy, r2_proxy = fit_proxy_c(df)
        phys = ProxyPhysics(c_proxy)
        print('[proxy] c = %.6g  拟合 R² = %.5f   （P_tur = c·u³·Σ4a(1−a)²）'
              % (c_proxy, r2_proxy))

    # ---- critic state ----
    phase_table = phase_daily_table() if int(args.phase_feat) >= 2 else None
    wind_day_df = None
    if int(args.critic_wind5) and args.wind5_mode == 'day':
        try:
            from env import FIXED_DAY_SEQUENCE, WIND_CSV
            wind_day_df = pd.read_csv(WIND_CSV)
            print('[wind5] day 模式：%s，%d 天' % (WIND_CSV, len(FIXED_DAY_SEQUENCE)))
        except Exception as e:                                   # pragma: no cover
            print('[wind5] day 模式不可用 (%s) ⇒ 退回 rep' % e)
            wind_day_df = None

    # ---- critic：先加载 ⇒ 顺便**自动推断** critic state 的口径 ----
    critics = None
    if args.run or args.ckpt:
        try:
            critics, critic_dim = load_critics(
                args.run or os.path.dirname(csv_path), args.ckpt,
                critic_nb_states=4 + PHASE_DIM[int(args.phase_feat)]
                + 4 * int(bool(args.critic_wind5)),
                arch=args.critic_arch, act=args.critic_act,
                out_init=args.critic_out_init,
                feat_power=bool(args.critic_feat_power))
            p_auto, w_auto = spec_from_ns(critic_dim)
            if p_auto is not None and (p_auto != int(args.phase_feat)
                                       or w_auto != int(args.critic_wind5)):
                print('[auto] checkpoint 的 critic state = %d 维 ⇒ phase_feat=%d, '
                      'critic_wind5=%d（覆盖命令行默认值）' % (critic_dim, p_auto, w_auto))
                args.phase_feat, args.critic_wind5 = int(p_auto), int(w_auto)
        except SystemExit as e:
            print('[!] %s' % e)
            print('    ⇒ 只做 CSV 侧诊断（动作增量 R² / 覆盖），跳过梯度对比')
            critics = None

    def wind5_for(row):
        if not int(args.critic_wind5):
            return None
        if wind_day_df is None:
            return None
        from env import FIXED_DAY_SEQUENCE
        st = int(row['step'])
        if st < 0:
            return None
        ep = st // max(1, int(args.day_steps))
        idx = (ep // max(1, int(args.day_repeat))) % max(1, len(FIXED_DAY_SEQUENCE))
        j = int(FIXED_DAY_SEQUENCE[idx])
        cw = st % max(1, int(args.day_steps))
        n = len(wind_day_df)
        return np.asarray([wind_day_df.iloc[(cw + 1 + i) % n, j] for i in range(4)],
                          dtype=float)

    # phase_table 依赖最终的 phase_feat（可能被自动推断改过）
    if int(args.phase_feat) >= 2 and phase_table is None:
        phase_table = phase_daily_table()
    S = np.stack([build_state_vector(r, bool(args.critic_wind5), int(args.phase_feat),
                                     phase_table, wind5_for(r)) for r in rows])
    A6 = np.stack([np.asarray(r['a'], dtype=float).reshape(-1)[:NA] for r in rows])
    A6[:, :NT] = np.clip(A6[:, :NT], 0.0, 0.5)
    A = A6[:, :NT].copy()
    print('[state] 维度 = %d（critic_nb_states）| 行数 = %d' % (S.shape[1], S.shape[0]))
    if critics is not None and S.shape[1] != critic_dim:
        print('[!] 构造的 state 维数 %d ≠ checkpoint 的 %d ⇒ 跳过梯度对比'
              % (S.shape[1], critic_dim))
        critics = None

    # ---- 逐行物理梯度 ----
    # 只在动作**远离边界**的行上做差分（否则会撞到 [0,0.5] 的 clip，梯度失真）
    h = 1e-3
    keep = np.all((A > h) & (A < 0.5 - h), axis=1)
    n_drop = int((~keep).sum())
    if n_drop:
        print('[fd] 剔除 %d 行（动作贴边，差分会被 clip 截断）' % n_drop)
    rows_k = [r for r, k in zip(rows, keep) if k]
    A_k = A[keep]
    S_k = S[keep]
    A6_k = A6[keep]

    if critics is None or len(rows_k) == 0:
        res = action_increment(df, c1_mode=args.c1_mode, smooth=bool(args.c1_smooth))
        print('\n[动作增量 R²]  state-only R² = %.4f | state+action R² = %.4f | Δ = %.4f (n=%d)'
              % (res['r2_state'], res['r2_full'], res['delta_r2'], res['n']))
        return 0

    gq, qv = dq_da(critics, S_k, A6_k)
    gq = gq * float(sc)                     # 口径还原：critic 吃的是 c1/s_c

    gc = np.zeros_like(gq)
    pf = np.zeros(len(rows_k))
    for i, r in enumerate(rows_k):
        g, v, _ = dc1_da(phys, r)
        gc[i] = g
        pf[i] = v

    # ---- 指标 ----
    thr = 1e-8
    m = np.abs(gc) > thr
    sign_agree = float(np.mean(np.sign(gq[m]) == np.sign(gc[m]))) if m.any() else float('nan')
    # Pearson（按行算再取中位 + 全展平两个口径）
    pears = []
    for i in range(len(gq)):
        if np.std(gc[i]) > 1e-12 and np.std(gq[i]) > 1e-12:
            pears.append(float(np.corrcoef(gq[i], gc[i])[0, 1]))
    pears = np.asarray(pears) if pears else np.zeros(0)
    flat = float(np.corrcoef(gq.reshape(-1), gc.reshape(-1))[0, 1]) \
        if np.std(gc) > 1e-12 and np.std(gq) > 1e-12 else float('nan')
    # K_c：斜率（Q_c 相对真值的倍数）
    denom = float(np.sum(gc * gc))
    K_c = float(np.sum(gq * gc) / denom) if denom > 0 else float('nan')
    # 折点比：P_F ∈ [2,4] 的行
    fold = (pf >= 2.0) & (pf <= 4.0)
    ratio_fold = (float(np.median(np.abs(gq[fold]))) / max(1e-12, float(np.median(np.abs(gc[fold]))))
                  if fold.any() else float('nan'))
    ratio_all = (float(np.median(np.abs(gq))) / max(1e-12, float(np.median(np.abs(gc)))))

    print('\n' + '-' * 78)
    print('  ∂Q_c/∂a vs 物理 ∂c1/∂a   (n=%d, 物理=%s, s_c=%.4g)' % (len(gq), phys.name, sc))
    print('-' * 78)
    print('  sign 一致率           : %6.2f %%      （计划目标 ≥ 65%%）' % (100 * sign_agree))
    print('  Pearson（逐行中位）   : %+8.3f      （>0.8 才算"方向对"）'
          % (float(np.median(pears)) if pears.size else float('nan')))
    print('  Pearson（全展平）     : %+8.3f' % flat)
    print('  K_c（Q 斜率/真值）    : %+9.3f' % K_c)
    print('  幅度比 中位（全行）   : %9.3f      （|∂Q/∂a| ÷ |∂c1/∂a| 的绝对倍数）'
          % ratio_all)
    print('  幅度比 中位（折点带） : %9.3f      （%d 行 P_F∈[2,4]）'
          % (ratio_fold, int(fold.sum())))
    print('  Q_c 均值              : %9.4f      （量级应与平稳参考同阶）' % float(np.mean(qv)))
    print('  |∂Q/∂a| 中位          : %9.3e' % float(np.median(np.abs(gq))))
    print('  |∂c1/∂a| 中位         : %9.3e' % float(np.median(np.abs(gc))))

    # ---- 折点放大比（计划 §7 的"折点梯度比"本义）----
    amp = fold_amplification(critics, phys, rows_k, S_k, A6_k, sc)
    amp_p = float(np.median(amp['phys'])) if amp['phys'] else float('nan')
    amp_q = float(np.median(amp['critic'])) if amp['critic'] else float('nan')
    print('\n  折点放大比（物理/真值）: %8.2f   （%d 个跨点；uniform 模式解析值 = 21）'
          % (amp_p, amp['n']))
    print('  折点放大比（critic）   : %8.2f   ← **计划目标 ≥ 10**' % amp_q)

    # ---- 覆盖 / 增量 R²（计划 §7）----
    A_all = np.stack([np.asarray(r['a'], dtype=float)[:NT] for r in rows])
    cov_lo = float(np.mean(A_all < 0.05))
    cov_hi = float(np.mean(A_all > 0.45))
    pf_all = np.asarray([r['P_F'] for r in rows], dtype=float)
    cov_fold = float(np.mean((pf_all >= 2.0) & (pf_all <= 4.0)))
    print('\n  覆盖（本抽样）: a<0.05 %.1f%% | a>0.45 %.1f%% | P_F∈[2,4] %.1f%%'
          % (100 * cov_lo, 100 * cov_hi, 100 * cov_fold))
    res = action_increment(df, c1_mode=args.c1_mode, smooth=bool(args.c1_smooth))
    print('  动作增量 R²   : state-only %.4f → state+action %.4f | Δ = %+.4f (n=%d)'
          % (res['r2_state'], res['r2_full'], res['delta_r2'], res['n']))

    # ---- 结论行（照计划 §7 的口径自动判定）----
    ok_sign = (sign_agree >= 0.65) if np.isfinite(sign_agree) else False
    ok_fold = (amp_q >= 10.0) if np.isfinite(amp_q) else False
    ok_r2 = (res['delta_r2'] >= 0.05) if np.isfinite(res['delta_r2']) else False
    print('\n' + '-' * 78)
    print('  判定: sign %s | 折点放大比 %s | 动作增量 R² %s'
          % ('PASS' if ok_sign else 'FAIL', 'PASS' if ok_fold else 'FAIL',
             'PASS' if ok_r2 else 'FAIL'))
    print('-' * 78)

    if args.out:
        out = pd.DataFrame({
            'step': [r['step'] for r in rows_k],
            'P_F': pf,
            'Q_c': qv,
            **{'dq_da%d' % (i + 1): gq[:, i] for i in range(NT)},
            **{'dc1_da%d' % (i + 1): gc[:, i] for i in range(NT)},
        })
        out.to_csv(args.out, index=False)
        print('逐行梯度已写: %s' % args.out)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
