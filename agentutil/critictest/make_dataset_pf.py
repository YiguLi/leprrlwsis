#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""单步功率波动 `PF=|P(u_cur,a)−P(u_prev,a)|` 数据集（scalar 1D 动作）。

与五步 F5 的区别：单步 PF 在给定 `(u_prev,u_cur,a)` 时是**确定性函数**，
没有未来风速噪声；这是检验 critic 成本判别与行为梯度的干净任务。

数据
----
* 从 `FFC_speed_noise_1.csv` 同一列取相邻两步 `(u_prev,u_cur)`，
  两步都在 `[3.5,25]`；
* 按**天**划分 train/val/test（同一天不跨 split），real 用 test 天 + 结构化动作；
* 动作：1D `a0∈[0,0.5]`，广播到 5 台；`u_prev/u_cur` 用同一个 `a`。

产物 `data/aifflor_pf_scalar.npz`：
    u_prev, u_cur, a, a_eff, p_prev, p_cur, pf, split, family, day
"""
import argparse
import datetime
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENTUTIL = os.path.dirname(_HERE)
ROOT = os.path.dirname(_AGENTUTIL)
os.chdir(ROOT)
for _p in (ROOT, _AGENTUTIL, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

FAMILIES = ('level', 'cp_opt', 'farm_opt', 'low', 'high')


def _scalar_actions(family, m, rng):
    if family == 'level':
        c = rng.uniform(0.05, 0.45, size=m)
    elif family == 'cp_opt':
        c = 1.0 / 3.0 + rng.normal(0.0, 0.04, size=m)
    elif family == 'farm_opt':
        c = 0.20 + rng.normal(0.0, 0.04, size=m)
    elif family == 'low':
        c = rng.uniform(0.025, 0.20, size=m)
    elif family == 'high':
        c = rng.uniform(0.30, 0.48, size=m)
    else:
        raise ValueError(family)
    return np.clip(c, 0.02, C.A_HI).reshape(-1, 1)


def main():
    ap = argparse.ArgumentParser(description='单步 PF 数据集')
    ap.add_argument('--n_train', type=int, default=40000)
    ap.add_argument('--n_val', type=int, default=5000)
    ap.add_argument('--n_test', type=int, default=5000)
    ap.add_argument('--n_real_per_family', type=int, default=600)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='agentutil/critictest/data/aifflor_pf_scalar.npz')
    ap.add_argument('--chunk', type=int, default=4096)
    args = ap.parse_args()

    t0 = time.time()
    C.set_seeds(args.seed)
    rng = np.random.RandomState(args.seed)
    phys = C.build_physics(calibrated=False, power_scale=C.AT.LEVEL_COEF)
    df = pd.read_csv(C.WIND_CSV)
    days = df.iloc[:, 1:].values.astype(np.float64)

    up = days[:-1, :]
    uc = days[1:, :]
    ok = ((up >= C.U_MIN) & (up <= C.U_MAX) &
          (uc >= C.U_MIN) & (uc <= C.U_MAX))
    rows, cols = np.nonzero(ok)
    print('合法相邻步 %d / %d（%.1f%%）'
          % (rows.size, up.size, 100.0 * rows.size / up.size))

    n_day = days.shape[1]
    day_perm = rng.permutation(n_day)
    n_tr = int(round(0.7 * n_day))
    n_va = int(round(0.15 * n_day))
    tr_days = set(day_perm[:n_tr].tolist())
    va_days = set(day_perm[n_tr:n_tr + n_va].tolist())
    te_days = set(day_perm[n_tr + n_va:].tolist())
    print('按天划分：train %d 天 / val %d 天 / test %d 天'
          % (len(tr_days), len(va_days), len(te_days)))

    def sample(mask, n):
        idx = np.flatnonzero(mask)
        if idx.size < n:
            raise SystemExit('可用样本 %d < 需要 %d' % (idx.size, n))
        return rng.choice(idx, size=n, replace=False)

    m_tr = np.isin(cols, list(tr_days))
    m_va = np.isin(cols, list(va_days))
    m_te = np.isin(cols, list(te_days))

    splits = []
    for name, mask, n in (('train', m_tr, args.n_train),
                          ('val', m_va, args.n_val),
                          ('test', m_te, args.n_test)):
        sel = sample(mask, n)
        r_, c_ = rows[sel], cols[sel]
        u_prev = days[r_, c_]
        u_cur = days[r_ + 1, c_]
        a = rng.uniform(C.A_LO, C.A_HI, size=(n, 1))
        p_prev, a_eff = C.power_of(phys, u_prev, a, chunk=args.chunk)
        p_cur, _ = C.power_of(phys, u_cur, a_eff, chunk=args.chunk)
        pf = np.abs(p_cur - p_prev)
        splits.append(dict(name=name, code=len(splits), u_prev=u_prev,
                           u_cur=u_cur, a=a, a_eff=a_eff, p_prev=p_prev,
                           p_cur=p_cur, pf=pf,
                           day=np.full(n, -1, dtype=np.int16),
                           family=np.full(n, -1, dtype=np.int16)))
        print('  %-5s n=%6d  PF mean=%.4f sd=%.4f range=[%.3f,%.3f]  用时 %.1fs'
              % (name, n, pf.mean(), pf.std(), pf.min(), pf.max(),
                 time.time() - t0))

    # real：test 天 + 结构化动作
    per = int(args.n_real_per_family)
    sel = sample(m_te, per * len(FAMILIES))
    r_, c_ = rows[sel], cols[sel]
    u_prev_r = days[r_, c_]
    u_cur_r = days[r_ + 1, c_]
    a_list, fam_list = [], []
    for j, fam in enumerate(FAMILIES):
        a_list.append(_scalar_actions(fam, per, rng))
        fam_list.append(np.full(per, j, dtype=np.int16))
    a_r = np.vstack(a_list)
    fam_r = np.concatenate(fam_list)
    day_r = c_.astype(np.int16)
    p_prev_r, ae_r = C.power_of(phys, u_prev_r, a_r, chunk=args.chunk)
    p_cur_r, _ = C.power_of(phys, u_cur_r, ae_r, chunk=args.chunk)
    pf_r = np.abs(p_cur_r - p_prev_r)
    splits.append(dict(name='real', code=3, u_prev=u_prev_r, u_cur=u_cur_r,
                       a=a_r, a_eff=ae_r, p_prev=p_prev_r, p_cur=p_cur_r,
                       pf=pf_r, day=day_r, family=fam_r))
    print('  real  n=%6d  PF mean=%.4f sd=%.4f' % (pf_r.size, pf_r.mean(), pf_r.std()))

    UP = np.concatenate([sp['u_prev'] for sp in splits]).astype(np.float32)
    UC = np.concatenate([sp['u_cur'] for sp in splits]).astype(np.float32)
    A = np.concatenate([sp['a'] for sp in splits]).astype(np.float32)
    AE = np.concatenate([sp['a_eff'] for sp in splits]).astype(np.float32)
    PP = np.concatenate([sp['p_prev'] for sp in splits]).astype(np.float32)
    PC = np.concatenate([sp['p_cur'] for sp in splits]).astype(np.float32)
    PF = np.concatenate([sp['pf'] for sp in splits]).astype(np.float32)
    SPLIT = np.concatenate([np.full(sp['pf'].size, sp['code'], dtype=np.int8)
                            for sp in splits])
    FAM = np.concatenate([sp['family'] for sp in splits]).astype(np.int16)
    DAY = np.concatenate([sp['day'] for sp in splits]).astype(np.int16)

    for arr, name in ((UP, 'u_prev'), (UC, 'u_cur'), (A, 'a'), (AE, 'a_eff'),
                      (PP, 'p_prev'), (PC, 'p_cur'), (PF, 'pf')):
        if not np.isfinite(arr).all():
            raise SystemExit('列 %s 非有限' % name)

    per_split = {}
    for sp in splits:
        per_split[sp['name']] = dict(
            n=int(sp['pf'].size), pf_mean=float(sp['pf'].mean()),
            pf_std=float(sp['pf'].std()), pf_min=float(sp['pf'].min()),
            pf_max=float(sp['pf'].max()), p_mean=float(sp['p_cur'].mean()))
    per_family = {}
    for j, fam in enumerate(FAMILIES):
        m = FAM == j
        if np.any(m):
            per_family[fam] = dict(n=int(m.sum()), pf_mean=float(PF[m].mean()),
                                   pf_std=float(PF[m].std()))

    meta = dict(created=datetime.datetime.now().isoformat(timespec='seconds'),
                source='aifflor/FLORIS, FFC adjacent pairs, scalar action',
                seed=int(args.seed), window=1, fluct=True,
                target_def='PF=|P(u_cur,a)-P(u_prev,a)|',
                state_note='u_prev/u_cur 两步风速；a 在两步相同（受控口径）',
                split_note='按天划分，real 用 test 天+结构化动作',
                action_dim=1, scalar_action=True,
                real_families=list(FAMILIES),
                per_split=per_split, per_family=per_family,
                physics=C.physics_meta(phys))
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, u_prev=UP, u_cur=UC, a=A, a_eff=AE,
                        p_prev=PP, p_cur=PC, pf=PF, split=SPLIT,
                        family=FAM, day=DAY,
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    C.save_json(os.path.splitext(out)[0] + '.json', meta)
    print('\n保存 %s (%.2f MB)' % (out, os.path.getsize(out) / 1e6))
    print('总样本 %d | train %d / val %d / test %d / real %d'
          % (PF.size, (SPLIT == 0).sum(), (SPLIT == 1).sum(),
             (SPLIT == 2).sum(), (SPLIT == 3).sum()))
    print('PF mean %.4f sd %.4f range [%.4f, %.4f]'
          % (PF.mean(), PF.std(), PF.min(), PF.max()))
    print('总用时 %.1fs' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
