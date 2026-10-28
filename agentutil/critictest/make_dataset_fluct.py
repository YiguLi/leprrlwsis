#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""生成**五步功率波动和**数据集：`(u1, a) → F5 = Σ_{t=1..5} |P_t − P_{t-1}|`。

口径（与 env 块级 `P_F` 一致，去掉 c1 折点）
--------------------------------------------
* 风速窗口：从 `FFC_speed_noise_1.csv` 同一列取连续 5 行 `u1..u5`；
  另外取窗口**前一行** `u0`（只用于标签，不喂网络）；
  `u0..u5` 全部在 `[3.5,25]`。
* 动作：`a∈[0,0.5]^5`，`u0..u5` 上**同一个 a**（受控口径）；窗口内 5 步不变。
* 标签：
      F5 = |P(u1,a) − P(u0,a)| + Σ_{t=2..5} |P(u_t,a) − P(u_{t-1},a)|
  同时保存 `F4 = Σ_{t=2..5}|…|`（窗口内 4 项，不含 u0）作对照。
* 网络输入：只有 `u1`（`u1/10`）；`u0` 与 `u2..u5` 都不给网络。
* 主目标（推荐）：`f5n = F5 / E[F5|u1]`，预测时乘回；另提供 `f5`（F5/10）、
  `f5u3`（1000·F5/(5u1³)）、`logf5`。

产物
----
`data/aifflor_ua_f5.npz`：
    u (N,), u_seq (N,5), u0 (N,), a (N,5), a_eff (N,5)
    p (N,) = F5；f5 (N,) 同；f4 (N,)；p5 (N,) 五步功率和
    p_steps (N,5)；p_prev (N,) = P(u0,a)
    split (N,) 0/1/2/3；family (N,) -1/0..4；meta
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

import common as C  # noqa: E402


SCALAR_FAMILIES = ('level', 'cp_opt', 'farm_opt', 'low', 'high')


def _scalar_actions(family, m, rng):
    """单动作（五台同因子）的真实工况族；返回 (m,1)。"""
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
        raise ValueError('未知标量动作族 %r' % (family,))
    return np.clip(c, 0.02, C.A_HI).reshape(-1, 1)


def _stats(name, u, a, f5, f4, p5):
    u1 = u
    return dict(
        name=name, n=int(u1.size),
        u=dict(min=float(u1.min()), max=float(u1.max()), mean=float(u1.mean())),
        a=dict(min=float(a.min()), max=float(a.max()), mean=float(a.mean())),
        f5=dict(min=float(f5.min()), max=float(f5.max()), mean=float(f5.mean()),
                std=float(f5.std())),
        f4=dict(min=float(f4.min()), max=float(f4.max()), mean=float(f4.mean()),
                std=float(f4.std())),
        p5=dict(mean=float(p5.mean()), std=float(p5.std())),
    )


def main():
    ap = argparse.ArgumentParser(description='用 aifflor 生成五步功率波动和数据集')
    ap.add_argument('--n_train', type=int, default=40000)
    ap.add_argument('--n_val', type=int, default=5000)
    ap.add_argument('--n_test', type=int, default=5000)
    ap.add_argument('--n_real_per_family', type=int, default=600)
    ap.add_argument('--window', type=int, default=C.WINDOW)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None,
                    help='默认：5 维动作用 aifflor_ua_f5.npz，'
                         '标量动作用 aifflor_ua_f5_scalar.npz')
    ap.add_argument('--scalar_action', action='store_true',
                    help='一个动作控制五台（a_i=a0，动作输入 1 维）')
    ap.add_argument('--calibrated', action='store_true')
    ap.add_argument('--power_scale', type=float, default=None)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--wind_csv', default=None)
    args = ap.parse_args()

    window = int(args.window)
    if window != C.WINDOW:
        raise SystemExit('本脚本固定 window=%d' % C.WINDOW)
    scalar = bool(args.scalar_action)
    action_dim = 1 if scalar else C.NT
    if args.out is None:
        args.out = ('agentutil/critictest/data/aifflor_ua_f5_scalar.npz'
                    if scalar else 'agentutil/critictest/data/aifflor_ua_f5.npz')
    t0 = time.time()
    C.set_seeds(args.seed)
    rng = np.random.RandomState(args.seed)

    print('=' * 100)
    print('critictest 五步功率波动和数据集：aifflor / FLORIS   window=%d  %s  seed=%d'
          % (window, 'scalar action (1D)' if scalar else '5D action', args.seed))
    print('=' * 100)
    phys = C.build_physics(calibrated=args.calibrated,
                           power_scale=args.power_scale)
    pmeta = C.physics_meta(phys)
    print('物理口径: calibrated=%s | wake=%s | power_scale=%.6f | FLORIS %s'
          % (pmeta['calibrated'], pmeta['wake_model'], pmeta['power_scale'],
             pmeta['floris_version']))
    anchor = float(phys.power(10.0, np.full(C.NT, C.AT.A_IND)))
    print('锚点 u=10, a=1/3 → %.6f MW（单步）' % anchor)

    days, starts, cols = C.wind_window_pool(window=window, path=args.wind_csv)
    # 需要前一行 u0；且 u0 也要在 env 守卫区间
    keep = starts >= 1
    u0_pool = days[np.maximum(starts - 1, 0), cols]
    keep &= (u0_pool >= C.U_MIN) & (u0_pool <= C.U_MAX)
    starts, cols = starts[keep], cols[keep]
    print('合法窗口（含 u0）%d；来自 %d 天 × %d 步'
          % (starts.size, days.shape[1], days.shape[0]))

    families = SCALAR_FAMILIES if scalar else C.REAL_FAMILIES
    n_real = int(args.n_real_per_family) * len(families)
    need = int(args.n_train) + int(args.n_val) + int(args.n_test) + n_real
    if need > starts.size:
        raise SystemExit('需要 %d 个窗口，但只有 %d 个合法窗口' % (need, starts.size))
    perm = rng.permutation(starts.size)
    pos = 0
    idx_tr = perm[pos:pos + args.n_train]; pos += args.n_train
    idx_va = perm[pos:pos + args.n_val]; pos += args.n_val
    idx_te = perm[pos:pos + args.n_test]; pos += args.n_test
    idx_re = perm[pos:pos + n_real]

    splits = []
    for name, idx in (('train', idx_tr), ('val', idx_va), ('test', idx_te)):
        u_seq = C.windows_from_pool(days, starts[idx], cols[idx], window=window)
        u0 = days[starts[idx] - 1, cols[idx]]
        a = rng.uniform(C.A_LO, C.A_HI, size=(u_seq.shape[0], action_dim))
        t = time.time()
        f5, f4, p_steps, p_prev, a_eff = C.power_fluct(
            phys, u0, u_seq, a, chunk=args.chunk, verbose=True, window=window)
        p5 = p_steps.sum(1)
        splits.append(dict(name=name, code=len(splits), u_seq=u_seq, u0=u0,
                           a=a, a_eff=a_eff, p_steps=p_steps, p_prev=p_prev,
                           f5=f5, f4=f4, p5=p5,
                           family=np.full(u_seq.shape[0], -1, dtype=np.int16)))
        print('  %-5s n=%6d  用时 %5.1fs  F5=[%.3f, %.3f]  mean=%.4f sd=%.4f  '
              'F4 mean=%.4f  F5/F4=%.3f'
              % (name, u_seq.shape[0], time.time() - t, f5.min(), f5.max(),
                 f5.mean(), f5.std(), f4.mean(), float(np.mean(f5 / np.maximum(f4, 1e-6)))))

    print('\n[real] 5 种动作族 ...')
    u_seq_r = C.windows_from_pool(days, starts[idx_re], cols[idx_re],
                                  window=window)
    u0_r = days[starts[idx_re] - 1, cols[idx_re]]
    a_list, fam_list = [], []
    per = int(args.n_real_per_family)
    for j, fam in enumerate(families):
        a_list.append(_scalar_actions(fam, per, rng) if scalar
                      else C._family_actions(fam, per, rng))
        fam_list.append(np.full(per, j, dtype=np.int16))
    a_r = np.vstack(a_list)
    fam_r = np.concatenate(fam_list)
    t = time.time()
    f5_r, f4_r, p_steps_r, p_prev_r, ae_r = C.power_fluct(
        phys, u0_r, u_seq_r, a_r, chunk=args.chunk, verbose=True, window=window)
    p5_r = p_steps_r.sum(1)
    splits.append(dict(name='real', code=3, u_seq=u_seq_r, u0=u0_r, a=a_r,
                       a_eff=ae_r, p_steps=p_steps_r, p_prev=p_prev_r,
                       f5=f5_r, f4=f4_r, p5=p5_r, family=fam_r))
    print('  real  n=%6d  用时 %5.1fs  F5=[%.3f, %.3f]  mean=%.4f sd=%.4f'
          % (u_seq_r.shape[0], time.time() - t, f5_r.min(), f5_r.max(),
             f5_r.mean(), f5_r.std()))

    U1 = np.concatenate([sp['u_seq'][:, 0] for sp in splits]).astype(np.float32)
    USEQ = np.concatenate([sp['u_seq'] for sp in splits]).astype(np.float32)
    U0 = np.concatenate([sp['u0'] for sp in splits]).astype(np.float32)
    A = np.concatenate([sp['a'] for sp in splits]).astype(np.float32)
    AE = np.concatenate([sp['a_eff'] for sp in splits]).astype(np.float32)
    F5 = np.concatenate([sp['f5'] for sp in splits]).astype(np.float32)
    F4 = np.concatenate([sp['f4'] for sp in splits]).astype(np.float32)
    P5 = np.concatenate([sp['p5'] for sp in splits]).astype(np.float32)
    PS = np.concatenate([sp['p_steps'] for sp in splits]).astype(np.float32)
    PP = np.concatenate([sp['p_prev'] for sp in splits]).astype(np.float32)
    SPLIT = np.concatenate([np.full(sp['u_seq'].shape[0], sp['code'],
                                    dtype=np.int8) for sp in splits])
    FAM = np.concatenate([sp['family'] for sp in splits]).astype(np.int16)

    for arr, name in ((U1, 'u'), (USEQ, 'u_seq'), (U0, 'u0'), (A, 'a'),
                      (AE, 'a_eff'), (F5, 'f5'), (F4, 'f4'), (P5, 'p5'),
                      (PS, 'p_steps'), (PP, 'p_prev')):
        if not np.isfinite(arr).all():
            raise SystemExit('数据集列 %s 出现非有限值，拒绝保存' % name)

    per_split = {}
    for sp in splits:
        per_split[sp['name']] = _stats(sp['name'], sp['u_seq'][:, 0], sp['a'],
                                       sp['f5'], sp['f4'], sp['p5'])
    per_family = {}
    for j, fam in enumerate(families):
        m = FAM == j
        if np.any(m):
            per_family[fam] = dict(
                n=int(m.sum()), u1_mean=float(U1[m].mean()),
                f5_mean=float(F5[m].mean()), f5_min=float(F5[m].min()),
                f5_max=float(F5[m].max()))

    meta = dict(
        created=datetime.datetime.now().isoformat(timespec='seconds'),
        source='aifflor / FLORIS (quasi-steady), FFC_speed_noise_1.csv windows',
        seed=int(args.seed), window=window, fluct=True,
        action_dim=int(action_dim), scalar_action=bool(scalar),
        physics=pmeta, anchor_u10_a13_mw=anchor,
        target_def='F5 = |P(u1,a)-P(u0,a)| + sum_{t=2..5}|P(ut,a)-P(u(t-1),a)|; '
                   'F4 = sum_{t=2..5}|...|',
        u_note='u 只给 u1；u0/u_seq 只用于标签/真值梯度，不喂网络',
        a_note=('一个动作控制五台：a_i=a0，动作输入 1 维' if scalar
                else '动作在 u0..u5 上保持同一个 5 维 a（受控口径）'),
        action_note=('scalar action：a0 广播到 5 台' if scalar
                     else 'independent 5D action'),
        wind_note='同一列连续 %d 行 + 前一行 u0；全部在 [%.1f,%.1f]'
                  % (window, C.U_MIN, C.U_MAX),
        splits=dict(train=int(args.n_train), val=int(args.n_val),
                    test=int(args.n_test), real=int(n_real)),
        split_codes=dict(train=0, val=1, test=2, real=3),
        real_families=list(families),
        per_split=per_split, per_family=per_family,
    )

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, u=U1, u_seq=USEQ, u0=U0, a=A, a_eff=AE,
                        p=F5, f5=F5, f4=F4, p5=P5, p_steps=PS, p_prev=PP,
                        split=SPLIT, family=FAM,
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    report = os.path.splitext(out)[0] + '.json'
    C.save_json(report, meta)
    print('\n' + '=' * 100)
    print('保存 %s  (%.2f MB)' % (out, os.path.getsize(out) / 1e6))
    print('保存 %s' % report)
    print('总样本 %d | train %d / val %d / test %d / real %d | 用时 %.1fs'
          % (F5.size, (SPLIT == 0).sum(), (SPLIT == 1).sum(),
             (SPLIT == 2).sum(), (SPLIT == 3).sum(), time.time() - t0))
    print('F5: [%.4f, %.4f] MW, mean %.4f, sd %.4f'
          % (F5.min(), F5.max(), F5.mean(), F5.std()))
    print('F4: mean %.4f, sd %.4f；F5/F4 mean %.3f'
          % (F4.mean(), F4.std(), float(np.mean(F5 / np.maximum(F4, 1e-6)))))
    print(phys.summary())
    print('=' * 100)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
