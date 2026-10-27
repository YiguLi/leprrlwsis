#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""生成**五步功率和**数据集：`(u_1, a) → P5 = Σ_{t=1..5} P(u_t, a)`。

任务口径（比单步难在"只看得到窗口首步风速"）
------------------------------------------------
* 风速窗口：从 `FFC_speed_noise_1.csv` 的**同一列连续 5 行**取；
  要求 5 个风速都在 `[3.5, 25] m/s`（env 守卫区间）。
* 动作：`a ∈ [0,0.5]^5`，**窗口内 5 步保持不变**。
* 网络输入：只有窗口首步 `u_1`（`u1/10`）；后续 `u_2..u_5` 不给网络。
* 标签：`P5 = Σ_t P(u_t, a)` [MW]；同时保存逐点 `p_steps` 供诊断。
* 目标（主臂）：`s5 = 1000·P5/(5·u1³)`，即"五步平均 specific power"。
  预测时 `P5̂ = s5̂·5·u1³/1000`。
* 可约/不可约：`P5 = s(a)·Σu³/1000`，而给定 `u1` 后 `Σu³` 仍有很大波动
  （`R=Σu³/(5u1³)` 的 sd≈0.36），所以**只看 u1 时存在不可约噪声**；
  `evaluate.py` 会用 `s(a)·E[Σu³|u1]/1000` 的 oracle 给出拟合天花板。

数据划分
--------
所有合法窗口先打乱，再按 **窗口级** 切分（不重叠）：
train 40k / val 5k / test 5k / real 3k（real 用 5 种结构化动作族）。

产物
----
`data/aifflor_ua_p5.npz`：
    u        (N,)      窗口首步 u1（网络输入）
    u_seq    (N,5)     完整窗口（只用于真值梯度/诊断，不喂网络）
    a        (N,5)     动作
    a_eff    (N,5)     aifflor 物理层映射后的动作
    p        (N,)      P5 = Σ_t P(u_t,a)
    p_steps  (N,5)     逐点功率 [MW]
    s        (N,)      s5 = 1000·P5/(5·u1³)（主目标）
    ratio    (N,)      R = Σu³/(5·u1³)（诊断不可约噪声）
    split    (N,)      0=train 1=val 2=test 3=real
    family   (N,)      -1=iid；0..4=REAL_FAMILIES
    meta               0-d JSON 字符串
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


def _splits_stats(name, u_seq, a, p, s, ratio):
    u1 = u_seq[:, 0]
    return dict(
        name=name, n=int(u1.size),
        u1=dict(min=float(u1.min()), max=float(u1.max()), mean=float(u1.mean())),
        a=dict(min=float(a.min()), max=float(a.max()), mean=float(a.mean())),
        p=dict(min=float(p.min()), max=float(p.max()), mean=float(p.mean()),
               std=float(p.std())),
        s=dict(min=float(s.min()), max=float(s.max()), mean=float(s.mean()),
               std=float(s.std())),
        ratio=dict(min=float(ratio.min()), max=float(ratio.max()),
                   mean=float(ratio.mean()), std=float(ratio.std())),
    )


def main():
    ap = argparse.ArgumentParser(description='用 aifflor 生成五步功率和数据集')
    ap.add_argument('--n_train', type=int, default=40000)
    ap.add_argument('--n_val', type=int, default=5000)
    ap.add_argument('--n_test', type=int, default=5000)
    ap.add_argument('--n_real_per_family', type=int, default=600)
    ap.add_argument('--window', type=int, default=C.WINDOW)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='agentutil/critictest/data/aifflor_ua_p5.npz')
    ap.add_argument('--calibrated', action='store_true')
    ap.add_argument('--power_scale', type=float, default=None)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--wind_csv', default=None)
    args = ap.parse_args()

    window = int(args.window)
    if window <= 1:
        raise SystemExit('make_dataset5.py 只用于 window>1；单步请用 make_dataset.py')
    t0 = time.time()
    C.set_seeds(args.seed)
    rng = np.random.RandomState(args.seed)

    print('=' * 100)
    print('critictest 五步功率和数据集：aifflor / FLORIS   window=%d  seed=%d'
          % (window, args.seed))
    print('=' * 100)
    phys = C.build_physics(calibrated=args.calibrated,
                           power_scale=args.power_scale)
    pmeta = C.physics_meta(phys)
    print('物理口径: calibrated=%s | wake=%s | power_scale=%.6f | FLORIS %s'
          % (pmeta['calibrated'], pmeta['wake_model'], pmeta['power_scale'],
             pmeta['floris_version']))
    anchor = float(phys.power(10.0, np.full(C.NT, C.AT.A_IND)))
    print('锚点 u=10, a=1/3 → %.6f MW（单步）' % anchor)

    days, starts, cols = C.wind_window_pool(window=window,
                                            path=args.wind_csv)
    print('合法窗口 %d / %d（%.1f%%），来自 %d 天 × %d 步'
          % (starts.size, days.shape[1] * (days.shape[0] - window + 1),
             100.0 * starts.size / (days.shape[1] * (days.shape[0] - window + 1)),
             days.shape[1], days.shape[0]))

    n_real = int(args.n_real_per_family) * len(C.REAL_FAMILIES)
    need = int(args.n_train) + int(args.n_val) + int(args.n_test) + n_real
    if need > starts.size:
        raise SystemExit('需要 %d 个窗口，但只有 %d 个合法窗口' % (need, starts.size))
    perm = rng.permutation(starts.size)
    pos = 0
    idx_tr = perm[pos:pos + args.n_train]; pos += args.n_train
    idx_va = perm[pos:pos + args.n_val]; pos += args.n_val
    idx_te = perm[pos:pos + args.n_test]; pos += args.n_test
    idx_re = perm[pos:pos + n_real]

    # ---------------- 覆盖集 ----------------
    splits = []
    for name, idx in (('train', idx_tr), ('val', idx_va), ('test', idx_te)):
        u_seq = C.windows_from_pool(days, starts[idx], cols[idx], window=window)
        a = rng.uniform(C.A_LO, C.A_HI, size=(u_seq.shape[0], C.NT))
        t = time.time()
        p_steps, a_eff = C.power_steps(phys, u_seq, a, chunk=args.chunk,
                                       verbose=True, window=window)
        p = p_steps.sum(1)
        u1 = u_seq[:, 0]
        s = C.target_transform(p, u1, 'sp5', window=window)
        ratio = C.u3_sum(u_seq) / (window * np.power(u1, 3))
        splits.append(dict(name=name, code=len(splits), u_seq=u_seq, a=a,
                           a_eff=a_eff, p=p, p_steps=p_steps, s=s, ratio=ratio,
                           family=np.full(u_seq.shape[0], -1, dtype=np.int16)))
        print('  %-5s n=%6d  用时 %5.1fs  P5=[%.3f, %.3f] MW  mean(s5)=%.4f  '
              'R mean=%.3f sd=%.3f'
              % (name, u_seq.shape[0], time.time() - t, p.min(), p.max(),
                 float(s.mean()), float(ratio.mean()), float(ratio.std())))

    # ---------------- 真实工况集 ----------------
    print('\n[real] 5 种动作族 ...')
    u_seq_r = C.windows_from_pool(days, starts[idx_re], cols[idx_re],
                                  window=window)
    a_list, fam_list = [], []
    per = int(args.n_real_per_family)
    for j, fam in enumerate(C.REAL_FAMILIES):
        a_list.append(C._family_actions(fam, per, rng))
        fam_list.append(np.full(per, j, dtype=np.int16))
    a_r = np.vstack(a_list)
    fam_r = np.concatenate(fam_list)
    if a_r.shape[0] != u_seq_r.shape[0]:
        raise RuntimeError('real 动作/窗口行数不一致: %d vs %d'
                           % (a_r.shape[0], u_seq_r.shape[0]))
    t = time.time()
    p_steps_r, ae_r = C.power_steps(phys, u_seq_r, a_r, chunk=args.chunk,
                                    verbose=True, window=window)
    p_r = p_steps_r.sum(1)
    u1_r = u_seq_r[:, 0]
    s_r = C.target_transform(p_r, u1_r, 'sp5', window=window)
    ratio_r = C.u3_sum(u_seq_r) / (window * np.power(u1_r, 3))
    splits.append(dict(name='real', code=3, u_seq=u_seq_r, a=a_r, a_eff=ae_r,
                       p=p_r, p_steps=p_steps_r, s=s_r, ratio=ratio_r,
                       family=fam_r))
    print('  real  n=%6d  用时 %5.1fs  P5=[%.3f, %.3f] MW  mean(s5)=%.4f  '
          'R mean=%.3f sd=%.3f'
          % (u_seq_r.shape[0], time.time() - t, p_r.min(), p_r.max(),
             float(s_r.mean()), float(ratio_r.mean()), float(ratio_r.std())))

    # ---------------- 拼接 / 保存 ----------------
    U1 = np.concatenate([sp['u_seq'][:, 0] for sp in splits]).astype(np.float32)
    USEQ = np.concatenate([sp['u_seq'] for sp in splits]).astype(np.float32)
    A = np.concatenate([sp['a'] for sp in splits]).astype(np.float32)
    AE = np.concatenate([sp['a_eff'] for sp in splits]).astype(np.float32)
    P = np.concatenate([sp['p'] for sp in splits]).astype(np.float32)
    PS = np.concatenate([sp['p_steps'] for sp in splits]).astype(np.float32)
    S = np.concatenate([sp['s'] for sp in splits]).astype(np.float32)
    RAT = np.concatenate([sp['ratio'] for sp in splits]).astype(np.float32)
    SPLIT = np.concatenate([np.full(sp['u_seq'].shape[0], sp['code'],
                                    dtype=np.int8) for sp in splits])
    FAM = np.concatenate([sp['family'] for sp in splits]).astype(np.int16)

    for arr, name in ((U1, 'u'), (USEQ, 'u_seq'), (A, 'a'), (AE, 'a_eff'),
                      (P, 'p'), (PS, 'p_steps'), (S, 's'), (RAT, 'ratio')):
        if not np.isfinite(arr).all():
            raise SystemExit('数据集列 %s 出现非有限值，拒绝保存' % name)

    per_split = {}
    for sp in splits:
        per_split[sp['name']] = _splits_stats(sp['name'], sp['u_seq'], sp['a'],
                                              sp['p'], sp['s'], sp['ratio'])
    per_family = {}
    for j, fam in enumerate(C.REAL_FAMILIES):
        m = FAM == j
        if np.any(m):
            per_family[fam] = dict(
                n=int(m.sum()), u1_mean=float(U1[m].mean()),
                p_mean=float(P[m].mean()), p_min=float(P[m].min()),
                p_max=float(P[m].max()), s_mean=float(S[m].mean()))

    # 不可约噪声的粗估：R 的变异系数（对同一 u1，P5 的乘性波动）
    r_cv = float(np.std(RAT) / np.mean(RAT))
    meta = dict(
        created=datetime.datetime.now().isoformat(timespec='seconds'),
        source='aifflor / FLORIS (quasi-steady), FFC_speed_noise_1.csv windows',
        seed=int(args.seed),
        window=window,
        physics=pmeta,
        anchor_u10_a13_mw=anchor,
        target_def='P5 = sum_{t=1..5} P(u_t, a); s5 = 1000*P5/(5*u1^3)',
        u_note='u 只给窗口首步 u1；u_seq 仅用于真值梯度/诊断，不作为网络输入',
        a_note='动作在 5 步内保持不变；a_eff = aifflor 物理层映射值',
        wind_note='同一列连续 %d 行，5 个风速都在 [%.1f, %.1f]'
                  % (window, C.U_MIN, C.U_MAX),
        splits=dict(train=int(args.n_train), val=int(args.n_val),
                    test=int(args.n_test), real=int(n_real)),
        split_codes=dict(train=0, val=1, test=2, real=3),
        real_families=list(C.REAL_FAMILIES),
        ratio_note='R = sum(u^3)/(window*u1^3)；给定 u1 后 R 的波动是不可约噪声',
        ratio_cv=r_cv,
        per_split=per_split,
        per_family=per_family,
    )

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, u=U1, u_seq=USEQ, a=A, a_eff=AE, p=P, p_steps=PS,
                        s=S, ratio=RAT, split=SPLIT, family=FAM,
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    report = os.path.splitext(out)[0] + '.json'
    C.save_json(report, meta)
    print('\n' + '=' * 100)
    print('保存 %s  (%.2f MB)' % (out, os.path.getsize(out) / 1e6))
    print('保存 %s' % report)
    print('总样本 %d | train %d / val %d / test %d / real %d | 用时 %.1fs'
          % (U1.size, (SPLIT == 0).sum(), (SPLIT == 1).sum(),
             (SPLIT == 2).sum(), (SPLIT == 3).sum(), time.time() - t0))
    print('P5: [%.4f, %.4f] MW, mean %.4f, sd %.4f'
          % (P.min(), P.max(), P.mean(), P.std()))
    print('s5: [%.4f, %.4f], mean %.4f, sd %.4f'
          % (S.min(), S.max(), S.mean(), S.std()))
    print('R : mean %.4f, sd %.4f, CV %.3f（给定 u1 的不可约乘性波动）'
          % (RAT.mean(), RAT.std(), r_cv))
    print(phys.summary())
    print('=' * 100)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
