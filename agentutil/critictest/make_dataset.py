#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 `(u, a) → P_farm` 数据集（风场仿真 = aifflor / FLORIS）。

数据口径
--------
* 覆盖集（train / val / test）：`u ~ U(3.5,25)`，`a ~ U(0,0.5)^5`。
  这是**函数覆盖**采样，不模拟策略分布；目的是检验网络在整个动作/风速域上的
  拟合能力。
* 真实工况集（real）：`u` 从 `FFC_speed_noise_1.csv` 的实测序列随机抽，
  `a` 取 5 种结构化动作族（整场同水平 / 沿程斜坡 / 单机 Cp 最优附近 /
  场级最优附近 / 逐台独立），检验在真实运行区域的拟合。
* 目标 `P` 由 `AifflorPhysics.power()` 给出 [MW]，默认 aifflor 生产配置
  （`calibrated=False` + `power_scale=LEVEL_COEF`）。
* 同时保存 `s = 1000·P/u³`（specific power），作为主训练目标。

产物
----
`data/aifflor_ua_p.npz`（压缩）：
    u, a, a_eff, p, s : float32
    split              : int8   0=train 1=val 2=test 3=real
    family             : int16  -1=覆盖集；0..4 = REAL_FAMILIES
    meta               : 0-d JSON 字符串

用法
----
    python agentutil/critictest/make_dataset.py
    python agentutil/critictest/make_dataset.py --n_train 60000 --seed 1
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


def _stats(name, u, a, p, s):
    return dict(
        name=name, n=int(u.size),
        u=dict(min=float(u.min()), max=float(u.max()), mean=float(u.mean())),
        a=dict(min=float(a.min()), max=float(a.max()), mean=float(a.mean())),
        p=dict(min=float(p.min()), max=float(p.max()), mean=float(p.mean()),
               std=float(p.std())),
        s=dict(min=float(s.min()), max=float(s.max()), mean=float(s.mean()),
               std=float(s.std())),
    )


def main():
    ap = argparse.ArgumentParser(description='用 aifflor 生成 (u,a)->P 数据集')
    ap.add_argument('--n_train', type=int, default=40000)
    ap.add_argument('--n_val', type=int, default=5000)
    ap.add_argument('--n_test', type=int, default=5000)
    ap.add_argument('--n_real_per_family', type=int, default=600,
                    help='真实工况集每个动作族的条数（5 族 ⇒ 总数 ×5）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='agentutil/critictest/data/aifflor_ua_p.npz')
    ap.add_argument('--calibrated', action='store_true',
                    help='用 floris_calibrated_defaults 的标定参数（默认用 aifflor 生产配置：'
                         'FLORIS 原厂参数 + LEVEL_COEF 电平系数）')
    ap.add_argument('--power_scale', type=float, default=None,
                    help='覆盖出口电平系数；默认按 --calibrated 自动选')
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--wind_csv', default=None, help='真实风速池 CSV（默认 FFC_speed_noise_1.csv）')
    args = ap.parse_args()

    t0 = time.time()
    C.set_seeds(args.seed)
    rng = np.random.RandomState(args.seed)

    print('=' * 100)
    print('critictest 数据集：aifflor / FLORIS   seed=%d' % args.seed)
    print('=' * 100)
    phys = C.build_physics(calibrated=args.calibrated,
                           power_scale=args.power_scale)
    pmeta = C.physics_meta(phys)
    print('物理口径: calibrated=%s | wake=%s | power_scale=%.6f | FLORIS %s'
          % (pmeta['calibrated'], pmeta['wake_model'], pmeta['power_scale'],
             pmeta['floris_version']))
    anchor = float(phys.power(10.0, np.full(C.NT, C.AT.A_IND)))
    print('锚点 u=10, a=1/3 → %.6f MW' % anchor)

    # ---------------- 覆盖集 ----------------
    print('\n[1/2] 覆盖集采样 ...')
    splits = []
    for name, n in (('train', args.n_train), ('val', args.n_val), ('test', args.n_test)):
        u, a = C.sample_iid(n, rng)
        t = time.time()
        p, a_eff = C.power_of(phys, u, a, chunk=args.chunk, verbose=True)
        splits.append(dict(name=name, code=len(splits), u=u, a=a, a_eff=a_eff, p=p))
        print('  %-5s n=%6d  用时 %5.1fs  P=[%.3f, %.3f] MW  mean(s)=%.4f'
              % (name, n, time.time() - t, p.min(), p.max(),
                 float(np.mean(C.S_SCALE * p / np.power(u, 3)))))

    # ---------------- 真实工况集 ----------------
    print('\n[2/2] 真实工况集采样（5 种动作族）...')
    u_r, a_r, fam_r = C.sample_realistic(args.n_real_per_family, rng,
                                         wind_pool=C.load_wind_pool(args.wind_csv))
    t = time.time()
    p_r, ae_r = C.power_of(phys, u_r, a_r, chunk=args.chunk, verbose=True)
    splits.append(dict(name='real', code=3, u=u_r, a=a_r, a_eff=ae_r, p=p_r))
    print('  real  n=%6d  用时 %5.1fs  P=[%.3f, %.3f] MW  mean(s)=%.4f'
          % (u_r.size, time.time() - t, p_r.min(), p_r.max(),
             float(np.mean(C.S_SCALE * p_r / np.power(u_r, 3)))))

    # ---------------- 拼接 ----------------
    U, A, AE, P = [], [], [], []
    SPLIT, FAM = [], []
    for sp in splits:
        U.append(sp['u'].astype(np.float32))
        A.append(sp['a'].astype(np.float32))
        AE.append(sp['a_eff'].astype(np.float32))
        P.append(sp['p'].astype(np.float32))
        SPLIT.append(np.full(sp['u'].size, sp['code'], dtype=np.int8))
        FAM.append(np.full(sp['u'].size, -1, dtype=np.int16))
    FAM[-1] = fam_r.astype(np.int16)

    U = np.concatenate(U)
    A = np.concatenate(A)
    AE = np.concatenate(AE)
    P = np.concatenate(P)
    SPLIT = np.concatenate(SPLIT)
    FAM = np.concatenate(FAM)
    S = (C.S_SCALE * P.astype(np.float64) / np.power(U.astype(np.float64), 3)).astype(np.float32)

    if not np.isfinite(P).all() or not np.isfinite(S).all():
        raise SystemExit('数据集中出现非有限值，拒绝保存')

    # 分集统计（写进 JSON，便于复查；不依赖重新加载）
    per_split = {}
    for sp in splits:
        per_split[sp['name']] = _stats(sp['name'], sp['u'], sp['a'], sp['p'],
                                       C.S_SCALE * sp['p'] / np.power(sp['u'], 3))
    # 每族统计
    per_family = {}
    for j, fam in enumerate(C.REAL_FAMILIES):
        m = FAM == j
        if np.any(m):
            per_family[fam] = dict(
                n=int(m.sum()),
                u_mean=float(U[m].mean()),
                p_mean=float(P[m].mean()),
                p_min=float(P[m].min()),
                p_max=float(P[m].max()),
                s_mean=float(S[m].mean()),
            )

    meta = dict(
        created=datetime.datetime.now().isoformat(timespec='seconds'),
        source='aifflor / FLORIS (quasi-steady)',
        seed=int(args.seed),
        physics=pmeta,
        anchor_u10_a13_mw=anchor,
        target_def='s = 1000 * P / u^3  (MW / (m/s)^3, x1000)',
        splits=dict(train=int(args.n_train), val=int(args.n_val), test=int(args.n_test),
                    real=int(u_r.size)),
        split_codes=dict(train=0, val=1, test=2, real=3),
        real_families=list(C.REAL_FAMILIES),
        a_note='a 为原始动作 [0,0.5]；a_eff 为 aifflor 物理层映射后的值 '
               '(a<0.025 抬到 0.025)。网络输入用 a。',
        per_split=per_split,
        per_family=per_family,
        u_range=[C.U_MIN, C.U_MAX],
        a_range=[C.A_LO, C.A_HI],
    )

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, u=U, a=A, a_eff=AE, p=P, s=S,
                        split=SPLIT, family=FAM,
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    report = os.path.splitext(out)[0] + '.json'
    C.save_json(report, meta)

    print('\n' + '=' * 100)
    print('保存 %s  (%.2f MB)' % (out, os.path.getsize(out) / 1e6))
    print('保存 %s' % report)
    print('总样本 %d | train %d / val %d / test %d / real %d | 用时 %.1fs'
          % (U.size, (SPLIT == 0).sum(), (SPLIT == 1).sum(), (SPLIT == 2).sum(),
             (SPLIT == 3).sum(), time.time() - t0))
    print('P: [%.4f, %.4f] MW, mean %.4f, sd %.4f'
          % (P.min(), P.max(), P.mean(), P.std()))
    print('s: [%.4f, %.4f], mean %.4f, sd %.4f'
          % (S.min(), S.max(), S.mean(), S.std()))
    print(phys.summary())
    print('=' * 100)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
