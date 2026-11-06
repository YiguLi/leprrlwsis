#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""给五步波动数据集补上**逐台逐点有效风速** `u_eff(t,i)`。

`u_eff` 取自 FLORIS 的 `turbine_average_velocities`（尾流后的转子平均风速），
形状 `(N, 5 步, 5 台)`；另存 `u_eff0`（窗口前一步 `u0` 的逐台有效风速）。
这是"每个风机逐点风速"的**oracle 特征**：它依赖当前动作（尾流随 `a` 变化），
推理时若换动作需要重跑 FLORIS；它用来测"信息量上限"，不是可直接部署的输入。

用法
----
    python agentutil/critictest/add_ueff_features.py \
        --data agentutil/critictest/data/aifflor_ua_f5_scalar.npz \
        --out  agentutil/critictest/data/aifflor_ua_f5_scalar_ueff.npz
"""
import argparse
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
import agentutil.probe_engine_random_actions as R  # noqa: E402


def effective_velocities(physics, u_seq, a_eff, chunk=2048):
    """返回 `(u_eff (N,5,5), turb_mw (N,5))`；`u_eff[:, t, i]` = t 步第 i 台。"""
    u_seq = np.asarray(u_seq, dtype=np.float64)
    a_eff = np.asarray(a_eff, dtype=np.float64)
    n, w = u_seq.shape
    u_eff = np.empty((n, w, C.NT), dtype=np.float64)
    turb = np.empty((n, w, C.NT), dtype=np.float64)
    m = physics.model
    for s in range(0, n, int(chunk)):
        e = min(s + int(chunk), n)
        us = u_seq[s:e].reshape(-1)
        acts = np.repeat(a_eff[s:e], w, axis=0)
        _farm, tb = R._run_batch(m, us, acts)
        vel = np.asarray(m.turbine_average_velocities, dtype=np.float64)
        if vel.shape[0] != (e - s) * w:
            raise RuntimeError('velocity 形状 %s 与批大小 %d 不一致'
                               % (vel.shape, (e - s) * w))
        u_eff[s:e] = vel.reshape(e - s, w, C.NT)
        turb[s:e] = np.asarray(tb, dtype=np.float64).reshape(e - s, w, C.NT)
    return u_eff, turb


def main():
    ap = argparse.ArgumentParser(description='给 F5 数据集补逐台逐点有效风速')
    ap.add_argument('--data',
                    default='agentutil/critictest/data/aifflor_ua_f5_scalar.npz')
    ap.add_argument('--out', default=None,
                    help='默认 = <data stem>_ueff.npz')
    ap.add_argument('--chunk', type=int, default=2048)
    args = ap.parse_args()

    t0 = time.time()
    ds = np.load(args.data, allow_pickle=False)
    if 'f5' not in ds.files:
        raise SystemExit('需要五步波动数据集')
    phys = C.build_physics(calibrated=False, power_scale=C.AT.LEVEL_COEF)
    u_seq = ds['u_seq'].astype(np.float64)
    u0 = ds['u0'].astype(np.float64)
    a_eff = ds['a_eff'].astype(np.float64)
    p_steps = ds['p_steps'].astype(np.float64)
    p_prev = ds['p_prev'].astype(np.float64)

    print('计算逐台逐点有效风速：%d 窗口 × 5 步 ...' % u_seq.shape[0])
    t = time.time()
    u_eff, turb = effective_velocities(phys, u_seq, a_eff, chunk=args.chunk)
    print('  u_eff 用时 %.1fs  shape=%s  range=[%.3f, %.3f]'
          % (time.time() - t, u_eff.shape, u_eff.min(), u_eff.max()))

    # 对齐自检：turb * power_scale 应等于数据集里的 p_steps
    p_hat = turb.sum(axis=2) * float(phys.power_scale)
    maxdiff = float(np.max(np.abs(p_hat - p_steps)))
    print('  逐台功率自检：max|turb·scale − p_steps| = %.3e MW' % maxdiff)
    if maxdiff > 1e-4:
        raise SystemExit('u_eff 与 p_steps 不对齐（maxdiff=%.3e），拒绝保存' % maxdiff)

    t = time.time()
    u_eff0, turb0 = effective_velocities(
        phys, u0.reshape(-1, 1), a_eff, chunk=args.chunk)
    u_eff0 = u_eff0[:, 0, :]
    p_prev_hat = turb0[:, 0, :].sum(1) * float(phys.power_scale)
    maxdiff0 = float(np.max(np.abs(p_prev_hat - p_prev)))
    print('  u_eff0 用时 %.1fs  shape=%s  p_prev 自检 max|diff|=%.3e'
          % (time.time() - t, u_eff0.shape, maxdiff0))
    if maxdiff0 > 1e-4:
        raise SystemExit('u_eff0 与 p_prev 不对齐（maxdiff=%.3e）' % maxdiff0)

    if not (np.isfinite(u_eff).all() and np.isfinite(u_eff0).all()):
        raise SystemExit('u_eff 出现非有限值，拒绝保存')

    meta = json.loads(str(ds['meta']))
    meta['ueff'] = True
    meta['ueff_note'] = ('u_eff[t,i] = 第 t 步第 i 台的 FLORIS 转子平均风速；'
                         '依赖动作（尾流），是 oracle 特征')
    meta['ueff_shape'] = list(u_eff.shape)
    meta['u_eff0_note'] = '窗口前一步 u0 的逐台有效风速'
    # 统计：逐台均值 / 与自由来流的比值
    ratios = u_eff / u_seq[:, :, None]
    meta['ueff_ratio'] = dict(
        mean=float(ratios.mean()), min=float(ratios.min()),
        max=float(ratios.max()),
        per_turbine_mean=[float(v) for v in ratios.mean(axis=(0, 1))])

    out = args.out or (os.path.splitext(args.data)[0] + '_ueff.npz')
    out = os.path.abspath(out)
    arrays = {k: ds[k] for k in ds.files if k != 'meta'}
    arrays['u_eff'] = u_eff.astype(np.float32)
    arrays['u_eff0'] = u_eff0.astype(np.float32)
    arrays['meta'] = np.array(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(out, **arrays)
    print('\n保存 %s (%.2f MB)' % (out, os.path.getsize(out) / 1e6))
    print('  u_eff 与自由来流比值: mean %.3f, range [%.3f, %.3f]'
          % (ratios.mean(), ratios.min(), ratios.max()))
    print('  逐台均值:', np.round(ratios.mean(axis=(0, 1)), 3).tolist())
    print('总用时 %.1fs' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
