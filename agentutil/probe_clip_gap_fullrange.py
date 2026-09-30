# -*- coding: utf-8 -*-
"""全风速域（3–20 m/s，0.5 m/s 粒度，每档 10 个随机行为）的 **clip(u<0→0) 影响**与
**FLORIS vs WFSim 差距**检查。

背景
----
方案 A（接受并声明）的前提是"`u<0` 的影响可忽略"。要验证它，需要两件事：

① **clip(u<0→0) 到底改变了多少功率？**
   ⚠ 注意：`aifflor` 的功率路径**已经** clip 了 —— `_power_of_u` 里
   `u = np.clip(u, 0, None)` 之后才算 `cbrt(mean(u³))`。所以"clip 到 0"是当前行为。
   真正要量的是**如果不 clip 会差多少**（即 clip 是否掩盖了显著差异）。
   本脚本直接读转子平面 `flow_field.u`，同时算 clip / 不 clip 两个口径的
   `Ue_eff = (mean(u³))^(1/3)`，再转成功率比 `(Ue_clip/Ue_noclip)³`。

② **FLORIS 与 WFSim 的差距**（在同批点上）
   需要 WFSim 参考值。本脚本会：
     * 若 `output/clip_check_wfsim.json` 存在 → 直接用它对比；
     * 否则生成 `output/clip_check_plan.json` + `agentutil/run_clip_check_wfsim.m`，
       供 MATLAB 侧执行（`agentutil/run_wfsim` 需要 MATLAB 引擎，沙箱下起不来）。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_clip_gap_fullrange.py
    ... --u-lo 3.0 --u-hi 20.0 --du 0.5 --n-per-bin 10 --seed 0
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

from floris_import_guard import check_interpreter as _chk, sanitize as _sanitize  # noqa: E402
_chk()
_sanitize()

import numpy as np  # noqa: E402

import agentutil.probe_engine_random_actions as R  # noqa: E402
import aifflor.aifflor_turbine as AT  # noqa: E402

NT = AT.NT
PLAN = os.path.join('output', 'clip_check_plan.json')
WREF = os.path.join('output', 'clip_check_wfsim.json')
OUT = os.path.join('output', 'clip_check_fullrange.json')


def build_plan(u_lo, u_hi, du, n_per_bin, seed):
    rng = np.random.default_rng(seed)
    bins = np.arange(u_lo, u_hi + 1e-9, du)
    us, As, bs = [], [], []
    for b in bins:
        for _ in range(n_per_bin):
            us.append(float(b))
            As.append(rng.uniform(0.0, 0.5, size=NT))
            bs.append(float(b))
    return np.asarray(us), np.asarray(As), np.asarray(bs), bins


def run_floris(us, A):
    """返回逐台 (Ue_clip, Ue_noclip) 与农场功率（两种口径）。

    ⚠ 通过**逐样本**调用以获得转子平面 u（`_run_batch` 是批量，flow_field 只留最后一批）。
    """
    R.CT_SCALE = 1.0
    m, _ = AT.build_model(calibrated=True)
    n = len(us)
    farm_clip = np.zeros(n)
    farm_noclip = np.zeros(n)
    turb_clip = np.zeros((n, NT))
    turb_noclip = np.zeros((n, NT))
    ue_clip = np.zeros((n, NT))
    ue_noclip = np.zeros((n, NT))
    n_neg = np.zeros(n, dtype=int)
    min_u = np.full(n, np.nan)
    import yawflor.yawflor_turbine as YT
    for i in range(n):
        R._A_SET = A[i:i + 1]
        m.set(wind_speeds=[float(us[i])],
              wind_directions=[YT.WIND_DIR_ALIGNED],
              turbulence_intensities=[YT.TI], yaw_angles=np.zeros((1, NT)))
        with R._solver_hook()():
            m.run()
        u = np.asarray(m.core.flow_field.u, dtype=float)[0]        # (NT, gz, gy)
        n_neg[i] = int((u < 0).sum())
        min_u[i] = float(u.min())
        uc = np.clip(u, 0.0, None)
        a_eff = np.cbrt(np.mean(uc ** 3, axis=(1, 2)))             # clip 口径
        a_no = np.cbrt(np.mean(u ** 3, axis=(1, 2)))               # 不 clip（可能为负/复数）
        ue_clip[i] = a_eff
        ue_noclip[i] = np.real(a_no)
        # 功率 ∝ Cp_eff(a)·Ue³，且 `Cp_eff` 与 `Ue` **无关** ⇒ 两个口径的比值
        # 恰为 (Ue_clip/Ue_noclip)³，无需反算任何常数。
        pw = np.asarray(m.get_turbine_powers(), dtype=float).reshape(-1)[:NT] / 1e6
        turb_clip[i] = pw
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio_t = np.where(np.abs(ue_noclip[i]) > 1e-12,
                               (a_eff / ue_noclip[i]) ** 3, np.nan)
        turb_noclip[i] = pw / np.where(np.abs(ratio_t) > 1e-12, ratio_t, np.nan)
        farm_clip[i] = turb_clip[i].sum()
        farm_noclip[i] = np.nansum(turb_noclip[i])
    return dict(farm_clip=farm_clip, farm_noclip=farm_noclip,
                turb_clip=turb_clip, turb_noclip=turb_noclip,
                ue_clip=ue_clip, ue_noclip=ue_noclip,
                n_neg=n_neg, min_u=min_u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--u-lo', type=float, default=3.0)
    ap.add_argument('--u-hi', type=float, default=20.0)
    ap.add_argument('--du', type=float, default=0.5)
    ap.add_argument('--n-per-bin', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    us, A, binv, bins = build_plan(args.u_lo, args.u_hi, args.du, args.n_per_bin,
                                   args.seed)
    print('=' * 100)
    print('计划: u ∈ [%.1f, %.1f] 步长 %.1f，每档 %d 个随机行为 ⇒ 共 %d 点'
          % (args.u_lo, args.u_hi, args.du, args.n_per_bin, len(us)))
    print('=' * 100)

    print('\n[FLORIS] 逐样本求值（%d 点，含转子平面读数）...' % len(us))
    r = run_floris(us, A)
    print('[FLORIS] 完成。')

    # ---------- ① clip 影响 ----------
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where((r['farm_noclip'] != 0) & np.isfinite(r['farm_noclip']),
                         r['farm_clip'] / r['farm_noclip'], np.nan)
    print('\n' + '=' * 100)
    print('① **clip(u<0→0) 的影响**（当前行为 = clip；对比"不 clip"）')
    print('=' * 100)
    has = r['n_neg'] > 0
    print('  出现 u<0 的点: %d / %d (%.1f%%)' % (has.sum(), len(us), 100 * has.mean()))
    print('  出现 u<0 的点里，农场合计功率比 (clip/noclip):')
    if has.any():
        rr = ratio[has]
        rr = rr[np.isfinite(rr)]
        print('     min=%.6f  p25=%.6f  中位=%.6f  p75=%.6f  max=%.6f'
              % (rr.min(), np.percentile(rr, 25), np.median(rr),
                 np.percentile(rr, 75), rr.max()))
        print('     |1−比| 的最大值 = %.3e  (即 clip 最多改变出力 %.4f%%)'
              % (np.abs(1 - rr).max(), 100 * np.abs(1 - rr).max()))
    print('  未出现 u<0 的点: 比恒为 1（clip 无作用）')
    worst = int(np.nanargmax(np.abs(1 - ratio))) if np.isfinite(ratio).any() else None
    if worst is not None:
        print('\n  最坏点: u=%.2f  a=%s' % (us[worst],
                                          np.array2string(A[worst], precision=3)))
        print('          n_neg=%d  min_u=%.4f' % (r['n_neg'][worst], r['min_u'][worst]))
        print('          Ue_clip   = %s' % np.array2string(r['ue_clip'][worst], precision=4))
        print('          Ue_noclip = %s' % np.array2string(r['ue_noclip'][worst], precision=4))
        print('          P_clip=%.6f  P_noclip=%.6f  MW' % (r['farm_clip'][worst],
                                                            r['farm_noclip'][worst]))

    # ---------- ② 逐风速档汇总 ----------
    print('\n' + '=' * 100)
    print('② 逐风速档汇总（FLORIS clip 口径）')
    print('=' * 100)
    print('  %-8s %8s %10s %12s %12s %12s' %
          ('u 档', 'n', '有u<0点', 'min_u 均值', 'P_FLORIS', 'clip 影响%'))
    rows = []
    for b in bins:
        m_ = binv == b
        if not m_.any():
            continue
        nneg = int((r['n_neg'][m_] > 0).sum())
        imp = 100 * np.nanmax(np.abs(1 - ratio[m_])) if np.isfinite(ratio[m_]).any() else 0.0
        rows.append(dict(u=float(b), n=int(m_.sum()), n_bad=nneg,
                         min_u_mean=float(np.nanmean(r['min_u'][m_])),
                         p_floris=float(r['farm_clip'][m_].mean()),
                         clip_impact_pct=float(imp)))
        print('  %-8.1f %8d %10d %12.4f %12.4f %12.4f'
              % (b, m_.sum(), nneg, np.nanmean(r['min_u'][m_]),
                 r['farm_clip'][m_].mean(), imp))

    # ---------- ③ 与 WFSim 对比（若有参考） ----------
    print('\n' + '=' * 100)
    print('③ 与 WFSim 对比')
    print('=' * 100)
    if os.path.exists(WREF):
        w = json.load(open(WREF, encoding='utf-8'))
        pw = np.asarray(w['p_wfsim'], dtype=float)
        if len(pw) == len(us):
            rel = 100 * (r['farm_clip'] - pw) / np.maximum(pw, 1e-9)
            print('  有 WFSim 参考（%d 点）: 相对误差 min=%.3f%% 中位=%.3f%% max=%.3f%%'
                  % (len(pw), rel.min(), np.median(rel), rel.max()))
            print('  %-8s %10s %12s %12s %12s' %
                  ('u 档', 'n', 'P_WFSim', 'P_FLORIS', '相对误差%'))
            for b in bins:
                m_ = binv == b
                if not m_.any():
                    continue
                e = 100 * (r['farm_clip'][m_].mean() - pw[m_].mean()) / max(pw[m_].mean(), 1e-9)
                print('  %-8.1f %10d %12.4f %12.4f %12.3f'
                      % (b, m_.sum(), pw[m_].mean(), r['farm_clip'][m_].mean(), e))
        else:
            print('  ⚠ %s 点数 %d ≠ 计划 %d，跳过逐档对比' % (WREF, len(pw), len(us)))
    else:
        print('  ❌ 没有 WFSim 参考 `%s`。已生成执行计划与 MATLAB 脚本：' % WREF)
        print('     · 计划: %s' % PLAN)
        print('     · 脚本: agentutil/run_clip_check_wfsim.m')
        print('     · 跑完后把结果存为 %s（字段 p_wfsim/p_wfsim_turbine）' % WREF)
        print('     · 再重跑本脚本即可给出逐档对比。')

    # ---------- 落盘 ----------
    json.dump(dict(plan=dict(u_lo=args.u_lo, u_hi=args.u_hi, du=args.du,
                             n_per_bin=args.n_per_bin, seed=args.seed,
                             nt=NT, n=len(us)),
                   u=[float(x) for x in us],
                   A=[[float(v) for v in row] for row in A],
                   p_floris_clip=[float(x) for x in r['farm_clip']],
                   p_floris_noclip=[float(x) for x in r['farm_noclip']],
                   n_neg=[int(x) for x in r['n_neg']],
                   min_u=[float(x) for x in r['min_u']],
                   by_bin=rows),
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(dict(config=dict(u_lo=args.u_lo, u_hi=args.u_hi, du=args.du,
                               n_per_bin=args.n_per_bin, seed=args.seed, nt=NT),
                   u=[float(x) for x in us],
                   A=[[float(v) for v in row] for row in A]),
              open(PLAN, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n已写入: %s （FLORIS 结果）' % OUT)
    print('已写入: %s （WFSim 执行计划）' % PLAN)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
