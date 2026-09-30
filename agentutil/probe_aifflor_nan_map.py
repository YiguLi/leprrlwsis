# -*- coding: utf-8 -*-
"""**标定后 FLORIS 的 NaN 危险区地图** —— 量化"多少动作组合会被静默退回标称"。

背景（aifflor 训练实测）
------------------------
`aifflor` 跑 4000 步时，FLORIS 层统计：评估点 **5748**，整批坏点 **251（4.367%）**，
且**逐条重试救回 0** ⇒ 全部走"退回标称动作 `a=1/3`"。
也就是说：**约 4.4% 的 `(u, a)` 组合返回的不是该动作的功率，而是标称动作的功率**
（静默替换）。这既是原 `assert priority > 0` 崩溃的根因，也是"把标定后 FLORIS
当训练环境用"必须知道的偏置来源。

本脚本扫描动作空间，回答：
  ① 坏点率随**风速**怎么变；
  ② 坏点率随**动作水平** `mean(a)` 怎么变；
  ③ 坏点**集中在哪几台机组**（逐台 a 的边缘区？）；
  ④ 与"可能出现负速度"的直觉区域是否一致。

用法（不需 MATLAB）
------------------
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_aifflor_nan_map.py
    ... --n 200 --json output/aifflor_nan_map.json
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

import aifflor.aifflor_turbine as AT  # noqa: E402

NT = AT.NT


def raw_batch(m, us, acts):
    """**不经** `AT.eval_batch` 的逐条重试/兜底，直接拿 FLORIS 的原始结果。

    用于测"**裸引擎**的 NaN 率"——`AT.eval_batch` 会把它修好，反而测不到。

    ⚠ 读法：本探针测的是**引擎层**的死点，**不是生产路径的行为**。
    生产路径（`AifflorPhysics.a_clip`）会把 `a` 抬到 `A_PHYS_LO = 0.025`，
    因此下图里 `a = 0` 那一列的 100% 在实际训练中**不会出现**
    （`aifflor/README.md` §5 有实测：训练 4000 步坏点率 0）。
    """
    import agentutil.probe_engine_random_actions as R
    keep = np.clip(np.atleast_1d(np.asarray(us, dtype=float)), 1e-6, None)
    a2 = np.atleast_2d(np.asarray(acts, dtype=float))
    farm, turb = R._run_batch(m, keep, a2)
    return np.asarray(farm, dtype=float), np.asarray(turb, dtype=float)


def main():
    ap = argparse.ArgumentParser(description='标定后 FLORIS 的 NaN 危险区地图')
    ap.add_argument('--n', type=int, default=200, help='每个风速的采样点数')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    m, info = AT.build_model(verbose=False)
    print('FLORIS %s | 标定=%s' % (info['floris_version'], info['calibrated']))
    print('θ = %s' % dict((('%s.%s' % k), round(v, 5))
                          for k, v in AT.CALIBRATED_WAKE_PARAMS.items()))

    # ---------- ① 按风速 ----------
    print('\n' + '=' * 88)
    print('① 坏点率 vs 风速（每个风速 %d 个随机动作，a∈[0,0.5]^5）' % args.n)
    print('=' * 88)
    print('  %-10s %10s %10s %12s' % ('u', 'n', '坏点', '坏点率'))
    rows_u = []
    for u in (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0):
        A = rng.uniform(0.0, 0.5, size=(args.n, NT))
        us = np.full(args.n, u)
        farm, _ = raw_batch(m, us, A)
        bad = int((~np.isfinite(farm)).sum())
        rows_u.append((u, args.n, bad, bad / args.n))
        print('  %-10.1f %10d %10d %11.3f%%' % (u, args.n, bad, 100 * bad / args.n))

    # ---------- ② 按动作水平 ----------
    print('\n' + '=' * 88)
    print('② 坏点率 vs 动作水平 mean(a)（u=10，每档 %d 点）' % args.n)
    print('=' * 88)
    print('  %-14s %10s %10s %12s' % ('mean(a) 区间', 'n', '坏点', '坏点率'))
    rows_a = []
    edges = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
    for lo, hi in edges:
        cnt = tot = 0
        while tot < args.n:
            A = rng.uniform(lo, hi, size=(args.n, NT))
            farm, _ = raw_batch(m, np.full(args.n, 10.0), A)
            cnt += int((~np.isfinite(farm)).sum())
            tot += args.n
        rows_a.append((lo, hi, tot, cnt, cnt / tot))
        print('  %4.1f-%-9.1f %10d %10d %11.3f%%' % (lo, hi, tot, cnt, 100 * cnt / tot))

    # ---------- ③ 单台极端：只动一台 ----------
    print('\n' + '=' * 88)
    print('③ 只把**某一台**推到带内极值（其余 = 1/3），看坏点率（u=10）')
    print('=' * 88)
    base = np.full(NT, 1.0 / 3.0)
    print('  %-10s %-10s %10s %10s %12s' % ('机组', 'a_该台', 'n', '坏点', '坏点率'))
    rows_t = []
    for j in range(NT):
        for av in (0.0, 0.05, 0.45, 0.5):
            A = np.tile(base, (args.n, 1))
            A[:, j] = av
            farm, _ = raw_batch(m, np.full(args.n, 10.0), A)
            bad = int((~np.isfinite(farm)).sum())
            rows_t.append((j, av, args.n, bad, bad / args.n))
            print('  %-10d %-10.2f %10d %10d %11.3f%%'
                  % (j, av, args.n, bad, 100 * bad / args.n))

    # ---------- ④ 全场同时极端 ----------
    print('\n' + '=' * 88)
    print('④ 全场同时取同一极端值（u=10）—— 与"标定把尾流压窄"是否相关')
    print('=' * 88)
    print('  %-10s %10s %10s %12s' % ('全场 a', 'n', '坏点', '坏点率'))
    rows_c = []
    for av in (0.0, 0.025, 0.05, 0.1, 0.2, 1 / 3, 0.4, 0.45, 0.5):
        A = np.full((args.n, NT), av)
        farm, _ = raw_batch(m, np.full(args.n, 10.0), A)
        bad = int((~np.isfinite(farm)).sum())
        rows_c.append((av, args.n, bad, bad / args.n))
        print('  %-10.4f %10d %10d %11.3f%%' % (av, args.n, bad, 100 * bad / args.n))

    # ---------- 汇总 ----------
    tot_n = sum(r[1] for r in rows_u)
    tot_b = sum(r[2] for r in rows_u)
    print('\n' + '=' * 88)
    print('汇总：① 全部风速合并坏点率 = %d/%d = %.3f%%' % (tot_b, tot_n, 100 * tot_b / tot_n))
    worst = max(rows_t, key=lambda r: r[4])
    print('      ③ 最敏感的单台设置：机组 %d, a=%.2f ⇒ %.2f%%'
          % (worst[0], worst[1], 100 * worst[4]))
    print('=' * 88)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(dict(theta=dict((('%s.%s' % k), v)
                                  for k, v in AT.CALIBRATED_WAKE_PARAMS.items()),
                       by_u=[dict(u=float(a), n=int(b), bad=int(c), rate=float(d))
                             for a, b, c, d in rows_u],
                       by_amean=[dict(lo=float(a), hi=float(b), n=int(c), bad=int(d),
                                      rate=float(e)) for a, b, c, d, e in rows_a],
                       by_turbine=[dict(turb=int(a), a=float(b), n=int(c), bad=int(d),
                                        rate=float(e)) for a, b, c, d, e in rows_t],
                       by_common=[dict(a=float(a), n=int(b), bad=int(c), rate=float(d))
                                  for a, b, c, d in rows_c]),
                  open(args.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('已写入 %s' % args.json)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
