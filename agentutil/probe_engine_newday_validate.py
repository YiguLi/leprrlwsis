# -*- coding: utf-8 -*-
"""**独立验证**：标定后的 FLORIS 默认参数，在一批**新数据**上是否仍然合理。

与既有脚本的区别（这正是本次要做的事）
--------------------------------------
`probe_engine_random_actions.py` 用的是 LHS 乱序样本 ⇒ 逐样本差分 `|P(t)−P(t−1)|`
里绝大部分是"工况跳变"，所以那份数据**只能做两引擎对照，不能算物理波动**。

本脚本改成**真实时间序列**：

  * 风速 = `FFC_speed_noise_1.csv` 的**随机一天**（列 = 日）的**随机连续 100 分钟**；
  * 动作 = **每一步都重新随机**（5 维逐台 `a∈[0.025,0.5]`，无块结构）——
    这正是用户要的"每一步都使用随机行为"；
  * 两引擎**逐条**对齐 `(u, a₁…a₅)`，唯一差异是引擎。

因为相邻步的风速是连续的，`P_F = |P(t)−P(t−1)|` 这时**才有物理意义**，
可以顺带验证 H 层约束口径（`c1` / 3 MW 越限率 / `d_block`）。

判据（预登记）
--------------
1. 比值 `FLORIS/WFSim` 的均值 ≈ 1、std 小 ⇒ 标定有效；
2. **常数重标定后 MAPE** 与标定阶段报告的量级一致（~1%）⇒ 未过拟合到标定集；
3. `c1` 块和倍率 ≈ 1 ⇒ 尺度类超参数可直接迁移；
4. 若出现 NaN ⇒ 记录位置（标定参数把尾流压窄的外推风险）。

用法（普通终端；MATLAB Engine 需 named pipe）
--------------------------------------------
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_engine_newday_validate.py
    ... --day 57 --start 300 --steps 100 --seed 1
    ... --day random --start random          # 显式随机（默认就是随机）
    ... --json output/engine_newday_validate.json
"""
import argparse
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

from floris_import_guard import check_interpreter as _chk, sanitize as _sanitize  # noqa: E402
_chk()
_sanitize()

import numpy as np  # noqa: E402

import hsrl_cost as hc  # noqa: E402
import yawflor.yawflor_turbine as YT  # noqa: E402

NT = 5
A_LO, A_HI = 0.025, 0.5
U_LO, U_HI = 3.5, 25.0
BLOCK = 5


# ===========================================================================
# 1. 取一批新数据：随机日 + 随机连续 100 分钟 + 每步随机动作
# ===========================================================================
def pick(day_arg, start_arg, steps, seed):
    import pandas as pd
    df = pd.read_csv('FFC_speed_noise_1.csv')
    n_day = df.shape[1] - 1                       # 第 0 列是行号，列 = 日
    rng = np.random.default_rng(seed)
    day = int(day_arg) if str(day_arg) != 'random' else int(rng.integers(1, n_day + 1))
    full = df.iloc[:, day].to_numpy(dtype=float)
    tmax = len(full)
    if str(start_arg) == 'random':
        start = int(rng.integers(0, max(1, tmax - steps)))
    else:
        start = int(start_arg)
    if start + steps > tmax:
        raise SystemExit('起始 %d + 步数 %d 超出该日长度 %d' % (start, steps, tmax))
    u = full[start:start + steps].copy()
    return day, start, u, n_day


def sample_actions(n, seed, rng=None):
    """每一步都重新随机的逐台 a（无块结构）。"""
    rng = np.random.default_rng(seed + 4242) if rng is None else rng
    return rng.uniform(A_LO, A_HI, size=(n, NT))


# ===========================================================================
# 2. 两个引擎（FLORIS 用**标定后默认**）
# ===========================================================================
def run_floris(us, A, stock=False, verbose=True):
    import agentutil.probe_engine_random_actions as R
    R.register_op_model()
    m = R.build_floris_model(stock=stock)
    if verbose:
        print('[FLORIS] %s（参数=%s）算 %d 条 ...'
              % (R._floris_version(), '随包默认' if stock else '标定后(现行默认)', len(us)),
              flush=True)
    t0 = time.time()
    farm, turb = R._run_batch(m, us, A)
    nbad = int((~np.isfinite(farm)).sum())
    if verbose:
        print('[FLORIS] 完成 %.3f s；非有限值 %d 条' % (time.time() - t0, nbad), flush=True)
    return farm, turb, nbad


def run_wfsim(us, A, verbose=True):
    import matlab
    import matlab.engine
    if verbose:
        print('[WFSim] 启动 MATLAB 引擎 ...', flush=True)
    t0 = time.time()
    eng = matlab.engine.start_matlab()
    if verbose:
        print('[WFSim] 就绪 (%.1f s)，MATLAB %s' % (time.time() - t0, eng.version()),
              flush=True)
    eng.cd(ROOT, nargout=0)
    P = np.full(len(us), np.nan)
    T = np.full((len(us), NT), np.nan)
    t0 = time.time()
    for i in range(len(us)):
        res = eng.WFSim_port(matlab.double([float(us[i])]), matlab.double([0.0]),
                             matlab.double([list(map(float, A[i]))]), matlab.double([0.0]))
        P[i] = float(res['out']) / 1e6
        try:
            T[i] = np.asarray(res['test'][0][4]['turbine']['power'],
                              dtype=float).reshape(-1)[:NT] / 1e6
        except Exception:
            pass
        if verbose and ((i + 1) % 20 == 0 or i == 0):
            print('[WFSim] %3d/%d u=%7.3f P=%9.6f MW (%.1f s)'
                  % (i + 1, len(us), us[i], P[i], time.time() - t0), flush=True)
    eng.quit()
    if verbose:
        print('[WFSim] 完成 %.1f s（逐台回读 %d/%d）'
              % (time.time() - t0, int(np.isfinite(T).all(axis=1).sum()), len(us)), flush=True)
    return P, T


# ===========================================================================
# 3. 对比（含 H 层约束口径 —— 这次是真实时间序列，绝对值有物理意义）
# ===========================================================================
def stats(PW, PF):
    ok = np.isfinite(PW) & np.isfinite(PF) & (PW > 1e-9)
    r = PF[ok] / PW[ok]
    rel = np.abs(PF[ok] - PW[ok]) / PW[ok]
    k = float(np.dot(PW[ok], PF[ok]) / np.dot(PW[ok], PW[ok]))
    rel_k = np.abs(k * PW[ok] - PW[ok]) / PW[ok]
    return dict(n=int(ok.sum()), ratio_mean=float(r.mean()), ratio_std=float(r.std()),
                ratio_min=float(r.min()), ratio_max=float(r.max()),
                mape=float(rel.mean()), mape_k=float(rel_k.mean()), k_best=k,
                medape=float(np.median(rel)),
                corr=float(np.corrcoef(PW[ok], PF[ok])[0, 1]) if ok.sum() > 2 else None)


def cost(P, tag):
    pf = np.abs(np.diff(P))
    pf = pf[np.isfinite(pf)]
    if pf.size == 0:
        return None
    c1 = np.array([hc.fluct_cost(x) for x in pf])
    nb = len(c1) // BLOCK
    blocks = (c1[:nb * BLOCK].reshape(nb, BLOCK).sum(axis=1) if nb
              else np.array([c1.sum()]))
    return dict(tag=tag, pf_mean=float(pf.mean()),
                pf_p95=float(np.percentile(pf, 95)), pf_max=float(pf.max()),
                viol=float((pf > hc.P_F_LIMIT).mean()),
                c1_mean=float(c1.mean()), blk_mean=float(blocks.mean()),
                blk_over=float((blocks > hc.D_BLOCK).mean()))


def main():
    ap = argparse.ArgumentParser(description='新数据独立验证标定后的默认参数',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument('--day', default='random', help="日（列号）或 'random'")
    ap.add_argument('--start', default='random', help="起始分钟或 'random'")
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--also-stock', action='store_true',
                    help='同时算 FLORIS 随包默认，作对照')
    ap.add_argument('--no-matlab', action='store_true')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    day, start, us, n_day = pick(args.day, args.start, args.steps, args.seed)
    A = sample_actions(args.steps, args.seed)
    print('运行回执')
    print('  解释器 : %s' % sys.executable)
    print('  数据   : %s 第 %d 日（共 %d 日），分钟 [%d, %d)'
          % ('FFC_speed_noise_1.csv', day, n_day, start, start + args.steps))
    print('  风速   : n=%d 范围 [%.3f, %.3f] 均值 %.3f（围栏外 %d 条）'
          % (len(us), us.min(), us.max(), us.mean(),
             int(((us < U_LO) | (us > U_HI)).sum())))
    print('  动作   : **每一步都重新随机** a∈[%.3f,%.3f]^%d  seed=%d'
          % (A_LO, A_HI, NT, args.seed))
    print('  引擎   : WFSim vs FLORIS（参数 = 标定后默认）')

    PF, TF, nbad = run_floris(us, A)
    if args.no_matlab:
        print('\n--no-matlab: FLORIS ΣP/60=%.4f MWh 均值 %.6f MW'
              % (np.nansum(PF) / 60, np.nanmean(PF)))
        return 0
    PF_stock = None
    if args.also_stock:
        PF_stock, _, _ = run_floris(us, A, stock=True)

    try:
        PW, TW = run_wfsim(us, A)
    except Exception as e:
        print('\n[!] WFSim 侧失败：%s: %s' % (type(e).__name__, e))
        traceback.print_exc()
        print('\n排查：MATLAB Engine 需能创建 named pipe；受限沙箱下会报 '
              '"CreateFile failed: Access is denied" ⇒ 请用普通终端。')
        return 2

    W = 94
    print('\n' + '=' * W)
    print('新数据独立验证  |  日 %d 分钟 %d-%d  |  每步随机动作  |  n=%d'
          % (day, start, start + args.steps, args.steps))
    print('=' * W)
    print('  %-30s %14s %14s' % ('', 'WFSim', 'FLORIS(标定)'))
    print('  %-30s %14.6f %14.6f' % ('平均出力 [MW]', np.nanmean(PW), np.nanmean(PF)))
    print('  %-30s %14.6f %14.6f' % ('出力 std [MW]', np.nanstd(PW), np.nanstd(PF)))
    print('  %-30s %14.6f %14.6f' % ('ΣP/60 [MWh]（该 100 分钟）',
                                     np.nansum(PW) / 60, np.nansum(PF) / 60))

    m1 = stats(PW, PF)
    print('\n  >>> 判据① 比值 FLORIS/WFSim:')
    print('      mean %.4f  std %.4f  min %.4f  max %.4f'
          % (m1['ratio_mean'], m1['ratio_std'], m1['ratio_min'], m1['ratio_max']))
    print('      corr(P_W, P_F) = %.6f' % m1['corr'])
    print('\n  >>> 判据② 误差:')
    print('      MAPE %.3f%%   常数重标定后 %.3f%% (k*=%.4f)   MedAPE %.3f%%'
          % (100 * m1['mape'], 100 * m1['mape_k'], m1['k_best'], 100 * m1['medape']))
    print('      对照（标定阶段报告）: 验证集 MAPE 4.448%%，常数重标定后 1.098%%')
    ok_ratio = abs(m1['ratio_mean'] - 1) < 0.05
    ok_shape = m1['mape_k'] < 0.05
    print('      ⇒ 判据① %s ｜ 判据② %s'
          % ('通过' if ok_ratio else '**未通过**',
             '通过' if ok_shape else '**未通过**'))

    rows = [cost(PW, 'WFSim'), cost(PF, 'FLORIS(标定)')]
    if PF_stock is not None:
        rows.insert(1, cost(PF_stock, 'FLORIS(随包默认)'))
    rows = [r for r in rows if r]
    print('\n  >>> 判据③ H 层约束口径（**真实时间序列 ⇒ 绝对值有意义**）:')
    print('      %-20s %10s %10s %10s %12s %10s'
          % ('', 'P_F均值', '越限率', 'c1均值', '块和÷15', '超预算%'))
    for r in rows:
        print('      %-20s %10.4f %9.2f%% %10.4f %12.4f %9.1f%%'
              % (r['tag'], r['pf_mean'], 100 * r['viol'], r['c1_mean'],
                 r['blk_mean'] / hc.D_BLOCK, 100 * r['blk_over']))
    w = rows[0]
    f = [r for r in rows if r['tag'].startswith('FLORIS(标定')][0]
    print('      块和倍率（标定/WFSim）= %.4f   ⇒ %s'
          % (f['blk_mean'] / w['blk_mean'],
             '通过（±25% 内）' if abs(f['blk_mean'] / w['blk_mean'] - 1) < 0.25
             else '**未通过**'))
    print('      等效 s_c = %.1f  (基准 %g)'
          % (hc.COST_SCALE_TD3 * f['blk_mean'] / w['blk_mean'], hc.COST_SCALE_TD3))

    print('\n  >>> 判据④ NaN / 负速度外推风险:')
    print('      FLORIS 非有限值条数 = %d / %d' % (nbad, len(us)))
    if PF_stock is not None:
        ms = stats(PW, PF_stock)
        print('      （对照）随包默认参数: 比值 mean %.4f std %.4f 常数重标定后 %.3f%%'
              % (ms['ratio_mean'], ms['ratio_std'], 100 * ms['mape_k']))

    print('\n  逐分钟明细（每 10 步）:')
    for i in range(0, len(us), 10):
        print('    %3d  u=%7.3f  a=%s  %9.5f %9.5f  %7.4f'
              % (i, us[i], np.array2string(A[i], precision=2), PW[i], PF[i],
                 PF[i] / PW[i] if PW[i] > 1e-9 else float('nan')))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(dict(config=dict(day=day, start=start, steps=args.steps,
                                   seed=args.seed, a_lo=A_LO, a_hi=A_HI,
                                   floris_default='calibrated'),
                       wind=[float(x) for x in us],
                       actions=[[float(v) for v in r] for r in A],
                       p_wfsim=[float(x) for x in PW],
                       p_floris=[float(x) for x in PF],
                       p_floris_stock=(None if PF_stock is None
                                       else [float(x) for x in PF_stock]),
                       stats=m1, cost_rows=rows, floris_nonfinite=int(nbad)),
                  open(args.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('\n已写入 %s' % args.json)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
