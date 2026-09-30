# -*- coding: utf-8 -*-
"""WFSim(MATLAB) vs FLORIS —— **min 级出力对比**（单次运行，100 步即可）。

统一行为（两边逐项对齐，全部可追溯到源码）
--------------------------------------------
               WFSim (MATLAB)                        FLORIS 4.6.6
  入口         WFSim_port(u, 0, [a]*5, 0)            yawflor_turbine.eval_batch(us, 0)
  布局         单排 5 台 5D, D=100 m (meshing.m)     同 (yawflor_turbine.build_model)
  密度         1.20 (meshing.m Rho)                  1.20 (表 ref_air_density)
  切变         0.12 (meshing.m)                      0.12 (build_model 默认)
  单机口径     CT=4a(1−a)²(=Cp), F=½ρU²·CT/(1−a)     CP_eff=0.95·4a(1−a)², CT=8/9
  轴向因子     a=0.3333 (env 标称值)                 a=1/3 固定（不是动作）
  航向/phi     0 (Control.phi 恒 0)                  0 (yaw_angles 全 0)
  围栏         3.5 ≤ u ≤ 25 (env.py)                 3.5 ≤ u ≤ 25 (env.py / yawflor)

**唯一差异是引擎本身**。已知预期（`yawflor/README.md` §0 实测）：u=10 时 FLORIS 绝对电平
是 WFSim 的 **1.74×**（WFSim 隐含 CT=1.3333，FLORIS 的 CT 上限为 1）。所以本次要看的
**不是"哪个更大"，而是比值稳不稳、形状（corr）一致不一致**。

自己跑（**不需要任何沙箱放宽**，在你的普通终端里执行）
-----------------------------------------------------
    cd "C:\\Users\\SCP Site19\\Desktop\\llm enhanced wsis\\leprrlwsis"
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_engine_minute_compare.py

常用变体：
    ... --steps 100 --day 1            # 用 FFC_speed_noise_1.csv 第 1 日列（默认）
    ... --source ramp --u 4 12         # 用 4→12 m/s 线性爬升（不读 CSV）
    ... --no-matlab                    # 只跑 FLORIS（自检 / 没有 MATLAB 时）
    ... --json output/engine_minute_compare.json

依赖：`matlab.engine`（MATLAB R2025a 已装在 F:\\Matlab）与 `floris 4.6.6`
（装在 F:\\anaconda\\envs\\wsis）。**必须用这个解释器**，系统默认 python 没有 floris。
"""
import argparse
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)                       # 相对数据/输出路径一律以仓库根为基准
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

A_NOMINAL = 0.3333        # env.WindBattery.nominal_action_value（旧环境 5 维轴向因子标称）
NT = 5
U_LO, U_HI = 3.5, 25.0    # env.py::PowerOutput 的围栏


# ===========================================================================
# 1. 风速序列（两边共用，逐位同一组数）
# ===========================================================================
def wind_sequence(n, source, day, u_lo, u_hi):
    if source == 'ramp':
        return np.linspace(u_lo, u_hi, n)
    import pandas as pd
    df = pd.read_csv('FFC_speed_noise_1.csv')
    if day < 1 or day >= df.shape[1]:
        raise SystemExit('--day 必须在 1..%d 之间（第 0 列是行号）' % (df.shape[1] - 1))
    seq = df.iloc[:n, day].to_numpy(dtype=float)
    if seq.size < n:
        raise SystemExit('该列只有 %d 行，不足 %d' % (seq.size, n))
    return seq


# ===========================================================================
# 2. WFSim（MATLAB engine）
# ===========================================================================
def run_wfsim(us, verbose=True):
    import matlab
    import matlab.engine

    if verbose:
        print('[WFSim] 启动 MATLAB 引擎 ...', flush=True)
    t0 = time.time()
    eng = matlab.engine.start_matlab()
    if verbose:
        print('[WFSim] 引擎就绪 (%.1f s)，MATLAB %s' % (time.time() - t0, eng.version()),
              flush=True)
    eng.cd(ROOT, nargout=0)          # 保证 WFSim_port 内 run('WFSim_addpaths.m') 能找到

    pw = np.zeros(len(us), dtype=float)
    tw = np.full((len(us), NT), np.nan, dtype=float)
    a_vec = [A_NOMINAL] * NT
    t0 = time.time()
    for i, u in enumerate(us):
        res = eng.WFSim_port(matlab.double([float(u)]), matlab.double([0.0]),
                             matlab.double([a_vec]), matlab.double([0.0]))
        pw[i] = float(res['out']) / 1e6                      # W → MW
        if i == 0 or (verbose and (i + 1) % 20 == 0):
            print('[WFSim] %3d/%d  u=%7.3f  P=%9.6f MW  (%.1f s)'
                  % (i + 1, len(us), u, pw[i], time.time() - t0), flush=True)
    # 逐台功率单独取（失败不影响主流程）
    try:
        for i, u in enumerate(us):
            res = eng.WFSim_port(matlab.double([float(u)]), matlab.double([0.0]),
                                 matlab.double([a_vec]), matlab.double([0.0]))
            sol = res['test']
            t = sol[0][4]['turbine']
            tw[i] = np.asarray(t['power'], dtype=float).reshape(-1)[:NT] / 1e6
    except Exception as e:
        print('[WFSim] ⚠ 逐台功率回读失败（不影响总出力对比）: %s: %s'
              % (type(e).__name__, e))
    eng.quit()
    if verbose:
        print('[WFSim] 完成，用时 %.1f s' % (time.time() - t0), flush=True)
    return pw, tw


# ===========================================================================
# 3. FLORIS
# ===========================================================================
def run_floris(us, verbose=True):
    from yawflor import yawflor_turbine as YT
    model, info = YT.build_model()
    if verbose:
        print('[FLORIS] %s 就绪，向量化一次算完 %d 条 ...'
              % (info.get('floris_version'), len(us)), flush=True)
    t0 = time.time()
    farm, turb = YT.eval_batch(model, us, np.zeros((len(us), NT)))
    if verbose:
        print('[FLORIS] 完成，用时 %.3f s' % (time.time() - t0), flush=True)
    return np.asarray(farm, dtype=float), np.asarray(turb, dtype=float), info


# ===========================================================================
# 4. 对比
# ===========================================================================
def _stats(d):
    return (float(d.mean()), float(np.percentile(d, 95)), float(d.max()), float(d.std()))


def compare(us, pw, pf, tw, tf, day, source, info):
    m = (us >= U_LO) & (us <= U_HI)
    d = pf - pw
    rel = np.divide(d, pw, out=np.full_like(d, np.nan), where=np.abs(pw) > 1e-9)
    big = m & (pw > 0.5)                       # 只在大功率点上算比值，避免小分母噪声
    ratio = np.divide(pf, pw, out=np.full_like(pf, np.nan), where=pw > 0.5)

    W = 100
    print('\n' + '=' * W)
    print('min 级出力对比  |  风速源=%s%s  n=%d  |  a≡%.4f  γ≡0  |  FLORIS %s'
          % (source, (' day=%d' % day) if source == 'real' else '',
             len(us), A_NOMINAL, info.get('floris_version')))
    print('=' * W)
    print('  %-26s %14s %14s %10s' % ('', 'WFSim', 'FLORIS', 'FL/WS'))
    print('  %-26s %14.4f %14.4f %10.4f'
          % ('全天电量 [MWh] (ΣP/60)', pw.sum() / 60, pf.sum() / 60,
             pf.sum() / max(pw.sum(), 1e-9)))
    print('  %-26s %14.6f %14.6f %10.4f'
          % ('平均出力 [MW]', pw.mean(), pf.mean(), pf.mean() / max(pw.mean(), 1e-9)))
    print('  %-26s %14.6f %14.6f %10.4f'
          % ('出力标准差 [MW]', pw.std(), pf.std(), pf.std() / max(pw.std(), 1e-9)))
    print('  %-26s %14.4f %14.4f %10.4f'
          % ('峰谷差 [MW]', pw.max() - pw.min(), pf.max() - pf.min(),
             (pf.max() - pf.min()) / max(pw.max() - pw.min(), 1e-9)))
    print('  %-26s %14.6f %14.6f %10s'
          % ('平均电平差 Δ (F−W) [MW]', np.nan, d[m].mean(), ''))
    print('  %-26s %14s %13.3f%% %10s' % ('平均相对差 mean(Δ/P)', '', 100 * np.nanmean(rel), ''))
    print('  %-26s %14s %13.3f%% %10s' % ('中位相对差 median(Δ/P)', '', 100 * np.nanmedian(rel), ''))

    print('\n  >>> 比值 FLORIS/WFSim（仅 P_WFSim>0.5 MW 的点, n=%d）为本次核心指标：' % big.sum())
    if big.sum() > 0:
        print('      最小 %.4f | 中位 %.4f | 均值 %.4f | 最大 %.4f | 标准差 %.4f'
              % (np.nanmin(ratio[big]), np.nanmedian(ratio[big]), np.nanmean(ratio[big]),
                 np.nanmax(ratio[big]), np.nanstd(ratio[big])))
    else:
        print('      （没有功率 >0.5 MW 的点，跳过）')

    dw, df = np.abs(np.diff(pw)), np.abs(np.diff(pf))
    sw, sf = _stats(dw), _stats(df)
    print('\n  逐分钟波动 |ΔP| [MW]:')
    print('  %-26s %14s %14s %10s' % ('', 'WFSim', 'FLORIS', 'FL/WS'))
    for k, nm in enumerate(('均值', 'p95', 'max', '标准差')):
        print('  %-26s %14.4f %14.4f %10.4f'
              % ('  ' + nm, sw[k], sf[k], sf[k] / max(sw[k], 1e-12)))

    rho = k = None
    if big.sum() > 5:
        rho = float(np.corrcoef(pw[big], pf[big])[0, 1])
        k = float(np.dot(pw[big], pf[big]) / np.dot(pw[big], pw[big]))
        resid = pf[big] - k * pw[big]
        rms = float(np.sqrt((resid ** 2).mean()))
        print('\n  形状一致性（把握平掉后比形状）:')
        print('    corr(P_WFSim, P_FLORIS)      = %.6f' % rho)
        print('    最优比例 k                   = %.6f   (min ||P_F − k·P_W||)'
              % k)
        print('    归一化后残差 RMS             = %.4f MW  (占 FLORIS 均值 %.3f%%)'
              % (rms, 100 * rms / pf[big].mean()))

    if np.isfinite(tw).any() and np.isfinite(tf).any():
        print('\n  逐台平均出力 [MW]（上游 → 下游）:')
        mw, mf = np.nanmean(tw, axis=0), np.nanmean(tf, axis=0)
        print('    WFSim  : %s' % np.array2string(mw, precision=4))
        print('    FLORIS : %s' % np.array2string(mf, precision=4))
        print('    比值   : %s' % np.array2string(mf / mw, precision=4))

    print('\n  逐分钟明细（min, u, P_WFSim, P_FLORIS, Δ, 比值）:')
    for i in range(len(us)):
        if i < 12 or i % 10 == 0 or i == len(us) - 1:
            r = pf[i] / pw[i] if pw[i] > 1e-9 else float('nan')
            print('  %4d  u=%7.3f  %10.6f  %10.6f  %+10.6f  %8.4f'
                  % (i, us[i], pw[i], pf[i], pf[i] - pw[i], r))

    return dict(
        n=int(len(us)), source=source, day=(day if source == 'real' else None),
        a_nominal=A_NOMINAL, floris_version=info.get('floris_version'),
        e_wfsim_mwh=float(pw.sum() / 60), e_floris_mwh=float(pf.sum() / 60),
        mean_wfsim_mw=float(pw.mean()), mean_floris_mw=float(pf.mean()),
        std_wfsim_mw=float(pw.std()), std_floris_mw=float(pf.std()),
        mean_rel_diff=float(np.nanmean(rel)) if np.isfinite(np.nanmean(rel)) else None,
        median_rel_diff=(float(np.nanmedian(rel))
                         if np.isfinite(np.nanmedian(rel)) else None),
        ratio_min=float(np.nanmin(ratio[big])) if big.sum() else None,
        ratio_median=float(np.nanmedian(ratio[big])) if big.sum() else None,
        ratio_mean=float(np.nanmean(ratio[big])) if big.sum() else None,
        ratio_max=float(np.nanmax(ratio[big])) if big.sum() else None,
        ratio_std=float(np.nanstd(ratio[big])) if big.sum() else None,
        corr=rho, k_best=k,
        dp_wfsim_mean=sw[0], dp_floris_mean=sf[0],
        dp_wfsim_max=sw[2], dp_floris_max=sf[2],
    )


# ===========================================================================
# 5. 主流程
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description='WFSim vs FLORIS min 级出力对比',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--steps', type=int, default=100, help='对比的分钟数（默认 100）')
    ap.add_argument('--source', choices=['real', 'ramp'], default='real',
                    help='real=FFC_speed_noise_1.csv；ramp=线性爬升')
    ap.add_argument('--day', type=int, default=1,
                    help='real 模式用哪一列（列 = 日；第 0 列是行号，默认 1）')
    ap.add_argument('--u', type=float, nargs=2, default=[4.0, 12.0],
                    metavar=('U_LO', 'U_HI'), help='ramp 模式的起止风速（默认 4 12）')
    ap.add_argument('--no-matlab', action='store_true', help='只跑 FLORIS')
    ap.add_argument('--json', default=None, help='把风速/逐分钟出力/汇总写 JSON')
    args = ap.parse_args()

    us = wind_sequence(args.steps, args.source, args.day, args.u[0], args.u[1])
    print('运行回执')
    print('  解释器      : %s' % sys.executable)
    print('  工作目录    : %s' % os.getcwd())
    print('  风速源      : %s%s' % (args.source,
                                    ('  (FFC_speed_noise_1.csv 第 %d 列)' % args.day)
                                    if args.source == 'real' else
                                    '  (%.1f → %.1f m/s)' % (args.u[0], args.u[1])))
    print('  风速        : n=%d  范围 [%.3f, %.3f]  均值 %.3f  (围栏 %.1f~%.1f 内 %d 条)'
          % (len(us), us.min(), us.max(), us.mean(), U_LO, U_HI,
             int(((us >= U_LO) & (us <= U_HI)).sum())))
    print('  统一行为    : 单排 5 台 5D | A_NOMINAL≡%.4f | γ≡0 | ρ=1.20 | shear=0.12' % A_NOMINAL)

    pf, tf, info = run_floris(us)
    if args.no_matlab:
        print('\n--no-matlab：只输出 FLORIS 侧。')
        print('  ΣP/60 = %.4f MWh | 均值 %.6f MW | 前 8 条 = %s'
              % (pf.sum() / 60, pf.mean(), np.array2string(pf[:8], precision=6)))
        if args.json:
            _dump_json(args.json, us, None, pf, None, tf, info, args)
        return 0

    try:
        pw, tw = run_wfsim(us)
    except Exception as e:
        print('\n[!] WFSim 侧失败：%s: %s' % (type(e).__name__, e))
        traceback.print_exc()
        print('\n排查提示：')
        print('  1) MATLAB Engine 需要能创建 named pipe。若在受限沙箱/服务账号下运行，'
              '会报 "CreateFile failed: Access is denied" ⇒ 请在普通终端里跑本脚本。')
        print('  2) 确认 MATLAB 已安装且与 Python 版本匹配（本机 F:\\Matlab R2025a + '
              'F:\\anaconda\\envs\\wsis 的 matlabengine）。')
        print('  3) 可先用 --no-matlab 确认 FLORIS 侧正常。')
        return 2

    res = compare(us, pw, pf, tw, tf, args.day, args.source, info)
    if args.json:
        _dump_json(args.json, us, pw, pf, tw, tf, info, args, res)
    return 0


def _dump_json(path, us, pw, pf, tw, tf, info, args, res=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    obj = dict(
        config=dict(steps=args.steps, source=args.source, day=args.day,
                    a_nominal=A_NOMINAL, nt=NT, floris=info.get('floris_version'),
                    interpreter=sys.executable),
        wind=[float(x) for x in us],
        p_floris=[float(x) for x in pf],
        summary=res)
    if pw is not None:
        obj['p_wfsim'] = [float(x) for x in pw]
    if tw is not None and np.isfinite(tw).any():
        obj['p_wfsim_turbine'] = [[None if not np.isfinite(v) else float(v) for v in row]
                                  for row in tw]
    obj['p_floris_turbine'] = [[float(v) for v in row] for row in tf]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print('\nJSON 已写入 %s' % path)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
