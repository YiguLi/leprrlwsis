# -*- coding: utf-8 -*-
"""**用 WFSim 标定 FLORIS 的尾流参数** —— 让"快"的引擎尽量贴近"准"的引擎。

问题
----
`output/engine_random_actions.json`（LHS n=100）实测：固定参数下两引擎的比值
**不是常数**（std 0.1388、极差 1.517），常数重标定后残差 **10.58%**。
原因是两引擎的尾流模型结构不同（WFSim 是 2D 不可压流场 + `forcescale=1.5` 的动量汇
+ 4D 宽封闭风洞；FLORIS 是无界高斯解析尾流）。

做法
----
不追求"结构相同"，只做**参数标定**：把 FLORIS 尾流模型的可调参数拟合到 WFSim 的输出上。
标定参数（都在 `build_model` 之后用 `set_param` 改，不改任何生产文件）：

  * `wake.velocity_model.ka` / `kb`   —— 尾流增长率 `k = ka·TI + kb`（主控"亏损伤得多快"）
  * `wake.velocity_model.alpha` / `beta` —— 近尾流/远尾流过渡
  * `wake.turbulence_model.ceps_alpha` / `ceps_beta` —— 湍流模型（`aI` 通道的强度）

目标函数（对**相对误差**，因为尾流误差本质是乘性的）：
    J(θ) = mean |P_F(θ) − P_W| / P_W     （同时报告"再乘一个常数后"的残差）

流程
----
1. **标定集**：LHS 采样 `(u, a₁…a₅)`，逐条调 WFSim（MATLAB）—— 这是唯一慢的部分；
2. **验证集**：另一组独立 LHS 采样，同样调 WFSim；
3. 用 `scipy.optimize.minimize`（Nelder-Mead，无梯度）在标定集上最小化 J；
4. 在验证集上报告 **标定前 / 标定后** 的误差与比值散布 —— 这就是"接近了多少"的答案。

⚠ 标定集与验证集必须**分开**，否则报出来的误差是自欺。

用法（普通终端；MATLAB Engine 需 named pipe）
--------------------------------------------
    # 先只生成数据（最慢的一步，可反复复用）
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\calib_floris_to_wfsim.py --gen --n-cal 120 --n-val 60

    # 再在已有数据上做标定（很快，可反复试不同参数组）
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\calib_floris_to_wfsim.py --fit

    # 一步到底
    ... --gen --fit
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

import numpy as np  # noqa: E402

import yawflor.yawflor_turbine as YT  # noqa: E402

NT = 5
A_LO, A_HI = 0.025, 0.5
U_LO, U_HI = 3.5, 25.0
DATA = os.path.join('output', 'calib_floris_to_wfsim.json')

# 标定参数：(FLORIS 参数路径关键字, 名字, 下界, 上界, 初值)
# 路径实测（`agentutil` 里核过）：
#   gauss 尾流参数在 `wake.wake_velocity_parameters['gauss']`
#     {'alpha':0.58,'beta':0.077,'ka':0.38,'kb':0.004}
#   crespo_hernandez 湍流参数在 `wake.wake_turbulence_parameters['crespo_hernandez']`
#     {'initial':0.1,'constant':0.5,'ai':0.8,'downstream':-0.32}
#   （⚠ 函数签名里的 `ceps_alpha` 不是参数键名；参数键是 `alpha`。这是踩过的坑。）
PARAMS = [
    ('velocity', 'ka',         0.050, 1.200, 0.380),
    ('velocity', 'kb',         0.000, 0.050, 0.004),
    ('velocity', 'alpha',      0.100, 1.500, 0.580),
    ('velocity', 'beta',       0.010, 0.300, 0.077),
    ('turbulence', 'ai',       0.050, 2.000, 0.800),
    ('turbulence', 'constant', 0.050, 1.500, 0.500),
]


# ===========================================================================
# 1. 采样（固定种子；两引擎共用）
# ===========================================================================
def sample(n, seed):
    from scipy.stats import qmc
    A = A_LO + qmc.LatinHypercube(d=NT, seed=seed).random(n) * (A_HI - A_LO)
    u = np.random.default_rng(seed + 999).uniform(U_LO, U_HI, n)
    return u, A


# ===========================================================================
# 2. 两个引擎
# ===========================================================================
def _apply_theta(m, theta):
    """把标定参数写进 FLORIS 的 wake 参数表（不改任何生产文件）。"""
    vm = m.core.wake.model_strings['velocity_model']
    tm = m.core.wake.model_strings['turbulence_model']
    for (kind, name, _lo, _hi, _d), v in zip(PARAMS, theta):
        path = (['wake', 'wake_velocity_parameters', vm, name] if kind == 'velocity'
                else ['wake', 'wake_turbulence_parameters', tm, name])
        m.set_param(path, float(v))


def floris_eval(m, us, A, theta=None):
    """给一批 (u, a₁..a₅) 算 FLORIS 风场功率 [MW]。"""
    import agentutil.probe_engine_random_actions as R
    n = len(us)
    if theta is not None:
        _apply_theta(m, theta)
    R._A_SET = np.asarray(A, dtype=float)
    m.set(wind_speeds=list(np.asarray(us, dtype=float)),
          wind_directions=[YT.WIND_DIR_ALIGNED] * n,
          turbulence_intensities=[YT.TI] * n, yaw_angles=np.zeros((n, NT)))
    with R._solver_hook()():
        m.run()
    return np.asarray(m.get_turbine_powers(), dtype=float).sum(axis=1) / 1e6


def wfsim_eval(us, A, verbose=True):
    import matlab
    import matlab.engine
    if verbose:
        print('[WFSim] 启动 MATLAB 引擎 ...', flush=True)
    t0 = time.time()
    eng = matlab.engine.start_matlab()
    if verbose:
        print('[WFSim] 就绪 (%.1f s)' % (time.time() - t0), flush=True)
    eng.cd(ROOT, nargout=0)
    P = np.zeros(len(us), dtype=float)
    t0 = time.time()
    for i in range(len(us)):
        res = eng.WFSim_port(matlab.double([float(us[i])]), matlab.double([0.0]),
                             matlab.double([list(map(float, A[i]))]), matlab.double([0.0]))
        P[i] = float(res['out']) / 1e6
        if verbose and ((i + 1) % 20 == 0 or i == 0):
            print('[WFSim] %3d/%d (%.1f s)' % (i + 1, len(us), time.time() - t0),
                  flush=True)
    eng.quit()
    if verbose:
        print('[WFSim] 完成，用时 %.1f s' % (time.time() - t0), flush=True)
    return P


# ===========================================================================
# 3. 误差口径
# ===========================================================================
def metrics(PW, PF):
    ok = np.isfinite(PW) & np.isfinite(PF) & (PW > 1e-9)
    r = PF[ok] / PW[ok]
    rel = np.abs(PF[ok] - PW[ok]) / PW[ok]
    k = float(np.dot(PW[ok], PF[ok]) / np.dot(PW[ok], PW[ok]))
    rel_k = np.abs(k * PW[ok] - PW[ok]) / PW[ok]           # 只按常数重标定后的相对差
    resid_k = np.abs(PF[ok] - k * PW[ok]) / PF[ok].mean()   # 形状残差（相对 FLORIS 均值）
    return dict(n=int(ok.sum()),
                ratio_mean=float(r.mean()), ratio_std=float(r.std()),
                ratio_min=float(r.min()), ratio_max=float(r.max()),
                mape=float(rel.mean()), mape_k=float(rel_k.mean()),
                k_best=k, resid_k=float(resid_k.mean()))


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description='用 WFSim 标定 FLORIS 尾流参数',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument('--gen', action='store_true', help='生成/追加标定与验证数据（调 MATLAB）')
    ap.add_argument('--fit', action='store_true', help='在已有数据上做标定')
    ap.add_argument('--n-cal', type=int, default=120)
    ap.add_argument('--n-val', type=int, default=60)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--json', default=None, help='把标定结果另存')
    args = ap.parse_args()
    if not (args.gen or args.fit):
        args.gen = args.fit = True

    data = {}
    if os.path.exists(args.data):
        data = json.load(open(args.data, encoding='utf-8'))
        print('已有数据: %s（cal %d / val %d）'
              % (args.data, len(data.get('cal', {}).get('u', [])),
                 len(data.get('val', {}).get('u', []))))

    # ---------- 生成数据 ----------
    if args.gen:
        need = {}
        for tag, n, sd in (('cal', args.n_cal, args.seed),
                           ('val', args.n_val, args.seed + 5000)):
            if tag in data and len(data[tag]['u']) >= n:
                continue
            u, A = sample(n, sd)
            need[tag] = (u, A)
        if need:
            tot = sum(len(v[0]) for v in need.values())
            print('需要新跑 %d 条 WFSim（cal+val）' % tot)
            try:
                for tag, (u, A) in need.items():
                    print('\n--- 采样集 %s：%d 条 ---' % (tag, len(u)))
                    P = wfsim_eval(u, A)
                    data[tag] = dict(u=[float(x) for x in u],
                                     A=[[float(v) for v in r] for r in A],
                                     P_wfsim=[float(x) for x in P])
                    with open(args.data, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=1)
                    print('已写入 %s' % args.data)
            except Exception as e:
                print('\n[!] WFSim 失败：%s: %s' % (type(e).__name__, e))
                traceback.print_exc()
                print('\n排查：MATLAB Engine 需建 named pipe；受限沙箱下报 '
                      '"CreateFile failed: Access is denied" ⇒ 请用普通终端。')
                return 2

    if not args.fit:
        return 0
    if 'cal' not in data or 'val' not in data:
        raise SystemExit('缺少 cal/val 数据，先跑 --gen')

    # ---------- 载入 ----------
    import agentutil.probe_engine_random_actions as R
    R.register_op_model()
    m = R.build_floris_model()
    cal_u = np.asarray(data['cal']['u'], dtype=float)
    cal_A = np.asarray(data['cal']['A'], dtype=float)
    cal_W = np.asarray(data['cal']['P_wfsim'], dtype=float)
    val_u = np.asarray(data['val']['u'], dtype=float)
    val_A = np.asarray(data['val']['A'], dtype=float)
    val_W = np.asarray(data['val']['P_wfsim'], dtype=float)

    theta0 = np.array([p[4] for p in PARAMS], dtype=float)
    lo = np.array([p[2] for p in PARAMS], dtype=float)
    hi = np.array([p[3] for p in PARAMS], dtype=float)

    def J(theta):
        th = np.clip(theta, lo, hi)
        PF = floris_eval(m, cal_u, cal_A, theta=th)
        j = metrics(cal_W, PF)['mape']
        print('    J=%.5f  θ=%s' % (j, np.array2string(th, precision=4)), flush=True)
        return j

    # ---------- 标定前 ----------
    print('\n' + '=' * 96)
    print('标定前（FLORIS 随包默认）: %s'
          % dict(zip(['%s.%s' % (p[0], p[1]) for p in PARAMS],
                     [round(float(v), 5) for v in theta0])))
    print('=' * 96)
    PFv0 = floris_eval(m, val_u, val_A, theta=theta0)
    m0 = metrics(val_W, PFv0)
    _print_metrics('验证集', m0)

    # ---------- 优化 ----------
    print('\n' + '=' * 96)
    print('Nelder-Mead 标定（目标 = 标定集上的相对误差 MAPE）')
    print('=' * 96)
    from scipy.optimize import minimize
    t0 = time.time()
    res = minimize(J, theta0, method='Nelder-Mead',
                   options=dict(maxiter=120, xatol=1e-3, fatol=1e-5, disp=False))
    theta = np.clip(res.x, lo, hi)
    print('标定完成，用时 %.1f s，%d 次目标函数评估'
          % (time.time() - t0, int(res.nfev)))
    print('  θ* = %s' % dict(zip(['%s.%s' % (p[0], p[1]) for p in PARAMS],
                                 [round(float(v), 5) for v in theta])))
    print('  标定集 J: %.5f → %.5f' % (J(theta0), res.fun))

    # ---------- 标定后 ----------
    print('\n' + '=' * 96)
    print('标定后（在**独立验证集**上评估）')
    print('=' * 96)
    PFv1 = floris_eval(m, val_u, val_A, theta=theta)
    m1 = metrics(val_W, PFv1)
    _print_metrics('验证集', m1)

    print('\n' + '=' * 96)
    print('对照（验证集）')
    print('=' * 96)
    print('  %-24s %12s %12s %10s' % ('', 'MAPE', '常数重标后', '比值std'))
    print('  %-24s %11.3f%% %11.3f%% %10.4f'
          % ('标定前（默认参数）', 100 * m0['mape'], 100 * m0['mape_k'], m0['ratio_std']))
    print('  %-24s %11.3f%% %11.3f%% %10.4f'
          % ('标定后', 100 * m1['mape'], 100 * m1['mape_k'], m1['ratio_std']))
    print('  %-24s %11s %11.3f%% %10.4f'
          % ('（参考）随机动作基线', '—', 100 * 0.1058, 0.1388))
    print('\n  ⇒ "常数重标后"那一列是真正要看的：它衡量**形状**还剩多少差异。')

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(dict(params=[p[0] for p in PARAMS],
                       theta0=[float(x) for x in theta0],
                       theta_star=[float(x) for x in theta],
                       before=m0, after=m1,
                       nfev=int(res.nfev)),
                  open(args.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('\n标定结果已写入 %s' % args.json)
    return 0


def _print_metrics(tag, mm):
    print('  %s n=%d' % (tag, mm['n']))
    print('    比值 mean %.4f  std %.4f  min %.4f  max %.4f'
          % (mm['ratio_mean'], mm['ratio_std'], mm['ratio_min'], mm['ratio_max']))
    print('    MAPE（相对误差均值）        %.3f%%' % (100 * mm['mape']))
    print('    最优常数重标定后 MAPE       %.3f%%   (k*=%.4f)'
          % (100 * mm['mape_k'], mm['k_best']))
    print('    形状残差（相对 FLORIS 均值）%.3f%%' % (100 * mm['resid_k']))


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
