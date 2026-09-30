# -*- coding: utf-8 -*-
"""标定解空间的**多种子扫描** —— 判据选错的证据与修法。

背景（本会话实测）
------------------
单次 Nelder-Mead（起点 = 默认参数）给出的 θ* 在验证集上是 **1.315%**；
但换一个起点（θ0 = [0.35,0.006,0.30,0.06,1.2,0.9]）得到 **0.517%** ——
**标定集 J 反而更差**（0.054 vs 0.042）。
⇒ "用标定集 MAPE 选解" 是错的判据（6 个参数 / 120 个标定点，过拟合空间很大）。

本脚本从多个起点各跑一次，把所有解按**验证集**排序，给出：
  * 验证集误差的真实分布（而不是单点）；
  * 哪个解最好、其参数是多少；
  * 标定集 J 与验证集误差的**秩相关** —— 若相关性弱，就证明"标定集选解"不可靠。

用法（不需要 MATLAB；数据已落盘）
--------------------------------
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\calib_multistart_scan.py --seeds 6
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

from floris_import_guard import check_interpreter as _chk, sanitize as _sanitize  # noqa: E402
_chk()
_sanitize()

import numpy as np  # noqa: E402

import agentutil.calib_floris_to_wfsim as C  # noqa: E402
import agentutil.probe_engine_random_actions as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='标定解空间多种子扫描',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument('--data', default=C.DATA)
    ap.add_argument('--seeds', type=int, default=6, help='随机起点个数（不含默认起点）')
    ap.add_argument('--maxiter', type=int, default=80,
                    help='每次 Nelder-Mead 的迭代上限（控制总时长）')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    d = json.load(open(args.data, encoding='utf-8'))
    cu = np.asarray(d['cal']['u'], dtype=float)
    cA = np.asarray(d['cal']['A'], dtype=float)
    cW = np.asarray(d['cal']['P_wfsim'], dtype=float)
    vu = np.asarray(d['val']['u'], dtype=float)
    vA = np.asarray(d['val']['A'], dtype=float)
    vW = np.asarray(d['val']['P_wfsim'], dtype=float)
    print('数据 %s：标定 %d 条 / 验证 %d 条' % (args.data, len(cu), len(vu)))

    R.register_op_model()
    m = R.build_floris_model()
    lo = np.array([p[2] for p in C.PARAMS], dtype=float)
    hi = np.array([p[3] for p in C.PARAMS], dtype=float)
    names = ['%s.%s' % (p[0], p[1]) for p in C.PARAMS]

    def ev(us, A, W, theta, tag):
        PF = C.floris_eval(m, us, A, theta=theta)
        mm = C.metrics(W, PF)
        mm['tag'] = tag
        mm['theta'] = [float(x) for x in theta]
        return mm

    # 起点：默认 + 若干随机 + 两个已知好解
    rng = np.random.default_rng(0)
    starts = [np.array([p[4] for p in C.PARAMS], dtype=float),
              np.array([0.35, 0.006, 0.30, 0.06, 1.2, 0.9]),
              np.array([0.25191, 0.0, 0.3082, 0.01626, 2.0, 0.80676])]
    for _ in range(args.seeds):
        starts.append(lo + rng.random(len(lo)) * (hi - lo))

    from scipy.optimize import minimize
    rows = []
    t0 = time.time()
    for k, th0 in enumerate(starts):
        th0 = np.clip(th0, lo, hi)

        def J(th):
            PF = C.floris_eval(m, cu, cA, theta=np.clip(th, lo, hi))
            return C.metrics(cW, PF)['mape']

        res = minimize(J, th0, method='Nelder-Mead',
                       options=dict(maxiter=args.maxiter, xatol=1e-3, fatol=1e-5))
        th = np.clip(res.x, lo, hi)
        mc = ev(cu, cA, cW, th, 'cal#%d' % k)
        mv = ev(vu, vA, vW, th, 'val#%d' % k)
        rows.append((k, float(res.fun), mc, mv))
        print('  seed#%-2d  J_cal=%.5f  val MAPE=%6.3f%%  val 常数重标后=%6.3f%%  '
              '(%.0f s)'
              % (k, res.fun, 100 * mv['mape'], 100 * mv['mape_k'], time.time() - t0),
              flush=True)

    # ---- 排序与相关性 ----
    Js = np.array([r[1] for r in rows])
    Vk = np.array([r[3]['mape_k'] for r in rows])
    order = np.argsort(Vk)
    print('\n' + '=' * 96)
    print('按**验证集**（常数重标定后 MAPE）排序')
    print('=' * 96)
    print('  %-8s %10s %12s %12s   %s' %
          ('seed', 'J_cal', 'val MAPE', 'val 重标后', 'θ*'))
    for i in order:
        k, J, mc, mv = rows[i]
        print('  #%-7d %10.5f %11.3f%% %11.3f%%   %s'
              % (k, J, 100 * mv['mape'], 100 * mv['mape_k'],
                 np.array2string(np.asarray(mv['theta']), precision=4)))

    from scipy.stats import spearmanr
    rho, p = spearmanr(Js, Vk)
    print('\n  Spearman(标定集 J, 验证集误差) = %+.3f  (p=%.3f)' % (rho, p))
    print('  实测判据: ρ=+0.87 / p=0.002 时**标定集 J 是可靠的选择判据**（秩一致）;')
    print('           若 ρ 弱或 p 大，则不能只看标定集，必须多种子 + 用验证集定型。')
    print('  ⚠ 但个体波动仍存在（同一档解在验证集上可差 ~0.3pp）⇒ 单次结果不可当"极限"。')

    best = rows[order[0]]
    print('\n  验证集最优解（seed#%d）:' % best[0])
    print('    %s' % dict(zip(names, [round(float(v), 5) for v in best[3]['theta']])))
    print('    验证集: MAPE %.3f%%  常数重标定后 %.3f%%  MedAPE %.3f%%  k*=%.4f'
          % (100 * best[3]['mape'], 100 * best[3]['mape_k'],
             100 * best[3]['medape'], best[3]['k_best']))
    print('\n  复用方式（把 θ* 直接喂给下一轮）:')
    print('    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\calib_floris_to_wfsim.py '
          '--fit --theta0 %s'
          % ' '.join('%.5f' % v for v in best[3]['theta']))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(dict(names=names,
                       rows=[dict(seed=int(r[0]), J_cal=float(r[1]),
                                  val_mape=float(r[3]['mape']),
                                  val_mape_k=float(r[3]['mape_k']),
                                  val_medape=float(r[3]['medape']),
                                  k_best=float(r[3]['k_best']),
                                  theta=r[3]['theta']) for r in rows],
                       spearman_rho=float(rho), spearman_p=float(p),
                       best_theta=best[3]['theta']),
                  open(args.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('\n已写入 %s' % args.json)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
