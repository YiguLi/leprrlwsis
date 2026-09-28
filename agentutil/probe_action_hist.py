# -*- coding: utf-8 -*-
"""动作分布探针: 逐维直方图 + "堆两端"签名 + 整排翻面检验。

为什么需要它
------------
"tanh 一上线就两边倒、推到顶"与"最优解在约束边界上(角点解)"是同一个问题的
两种说法。要区分"真角点"和"critic 界外偏好", `probe_q_profile_boundary.py`
看的是 **Q(a_j) 剖面**; 本脚本看的是 **策略自己的动作分布**, 提供三个可证伪签名:

  (1) **两端堆积**: 动作质量堆在 box 两端, 中间稀薄 => 端部倍率 > 1
      (端部倍率 = 两端各 5% 宽度内的质量 / 均匀分布应有的 0.10)。
  (2) **端点饱和**: 越界占比与"贴边"(|a - bound| <= rail_eps)占比同时偏高。
  (3) **整排翻面**: 同一状态上多维也同时贴边(共享 trunk 被一起推出去), 表现为
      "贴边维数"分布偏向 >= 2, 且各维贴边指示量**互相关为正**。这是
      "两边倒"的机制签名 —— 单维各自独立贴边时相关≈0。

读数口径
--------
* 面板状态与 H 状态构造**复用 probe_q_profile_boundary**(同一口径, 不启 MATLAB)。
* 一律统计**确定性策略输出 A0 = actor(X)**(eval, 无探索噪声): 否则 `action_amp`
  的探索噪声自己就会把质量堆到两端, 看不出策略本身形状。
* 贴边判据 `|a - bound| <= --rail_eps`(默认 0.02); 直方图 bin 均分 box,
  越界质量单独画成 `<lo` / `>hi` 两根柱。

用法
----
    python agentutil/probe_action_hist.py run15
    python agentutil/probe_action_hist.py run15 --steps 30000 34000 38000 --dims 2
    python agentutil/probe_action_hist.py run13 --steps 40000 --penalty-coef 200
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np
import torch

from probe_q_profile_boundary import (DEFAULT_BOUNDS, build_panel, ckpt_steps,
                                      load_nets, run_dir)


def main():
    ap = argparse.ArgumentParser(description='动作分布/两端堆积/整排翻面 探针',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='run 名（自动补 output/TD3PER-<name>）或目录')
    ap.add_argument('--steps', nargs='*', type=int, default=None, help='checkpoint（默认最后一个）')
    ap.add_argument('--dims', nargs='*', type=int, default=None, help='要看的维（默认 0..5）')
    ap.add_argument('--panel', type=int, default=1500, help='面板状态数（默认 1500）')
    ap.add_argument('--every', type=int, default=20, help='读日志抽样步长（默认 20）')
    ap.add_argument('--bins', type=int, default=20, help='box 内直方图 bin 数（默认 20）')
    ap.add_argument('--rail-eps', type=float, default=0.02, help='贴边判据（默认 0.02）')
    args = ap.parse_args()

    dims = args.dims if args.dims else list(range(6))
    eps = args.rail_eps

    for name in args.runs:
        d = run_dir(name)
        steps = args.steps if args.steps else [ckpt_steps(d)[-1]]
        X, perr = build_panel(d, args.panel, args.every)
        X = torch.tensor(X, dtype=torch.float32)
        print('\n' + '=' * 112)
        print('=== %s: 面板 %d 状态 (相位自校验 %.1e) ===' % (os.path.basename(d.rstrip('/\\')), len(X), perr))
        print('=' * 112)

        for st in steps:
            if not os.path.exists(os.path.join(d, 'checkpoint_%d' % st, 'actor_6.pkl')):
                print('  checkpoint_%d 不存在, 跳过' % st)
                continue
            act, _ = load_nets(d, st, 'p')
            with torch.no_grad():
                A = act(X).numpy()

            print('\n--- checkpoint %d | 确定性策略 %d 样本 (eval, 无探索噪声) ---' % (st, len(A)))
            print('  %-5s %8s %7s %8s %8s %8s %9s %9s %8s %8s %9s   %s'
                  % ('dim', 'mean', 'std', 'p05', 'p50', 'p95', '越界下%', '越界上%', '贴下%', '贴上%', '端部倍率', '边界'))
            rail_ind = np.zeros((len(A), len(dims)), dtype=bool)
            for k, j in enumerate(dims):
                lo, hi = DEFAULT_BOUNDS[j]
                a = A[:, j]
                below, above = a < lo, a > hi
                w = hi - lo
                nb = max(4, args.bins)
                step_w = w / nb
                end_mass = float(((a >= lo) & (a < lo + 0.05 * w)).mean()
                                 + ((a > hi - 0.05 * w) & (a <= hi)).mean())
                near_lo, near_hi = np.abs(a - lo) <= eps, np.abs(a - hi) <= eps
                rail_ind[:, k] = near_lo | near_hi
                print('  dim%-2d %+8.3f %7.3f %+8.3f %+8.3f %+8.3f %9.3f %9.3f %8.3f %8.3f %9.2f   [%.1f, %.1f]'
                      % (j, a.mean(), a.std(), np.percentile(a, 5), np.percentile(a, 50),
                         np.percentile(a, 95), float(below.mean()), float(above.mean()),
                         float(near_lo.mean()), float(near_hi.mean()),
                         end_mass / 0.10, lo, hi))
                _ = step_w

            # ---- 逐维直方图: 单行密度串 (box 内 nb 段, 数字 = 9*cnt/max) ----
            nb = max(4, args.bins)
            print('\n  直方图 (%d 段均分 box, 数字越大质量越多; 左越界/右越界柱单独给):' % nb)
            for j in dims:
                lo, hi = DEFAULT_BOUNDS[j]
                a = A[:, j]
                cnt, _e = np.histogram(a, bins=np.linspace(lo, hi, nb + 1))
                n_lo, n_hi = int((a < lo).sum()), int((a > hi).sum())
                n_max = max(int(cnt.max()), n_lo, n_hi, 1)
                dens = ''.join(str(int(round(9.0 * c / n_max))) for c in cnt)
                print('    dim%d %+.1f..%+.1f  <lo:%-4d %s >hi:%-4d' % (j, lo, hi, n_lo, dens, n_hi))

            # ---- 整排翻面签名 ----
            nrail = rail_ind.sum(axis=1)
            dist = ' '.join('%d:%.1f%%' % (m, 100.0 * float((nrail == m).mean())) for m in range(len(dims) + 1))
            print('\n  贴边维数分布 (|a-bound|<=%.2f): %s' % (eps, dist))
            multi = float((nrail >= 2).mean())
            if len(dims) >= 2:
                Ri = rail_ind.astype(float)
                with np.errstate(invalid='ignore', divide='ignore'):
                    C = np.corrcoef(Ri.T)
                    off = C[~np.eye(len(dims), dtype=bool)]
                    r_rail = float(np.nanmean(off))
                    C2 = np.corrcoef(A[:, dims].T)
                    off2 = C2[~np.eye(len(dims), dtype=bool)]
                    r_act = float(np.nanmean(off2))
            else:
                r_rail = r_act = float('nan')
            s_rail = 'n/a' if not np.isfinite(r_rail) else '%+.3f' % r_rail
            s_act = 'n/a' if not np.isfinite(r_act) else '%+.3f' % r_act
            print('  多贴边(>=2 维同时贴边)占比 = %.1f%% | 贴边指示量平均相关 = %s | 原始动作平均相关 = %s'
                  % (100.0 * multi, s_rail, s_act))
            print('  读法: 端部倍率>1.5 且 多贴边占比高 且 贴边相关>0 => "堆两端 + 整排翻面" 签名成立;')
            print('        配合 probe_q_profile_boundary 的峰位: 峰在界上=真角点(良性), 峰在界外=critic 偏好界外(病)。')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
