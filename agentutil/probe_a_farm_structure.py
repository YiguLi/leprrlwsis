# -*- coding: utf-8 -*-
r"""WFSim 风场总功率的**动作结构**（集体水平 / 空间斜坡 / 逐台边际）—— 回答"这个 5 维动作空间好不好学、先验对不对"。

为什么要它
----------
2026-09-28 修掉"5 维轴向因子只有 dim0 生效"的接线后，动作空间第一次**真正逐台**作用于风场。
而当时动作侧的**先验**都是在故障态下定下的：`hsrl_warmup.py` 的 warmup 带
`[0.20, 0.45]`、`init_wt_action_bias(U(0.25, 0.40))`，理由统一是"该带内单机
`C_p = 4a(1−a)²` 保持 86%~100% 最优"。

问题在于 **单机 C_p 最优 ≠ 风场总功率最优**：上游机组降载可以把风留给下游（尾流亏损）。
本脚本用 WFSim 直接量四件事（下面 `[uniform]` 的输出**直接用于定 `WT_START/WT_LO/WT_HI`
与 `init_wt_action_bias`**，2026-09-28 已据此把先验改为 `0.20 / [0.15, 0.25] / U(0.17,0.23)`）：

  `[uniform]`  共同水平 `a` 从 0.00 扫到 0.50 ⇒ 风场总功率的**集体最优水平**落在哪。
  `[ramp]`     上游低 / 下游高的"降载斜坡"若干条 + 其**镜像** ⇒ 空间结构重不重要、方向对不对。
  `[marginal]` 全体固定在基线上，只抬/压**某一台** ±0.10 ⇒ 该维的**宽区间割线**。
               ⚠ 这是 0.10↔0.30 的割线而非局部斜率：价值面在 0.20 附近**强凹**，
                 割线被曲率主导（近峰的维会得到 ≈0 的割线），故只用来判"峰在哪一侧"。
  `[opt]`      基线 all 0.20，**逐维细扫** 0.10~0.35 ⇒ 每一台各自的**最优水平** a*_d。
               这决定 warmup 带 / init bias 该放在哪（两者都是"动作先验"，与网络结构无关）。

⚠ **需要 MATLAB**（与 main 相同: 只 `start_matlab()`）。**不改任何项目文件**。
⚠ 训练正在跑时不要改 `*.m`（MATLAB 按时间戳重载函数文件）。

用法
----
    python agentutil/probe_a_farm_structure.py                 # 四臂全跑 (u=10)
    python agentutil/probe_a_farm_structure.py --u 8 12        # 换风速
    python agentutil/probe_a_farm_structure.py --arms uniform opt
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

import hsrl_warmup as hw      # 单一真源: 先验带以它为准, 避免探针里的数字过期

NT = 5
BASE = 0.20            # 基线动作（= 实测峰位，也是 WT_START）
UNIFORM_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1.0 / 3, 0.40, 0.45, 0.50]
FINE_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

RAMPS = [
    ('greedy 1/3 全体',          [1 / 3] * 5),
    ('降载斜坡 缓 [.05 .10 .20 .33 .40]', [0.05, 0.10, 0.20, 0.33, 0.40]),
    ('降载斜坡 陡 [.00 .05 .15 .33 .45]', [0.00, 0.05, 0.15, 0.33, 0.45]),
    ('升载斜坡 镜像 [.40 .33 .20 .10 .05]', [0.40, 0.33, 0.20, 0.10, 0.05]),
    ('只降上游 [.05 .05 .20 .20 .20]',     [0.05, 0.05, 0.20, 0.20, 0.20]),
    ('只降上游 更强 [.00 .00 .20 .20 .20]', [0.00, 0.00, 0.20, 0.20, 0.20]),
]

# 只经生产入口；把逐台 CT/power 留在 base workspace 供回读
PORT_SCRIPT = r"""
po  = WFSim_port($U$, 0, [$A$], 0);
sol = po.test;
res = [sol{1,5}.turbine.CT_prime'; sol{1,5}.turbine.power'];
"""


def _fmt(v, nd=6):
    return '[' + ' '.join(('%.' + str(nd) + 'f') % x for x in np.ravel(v)) + ']'


def _run(eng, u, a):
    """经生产入口跑一次，返回 (逐台 CT, 逐台 power, 总功率 MW)。"""
    code = PORT_SCRIPT.replace('$U$', '%.6f' % u).replace('$A$', ' '.join('%.6f' % v for v in a))
    eng.eval(code, nargout=0)
    arr = np.array(eng.eval('res', nargout=1)).astype(float).reshape(2, NT)
    return arr[0], arr[1], float(arr[1].sum()) / 1e6


def main():
    ap = argparse.ArgumentParser(description='WFSim 风场总功率的动作结构 (需 MATLAB)',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--u', nargs='+', type=float, default=[10.0], help='风速列表 (默认 10)')
    ap.add_argument('--arms', nargs='+', default=['uniform', 'ramp', 'marginal', 'opt'],
                    choices=['uniform', 'ramp', 'marginal', 'opt'],
                    help='要跑的臂 (默认全跑)')
    args = ap.parse_args()

    import matlab
    import matlab.engine

    print('启动 MATLAB 引擎 (只 start_matlab())...', flush=True)
    eng = matlab.engine.start_matlab()
    print('引擎就绪。NT = %d，基线 a = %.2f\n' % (NT, BASE), flush=True)

    for u in args.u:
        print('#' * 100)
        print('#  u = %.1f m/s' % u)
        print('#' * 100)
        ct0, pw0, tot0 = _run(eng, u, [BASE] * NT)
        print('基线 all %.2f: CT=%s' % (BASE, _fmt(ct0, 6)))
        print('              逐台功率(MW)=%s  总=%.6f MW' % (_fmt(pw0 / 1e6, 4), tot0))
        print('              ↑ 上游 vs 下游落差 %.2fx ⇒ 尾流亏损量级\n'
              % (pw0[0] / pw0[-1]), flush=True)

        ct_g, pw_g, tot_greedy = _run(eng, u, [1.0 / 3] * NT)   # Cp 单机最优点（全体）
        print('参考: 全体 a=1/3 (单机 Cp 最优): 总=%.6f MW (vs 基线 %+.6f)'
              % (tot_greedy, tot_greedy - tot0))
        print('      ↑ 若基线(0.20) 反而更高, 说明"单机 Cp 最优"不是风场最优\n', flush=True)

        best_u, best_tot = None, None
        if 'uniform' in args.arms:
            print('=' * 100)
            print('[uniform] 共同水平 a 扫描 —— 风场总功率的集体最优在哪')
            print('=' * 100)
            rows = []
            for a in UNIFORM_GRID:
                ct, pw, tot = _run(eng, u, [a] * NT)
                rows.append((a, tot))
                flag = ''
                if best_tot is None or tot > best_tot:
                    best_u, best_tot, flag = a, tot, '  ← 目前最好'
                print('  a=%.4f  逐台功率=%s  总=%.6f MW  单机 Cp=%.4f%s'
                      % (a, _fmt(pw / 1e6, 4), tot, 4 * a * (1 - a) ** 2, flag), flush=True)
            print('\n  ⇒ 集体最优水平 a* = %.4f (%.6f MW); 基线 %.2f = %.6f MW; '
                  '单机 Cp 最优 1/3 = %.6f MW'
                  % (best_u, best_tot, BASE, tot0, tot_greedy))
            print('     当前 warmup 带为 [%.2f, %.2f] (WT_START=%.2f) ⇒ %s'
                  % (hw.WT_LO, hw.WT_HI, hw.WT_START,
                     'a* 在带内' if hw.WT_LO <= best_u <= hw.WT_HI else
                     '**a* 在带外 ⇒ 先验与实测峰位不一致**'))
            # ---- 平台宽度：决定 warmup 带 [WT_LO, WT_HI] 该放哪（网格量化到 UNIFORM_GRID 步长）
            an = np.array([a for a, _ in rows])
            pn = np.array([t for _, t in rows])
            ip = int(np.argmax(pn))
            print('\n  [平台] 峰 %.6f MW @ a=%.2f；"总功率 ≥ 峰的 (1−tol)"的连续区间:'
                  % (pn[ip], an[ip]))
            for tol in (0.01, 0.02, 0.05, 0.10):
                lo = hi = ip
                while lo - 1 >= 0 and pn[lo - 1] >= pn[ip] * (1 - tol):
                    lo -= 1
                while hi + 1 < len(pn) and pn[hi + 1] >= pn[ip] * (1 - tol):
                    hi += 1
                print('         tol=%2d%% ⇒ a ∈ [%.2f, %.2f]  (带宽 %.2f)'
                      % (tol * 100, an[lo], an[hi], an[hi] - an[lo]))
            print('         注: 网格步长 %.2f, 故上下界是量化值' % (UNIFORM_GRID[1] - UNIFORM_GRID[0]))
            print()

        if 'ramp' in args.arms:
            print('=' * 100)
            print('[ramp] 空间结构（上游降载 / 下游保载）—— 5 维是否真的"每台不同"')
            print('=' * 100)
            for name, a in RAMPS:
                ct, pw, tot = _run(eng, u, a)
                print('  %-34s a=%s' % (name, _fmt(a, 3)))
                print('  %-34s 逐台功率=%s' % ('', _fmt(pw / 1e6, 4)))
                print('  %-34s 总=%.6f MW  (vs 基线 %+.6f, vs 全体1/3 %+.6f)'
                      % ('', tot, tot - tot0, tot - tot_greedy), flush=True)
            print()

        if 'marginal' in args.arms:
            print('=' * 100)
            print('[marginal] 逐维**宽区间割线**（基线 all %.2f，每次只动一台 ±0.10）' % BASE)
            print('           ⚠ 0.10↔0.30 的割线被**曲率**主导，不是局部梯度：近峰的维割线≈0')
            print('=' * 100)
            print('  %-6s %-14s %-14s %-14s' % ('机组', 'a_d=0.10', 'a_d=0.30', '割线 ΔP/0.20'))
            slopes, peaks = [], []
            for d in range(NT):
                a_dn = [BASE] * NT
                a_up = [BASE] * NT
                a_dn[d] = BASE - 0.10
                a_up[d] = BASE + 0.10
                t_dn = _run(eng, u, a_dn)[2]
                t_up = _run(eng, u, a_up)[2]
                slope = (t_up - t_dn) / 0.20
                slopes.append(slope)
                peaks.append(max(t_dn, t_up) < tot0)      # 两侧都不如基线 ⇒ 基线是内点极大
                print('  dim%-3d %-14.6f %-14.6f %+.4f   %s'
                      % (d, t_dn, t_up, slope,
                         '峰在 0.10~0.30 内' if peaks[-1] else '单侧仍能超过基线'), flush=True)
            s = np.array(slopes)
            print('\n  峰值在 0.10~0.30 内的维数: %d/%d'
                  % (sum(peaks), NT))
            print('  割线量级: %s ⇒ 最大/最小 = %.2fx （近峰的维割线≈0, 是曲率不是"梯度小"）'
                  % (_fmt(np.abs(s), 4), np.abs(s).max() / max(np.abs(s).min(), 1e-9)))
            print()

        if 'opt' in args.arms:
            print('=' * 100)
            print('[opt] 逐维细扫 %.2f~%.2f（其余维固定在基线 %.2f）—— 每台各自的最优水平'
                  % (FINE_GRID[0], FINE_GRID[-1], BASE))
            print('=' * 100)
            best_a, best_tot = [], []
            for d in range(NT):
                row = []
                for a in FINE_GRID:
                    act = [BASE] * NT
                    act[d] = a
                    row.append((a, _run(eng, u, act)[2]))
                bi = int(np.argmax([t for _, t in row]))
                best_a.append(row[bi][0])
                best_tot.append(row[bi][1])
                peaks = ' '.join('%s%.2f:%.4f' % ('*' if i == bi else '', a, t)
                                 for i, (a, t) in enumerate(row))
                print('  dim%-3d %s  ⇒ a*_%d=%.2f (%.6f MW)' % (d, peaks, d, row[bi][0], row[bi][1]))
            a_comb = list(best_a)
            tot_comb = _run(eng, u, a_comb)[2]
            print('\n  逐维最优 a* = %s ⇒ 拼接 %.6f MW '
                  '(vs 基线 %.2f = %.6f, vs 1/3 全体 = %.6f)'
                  % (_fmt(a_comb, 3), tot_comb, BASE, tot0, tot_greedy))
            print('  各维最优的**离散度**: min=%.2f max=%.2f 极差=%.2f'
                  % (min(best_a), max(best_a), max(best_a) - min(best_a)))
            print('  ⚠ 若拼接值**低于**各维单独最优的期望, 说明维间存在**竞争**(共享同一股风),')
            print('    ⇒ 最优解不是把 5 个独立最优点拼起来, 网络必须学**维间相互作用**。')
            print()

    eng.quit()
    print('=' * 100)
    print('读法:')
    print('  * [uniform] 集体最优水平 a* 与**当前**先验 (warmup 带 [%.2f,%.2f] / init bias '
          'U(0.17,0.23), 由 hsrl_warmup 单一真源读出) 的位置'
          % (hw.WT_LO, hw.WT_HI))
    print('    关系: 若 a* 落在带外、或带内大部分区域明显更差 ⇒ **常量先验**错位,')
    print('    要改的是 hsrl_warmup.py 的 WT_START/WT_LO/WT_HI 与 init_wt_action_bias 的 (lo,hi), 不是网络。')
    print('  * 2026-09-28 已据此把先验从 [0.20,0.45]/U(0.25,0.40) 改为 0.20/[0.15,0.25]/U(0.17,0.23):')
    print('    u=6/8/10/12/14 的集体峰**全在 a=0.20**, 峰功率精确 ∝ u³ ⇒ 最优动作与风速无关。')
    print('  * [ramp] 若"降载斜坡"并不优于全体 1/3 ⇒ 最优动作**不**需要空间结构,')
    print('    动作空间虽 5 维但最优解接近常向量 ⇒ 比"5 个独立自由度"**更好学**。')
    print('  * [marginal] 割线被曲率主导, 只用来判峰在哪一侧; 真正的峰位看 [opt]。')
    print('  * [opt] 各维 a*_d 越接近 ⇒ 越接近"1 个自由度 + 微调", 网络越容易学;')
    print('    拼接值低于单独最优越多 ⇒ 维间竞争越强, 越依赖 critic 学相互作用(容量足够)。')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
