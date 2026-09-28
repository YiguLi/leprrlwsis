# -*- coding: utf-8 -*-
"""直接扫 WFSim: 风场功率 vs 单维轴向因子 a_j —— 判定"推到 a=0.5"对不对。

为什么要它
----------
`act_wind` 的一维偏效应从**日志回归**估出来是 **正的** (+2.4 MW/单位, run15),
但按 MATLAB 侧公式这是**不可能**的:

* `WFSim_port.m:10-11`  `c_prime = 4*a.*(1-a).^2`;
* `bin/core/Actuator.m:55`  `CT(kk) = input.CT_prime(kk)`  ← 用的是上面那个 **C_p**;
* `bin/core/Actuator.m:69`  `Power = powerscale*.5*Rho*Ar*CT*mean(Ue.^3)`
      ⇒ **单机功率 ∝ 4a(1−a)², 峰在 a=1/3, a=0.5 只有 84%**;
* `bin/core/Actuator.m:64`  `Fthrust = F*1/2*Rho*Ue^2*CT/(1-a)` = ½ρUe²·**4a(1−a)**
      ⇒ **推力在 a=0.5 取最大** (C_t→1, 动量理论失效 ⇒ 这就是 0.5 这个物理上界);
* `bin/core/Actuator.m:53`  `Ue = cos(phi)*U` ⇒ 有效风速与 a 无关。

⇒ a_j 升到 0.5 是双输: 自己 Cp 掉 16% + 推力变大让下游尾流亏损更深。
日志回归的 +2.4 只能来自 (a) WFSim 的**历史依赖**(见 probe_wfsim_memoryless.py)
或 (b) 布局侧向加速(阻塞效应)。两者都只有直接扫仿真才能定。

本脚本
------
固定其余四维 (默认 run15 的日志均值, 可用 --others 覆盖), 只扫一维, 每个
(u, a_j) 调一次 `WFSim_port`, 打印风场总功率、局部斜率、峰值位置;
并**倒序再扫一遍**做迟滞检验 (`--hysteresis`): 正反两遍差异 > 1% ⇒ 有历史依赖,
那时日志回归的偏效应就不可信。

⚠ **需要 MATLAB**(引擎启动与各 main 相同: 只 `start_matlab()`, 无 addpath;
`WFSim_port.m` 自己会 `run('WFSim_addpaths.m')`)。

用法
----
    python agentutil/probe_a_power_sweep.py --dim 2
    python agentutil/probe_a_power_sweep.py --dim 2 --u 8 10 12 --others 0.21 0.34 0.23 0.27
    python agentutil/probe_a_power_sweep.py --dim 2 --grid 0 0.5 0.05 --hysteresis
"""

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np

# run15 日志里前 5 维 (applied_action_H) 的均值, 作为"其它维固定值"的默认
RUN15_MEANS = [0.206, 0.340, 0.442, 0.228, 0.268]


def log_means(run='run15'):
    p = os.path.join('output', 'TD3PER-' + run, 'df_td3per.csv')
    if not os.path.exists(p):
        return None
    tot = np.zeros(5)
    n = 0
    with open(p, newline='') as f:
        rd = csv.reader(f)
        ix = {c.strip(): i for i, c in enumerate(next(rd))}
        ia = ix.get('applied_action_H')
        if ia is None:
            return None
        for r in rd:
            try:
                v = [float(x) for x in r[ia].strip().strip('[]').split()]
            except (ValueError, IndexError):
                continue
            tot += np.asarray(v[:5])
            n += 1
    return (tot / max(n, 1)).tolist()


def main():
    ap = argparse.ArgumentParser(description='WFSim 直接扫描: 风场功率 vs a_j (需 MATLAB)',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dim', type=int, default=2, help='要扫的维 (0..4, 默认 2)')
    ap.add_argument('--u', nargs='+', type=float, default=[8.0, 10.0, 12.0], help='风速点')
    ap.add_argument('--others', nargs=4, type=float, default=None,
                    help='其余四维固定值 (默认取 run15 日志均值)')
    ap.add_argument('--grid', nargs=3, type=float, default=[0.0, 0.5, 0.05],
                    metavar=('LO', 'HI', 'STEP'), help='扫描网格 (默认 0 0.5 0.05)')
    ap.add_argument('--hysteresis', action='store_true', help='倒序再扫一遍做迟滞检验')
    ap.add_argument('--out', default=None, help='把结果写成 CSV (可选)')
    args = ap.parse_args()

    import matlab
    import matlab.engine

    others = args.others
    if others is None:
        base = log_means()
        if base is None:
            base = RUN15_MEANS
            src = '内置 run15 均值 (日志不可读)'
        else:
            src = 'run15 日志均值'
        others = [base[k] for k in range(5) if k != args.dim]
        print('其余四维固定为 %s (来源: %s)' % (np.round(others, 3).tolist(), src))
    else:
        print('其余四维固定为 %s (来源: --others)' % (np.round(others, 3).tolist()))

    print('启动 MATLAB 引擎 (与 main 相同: 只 start_matlab())...', flush=True)
    eng = matlab.engine.start_matlab()
    print('引擎就绪。', flush=True)

    grid = np.round(np.arange(args.grid[0], args.grid[1] + 1e-9, args.grid[2]), 6)
    if args.hysteresis:
        grid = np.concatenate([grid, grid[::-1]])

    def power(u, a5):
        r = eng.WFSim_port(matlab.double([float(u)]), matlab.double([0.0]),
                           matlab.double([list(map(float, a5))]), matlab.double([0.0]))
        return float(r['out']) / 1e6

    rows = []
    for u in args.u:
        print('\n===== u = %.2f m/s | 扫 dim%d (其余 %s) =====' % (u, args.dim, np.round(others, 3).tolist()))
        print('  %8s %12s %12s' % ('a%d' % args.dim, '风场功率 MW', '局部斜率'))
        prev = None
        for k, v in enumerate(grid):
            a5 = np.array(others, dtype=float)
            a5 = np.insert(a5, args.dim, v)
            p = power(u, a5)
            slope = '' if prev is None else '%+10.3f' % ((p - prev[1]) / (v - prev[0]))
            rows.append((u, float(v), p, k >= len(grid) // 2 if args.hysteresis else False))
            print('  %8.3f %12.4f %12s' % (v, p, slope))
            prev = (v, p)
        # 峰值与 a=1/3 对比
        arr = np.array([(r[1], r[2]) for r in rows if r[0] == u])
        pk = arr[np.argmax(arr[:, 1])]
        i13 = int(np.argmin(np.abs(arr[:, 0] - 1.0 / 3.0)))
        print('  ⇒ 峰值 a=%.3f (%.4f MW) | a=1/3 处 %.4f MW | a=0.5 处 %.4f MW | 差(峰−1/3) = %+.4f MW'
              % (pk[0], pk[1], arr[i13, 1], arr[-1, 1], pk[1] - arr[i13, 1]))
        if args.hysteresis:
            n = len(grid) // 2
            up = arr[:n]
            dn = arr[n:][::-1]
            d = np.abs(up[:, 1] - dn[:, 1]).max()
            print('  ⇒ 迟滞检验: 正/反扫最大差 = %.4f MW (%.2f%%) ⇒ %s'
                  % (d, 100.0 * d / max(up[:, 1].mean(), 1e-9),
                     '有历史依赖, 日志回归不可信' if d > 0.01 * up[:, 1].mean() else '可视为无记忆'))

    if args.out:
        with open(args.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['u', 'a%d' % args.dim, 'power_MW'])
            for r in rows:
                w.writerow(['%.4f' % r[0], '%.4f' % r[1], '%.6f' % r[2]])
        print('\n已写出 %s' % args.out)
    print('\n读法: 单机功率 ∝ 4a(1−a)² 峰在 a=1/3 且推力 ∝ 4a(1−a) 在 0.5 最大 ⇒')
    print('      若总功率峰值就在 1/3 附近并在 0.5 处下降, 则"策略推到 0.5"与真 env **矛盾**;')
    print('      若总功率到头仍在升, 才说明该维角点是真的 (且与 Actuator.m 公式冲突, 需查历史依赖)。')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
