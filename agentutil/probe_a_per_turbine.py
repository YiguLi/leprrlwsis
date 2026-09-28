# -*- coding: utf-8 -*-
r"""让 MATLAB 打印**逐台机组的 CT_prime / power** —— 直接看 5 个动作维有没有接进模型。

背景（机制）
------------
`bin/core/meshing.m` 的 windTunnel 分支：
    'CT_prime', zeros(NT,1) + Control.c_prime,   % NT=5
    'a',        zeros(NT,1) + Control.a
`NT×1 + 1×K` 在 MATLAB (R2016b+) 里做**隐式扩展**成 **NT×K** 矩阵；而
`bin/core/Actuator.m:14` 取 `input = Wp.turbine.input(sol.k)`（标量 struct，字段是矩阵），
`:55 CT(kk) = input.CT_prime(kk)` 与 `:61 a(kk) = input.a(kk)` 都是**线性索引**，
`kk=1..5` 全部落在**第一列** ⇒ 每台机组都拿到 `c_prime(Control.a(1))`。

所以"5 个动作维有没有接进去"**完全取决于入口** `WFSim_port.m` 是否把动作压成
`NT×1` 列向量：压成列 ⇒ `zeros(NT,1)+a` 逐元素 ⇒ 5 台各自生效；留着 1×K 行 ⇒
只有 dim0 生效（这正是 2026-09-28 实测到的故障态）。

四个臂（`--arms`，默认全跑）
---------------------------
  `port`（**生产入口**）`env.py::PowerOutput` 走的就是它：`WFSim_port(u, 0, a, 0)`。
         判定:**逐台 CT 必须按位置响应**——`a_d=0.50` 时只有第 d 台的 CT 变，
         其余 4 台与 all-0.20 基线**逐位相同**；`ramp` 则每台 CT = `c_p(自己的 a)`
         （其中 a=0.20 那台与基线同值，**本就应不变**）。
  `raw` （`meshing.m` 直连，绕过入口）`Control.a` 被直接设成 1×K 行向量。
         这条臂**永远**打印 5 个相同的 CT —— 它展示的是 `meshing.m` 的固有行为，
         用来解释"为什么必须修入口"，**不是**"没修好"的证据（故只打印、不断言）。
  `env` （**整链**）Python 侧 `WindBattery.PowerOutput(u, 6 维动作)` 的返回值必须与
         `WFSim_port(u, 5 维前缀)` **逐位相同**，且第 6 维（SOC 信号）取任何值都不影响
         功率 ⇒ 证明"截到 num_turbine + clip"与下传路径一致、信号维被正确丢弃。
  `guard`（**入口维度守卫**）只接受**恰好 NT 维**：5 维通过，而 4 维（少传）与
         6/10 维（多传）都必须**响亮报错**。多传曾是静默"只取前 NT 个"（实测
         10 维 `[.2]*5+[.5]*5` 与 5 维逐位相同 3.717941 MW）——与"只有 dim0 生效"
         是同一类**静默失效**，只是换了一层，故一并纳入回归。

`port` / `env` / `guard` 三臂带**断言**，失败时退出码 1，故本脚本可直接当回归测试用。

⚠ **需要 MATLAB**（与 main 相同: 只 `start_matlab()`）。
⚠ 训练正在跑时**不要**改 `*.m`：MATLAB 会按时间戳重载函数文件，可能影响运行中的实验。

用法
----
    python agentutil/probe_a_per_turbine.py                     # 四臂全跑
    python agentutil/probe_a_per_turbine.py --arms port         # 只验生产入口
    python agentutil/probe_a_per_turbine.py --u 12 --arms port env guard
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

NT = 5          # windTunnel 场景机组数 (= meshing.m:187 Crx 长度)
MAX_E = 6.0

# (名称, 5 维动作)
VARIANTS = [
    ('all 0.20',             [0.20] * 5),
    ('all 0.33',             [0.33] * 5),
    ('a0=0.50  others 0.20', [0.50, 0.20, 0.20, 0.20, 0.20]),
    ('a2=0.50  others 0.20', [0.20, 0.20, 0.50, 0.20, 0.20]),
    ('a4=0.50  others 0.20', [0.20, 0.20, 0.20, 0.20, 0.50]),
    ('ramp 0.1..0.5',        [0.10, 0.20, 0.30, 0.40, 0.50]),
]

BASE_A = 0.20           # 基线动作（`all 0.20`）


def _cp(a):
    """与 WFSim_port.m 同口径: C_p = 4a(1-a)^2。"""
    return 4.0 * a * (1.0 - a) ** 2

# `raw` 臂：直接把 1×K 行向量塞进 Control（复刻生产前的旧口径，绕过 WFSim_port.m）
RAW_SCRIPT = r"""
run('WFSim_addpaths.m');
Wind.wind_speed = $U$; Wind.wind_angle = 0;
Control.phi = 0;
Control.a = [$A$];                              % 1xK 行向量 = 隐式扩展的触发条件
Control.c_prime = 4*Control.a.*(1-Control.a).^2;
sol = WFSim(Wind, Control);
res = [sol{1,5}.turbine.CT_prime'; sol{1,5}.turbine.power'];
"""

# `port` 臂：只经生产入口
PORT_SCRIPT = r"""
po  = WFSim_port($U$, 0, [$A$], 0);
sol = po.test;
res = [sol{1,5}.turbine.CT_prime'; sol{1,5}.turbine.power'];
"""


def _fmt(v, nd=6):
    return '[' + ' '.join(('%.' + str(nd) + 'f') % x for x in np.ravel(v)) + ']'


def _run_script(eng, script, u, a):
    """跑脚本并回读 base workspace 的 `res`（2×NT: 第一行 CT_prime, 第二行 power）。"""
    code = script.replace('$U$', '%.6f' % u).replace('$A$', ' '.join('%.6f' % v for v in a))
    eng.eval(code, nargout=0)
    arr = np.array(eng.eval('res', nargout=1)).astype(float).reshape(2, NT)
    return arr[0].copy(), arr[1].copy()


def main():
    ap = argparse.ArgumentParser(description='逐台机组 CT/power 读出 (确认 5 个动作维是否接入)',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--u', type=float, default=10.0, help='风速 (默认 10 m/s)')
    ap.add_argument('--arms', nargs='+', default=['port', 'raw', 'env', 'guard'],
                    choices=['port', 'raw', 'env', 'guard'],
                    help='要跑的臂: port=生产入口 / raw=meshing.m 直连 / env=整链 / '
                         'guard=维度守卫 (默认全跑)')
    args = ap.parse_args()

    import matlab
    import matlab.engine

    print('启动 MATLAB 引擎 (只 start_matlab())...', flush=True)
    eng = matlab.engine.start_matlab()
    print('引擎就绪。风速 u = %.2f m/s，NT = %d\n' % (args.u, NT), flush=True)

    failures = []
    ct_of = {}

    # ---------------- port 臂：生产入口 ----------------
    if 'port' in args.arms:
        print('=' * 98)
        print('[port] 生产入口 WFSim_port.m —— env.py::PowerOutput 走的就是它')
        print('=' * 98)
        pw_of = {}
        for name, a in VARIANTS:
            ct, pw = _run_script(eng, PORT_SCRIPT, args.u, a)
            ct_of[name], pw_of[name] = ct, pw
            print('--- %s ---' % name)
            print('    a_passed = %s' % _fmt(a, 4))
            print('    CT_prime = %s   (distinct = %d)'
                  % (_fmt(ct, 6), len(np.unique(np.round(ct, 9)))))
            print('    power    = %s' % _fmt(pw, 6))
            print('    total    = %.6f MW' % (pw.sum() / 1e6), flush=True)

        base_ct, base_total = ct_of['all 0.20'], pw_of['all 0.20'].sum() / 1e6
        print('\n[判定] 按位置响应 —— 逐台 CT 与 all-0.20 基线比对')
        print('       期望变化的机组 = {d : c_p(a_d) != c_p(0.20)}')
        print('       （注意 ramp 的第 1 台 a=0.20 与基线同值, 本来就应"不变"）')
        for name, a in VARIANTS:
            if name == 'all 0.20' or len(set('%.6f' % v for v in a)) == 1:
                continue                        # 均匀动作单列在上面判
            ct = ct_of[name]
            diff = [int(x) for x in np.flatnonzero(~np.isclose(ct, base_ct, rtol=0, atol=1e-9))]
            exp = [i for i in range(NT) if abs(_cp(a[i]) - _cp(BASE_A)) > 1e-9]
            ok = diff == exp
            print('    %-24s 实测变化=%-18s 期望=%-18s %s'
                  % (name, diff, exp, 'OK' if ok else 'FAIL'), flush=True)
            if not ok:
                failures.append('[port/%s] 逐台 CT 变化位置实测 %s != 期望 %s'
                                % (name, diff, exp))
        for name, a in VARIANTS:
            if len(set('%.6f' % v for v in a)) == 1 and name != 'all 0.20':
                if len(np.unique(np.round(ct_of[name], 9))) != 1:
                    failures.append('[port/%s] 均匀动作却得到逐台不同的 CT' % name)
        for name in ('a0=0.50  others 0.20', 'a2=0.50  others 0.20',
                     'a4=0.50  others 0.20', 'ramp 0.1..0.5'):
            d_tot = pw_of[name].sum() / 1e6 - base_total
            print('    %-24s Δtotal vs all-0.20 = %+.6f MW%s'
                  % (name, d_tot, '' if abs(d_tot) > 1e-9 else '   ⇒ FAIL: 与基线逐位相同'))
            if abs(d_tot) < 1e-9:
                failures.append('[port/%s] 总功率与 all-0.20 逐位相同 ⇒ 该维未生效' % name)
        print()

    # ---------------- raw 臂：meshing.m 直连 ----------------
    if 'raw' in args.arms:
        print('=' * 98)
        print('[raw] meshing.m 直连 (绕过入口, Control.a 是 1×K 行) —— 演示固有机制')
        print('      预期: 每组的 CT_prime 都是 5 个**相同**值, 且只随 a(1) 变 (只打印, 不断言)')
        print('=' * 98)
        for name, a in VARIANTS:
            ct, pw = _run_script(eng, RAW_SCRIPT, args.u, a)
            print('--- %s ---' % name)
            print('    a_in     = %s' % _fmt(a, 4))
            print('    CT_prime = %s   (distinct = %d)  ← 全同即为预期'
                  % (_fmt(ct, 6), len(np.unique(np.round(ct, 9)))))
            print('    total    = %.6f MW' % (pw.sum() / 1e6), flush=True)
        print()

    # ---------------- env 臂：整链 ----------------
    if 'env' in args.arms:
        print('=' * 98)
        print('[env] 整链 WindBattery.PowerOutput (6 维动作, 第 6 维=SOC 信号)')
        print('      要求与 WFSim_port(5 维前缀) 逐位相同 ⇒ 截断+下传一致、第 6 维被丢弃')
        print('=' * 98)
        from env import WindBattery
        wb = WindBattery(MAX_E, NT, False, eng)
        for name, a in VARIANTS:
            got = float(wb.PowerOutput(args.u, np.array(list(a) + [0.25], dtype=float)))
            ref = float(eng.WFSim_port(matlab.double([args.u]), matlab.double([0.0]),
                                       matlab.double([list(a)]),
                                       matlab.double([0.0]))['out']) / 1e6
            sig = float(wb.PowerOutput(args.u, np.array(list(a) + [0.0], dtype=float)))
            ok = abs(got - ref) < 1e-12 and abs(got - sig) < 1e-12
            print('%-24s 6维=%.6f MW  5维前缀=%.6f MW  Δ=%.2e  %s'
                  % (name, got, ref, abs(got - ref), 'OK' if ok else 'FAIL'), flush=True)
            if not ok:
                failures.append('[env/%s] PowerOutput(6维)=%.9f != WFSim_port(5维前缀)=%.9f'
                                % (name, got, ref))
        a5 = [0.20] * NT
        p_sig_pos = float(wb.PowerOutput(args.u, np.array(a5 + [0.25])))
        p_sig_neg = float(wb.PowerOutput(args.u, np.array(a5 + [-0.40])))
        if abs(p_sig_pos - p_sig_neg) > 1e-12:
            failures.append('[env] 第 6 维仍在影响风功率 (信号维泄漏进了风机动作)')
            print('\n第 6 维泄漏检查: FAIL (%.9f vs %.9f)' % (p_sig_pos, p_sig_neg))
        else:
            print('\n第 6 维泄漏检查: signal=+0.25 与 -0.40 的功率逐位相同 (%.6f MW) ⇒ 未泄漏'
                  % p_sig_pos)
        print()

    # ---------------- guard 臂：入口的维度守卫必须**对称** ----------------
    if 'guard' in args.arms:
        print('=' * 98)
        print('[guard] 入口维度守卫：只接受**恰好 NT 维**；少传与多传都必须**响亮报错**')
        print('        (多传曾是静默"只取前 NT 个" —— 实测 10 维 [.2]*5+[.5]*5 与 5 维')
        print('         逐位相同 3.717941 MW，与"只有 dim0 生效"是同一类静默失效)')
        print('=' * 98)
        cases = [
            ('恰好 5 维 [0.2]*5',        [0.20] * NT,              True),
            ('6 维 [0.2]*5+[信号]',      [0.20] * NT + [0.25],     False),
            ('4 维 [0.2]*4 (少传)',      [0.20] * (NT - 1),        False),
            ('10 维 [0.2]*10 (多传)',    [0.20] * (2 * NT),        False),
            ('10 维 [0.2]*5+[0.5]*5',    [0.20] * NT + [0.50] * NT, False),
        ]
        for label, a, should_pass in cases:
            try:
                tot = float(eng.WFSim_port(matlab.double([args.u]), matlab.double([0.0]),
                                           matlab.double([list(a)]),
                                           matlab.double([0.0]))['out']) / 1e6
                ok, detail = should_pass, '%.6f MW' % tot
            except Exception as e:                                    # noqa: BLE001
                ok, detail = (not should_pass), str(e).strip().splitlines()[0][:60]
            print('  %-26s 期望 %-8s 实测 %-28s %s'
                  % (label, '接受' if should_pass else '报错', detail, 'OK' if ok else 'FAIL'),
                  flush=True)
            if not ok:
                failures.append('[guard/%s] 期望 %s 但实际相反 ⇒ 守卫失效'
                                % (label, '接受' if should_pass else '报错'))
        print()

    eng.quit()

    print('=' * 98)
    if failures:
        print('复验失败 (%d 项):' % len(failures))
        for f in failures:
            print('  !! %s' % f)
        return 1
    print('复验通过: port 臂逐台 CT **按位置**响应各自的动作维; env 臂 6 维动作 == 5 维前缀;')
    print('       guard 臂维度守卫对称 (少传/多传都报错)。')
    print('       (raw 臂的"CT 全同"是 meshing.m 的固有行为, 与本项判定无关。)')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
