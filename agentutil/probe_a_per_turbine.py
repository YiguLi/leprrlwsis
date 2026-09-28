# -*- coding: utf-8 -*-
"""让 MATLAB 打印**逐台机组的 CT_prime / power** —— 直接看 5 个动作维有没有接进模型。

为什么需要它
------------
`bin/core/meshing.m` 的 windTunnel 分支：
    'CT_prime', zeros(NT,1) + Control.c_prime,   % NT=5, Control.c_prime 是 1×5
    'a',        zeros(NT,1) + Control.a
`NT×1 + 1×5` 在 MATLAB (R2016b+) 里做**隐式扩展**成 **NT×5 矩阵**；
而 `bin/core/Actuator.m:14` 取 `input = Wp.turbine.input(sol.k)`（标量 struct，字段是矩阵），
`:55 CT(kk) = input.CT_prime(kk)` 与 `:61 a(kk) = input.a(kk)` 都是**线性索引**，
kk=1..5 全部落在**第一列** ⇒ **每台机组都拿到 `c_prime(Control.a(1))`**，
即只有 H 的 **dim0** 生效，dim1~4 与 env 无关。

本脚本对若干组动作打印逐台 CT/power：
  * 若不同动作组下逐台 CT **完全相同** ⇒ 上述机制成立（只有 dim0 生效）；
  * 若逐台 CT 各不相同 ⇒ 动作维正常接入了，问题在别处。

⚠ **需要 MATLAB**（与 main 相同: 只 `start_matlab()`）。用 `eng.eval` 直接跑 `WFSim`，
不 import 项目模块、**不修改任何项目文件**。
⚠ 训练正在跑时**不要**改 `*.m`：MATLAB 会按时间戳重载函数文件，可能影响运行中的实验。

用法
----
    python agentutil/probe_a_per_turbine.py
    python agentutil/probe_a_per_turbine.py --u 12
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

TEMPLATE = r"""
run('WFSim_addpaths.m');
Wind.wind_speed = $U$; Wind.wind_angle = 0;
Control.phi = 0;
Control.a = [$A$];
Control.c_prime = 4*Control.a.*(1-Control.a).^2;
sol = WFSim(Wind, Control);
fprintf('    a_in     = %s\n', mat2str(Control.a, 4));
fprintf('    CT_prime = %s\n', mat2str(sol{1,5}.turbine.CT_prime, 5));
fprintf('    power    = %s\n', mat2str(sol{1,5}.turbine.power, 7));
fprintf('    total    = %.6f MW\n', sum(sol{1,5}.turbine.power(:))/1e6);
"""

VARIANTS = [
    ('all 0.20',            [0.20] * 5),
    ('all 0.33',            [0.33] * 5),
    ('a0=0.50  others 0.20', [0.50, 0.20, 0.20, 0.20, 0.20]),
    ('a2=0.50  others 0.20', [0.20, 0.20, 0.50, 0.20, 0.20]),
    ('a4=0.50  others 0.20', [0.20, 0.20, 0.20, 0.20, 0.50]),
    ('ramp 0.1..0.5',       [0.10, 0.20, 0.30, 0.40, 0.50]),
]


def main():
    ap = argparse.ArgumentParser(description='逐台机组 CT/power 读出 (确认动作维是否接入)',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--u', type=float, default=10.0, help='风速 (默认 10 m/s)')
    args = ap.parse_args()

    import matlab.engine

    print('启动 MATLAB 引擎 (只 start_matlab())...', flush=True)
    eng = matlab.engine.start_matlab()
    print('引擎就绪。逐组打印 (CT_prime 应按机组各不相同才算接对了):\n', flush=True)
    for name, a in VARIANTS:
        print('--- %s ---' % name, flush=True)
        code = TEMPLATE.replace('$U$', '%.4f' % args.u).replace('$A$', ' '.join('%.4f' % v for v in a))
        try:
            eng.eval(code, nargout=0)
        except Exception as e:                        # noqa: BLE001
            print('    !! MATLAB 报错: %s' % e, flush=True)
    print('\n读法: 若各组 CT_prime 都是 5 个相同值 (且只随 a0 变) ⇒ 只有 dim0 生效;')
    print('      若 CT_prime 逐台不同且随对应维变 ⇒ 动作维正常接入。')
    eng.quit()


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
