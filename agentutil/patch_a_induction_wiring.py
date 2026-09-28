# -*- coding: utf-8 -*-
"""把 5 维轴向因子**逐台**接进 WFSim 的一次性补丁（默认 dry-run，不写盘）。

问题
----
`bin/core/meshing.m` 的 windTunnel 分支：
    'CT_prime', zeros(NT,1) + Control.c_prime,      % NT=5, Control.c_prime 是 1×5 行
    'a',        zeros(NT,1) + Control.a
`NT×1 + 1×K` 在 MATLAB 里**隐式扩展**成 `NT×K` 矩阵；而
`bin/core/Actuator.m:14` 取 `input = Wp.turbine.input(sol.k)`（标量 struct，字段是矩阵），
`:55 CT(kk) = input.CT_prime(kk)`、`:61 a(kk) = input.a(kk)` 都是**线性索引**，
`kk=1..NT` 全部落在**第一列** ⇒ 每台机组都拿到 `c_prime(Control.a(1))`。
即 H 的 dim0 变成"全体机组共用的一个诱导因子"，**dim1~4 对 env 完全无效**。

实测（`agentutil/probe_a_per_turbine.py`，u=10, 其余 0.20）:
    a2=0.50 / a4=0.50 两组的 CT_prime 与逐台 power 与 all-0.20 **逐位相同**;
    ramp 0.1..0.5 只取 0.324 = c_p(0.1) 并复制给 5 台。

修法（单点，覆盖所有读 Control 的分支）
---------------------------------------
1. `WFSim_port.m`：把动作截到 NT 并**转成列向量**，使 `zeros(NT,1)+Control.a`
   成为逐元素运算。第 6 维（SOC 信号）本来就不该进风机动作，顺带丢掉。
2. `env.py::PowerOutput`：clip 前先截到 `self.num_turbine`，防止多传的维再次触发
   隐式扩展（防御性，和 1 是同一件事的两半）。

不采用"改 meshing.m"的理由：它有 4 处同样模式（`:79-81/:195-197/:243-245/:284-286`）
且另有 2 处（`:35-38/:159-162`）干脆硬写 `CT_prime = 2*ones(NT,1)` 完全不看 Control；
改 `WFSim_port.m` 一处即可让所有"会读 Control"的分支都正确。

⚠ **必须先停掉正在训练的 run**：MATLAB 会按文件时间戳重载函数，训练中改 `*.m`
有可能把运行中的实验从中间改掉。

验证（改完按顺序跑）
--------------------
    python agentutil\\probe_a_per_turbine.py            # 逐台 CT_prime 应各不相同
    python agentutil\\probe_a_power_sweep.py --dim 2 --hysteresis

用法
----
    python agentutil\\patch_a_induction_wiring.py            # dry-run, 打印 diff
    python agentutil\\patch_a_induction_wiring.py --apply    # 写盘
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

# ---- 1. WFSim_port.m -------------------------------------------------------
M_OLD = """Control.a = a;%degree
c_p = 4*Control.a.*(1-Control.a).^2;
Control.c_prime = c_p; %C_trust 
"""
M_NEW = """% ---- 逐台轴向因子接线 (原实现只让 dim0 生效) ----
% `zeros(NT,1)+Control.c_prime` 是 NT×K 隐式扩展, 而 Actuator.m:55/:61 用
% `input.CT_prime(kk)` / `input.a(kk)` 线性索引只取**第一列** ⇒ K>1 时每台机组
% 都拿到 c_prime(a(1))。这里截到 NT 并转成 **列向量**, 使加法成为逐元素运算。
% 证据: agentutil/probe_a_per_turbine.py (a2/a4 组的逐台 CT/power 与 all-0.20 逐位相同)。
NT = 5;                     % windTunnel 场景机组数 (meshing.m:187 `Crx` 长度)
if numel(a) < NT
    error('WFSim_port: 需要 %d 个轴向因子, 收到 %d 个', NT, numel(a));
end
Control.a = a(1:NT);
Control.a = Control.a(:);   % NT×1, 关键: 列向量才能逐台生效
Control.c_prime = 4*Control.a.*(1-Control.a).^2;   % C_p = 4a(1-a)^2 (沿用原口径)
"""

# ---- 2. env.py::PowerOutput ------------------------------------------------
E_OLD = """        if 3.5 <= u0 <= 25:
            act_wind = np.clip(act_wind, 0., 0.5)
            act_wind = act_wind.astype(float)
"""
E_NEW = """        if 3.5 <= u0 <= 25:
            # 只取前 num_turbine 维: H 动作是 6 维 (5 台 + SOC 信号), 多传的维会经
            # `zeros(NT,1)+Control.c_prime` 隐式扩展成 NT×K 矩阵, 而 Actuator.m
            # 的线性索引只取第一列 ⇒ 只有 dim0 生效 (见 agentutil/probe_a_per_turbine.py)。
            act_wind = np.asarray(act_wind, dtype=float).reshape(-1)[:self.num_turbine]
            act_wind = np.clip(act_wind, 0., 0.5)
"""

PATCHES = [('WFSim_port.m', M_OLD, M_NEW), ('env.py', E_OLD, E_NEW)]


def main():
    ap = argparse.ArgumentParser(description='把逐台轴向因子接进 WFSim (默认 dry-run)',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true', help='真正写盘 (默认只打印)')
    args = ap.parse_args()

    plans = []
    for path, old, new in PATCHES:
        if not os.path.exists(path):
            raise SystemExit('找不到 %s (请在仓库根运行)' % path)
        raw = open(path, 'rb').read()
        nl = '\r\n' if raw.count(b'\r\n') * 2 >= raw.count(b'\n') else '\n'
        with open(path, encoding='utf-8', errors='replace') as f:
            src = f.read()
        n = src.count(old)
        if n != 1:
            raise SystemExit('%s: 锚点命中 %d 次 (要求恰好 1 次) ⇒ 文件已被改过或版本不符, 中止'
                             % (path, n))
        plans.append((path, src.replace(old, new), nl))

    for path, out, _nl in plans:
        old_lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
        new_lines = out.splitlines()
        print('\n' + '=' * 90)
        print('=== %s: %d 行 -> %d 行 ===' % (path, len(old_lines), len(new_lines)))
        print('=' * 90)
        import difflib
        for ln in difflib.unified_diff(old_lines, new_lines, lineterm='',
                                       fromfile=path + ' (现在)', tofile=path + ' (改后)'):
            print(ln)

    if not args.apply:
        print('\n[dry-run] 未写盘。确认后加 --apply；**务必先停掉正在训练的 run** ——')
        print('         MATLAB 会按时间戳重载 *.m, 训练中改文件可能改掉运行中的实验。')
        return
    for path, out, nl in plans:
        with open(path, 'w', encoding='utf-8', newline=nl) as f:
            f.write(out)
        print('已写入 %s (换行 %s)' % (path, 'CRLF' if nl == '\r\n' else 'LF'))
    print('\n下一步验证: python agentutil\\probe_a_per_turbine.py  (逐台 CT_prime 应各不相同)')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
