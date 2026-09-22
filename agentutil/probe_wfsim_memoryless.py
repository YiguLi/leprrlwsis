#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`WFSim_port` 是否**无记忆** —— 决定 test9 能否当确定性收敛判据。

背景
----
test9（`nine_case_test`）在 Python 侧已核实为零随机性：
  * 动作噪声被 `if self.is_training:` 关掉（`nine_case_test` 设 `is_training=False`）；
  * 场景 `build_9case_scenario(seed=42)` 固定种子；
  * `nine_case_test` 内无任何 `random` 调用；网络只有 Linear+ReLU，`cuda=False`；
  * `env_test` 的 Python 状态被 `ScenarioReset` 完整重置。

**唯一剩余的不确定源**：`env_test` 与训练环境 `env` **共用同一个 MATLAB 引擎**
（`td3prrbb_main.py:83` 只 `start_matlab()` 一次，928 行把同一个 engine 传给两个
`WindBattery`）。而 `PowerOutput` 是纯调用：

    target_result = self.engine.WFSim_port(u, 0.0, [act_wind], 0.0)

若 MATLAB 侧的 `WFSim_port` 含 persistent 变量 / 滤波器 / 依赖被训练循环改过的工作区
变量，则 test9 会**路径依赖**（不是随机，但同样会让"平坦"失去意义）。

三节
----
A. 同一组 (u, act) 连续调用 N 次 ⇒ 是否逐位相同（无记忆的前提）。
B. 穿插若干不同的 (u, act) 后再回到原参数 ⇒ 是否仍等于第一次的值（无历史依赖）。
C. **两个 `WindBattery` 实例共享同一 engine，来回穿插** ⇒ 各自是否仍等于单独调用时的值。
   这是真实架构（训练 env 与评测 env 共引擎）的决定性检验。

用法: python agentutil/probe_wfsim_memoryless.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # WindBattery 读 FFC_*.csv 用相对路径
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                                                  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

print('启动 MATLAB 引擎 (与 main 同样的方式: 只 start_matlab(), 无 addpath)...', flush=True)
t0 = time.time()
import matlab.engine                                                # noqa: E402
engine = matlab.engine.start_matlab()
print('  引擎就绪, 耗时 %.1f s' % (time.time() - t0), flush=True)

from env import WindBattery                                         # noqa: E402

MAX_E, NTURB = 6.0, 5
env_a = WindBattery(MAX_E, NTURB, False, engine)
env_b = WindBattery(MAX_E, NTURB, False, engine)      # 共享 engine 的第二个实例
print('  两个 WindBattery 实例均就绪 (共享同一 engine)\n', flush=True)

U0 = 10.0
A0 = np.array([0.33] * NTURB)
OTHERS = [(8.0, np.array([0.10] * NTURB)),
          (12.0, np.array([0.45] * NTURB)),
          (15.0, np.array([0.25] * NTURB)),
          (6.0, np.array([0.50] * NTURB))]


def call(env, u, a):
    t = time.time()
    v = env.PowerOutput(u, a)
    return float(v), time.time() - t


# ---------------------------------------------------------------- A
print('=' * 92)
print('A. 同一组 (u=%.1f, act=0.33×%d) 连续调用 20 次' % (U0, NTURB))
print('=' * 92)
vals, dts = [], []
for i in range(20):
    v, dt = call(env_a, U0, A0)
    vals.append(v)
    dts.append(dt)
    if i < 5:
        print('   #%2d  %.12f   (%.3f s)' % (i + 1, v, dt), flush=True)
uniq = sorted(set(repr(v) for v in vals))
print('  返回值唯一值个数 = %d  (期望 1)' % len(uniq))
if len(uniq) > 1:
    print('  取值: %s' % uniq[:6])
print('  单次调用耗时 中位 %.3f s / 最大 %.3f s' % (float(np.median(dts)), max(dts)))
A_ok = len(uniq) == 1

# ---------------------------------------------------------------- B
print()
print('=' * 92)
print('B. 穿插 4 组不同 (u, act) 各 3 次, 再回到原参数')
print('=' * 92)
for (u, a) in OTHERS:
    for _ in range(3):
        call(env_a, u, a)
v2, _ = call(env_a, U0, A0)
print('  回到 (%.1f, 0.33) 得到 %.12f' % (U0, v2))
print('  与 A 节首次值 %.12f 相同? %s' % (vals[0], '是' if v2 == vals[0] else '**否**'))
B_ok = (v2 == vals[0])

# ---------------------------------------------------------------- C
print()
print('=' * 92)
print('C. 决定性检验: 两实例共享 engine, 来回穿插 (模拟 训练env 与 评测env 交替调用)')
print('=' * 92)
ref_a, _ = call(env_a, U0, A0)
ref_b, _ = call(env_b, U0, A0)
print('  各自单独调用: env_a=%.12f  env_b=%.12f' % (ref_a, ref_b))
mismatch = 0
for round_ in range(10):
    call(env_a, *OTHERS[round_ % len(OTHERS)])      # 训练侧: 不同参数
    call(env_b, U0, A0)                             # 评测侧: 固定参数
    va, _ = call(env_a, U0, A0)
    if va != ref_a:
        mismatch += 1
        print('   ✗ 第 %d 轮: env_a 回到原参数得到 %.12f != %.12f' % (round_ + 1, va, ref_a))
print('  10 轮穿插后 env_a 偏差次数 = %d' % mismatch)
C_ok = (mismatch == 0)

# ---------------------------------------------------------------- 判读
print()
print('=' * 92)
print('判读')
print('=' * 92)
print('  A 无记忆 (同参数恒定)      : %s' % ('通过' if A_ok else '**失败**'))
print('  B 无历史依赖 (穿插后复原)  : %s' % ('通过' if B_ok else '**失败**'))
print('  C 共享引擎互不干扰         : %s' % ('通过' if C_ok else '**失败**'))
print()
if A_ok and B_ok and C_ok:
    print('  ⇒ WFSim_port 对 (u, act) 是无记忆的确定性映射。')
    print('     test9 全部随机/状态来源已排除 ⇒ 可作为**确定性**收敛判据')
    print('     (相邻两次 test9 的任何差异都只能来自策略权重变化)。')
else:
    print('  ⇒ WFSim_port **有状态或非确定** ⇒ test9 的结果依赖调用历史,')
    print('     "平坦"可能只是路径巧合, **不能**当确定性收敛判据。')
    print('     需要: 在 ScenarioReset 里重置 MATLAB 侧状态, 或每次评测重启引擎,')
    print('     或改为比较"策略输出轨迹"(actor(state) 直接前向) 而不依赖环境回放。')

engine.quit()
print('\n引擎已退出。')
