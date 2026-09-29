#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_floris_call_cost.py —— FLORIS **调用开销的固定项 vs 每条项**（只读，不需 MATLAB）。

为什么需要它
------------
`yawflor/README.md` §0 记的是"单条 40.5 ms；批量 0.58 ms/条 ⇒ **逐块预取后
~0.3 s/episode**（朴素口径 58.3 s/episode）"。P1 实现时实测却给出十几秒量级，
差两个数量级 —— 必须把两种开销分开量才能说清：

* **固定开销**（每次 `f.set(...)+f.run()` 都要付，与条数基本无关）：本机实测
  `n ≤ 12` 时约 **40~50 ms/次**（其中 `set` ~20 ms、`run` ~14 ms）；
* **每条开销**（边际）：`n = 288` 时 ≈ **0.52 ms/条**、`n = 1440` 时 ≈ **0.24 ms/条**
  —— 这就是 README 里"0.58 ms/条"的来源。

⇒ 结论（README 的 0.3 s/episode 是**一次算完 1440 条**那一档，不是"每块一次"）：

| 批量粒度 | 覆盖 1440 步所需次数 | 外推 s/episode |
|---|---|---|
| 1 条/次（朴素：每步一次） | 1440 | ~74 |
| **6 条/次（本项目逐块预取）** | **240** | **~13** |
| 288 条/次 | 5 | ~0.7 |
| 1440 条/次（动作全局固定才可能） | 1 | ~0.30 |

训练路径的动作**每块都变**，所以只能用"每块一次 6 条"这一档 ⇒ 本项目环境的
真实物理调用预算 ≈ **13~16 s/episode**（比朴素口径快 4~6 倍，但**不是** 0.3 s）。
本脚本是这条更正的证据，也是"为什么预取粒度选 6 条"的依据。

运行
----
    python agentutil/probe_floris_call_cost.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, 'output', '_mplcache'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import numpy as np                                          # noqa: E402
from yawflor import yawflor_turbine as YT                   # noqa: E402


def main():
    f, info = YT.build_model()
    z = np.zeros(5)
    rng = np.random.RandomState(0)
    print('FLORIS %s | turbine_grid_points=%s | 5 台 5D 单排 D=100'
          % (info['floris_version'], info['turbine_grid_points']))
    print()
    print('%-8s %12s %12s' % ('n', 'per call ms', 'per row ms'))
    for n in (1, 2, 6, 12, 60, 288, 1440):
        us = rng.uniform(4.0, 20.0, size=n)
        yaws = np.tile(z, (n, 1)) + rng.uniform(-5, 5, size=(n, 5))
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            YT.eval_batch(f, us, yaws)
            ts.append(time.perf_counter() - t0)
        per_call = float(np.median(ts)) * 1e3
        print('%-8d %12.2f %12.4f' % (n, per_call, per_call / n))

    print()
    print('%-10s %16s %16s' % ('批量粒度', '次数/1440 步', 's/episode'))
    for n in (1, 2, 6, 12, 60, 288, 1440):
        us = rng.uniform(4.0, 20.0, size=n)
        yaws = np.tile(z, (n, 1))
        t0 = time.perf_counter()
        YT.eval_batch(f, us, yaws)
        t = time.perf_counter() - t0
        print('%-10d %16.0f %16.2f' % (n, 1440.0 / n, t * 1440.0 / n))
    print()
    print('⇒ 本项目采用"每块一次 6 条"（H 的动作每块都变，无法更大批量）；'
          '实测 ~13~16 s/episode，比"每步一次"快 4~6 倍。')


if __name__ == '__main__':
    main()
