# -*- coding: utf-8 -*-
"""testPg 环境层：**固定风机行为**的 Pmax + 标量降载率 `rate`。

与 `aifflor_env.WindBatteryAIF` 的唯一差异
------------------------------------------
`PowerOutput(u, act)` 不再把 `act` 当 5 台机组的轴向因子，而是：

    a_fixed = [0.2, 0.2, 0.2, 0.2, 0.33]          # 固定（= hsrl_warmup.WT_PRIOR 的 0.33 版）
    Pmax(u) = physics.power(u, a_fixed)            # 每个环境步现算（可按 u 命中预取缓存）
    ptur    = rate * Pmax(u),  rate ∈ [0, 1]       # 由上层 actor 输出

动作分派（**只按长度**）
------------------------
* `len(act) == 2` ⇒ 上层动作 `[rate, goal]`：`goal` 第 2 维在此**不使用**（SOC 目标
  由 myfunc 读），`rate` 夹到 [0,1] 后乘 Pmax。
* `len(act) == 1` ⇒ 只给 rate（契约测试/手工调用方便）。
* `len(act) == num_turbine`（=5）⇒ `env.reset/TestReset/ScenarioReset` 的标称动作路径：
  **rate 恒 1.0**（开局"不降载"，P_G_prev = Pmax）。这是刻意的：reset 时 H 还没决策，
  用"满发"作为波动基准，与 aifflor 用标称动作定 P_G_prev 的位置同形。

为什么 Pmax 可以预取
--------------------
`a_fixed` 是常量 ⇒ `Pmax` 只是 `u` 的函数。上层每个 5 步块才决策一次，块内 `rate`
不变；主循环在块首调用 `prefetch_pmax()`，把 `u_{t+1..t+K}` 一次算完放进
`AifflorPhysics` 的缓存 ⇒ 块内其余步全部命中，块首当前 `u` 由上一步的预取覆盖。
缓存键含动作 ⇒ 与 aifflor 的缓存互不混淆；数值与逐点直算**逐位相同**。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))            # testpg/
_ROOT = os.path.dirname(_HERE)                                # 仓库根
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from aifflor.aifflor_env import WindBatteryAIF                # noqa: E402


#: 固定风机行为（用户口径）。与 `hsrl_warmup.WT_PRIOR` 的差别只在末台的 1/3 写成 0.33。
PG_FIXED_ACTION = np.array([0.2, 0.2, 0.2, 0.2, 0.33], dtype=float)
#: rate 的物理带（上层动作第 1 维）
PG_RATE_MIN, PG_RATE_MAX = 0.0, 1.0
#: 上层动作维数（rate, goal）
PG_N_H_ACTIONS = 2


class WindBatteryPG(WindBatteryAIF):
    """固定风机行为 + 标量降载率的风电场/储能环境（FLORIS 后端）。

    继承 `WindBatteryAIF` 的**全部**非物理语义：储能更新、电价/风速序列、观测布局
    `[pr, u, E, P_G_last]`、`3.5 ≤ u ≤ 25` 围栏、九例场景接口。仅覆写 `PowerOutput`。
    """

    FIXED_ACTION = PG_FIXED_ACTION
    N_H_ACTIONS = PG_N_H_ACTIONS

    def __init__(self, max_E, num_turbine, upload, engine=None, physics=None):
        super().__init__(max_E, num_turbine, upload, engine, physics=physics)
        if int(num_turbine) != int(self.FIXED_ACTION.size):
            raise ValueError('WindBatteryPG: num_turbine({}) != len(A_FIXED)({})'
                             .format(num_turbine, self.FIXED_ACTION.size))
        self._pmax_calls = 0

    # ------------------------------------------------------------------
    # 物理
    # ------------------------------------------------------------------
    def Pmax(self, u0):
        """固定风机行为下的**可发功率** [MW]（= physics.power(u, a_fixed)）。"""
        u = float(u0)
        if not (3.5 <= u <= 25.0):
            return 0.0
        self._pmax_calls += 1
        return float(self.physics.power(u, self.FIXED_ACTION))

    def PowerOutput(self, u0, act_wind):
        """环境物理出口：`rate * Pmax(u)`（动作分派见模块 docstring）。"""
        if not (3.5 <= float(u0) <= 25.0):
            return 0.0
        a = np.asarray(act_wind, dtype=float).reshape(-1)
        if a.size == self.N_H_ACTIONS:                 # 上层动作 [rate, goal]
            rate = float(np.clip(a[0], PG_RATE_MIN, PG_RATE_MAX))
        elif a.size == 1:                              # 只给 rate（测试/手工）
            rate = float(np.clip(a[0], PG_RATE_MIN, PG_RATE_MAX))
        elif a.size == self.num_turbine:               # reset/标称动作路径：满发
            rate = 1.0
        else:
            raise ValueError('WindBatteryPG.PowerOutput: 动作长度 {} 无法解释'
                             '（应为 2=[rate,goal] / 1=[rate] / {}=标称）'
                             .format(a.size, self.num_turbine))
        return rate * self.Pmax(float(u0))

    # ------------------------------------------------------------------
    # 预取（每块调一次；Pmax 只是 u 的函数）
    # ------------------------------------------------------------------
    def prefetch_pmax(self, k=None):
        """把**下一步起**的 k 个风速的 Pmax 一次算完（替换物理层缓存）。

        Returns
        -------
        n : int   本批风速个数（0 = 无可用序列/缓存关闭）
        """
        if not self.physics.cache_enabled:
            return 0
        us = self._upcoming_wind(self.PREFETCH_K if k is None else int(k))
        if us is None or not np.size(us):
            return 0
        self.physics.prefetch(np.asarray(us, dtype=float), self.FIXED_ACTION)
        self.n_prefetch += 1
        return int(np.size(us))

    # ------------------------------------------------------------------
    def summary(self):
        return ('[{}] 固定动作={} | Pmax 调用 {} 次 | 预取 {} 次 | {}'
                .format(type(self).__name__,
                        np.round(self.FIXED_ACTION, 4).tolist(),
                        self._pmax_calls, self.n_prefetch,
                        self.physics.summary()))
