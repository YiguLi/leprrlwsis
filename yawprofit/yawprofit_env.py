# -*- coding: utf-8 -*-
"""`yawprofit` 环境层：**无储能**的偏航 + FLORIS 被控对象。

这是什么
--------
"只调度 FLORIS、只最大化发电利润"的最小环境：被控对象只有 5 台机组的风机总出力
`P_tur(u, γ)`，电网功率恒等于它（`P_G ≡ P_tur`），因此

* **state** 里没有 SOC、没有由电池反解出来的量；
* **action** 里没有电池功率位；
* **reward** 里没有 `bess_profit` / 退化 / 可行域缺口。

与 `yawflor` 的关系（**唯一差异 = 把 BESS 从接口上抹掉**）
----------------------------------------------------------
* **物理后端不复制**：`FlorisPhysicsYaw`（逐块预取 + NaN 守卫 + 缓存）与
  "动作 `n ⇒ γ = 25°·n`" 的换算，单一真源仍是 `yawflor.yawflor_env` /
  `yawflor.yawflor_turbine` —— 口径表已冻结（`yawflor/README.md` §2），**绝不分叉**。
  本模块只把它们重新导出，便于 `test_five_days.py` 式的按项目 import。
* **被控对象**换成 `WindTurbineYaw(WindBatteryYaw)`：只覆写"观测布局"与"步进"。

观测（3 维；⚠ 第 2 维语义与 yawflor **不同**）
------------------------------------------------
    yawprofit:  [pr (元/MWh), u (m/s), P_tur (MW)]
    yawflor  :  [pr,          u,       E (SOC), P_G]
⇒ `yawprofit_myfunc.Normalize` 只取前 2 维 ⇒ agent 的 state =
  **电价 + 风速 + 24h 相位特征**（无 SOC、无上一分钟功率）。
`u` 与 `pr` 已经是这个 MDP 的充分统计量：FLORIS 是**稳态**模型（无时间记忆），
动作无执行器动态（速率/一阶延迟默认全关，见 `yawflor/README.md` §4），
奖励只依赖 `(pr, P_tur(u, γ))`。

波动量（**仅诊断**，不进任何奖励/成本）
--------------------------------------
    P_F = |P_tur(t) − P_tur(t−1)|      —— 无电池时电网波动恒等于风机出力变化
它作为 `info['P_F']` 返回、写进日志/CSV 与九例测试的 `pf_viol` 列，但
`yawprofit` 的 c1 ≡ 0（无约束通道），不产生任何学习信号。

⚠ 与父类的**记账时点**差异（刻意）：`WindBattery.step` 的 `P_G(t+1)` 用的是**传进来的**
`pw_tru`（= 该步开始时算出的风机出力 ⇒ 电网功率比风机出力**滞后一步**，见 `env.py`
的 `GetBatteryActions(pw_tru, ...)`）—— 那是"电池在步内先动、风功率按步初值结算"的
遗留约定（reward 路径不受影响：`r1` 一直用当时的 `P_tur`）。无电池时这个滞后没有物理
含义（`P_G` 就是 `P_tur`），故本环境取 `P_G(t+1) = P_tur(t+1)`。`P_F` 在 yawprofit 里
**只作诊断**（c1 ≡ 0），这条差异不进入任何奖励/成本。

关于父类残留的诚实说明
----------------------
`WindBattery.reset()` 仍会写一个常数 `self.E = 3.`，父类也仍带着
`GetBatteryActions()` 等方法（**不改 `env.py`**：9 个既有方法共用它）。
本环境**从不读写 `self.E`**，并把 `GetBatteryActions()` 改成**显式抛错**，
使"电池通道已抹去"成为结构性事实而不是约定：
任何绕过本类接口去动电池的代码都会立刻失败，而不是静默地改动 SOC。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))        # yawprofit/
_ROOT = os.path.dirname(_HERE)                            # 仓库根
if _ROOT not in sys.path:                                 # 生产模块：env/util 在仓库根
    sys.path.insert(0, _ROOT)

# 物理后端与父类：**跨项目单一真源**（不复制 FLORIS 口径，见模块 docstring）
from yawflor.yawflor_env import FlorisPhysicsYaw, WindBatteryYaw   # noqa: E402,F401


class WindTurbineYaw(WindBatteryYaw):
    """无储能被控对象：观测 `[pr, u, P_tur]`；步进里**没有电池**。

    继承自 `WindBatteryYaw` 的部分（逐位不变，因此与 yawflor 同口径）：
    电价日程 `CalPrice`、风速序列 `CalWindSpeed/TestWindSpeed`、固定日序列、
    逐块预取 + NaN 守卫的 `PowerOutput`、九例场景接口的索引语义。

    覆写的部分（**这就是全部差异**）：
    * `reset/TestReset/ScenarioReset` → 3 维观测（把 yawflor 的 `E` 位换成 `P_tur`）；
    * `step/TestStep/ScenarioStep` → 不调 `GetBatteryActions`，`P_G ≡ P_tur`；
    * `GetBatteryActions` → 抛 `NotImplementedError`（结构性抹除电池通道）。
    """

    nominal_action_value = 0.0        # 继承（n = 0 ⇒ γ = 0°）；此处显式重申
    BLOCK = 5                         # H 决策周期（步）= 5 min；与主循环一致
    PREFETCH_K = 6                    # 继承自父类：块内 5 步 + 跨块那一次

    # ------------------------------------------------------------------
    # 观测：3 维，无 SOC
    # ------------------------------------------------------------------
    def _obs(self):
        """当前观测 `[pr, u, P_tur]`（`P_G_prev` 在无电池时就是上一分钟的风机出力）。"""
        return np.array([self.pr, self.u, self.P_G_prev], dtype=float)

    def reset(self):
        super().reset()                     # 父类：抽日序列 / 定初始 u、pr、P_G_prev 基准
        return self._obs()

    def TestReset(self, day=0):
        super().TestReset(day)
        return self._obs()

    def ScenarioReset(self, wind_seq, price_seq):
        super().ScenarioReset(wind_seq, price_seq)
        return self._obs()

    # ------------------------------------------------------------------
    # 步进：没有电池项
    # ------------------------------------------------------------------
    def step(self, applied_action_H):
        """`t → t+1`：推进电价与风速，用**同一个**上层动作重算风机出力。

        Returns
        -------
        (P_tur(t+1), obs2, info)
            obs2 = `[pr(t+1), u(t+1), P_tur(t+1)]`；info = `{'P_F':..., 'P_G':...}`
            （`P_G ≡ P_tur`；`P_F` 仅诊断）。
        与父类 `WindBattery.step` 的签名差异是**故意的**：它接收 `act_bat`、返回
        7 元组；本环境没有那条通道，硬留一个被忽略的参数只会让"电池是否还在被用"
        变得不可判定。
        """
        self.pr = self.CalPrice()                       # 与父类同一价格时钟
        self.u = self.CalWindSpeed()                    # 与父类同一风速序列
        pw_tru2 = self.PowerOutput(self.u, applied_action_H[:self.num_turbine])
        return self._advance(pw_tru2)

    def TestStep(self, applied_action_H):
        """测试路径：固定电价 300 元/MWh（与父类 `TestStep` 逐位一致）。"""
        self.pr = 300.
        self.u = self.TestWindSpeed()
        pw_tru2 = self.PowerOutput(self.u, applied_action_H[:self.num_turbine])
        return self._advance(pw_tru2)

    def ScenarioStep(self, applied_action_H):
        """九例分段场景：wind/price 完全由外部序列给定（与父类逐位一致）。"""
        self.scenario_idx += 1
        if self.scenario_idx >= len(self.scenario_wind):
            self.scenario_idx = len(self.scenario_wind) - 1      # 末尾保持
        self.pr = self.scenario_price[self.scenario_idx]
        self.u = self.scenario_wind[self.scenario_idx]
        pw_tru2 = self.PowerOutput(self.u, applied_action_H[:self.num_turbine])
        return self._advance(pw_tru2)

    def _advance(self, pw_tru2):
        """三个步进函数的公共尾部：`P_G ≡ P_tur`、更新波动基准、给出新观测。"""
        P_G = float(pw_tru2)
        P_F = abs(P_G - self.P_G_prev)
        self.P_G_prev = P_G
        info = {'P_F': P_F, 'P_G': P_G}
        return P_G, np.array([self.pr, self.u, P_G], dtype=float), info

    # ------------------------------------------------------------------
    # 结构性抹除：电池通道不存在
    # ------------------------------------------------------------------
    def GetBatteryActions(self, *args, **kwargs):
        """**电池通道已抹去**：本环境不存在 SOC/充放电，调用即报错（结构性保证）。

        这不是"暂时不用"而是"没有"：父类 `step()` 里那句
        `applied_action_L, penalty_L, P_G = self.GetBatteryActions(...)`
        正是 `yawprofit` 要删掉的东西 —— 上面三个步进函数都没有它。
        """
        raise NotImplementedError(
            'WindTurbineYaw 无储能：BESS 通道已从 state/action/reward 中抹去；'
            '需要电池请用 yawflor.yawflor_env.WindBatteryYaw')
