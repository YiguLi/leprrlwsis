# -*- coding: utf-8 -*-
"""`aifflor` —— **`td3per` 的 FLORIS 引擎版**（同一算法、同一动作空间、换物理引擎）。

一句话
------
`aifflor` 基本上就是 `td3per` 把 **WFSim 风场仿真器换成 FLORIS**，其余完全一致。

与 `td3per` 的差异清单（**只有引擎**）
--------------------------------------
| 项 | td3per | aifflor |
|---|---|---|
| 物理引擎 | WFSim（MATLAB，`WFSim_port`） | FLORIS（纯 Python，**不启 MATLAB**） |
| env 类 | `env.WindBattery`（`physics=None` 走 MATLAB） | `WindBatteryAIF`（`physics=AifflorPhysics`） |
| 尾流参数 | WFSim `meshescale/forcescale` | FLORIS + **标定到 WFSim 的参数** |
| **动作语义** | 轴向因子 `a ∈ [0,0.5]`（5 维）+ SOC 信号 | **完全相同** |
| **奖励/成本** | `td3per_myfunc` | **逐字复用**（零改动） |
| 算法本体 | TD3+PER、EMA 对偶、n 步、相位特征 | **逐字复用**（零改动） |

⚠ 关键点：因为**动作语义相同**，`td3per_myfunc.py` 可以**逐字复制**——
这是 `aifflor` 与 `yawflor` 最大的不同（yawflor 把动作带换成了 `[−1,+1]`，
所以必须改 `CheckAction` / 越界罚系数 / `violation_amount` 边界）。

I/O 映射（**本次任务的核心**）
------------------------------
两套仿真器的接口完全不同，映射关系逐项记在 `aifflor_turbine.py` 顶部的表里，要点：

* **入口**：WFSim `WFSim_port(u, 0.0, a[5], 0.0)` ↔ FLORIS `eval_batch(m, u, a[5])`
* **动作**：都是**逐台轴向因子 `a`**，都由 `env.py` 夹到 `[0, 0.5]`；航向恒 0
  （WFSim 的 `Control.phi=0` ↔ FLORIS 的 `yaw_angles≡0`）
* **输出**：都是**风场总有功 [MW]**（WFSim `out/1e6` ↔ FLORIS Σ逐台）
* **第 6 维**：SOC 目标信号，**不进风机**（`env.PowerOutput` 已截到前 5 维）
* **围栏**：`3.5 ≤ u ≤ 25`，否则出力 0（两引擎共用 `env.py` 的守卫）

因为输出语义相同，`env.py` **一行都不用改** —— `PowerOutput` 是唯一物理出口，
`physics is not None` 时分派给后端即可。

文件
----
* `aifflor_turbine.py` —— 口径 / **I-O 映射表** / 模型构造 / 向量化求值（单一真源）
* `aifflor_env.py`     —— `AifflorPhysics`（非有限值守卫 + 预取）+ `WindBatteryAIF`
* `aifflor_ddpg.py`    —— **逐字复制** `td3per/td3per_ddpg.py`
* `aifflor_myfunc.py`  —— **逐字复制** `td3per/td3per_myfunc.py`
* `aifflor_hsrl.py`    —— **新写**：`--hsrl` 方法分支的全部口径（"块总放电量"语义，见其 docstring
  与 `aifflor/README.md` §3.8；未加 `--hsrl` 时一行都不会被调用）
* `aifflor_main.py`    —— `td3per/td3per_main.py` + 替换表（`agentutil/port_aifflor.py`）
  + `--hsrl` 分支的 `train_hsrl()`（与旧 `train()` 并列）

克隆的**唯一真源**是 `agentutil/port_aifflor.py`（替换表 + `--check` 可逆性校验）。

物理差异（声明，不试图抹平）
----------------------------
1. FLORIS 的尾流参数已按 WFSim **标定**（`agentutil/floris_calibrated_defaults.py`）：
   验证集相对误差 **66.9% → 1.1%**（见 `documentation/floris_calibration_to_wfsim.md`）。
2. 标定只针对 **5 台 5D 单排 + `a` 通道 + `γ≡0`** ⇒ `build_model` 会断言。
3. 两引擎绝对电平仍不同（u=10、a=1/3：FLORIS **3.294 MW** vs WFSim **3.446 MW**）
   ⇒ **跨引擎的 profit 不可直接比较**，只能同轨比较或归一化后比较。
"""

from aifflor.aifflor_ddpg import SP_TD3

__all__ = ['SP_TD3']
