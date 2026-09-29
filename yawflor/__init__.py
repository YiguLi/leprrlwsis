# -*- coding: utf-8 -*-
"""`yawflor` —— 偏航控制 + FLORIS 新环境（HSRL 方法 10）。

方案与验收锚见 `yawflor/README.md`；本包与既有 9 个方法的关系是
**同一算法的第二个动作空间 / 第二个物理引擎**（基线 = `td3per`）。

基线克隆的**唯一真源**是 `agentutil/port_yawflor.py`（替换表 + `--check` 可逆性校验）：

* `yawflor_ddpg.py`    —— **逐字复制** `td3per/td3per_ddpg.py`（零改动）
* `yawflor_myfunc.py`  —— 动作带 `[0,0.5] → [−1,+1]`、越界罚系数 `400 → 100`
                          （**满带越界仍是 200 元**）、`violation_amount` 带、文案
* `yawflor_main.py`    —— env 类（FLORIS，不启 MATLAB）/ `init_wt_action_bias` 带
                          `[−1,+1]` 且打印 γ / `--wt_noise 0.2` /
                          `hwarm.add_warmup_args(..., yaw=True)` / CLI 文案

算法本体（TD3+PER、EMA 对偶、奖励/成本、n 步回报、相位特征、日志与 CSV 列）
与基线**逐字相同**，保证"新方法 vs 基线"的对照只差"动作空间 + 物理引擎"。

本包另外两个模块**不是克隆**（新写的"口径单一真源"与"后端"）：

* `yawflor_turbine.py` —— 物理常量 / 功率表 / 模型构造 / 向量化求值（单一真源）
* `yawflor_env.py`     —— `FlorisPhysicsYaw`（逐块预取 + NaN 守卫）+ `WindBatteryYaw`
"""

from yawflor.yawflor_ddpg import SP_TD3

__all__ = ['SP_TD3']
