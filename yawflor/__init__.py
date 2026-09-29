# -*- coding: utf-8 -*-
"""`yawflor` —— 偏航控制 + FLORIS 新环境（HSRL 方法 10）。

方案与验收锚见 `yawflor/README.md`；本包与既有 9 个方法的关系是
**同一算法的第二个动作空间 / 第二个物理引擎**（基线 = `td3per`）。

模块
----
* `yawflor_turbine.py` —— 风机/功率表/口径**单一真源** + 模型构造入口 + 向量化求值
* `yawflor_env.py`     —— `FlorisPhysicsYaw`（FLORIS 后端 + 逐块预取 + NaN 守卫）
                          + `WindBatteryYaw`（继承 `env.WindBattery`，改物理与标称动作）
* `yawflor_myfunc.py`  —— 奖励/成本/越界罚（偏航带）+ `nine_case_test` 接线
* `yawflor_ddpg.py`    —— 逐字复制 `td3per/td3per_ddpg.py`（不改）
* `yawflor_main.py`    —— 主循环（除 env 类/动作带/env 相关常量和 CLI 文案外与基线一致）
"""
