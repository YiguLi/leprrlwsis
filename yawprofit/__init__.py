# -*- coding: utf-8 -*-
"""`yawprofit` —— **只调度 FLORIS 的 5-min 发电利润最大化智能体**（单层 TD3 + PER）。

存在理由：`yawflor` 的学习不稳定，需要一条**最小对照臂**把"层次结构 / 电池耦合 /
约束对偶"从被诊断的对象里摘出去 —— 本项目只剩

    每 5 min 决策一次的 5 维偏航 + "发多少电就卖多少钱"的纯利润奖励。

与 `yawflor` 的关系（**同一算法、同一个物理引擎，只把 BESS 从接口上抹掉**）
--------------------------------------------------------------------------
* 复用（不复制，口径单一真源）：
    - `yawflor.yawflor_turbine` —— FLORIS 口径表 / `γ = 25°·n` / 功率表
    - `yawflor.yawflor_env.FlorisPhysicsYaw` —— 逐块预取 + NaN 守卫的后端
    - `yawflor.yawflor_myfunc` —— 电价日程 / 24h 相位表 / 风况-电价分档
    - `hsrl_warmup` —— 偏航带 `YAW_*` 与 warmup 调度器
    - `model.py` / `PER_buffer.py` / `random_process.py` / `util.py` / `hsrl_agent_init.py`
* 新写：
    - `yawprofit_env.py`   —— `WindTurbineYaw`：观测 `[pr, u, P_tur]`、`P_G ≡ P_tur`、
                              `GetBatteryActions()` 直接抛错（结构性抹除）
    - `yawprofit_myfunc.py`—— 2 维观测归一化 + 5 维动作 + 纯利润 r1 / c1 ≡ 0 +
                              无 BESS 的九例测试（含"无偏航"参考臂）
    - `yawprofit_main.py`  —— 单层训练循环（无下层 / RBB / 对偶 / gap 罚）
* 克隆：`yawprofit_ddpg.py`（= `yawflor_ddpg.py`，差异只有两处，见该文件 docstring）

口径与差异表见 `yawprofit/README.md`；契约测试见
`agentutil/test_yawprofit_nobess.py`。
"""

from yawprofit.yawprofit_ddpg import SP_TD3

__all__ = ['SP_TD3']
