# -*- coding: utf-8 -*-
"""`aifflorpave` —— aifflor 的**PAVE 式 critic 几何正则**分支方法（验证臂）。

目的（本仓库实测驱动）：基线 critic 的动作梯度场几何很差 ——
  * 相邻决策 ∇_aQ 符号**翻转率 35%**（Lee et al. 2026 的 Base 为 9–35.5%）；
  * 5 台机组间 ∇_aQ **同号率 45.7%**（≈随机）；
  * 全 6×6 ∇²_aaQ **负定性率 4.5%**（该文 Base 14–47%）⇒ μ≈0。
该文的理论界是策略敏感度 L ≤ M/μ，并给出**只作用于 critic** 的三项辅助损失
MPR / VFC / Curv。本方法的可检验预言：**施加几何正则后翻转率应显著下降且成绩改善**；
若只加大网络或只调噪声，翻转率不变（已实测）。

与 `aifflor` 的关系：本包**不复制** ddpg 主体，而是
  * `aifflorpave_pave.py` —— 新增（CriticPAVE + MPR/VFC/Curv 实现）；
  * `aifflorpave_ddpg.py` —— `SP_TD3` 的**子类**（只加可切换激活 + 一个附加几何优化步）；
  * `aifflorpave_main.py` —— 由 `agentutil/port_aifflorpave.py` 从 aifflor_main **机械派生**
    （7 处替换，`--check` 可逆证明）。
`--critic_act relu --pave_w_* 0` 时本 arm **逐位等于 aifflor 基线**。

L 层（BESS 残差动作）仍用基类 `SP_TD3` ⇒ 残差/零初始化设计不受影响。
"""

from aifflorpave.aifflorpave_ddpg import SP_TD3_PAVE
from aifflorpave.aifflorpave_pave import CriticPAVE, pave_terms
