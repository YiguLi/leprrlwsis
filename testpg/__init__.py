# -*- coding: utf-8 -*-
"""`testPg` —— `aifflor` 的**固定风机行为 + 标量降载率**对照实验臂。

一句话
------
把 `aifflor` 上层策略的 6 维动作 `[a1..a5, goal]`（5 台机组的轴向诱导因子 + SOC
目标信号）换成 2 维动作 `[rate, goal]`：

    Pmax(u)  = P_FLORIS(u; [0.2, 0.2, 0.2, 0.2, 0.33])      # 固定风机行为，每步现算
    ptur(t)  = rate * Pmax(u_t)                              # rate ∈ [0, 1]，块内固定
    P_G(t)   = ptur(t) - pbrule(t)                           # 并网功率（env 原语义）

其余全部与 `aifflor` 同口径：FLORIS 物理、下层 RL BESS、SOC 目标映射、r1/c1/gap 罚、
r2/c2、λ1/λ2 对偶、PER、TD3、warmup 信号游走、训练中 9-case 测试。

科学问题
--------
`aifflor` 的收益里有多少来自**逐台风机的尾流协同（改变 a 的形状）**，又有多少只是
**整场降载（一个标量）**？testPg 把尾流形状钉在 warmup 期的结构先验上，只让策略
学一个降载率，因此是"去掉逐机尾流自由度"的单因子对照臂。

模块
----
* `testpg_env.py`    —— `WindBatteryPG`：固定动作 Pmax + rate 缩放 + Pmax 预取
* `testpg_myfunc.py` —— 2 维动作检查 / 罚、rate 越界乘性罚、r1/c1、L 观测、9-case
* `testpg_ddpg.py`   —— `SP_TD3_PG`：2 维动作的 TD3（噪声/随机动作/cost 裁剪）
* `testpg_warmup.py` —— warmup：rate ~ U(0,1) + goal 价格条件游走
* `testpg_main.py`   —— 训练入口（由 `aifflor_main.py` 的默认分支派生）

见 `README.md` 的口径表与"与 aifflor 的差异清单"。
"""
