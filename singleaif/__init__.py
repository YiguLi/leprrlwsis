"""
SINGLEAIF (Single Axial Induction Factor) Package
=================================================
从 TD3PER 改造而来: 上层 H 只输出**一个**风机轴向因子 a ∈ [0, 0.5], 广播到全部
5 台风机 (与 run12/run13 旧接线下"dim0 统一调度"语义一致, 但这里是**显式单维动作**);
其余内容 (下层 RL BESS 追 SOC 目标、非分段奖励、SOC 目标信号、24h 相位、n 步回报、
EMA 对偶、PER、Twin critics、目标策略平滑等) 与 td3per 完全一致。

与 td3per 的唯一结构差异 (H 动作维数 6 → 2):
  * H 动作 = [a_wt, soc_signal] (2 维); 广播: 单一风机动作作用到 5 台
    ⇒ env 看到的仍是 6 维 [a_wt]*5 + [soc_signal]。
  * penalty_NN_H: `CheckAction(len==2)` 只计**一次**风机越界 (广播后不 ×5),
    系数仍为 PENALTY_NN_H_COEF=400 元/单位; 信号维越界计罚不变。
  * `violation_derate(..., n_dims=1)`: 越界乘性系数只对单一风机维计越界。
  * 探索噪声 (nb_wt=1): 风机维 ±wt_noise·ε (无地板), 信号维 ±max(sig_noise_max·ε, sig_noise_min)。
  * warmup 调度: `agent_H.nb_turbine=1` ⇒ `WarmupSchedule(nb_turbine=1, nb_actions=2)`
    产出单风机动作随机游走 + 信号维恒 0 (经 `build_from_args` 自动跟随, 不改共享文件)。
  * agent 文件: H = actor_2.pkl / critic_{p,p2,c,c2}_2.pkl; L = actor_1.pkl (不变)。
  * 输出目录: `--env SINGLEAIF` ⇒ `output/SINGLEAIF-runN`。

Key differences from SP_DDPG (与 td3per 相同): Twin critics per channel (MIN profit
pair / MAX cost pair, cost bootstrap clamp≥0)、delayed policy update、target policy
smoothing、PER (twin-min TD)、逐维探索噪声。
"""

from singleaif.singleaif_ddpg import SP_TD3
