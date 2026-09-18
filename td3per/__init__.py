"""
TD3PER (TD3 + PER) Package — 上层 td3perrb 口径 + 下层 RL BESS 追 SOC 目标
==========================================================================
TD3 (Twin Delayed Deep Deterministic Policy Gradient) 结合 PER 与拉格朗日
成本约束, 由 PER 方法 (`PER_ddpg.SP_DDPG`) 改造而来; 分层结构沿用 td3per
原设计 (上层 H 学风机桨距 + 1 个信号, 下层 L 学 BESS 功率)。
【本轮】下层动作已改为**残余动作** `pb_cmd = 规则基线 + Δ` (见下)。

本轮"td3perrb 化"改造 (2026, 原地修改 td3per):
  * 上层 H 与 td3perrb 完全一致 —— 非分段奖励 (收益权重/c1 风险系数恒 1.0,
    W_GAP=10), 第 6 维动作语义由"影子电价"改为 **SOC 目标信号**
    (a6 → soc_target = 3 + 5·a6 ∈ [0.5, 5.5], 块级固定), 块末 SOC 目标
    可行性罚 uniform 时进 r1 (legacy 进 c1); H 观测 = 4 维基础 + 24h 相位特征;
    uniform 下 lam1 ∈ [0.967, 4.833] 初值 0.967 (乘性对偶)。
  * 下层 L 保留 td3per 自己的 BESS 操作 (TD3+PER+EMA 对偶+RBB 硬裁剪),
    只把信号/追踪目标换成 SOC 目标: 5 维观测 = [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5],
    **动作 = 残余动作** pb_cmd = 规则基线 (hst.rule_action ≡ SocTracker.act) + Δ (actor 输出,
    输出层零初始化 ⇒ t=0 时 Δ≡0 ⇒ 起点逐位等于规则下层, r2≡0 且 c2≡0; 入 buffer 的是 Δ,
    交给 RBB 的是命令值, 事后填入用 cmd_pb/rule_pb 核对; 探索半宽 --l_noise_amp 默认 1.0 MW),
    r2 = −1·min(clip_amount, 20) [MW] (命令与实际之差 = 两段 clip), c2 = +1·trackerr [MW] (两区间分治, 单一真源 hsrl_soctarget:
    参考集合 = Soctrack范围[δ/(g·n_rem), δ/g] ∩ 平抑窗口; 空交集 ⇒ **不存在范围,
    只有纯粹一个竭力值 r*** = 窗口里离该范围最近的端点, 顶到才不罚);
    uniform 下 λ2 ∈ [0.5, 2] (无量纲兑换率: 每 MW c2 至多值 2 MW r2 / 至少 0.5), 初值 1.0, 乘性对偶 (η2 = 0.02), d_block_L = 1.2 (⇔ 0.02 MWh/块)。

Key differences from SP_DDPG:
  1. Twin critics per channel: MIN of the profit pair in targets and
     policy loss (clipped double Q-learning); MAX of the cost pair
     (safe: avoid cost underestimation), with cost bootstrap targets
     clamped to >= 0 (costs are non-negative by construction).
  2. Delayed policy update (actor + target nets updated every
     policy_delay critic updates).
  3. Target policy smoothing: clipped Gaussian noise added to
     target actions (scaled per-dimension by action amplitude).
  4. PER priority based on twin-min TD errors (unchanged formula shape).
  5. 逐维探索噪声 (WT 5 维 ±wt_noise·ε; 信号维 ±max(sig_noise_max·ε,
     sig_noise_min)), 由 td3perrb 移植。
"""

from td3per.td3per_ddpg import SP_TD3
