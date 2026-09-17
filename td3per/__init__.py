"""
TD3PER (TD3 + PER) Package — 上层 td3perrb 口径 + 下层 RL BESS 追 SOC 目标
==========================================================================
TD3 (Twin Delayed Deep Deterministic Policy Gradient) 结合 PER 与拉格朗日
成本约束, 由 PER 方法 (`PER_ddpg.SP_DDPG`) 改造而来; 分层结构沿用 td3per
原设计 (上层 H 学风机桨距 + 1 个信号, 下层 L 学 BESS 功率 pb)。

本轮"td3perrb 化"改造 (2026, 原地修改 td3per):
  * 上层 H 与 td3perrb 完全一致 —— 非分段奖励 (收益权重/c1 风险系数恒 1.0,
    W_GAP=10), 第 6 维动作语义由"影子电价"改为 **SOC 目标信号**
    (a6 → soc_target = 3 + 5·a6 ∈ [0.5, 5.5], 块级固定), 块末 SOC 目标
    可行性罚进入 c1, H 观测 = 4 维基础 + 24h 相位特征, lam1 ∈ [3,100]。
  * 下层 L 保留 td3per 自己的 BESS 操作 (TD3+PER+EMA 对偶+RBB 硬裁剪),
    只把信号/追踪目标换成 SOC 目标: 5 维蒸馏观测第 3 维 = soc_target/10,
    r2 = −1·clip_amount, c2 = 10·slack (能力内追踪失败量:
    slack = max(0, |target−SOC'| − |target−SOC_best|),
    SOC_best = SOC + g·clip(δ/g, RBB 窗口) 即规则下层动作), lam2 ∈ [10,500]。

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
