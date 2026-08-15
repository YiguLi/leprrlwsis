# PRRBB (Wind × Price Segmented Reward + PER lower baseline) Project
# =============================================================================
# 机会/风险分离版分段奖励（prrbb_myfunc.py）:
#
# Upper layer (H) r1（机会侧，乘法分解）:
#   r1 = wT * (pr * ptur * time)          ← 风机收益, wT = wT_u(u) × wT_p(pr)
#      + wB * (-pr * pbrule * time)        ← BESS 收益, wB = wB_p(pr) × wB_soc(SOC,wind)
#      - 490.0 * max(-pbrule, 0) * time    ← 退化 (H 看 BESS 净收益)
#      - penalty_NN_H
#
#   Weight tables:
#     wT_u:   LOW=1.2  MEDIUM=1.0  HIGH=0.9    (高风轻度降载倾向)
#     wT_p:   LOW=1.0  MEDIUM=1.0  HIGH=1.05   (高价时段轻微加码发电)
#     wB_p:   LOW=0.8  MEDIUM=1.0  HIGH=1.2    (低价时少罚充电→鼓励低谷充电)
#     wB_soc: 带内(当前风速档安全带)=1.0, 带外=0.7 (SOC 贴边抑制套利)
#
#   Wind boundaries: LOW < 7.0 m/s, MEDIUM [7.0, 11.0), HIGH >= 11.0 m/s
#
# Upper layer (H) c1（风险侧）:
#   c1 = r_u(u) × [10*(P_F-3)+ + 0.1*P_F] + 50.0 * unfeasible_gap
#     r_u: LOW=0.8  MEDIUM=1.0  HIGH=1.2
#     unfeasible_gap: 平抑可行性缺口 (MW)，风速/SOC/轴向因子/BESS额定的综合体现
#
# Lower layer (L):
#   r2 = 10.0 * shadow_profit - 100.0 * clip_amount
#        + 0.5 * (-pr * pbrule - 490.0 * max(-pbrule, 0.0)) * time  ← 真实利润项
#   c2 = 700.0 * soc_deviation   (SOC-safe 项进入拉格朗日约束, 保留原系数)
#   soc_deviation: wind-segmented safe ranges
#     LOW [0.5, 5.5] / MEDIUM [1.0, 5.0] / HIGH [1.5, 4.5]
#
# Philosophy: Wind farm maximizes profit under safety constraints.
# 机会（利润）由收益权重编码；风险（波动/平抑可行性/SOC 越界）由 c1/c2 编码。
# High risk (high wind) → c1 加重 + 收益权重轻微降载倾向。
# BESS safety is the lower layer's responsibility; upper layer only
# adjusts BESS profit weights to steer shadow-price regulation.
#
# Shadow price signal (PER convention): range (-0.5, 0.5), kept unchanged.
# RBB clipping: same hard clipping before env.step (unchanged).
#
# Lagrange multiplier updates (prrbb_main.py):
#   H (EMA 方案, 实际成本): 每 5 步块
#     EMA   ← (1-0.02)·EMA + 0.02·sum_c1
#     lam1  ← clip(lam1 + 0.01·(EMA − d_block), 0.1, 10)
#     d_block 初始 1.2，curriculum 连续 100 次满足则减半，地板 0.3
#   L (critic 版, 维持现状): lam2 ∈ [1, 10]，cost_limit 减半地板 0.0
