"""
TD3PERRB — TD3 + PER + 规则下层 (Rule-Based BESS)
=================================================
`td3prrlrb` 的【非分段奖励】版本: 上层 TD3 学习、下层 SocTracker 规则控制,
唯一差异是 **不使用分段奖励函数**。

模块分工
--------
- `td3perrb_ddpg.py`      : SP_TD3 (twin critic + 延迟策略更新 + 目标平滑 + PER)
- `td3perrb_myfunc.py`    : 非分段奖励/成本、SOC 目标映射、CheckAction、
                            归一化与 24h 相位特征、9-case 测试
- `td3perrb_rule_bess.py` : RuleBasedBESS (可行窗口) + SocTracker (规则下层)
- `td3perrb_main.py`      : 训练主循环 (只训练 H 层)

与 td3prrlrb 的差异 (逐项)
--------------------------
| 项 | td3prrlrb (分段) | td3perrb (非分段) |
|---|---|---|
| 风机收益权重 `wT` | `WT_WIND[u] × WT_PRICE[pr]` | `W_TURBINE = 1.0` |
| BESS 收益权重 `wB` | `WB_PRICE[pr] × wB_soc(SOC, wind)` | `W_BESS = 1.0` |
| c1 风险系数 | `C1_RISK[wind] ∈ {0.8, 1.0, 1.2}` | `C1_RISK_COEF = 1.0` |
| SOC 安全带 (仅日志) | 按风况分档 | 单一档 `[1.5, 4.5]` |

其余一切 (下层规则、超参、观测、对偶 EMA、课程学习、记录列) 与 td3prrlrb 逐字一致。

运行
----
    python td3perrb/td3perrb_main.py
"""
