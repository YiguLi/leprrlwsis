"""
TD3LEPRRLRB (TD3 + LLM-Enhanced Piecewise Reward + Rule-Based BESS) Package
==========================================================================
TD3PRRLRB 的 LLM 增强版 —— 关系类比 td3prrbb → td3prle。

架构: 上层 TD3 是唯一学习器 (5 台风机桨距 + 1 个 SOC 目标信号),
      下层是确定性规则控制器 SocTracker (在 RBB 可行窗口内最大能力跟踪目标)。
      分层与奖励结构与 td3prrlrb 相同, 差异只有两处:
      (1) 【风速分段阈值】交给 LLM 在线微调 (启动按 --seg 定起点, ±1 m/s 硬约束);
      (2) wB 的 SOC 贴边折减取消 (WB_SOC_OUTSIDE 0.7 → 1.0, 与 td3prle 对齐),
          上层 BESS 盈利不再随 SOC 位置打折, 该约束改由 SOC 目标机制承担。

- Algorithm layer (from td3prrlrb): SP_TD3_LEPRRLRB — twin critics
  (profit channel: MIN aggregation / clipped double Q-learning;
  cost channel: MAX safe aggregation with bootstrap clamp >= 0),
  delayed policy update, target policy smoothing,
  LEPRRLRBBuffer (PER + context store for reward relabeling),
  EMA dual ascent on actual per-block cost, per-episode SOC random
  reset [0.5, 5.5].
- Reward layer (from prle): Cal_Rew_RLRB with DYNAMIC wind thresholds
  (WT_WIND and C1_RISK keyed by wind_level; wB keyed by price only, its
  SOC-edge discount cancelled),
  per-sub-step context accumulation, WindowStats + LLM threshold
  adjustment at --llm_window boundaries (default 7200 steps = 5 days),
  and batch relabeling of stored (r1, c1) whenever thresholds move.

Files
-----
  td3leprrlrb_ddpg.py      SP_TD3_LEPRRLRB agent (TD3 + LEPRRLRBBuffer)
  td3leprrlrb_buffer.py    LEPRRLRBBuffer (PER + relabeling context store)
  td3leprrlrb_myfunc.py    动态阈值奖励 / relabel 辅助 / 9-case 测试
  td3leprrlrb_llm.py       LLM 元控制器 (WindowStats + 提示词 + provider 工厂)
  td3leprrlrb_rule_bess.py RuleBasedBESS + SocTracker (与 td3prrlrb 逐字一致)
  td3leprrlrb_main.py      训练主循环 (含 LLM 窗口与 relabel 触发)
  llm_config.json          LLM provider / model / api_key
"""

from td3leprrlrb.td3leprrlrb_ddpg import SP_TD3_LEPRRLRB
