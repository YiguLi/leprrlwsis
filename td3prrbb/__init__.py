"""
TD3PRRBB (TD3 + PRRBB) Package
==============================
TD3 (Twin Delayed Deep Deterministic Policy Gradient) agents combined
with PRRBB's Wind × Price Segmented Reward structure.

- Algorithm layer (from td3per): SP_TD3 — twin critics with MIN
  aggregation on the profit channel (clipped double Q-learning), MAX
  (safe) aggregation on the cost channel (bootstrap targets clamped to
  >= 0), delayed policy update, target policy smoothing, PER buffer
  with importance sampling, and EMA dual ascent on actual per-block
  cost.
- Reward layer (from prrbb): Cal_Rew_PRRBB — 机会/风险分离的分段奖励:
  H 层 r1 用 wT/wB 分段收益权重分解风机/BESS 收益 (wT=wT_u(u)×wT_p(pr),
  wB=wB_p(pr)×wB_soc(SOC,wind)), c1 用 r_u(u) 风险系数
  + W_GAP*unfeasible_gap 编码波动/平抑可行性风险;
  L 层 (SOC 目标追踪口径, 单一真源 hsrl_soctarget):
      a6 → soc_target = clip(3+5a6, 0.5, 5.5); r2 = −1·clip_amount;
      c2 = 10·slack (能力内追踪失败量; 零成本动作 = 规则 SocTracker).

File layout follows prrbb (own myfunc / rule_bess / main) and td3per
(own ddpg agent / main):
  td3prrbb_ddpg.py      SP_TD3 agent (copy of td3per_ddpg.py)
  td3prrbb_myfunc.py    PRRBB segmented reward (copy of prrbb_myfunc.py)
  td3prrbb_rule_bess.py RuleBasedBESS (copy of prrbb_rule_bess.py)
  td3prrbb_main.py      merged training loop
"""

from td3prrbb.td3prrbb_ddpg import SP_TD3
