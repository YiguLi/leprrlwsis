"""
TD3PRLE (TD3 + PRLE) Package
============================
TD3 (Twin Delayed Deep Deterministic Policy Gradient) agents combined
with PRLE's dynamic wind×price segmented reward and LLM meta-controller.

- Algorithm layer (from td3prrbb): SP_TD3_PRLE — twin critics with MIN
  aggregation on the profit channel (clipped double Q-learning), MAX
  (safe) aggregation on the cost channel (bootstrap targets clamped to
  >= 0), delayed policy update, target policy smoothing, PRLEBuffer
  (PER + context store for reward relabeling) with importance sampling,
  EMA dual ascent on actual per-block cost, distilled 5-dim L
  observation, per-episode SOC random reset [0.5, 5.5].
- Reward layer (from prle): Cal_Rew_PRRBB — 机会/风险分离的分段奖励 with
  DYNAMIC wind thresholds (SEG_COEFFS by wind×price), sub-step context
  accumulation, WindowStats + LLM threshold adjustment at 1440-step
  boundaries, batch reward relabeling when thresholds change.

File layout follows prle (own myfunc / llm / buffer / main) and
td3prrbb (own ddpg agent / main / rule_bess):
  td3prle_ddpg.py      SP_TD3_PRLE agent (TD3 + PRLEBuffer)
  td3prle_buffer.py    PRLEBuffer (copy of prle/prle_buffer.py)
  td3prle_myfunc.py    PRLE dynamic-threshold reward (copy of
                       prle_myfunc.py + build_L_obs_distilled)
  td3prle_llm.py       LLM threshold analysis (copy of prle/prle_llm.py)
  td3prle_rule_bess.py RuleBasedBESS (copy of prrbb_rule_bess.py)
  td3prle_main.py      merged training loop
"""

from td3prle.td3prle_ddpg import SP_TD3_PRLE
