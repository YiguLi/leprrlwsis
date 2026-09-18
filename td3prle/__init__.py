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
  **下层 L 口径与 prle 逐位一致**（单一真源 hsrl_soctarget）：
  `a6 → soc_target = clip(3+5·a6, 0.5, 5.5)`、`r2 = −1·min(clip_amount, 20) [MW]`、
  `c2 = +1·trackerr [MW]`（实际施加 pb 到参考集合 = Soctrack范围 ∩ 平抑窗口 的距离；
  空交集 ⇒ **不存在范围，只有纯粹一个竭力值 r***，顶到才不罚）；
  块末追踪罚 `10·max(0, |target−SOC_末| − SOC_TOL)` 进 r1（uniform）/ c1（legacy）；
  `λ2 ∈ [0.5, 2]`（无量纲兑换率：每 MW c2 至多值 2 MW r2 / 至少 0.5，MAX·MIN = 1）、`d_block_L = 1.2`、`η2 = 0.02`（乘性对数步长）；L 层 buffer 不 relabel（c2 无风速依赖）。
  注: 2026-09-19 由"目标中心二次势 0.1·(SOC−target)² + clip_over ±10 封顶"同步为本式。

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
