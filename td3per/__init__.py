"""
TD3PER (TD3 + PER) Package
==========================
TD3 (Twin Delayed Deep Deterministic Policy Gradient) combined with
Prioritized Experience Replay (PER) and a Lagrangian cost constraint,
adapted from the PER method (PER_ddpg.SP_DDPG).

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
"""

from td3per.td3per_ddpg import SP_TD3
