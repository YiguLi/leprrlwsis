"""
TD3LEPRRLRB Extended Prioritized Replay Buffer
==============================================
Extends PrioritizedReplayBuffer (PER_buffer.py) with a parallel context store
so that stored H-layer transitions can be RE-LABELED in place whenever the LLM
meta-controller moves the wind segmentation thresholds.

Why relabeling is needed
------------------------
本项目的 r1 / c1 都依赖【风速分档】u ↔ (LOW_MAX, HIGH_MIN):

  r1 = wT(wind,price)·(pr·ptur·dt) + wB(price,wind,SOC)·(−pr·pbrule·dt)
       − 退化 − penalty_NN_H
  c1 = C1_RISK[wind]·[10·(P_F−3)⁺ + 0.1·P_F] + W_GAP·unfeasible_gap

阈值一变, 同一条历史轨迹的 wind 档就变了 → 存进回放池的 (r,c) 与"当前
奖励函数"不再一致。若不重算, 智能体会拿两套口径混在一起的目标去拟合
(非平稳奖励), 所以阈值实际变化后立即对全体 H 层转移重算 r/c。

Context storage (与 td3prrlrb 的 5 步块一一对应)
------------------------------------------------
  H 层每条转移 = 一个 5 步块, context = 5 个 sub-step dict:
      {'u', 'pr', 'turbine_raw', 'bess_raw', 'degradation',
       'penalty_NN_H', 'SOC', 'P_F', 'unfeasible_gap', 'gap_pen'}
  其中前三类 "raw" 项与阈值无关 (只含 pr/ptur/pbrule/dt), 阈值相关的只有
  分档权重 (wT/wB) 与风险系数 (C1_RISK), 所以 relabel 只需重算这两处。

  与 td3prle 的差异 (本项目修正): td3prle 的 blk 末尾 SOC 跟踪罚写在块级、
  未进 context, 于是 relabel 时被静默丢掉; 这里把 gap_pen 存进块内最后一个
  sub-step 并原样带回 (该项与风速阈值无关 → 恒等重算), 保证 relabel 前后
  成本口径完全一致。

Base class: PER_buffer.PrioritizedReplayBuffer
"""

import numpy as np  # noqa: F401  (kept for parity with PER_buffer / future use)

from PER_buffer import PrioritizedReplayBuffer


class LEPRRLRBBuffer(PrioritizedReplayBuffer):
    """Prioritized replay buffer + context store for threshold relabeling.

    The inherited sample() is untouched — relabeling is done in batch via
    relabel_all() at the (rare) moments when thresholds actually change.
    """

    def __init__(self, size, alpha, block_size=5):
        """
        Parameters
        ----------
        size : int
            Max number of transitions to store.
        alpha : float
            PER prioritization exponent (0 = uniform, 1 = full prioritization).
        block_size : int
            Sub-steps per transition (5 for the H-layer block transition).
        """
        super().__init__(size, alpha)
        self._contexts = []          # parallel to self._storage (parent's list)
        self._block_size = block_size
        self.n_relabel_calls = 0     # diagnostics: how often relabel ran

    # ------------------------------------------------------------------
    # storage
    # ------------------------------------------------------------------
    def append(self, obs_t, action, reward, cost, obs_tp1, done, context=None):
        """Append a transition plus its optional relabeling context.

        Parameters
        ----------
        obs_t, action, reward, cost, obs_tp1, done : standard transition
        context : list of dict or None
            H-layer: one dict per sub-step of the 5-step block (see module
            docstring). None = transition cannot be relabeled (skipped).
        """
        idx = self._next_idx
        # PrioritizedReplayBuffer.append updates the SUM/MIN segment trees.
        super().append(obs_t, action, reward, cost, obs_tp1, done)

        if idx >= len(self._contexts):
            self._contexts.append(context)
        else:
            self._contexts[idx] = context

    def get_context(self, idx):
        """Return the stored context at a given index (for debugging)."""
        if 0 <= idx < len(self._contexts):
            return self._contexts[idx]
        return None

    @property
    def n_contexts(self):
        """Number of transitions that carry a usable relabeling context."""
        return sum(1 for c in self._contexts[:len(self._storage)] if c is not None)

    # ------------------------------------------------------------------
    # relabeling
    # ------------------------------------------------------------------
    def _reset_priorities(self):
        """Set every stored priority to max_priority.

        After relabeling, the TD errors kept in the PER tree were computed
        against the OLD reward function, so they are stale. Putting all
        priorities back to max_priority makes sampling uniform again until
        the TD errors re-converge under the new reward.
        """
        for idx in range(len(self._storage)):
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha

    def relabel_all(self, relabel_r_fn, relabel_c_fn=None, discount=1.0):
        """Recompute every stored H-layer (reward, cost) pair.

        For each transition carrying a context, the reward is recomputed as
        the sum over its sub-steps of ``relabel_r_fn(...)``; if
        ``relabel_c_fn`` is given the cost is recomputed the same way.

        Parameters
        ----------
        relabel_r_fn : callable
            relabel_r_fn(u, pr, turbine_raw, bess_raw, degradation,
                         penalty_NN_H, SOC) -> r1
            Typically td3leprrlrb_myfunc.relabel_r1_substep.
        relabel_c_fn : callable or None
            relabel_c_fn(u, P_F, unfeasible_gap, gap_pen) -> c1
            Typically td3leprrlrb_myfunc.relabel_c1_substep. None leaves the
            stored cost untouched.

        Returns
        -------
        n_relabeled : int
            Number of transitions that were relabeled.
        """
        n_relabeled = 0
        for i in range(len(self._storage)):
            ctx = self._contexts[i] if i < len(self._contexts) else None
            if ctx is None:
                continue

            obs_t, action, _old_reward, cost, obs_tp1, done = self._storage[i]

            _blk = int(getattr(self, 'block_size', 5) or 5)
            _gam = float(discount)
            new_reward = 0.0
            for _i, sub in enumerate(ctx):
                new_reward += (_gam ** (_i // _blk)) * relabel_r_fn(
                    sub['u'],
                    sub['pr'],
                    sub['turbine_raw'],
                    sub['bess_raw'],
                    sub['degradation'],
                    sub['penalty_NN_H'],
                    sub.get('SOC', 3.0),
                )

            new_cost = cost
            if relabel_c_fn is not None:
                new_cost = 0.0
                for _i, sub in enumerate(ctx):
                    new_cost += (_gam ** (_i // _blk)) * relabel_c_fn(
                        sub['u'],
                        sub['P_F'],
                        sub.get('unfeasible_gap', 0.0),
                        sub.get('gap_pen', 0.0),
                    )

            # tuples are immutable -> rebuild with the new (r, c)
            self._storage[i] = (obs_t, action, new_reward, new_cost, obs_tp1, done)
            n_relabeled += 1

        self._reset_priorities()
        self.n_relabel_calls += 1
        return n_relabeled

    def relabel_check(self, relabel_r_fn, relabel_c_fn=None, n_samples=5,
                      rtol=1e-9):
        """Self-check: relabel a sample and report the max |delta| per field.

        Recomputes (r, c) for up to ``n_samples`` stored transitions using the
        CURRENT thresholds and compares against what is stored. When the
        thresholds have not moved since those transitions were collected the
        deltas must be ~0 — this is exactly the invariant that makes
        relabel_all() sound (context completeness).

        Returns
        -------
        dict with 'reward_max_abs_delta' and 'cost_max_abs_delta'.
        """
        d_r = d_c = 0.0
        checked = 0
        for i in range(len(self._storage)):
            ctx = self._contexts[i] if i < len(self._contexts) else None
            if ctx is None:
                continue
            _, _, reward, cost, _, _ = self._storage[i]
            r_new = sum(relabel_r_fn(
                s['u'], s['pr'], s['turbine_raw'], s['bess_raw'],
                s['degradation'], s['penalty_NN_H'], s.get('SOC', 3.0))
                for s in ctx)
            d_r = max(d_r, abs(r_new - reward))
            if relabel_c_fn is not None:
                c_new = sum(relabel_c_fn(
                    s['u'], s['P_F'], s.get('unfeasible_gap', 0.0),
                    s.get('gap_pen', 0.0)) for s in ctx)
                d_c = max(d_c, abs(c_new - cost))
            checked += 1
            if checked >= n_samples:
                break
        return {'checked': checked,
                'reward_max_abs_delta': float(d_r),
                'cost_max_abs_delta': float(d_c)}
