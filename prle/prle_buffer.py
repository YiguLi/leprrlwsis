"""
PRLE Extended Prioritized Replay Buffer
========================================
Extends PrioritizedReplayBuffer with a parallel context store for reward/cost
relabeling. When wind thresholds change via the LLM meta-controller, the
buffer can recompute all stored rewards and costs in-place using current
thresholds (relabel_all), without changing the sample path.

Context storage:
  - H-agent transitions: list of 5 sub-step dicts
    {'u', 'pr', 'turbine_raw', 'bess_raw', 'degradation', 'penalty_NN_H', 'P_F'}
  - L-agent transitions: None (r2/c2 have no wind dependency)

Base class: PER_buffer.PrioritizedReplayBuffer
"""

import numpy as np
import random

import PER.PER_utils as PER_utils
from PER_buffer import PrioritizedReplayBuffer


class PRLEBuffer(PrioritizedReplayBuffer):
    """
    Prioritized replay buffer with context storage for reward relabeling.

    Extends PrioritizedReplayBuffer with a parallel _contexts list.
    The sample() method is inherited unchanged — relabeling is done
    in batch via relabel_all() when thresholds change, not per-sample.
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
            Number of sub-steps per transition. 5 for H agent, 1 for L agent.
        """
        super().__init__(size, alpha)
        self._contexts = []       # parallel list to self._storage (from parent)
        self._block_size = block_size

    def __len__(self):
        return super().__len__()

    def append(self, obs_t, action, reward, cost, obs_tp1, done, context=None):
        """
        Append a transition with optional context for reward relabeling.

        Parameters
        ----------
        obs_t, action, reward, cost, obs_tp1, done : standard transition
        context : list of dicts or None
            H-agent: list of sub-step dicts with keys:
                'u', 'pr', 'turbine_raw', 'bess_raw', 'degradation', 'penalty_NN_H'
            L-agent: None (r2/c2 have no wind dependency)
        """
        idx = self._next_idx
        # Call parent's append for the base transition storage + PER tree update
        # We must bypass PrioritizedReplayBuffer.append and call ReplayBuffer.append
        # directly, because the parent's append calls super().append() internally.
        # Actually, PrioritizedReplayBuffer.append does:
        #   idx = self._next_idx
        #   super().append(*args, **kwargs)   # → ReplayBuffer.append
        #   self._it_sum[idx] = ...
        # So we can call super().append(...) which is PrioritizedReplayBuffer.append
        super().append(obs_t, action, reward, cost, obs_tp1, done)

        # Store context in parallel array
        if idx >= len(self._contexts):
            self._contexts.append(context)
        else:
            self._contexts[idx] = context

    def relabel_all(self, relabel_r_fn, relabel_c_fn=None):
        """
        Recompute all stored rewards (and optionally costs) using current wind
        thresholds.

        For every transition that has a non-None context, recomputes the
        reward by applying relabel_r_fn to each sub-step context and summing
        the results. If relabel_c_fn is provided, the cost is likewise
        recomputed (c1 now carries a wind-threshold-dependent risk coefficient,
        so it must be relabeled alongside r1 when thresholds change).

        Parameters
        ----------
        relabel_r_fn : callable
            Function with signature:
                relabel_r_fn(u, pr, turbine_raw, bess_raw, degradation, penalty_NN_H) → r1
            Typically prle_myfunc.relabel_r1_substep.
        relabel_c_fn : callable or None
            Function with signature:
                relabel_c_fn(u, P_F) → c1
            Typically prle_myfunc.relabel_c1_substep. None leaves cost unchanged.

        Returns
        -------
        n_relabeled : int
            Number of transitions that were relabeled.
        """
        n_relabeled = 0
        for i in range(len(self._storage)):
            ctx = self._contexts[i]
            if ctx is None:
                continue
            # Unpack current storage tuple
            obs_t, action, _old_reward, cost, obs_tp1, done = self._storage[i]

            new_reward = 0.0
            for sub in ctx:
                new_reward += relabel_r_fn(
                    sub['u'],
                    sub['pr'],
                    sub['turbine_raw'],
                    sub['bess_raw'],
                    sub['degradation'],
                    sub['penalty_NN_H'],
                )

            new_cost = cost
            if relabel_c_fn is not None:
                new_cost = 0.0
                for sub in ctx:
                    new_cost += relabel_c_fn(sub['u'], sub['P_F'])

            # Replace reward and (optionally) cost; tuple is immutable so recreate
            self._storage[i] = (obs_t, action, new_reward, new_cost, obs_tp1, done)
            n_relabeled += 1

        # Reset ALL priorities to max_priority after relabeling.
        # TD errors stored in the PER tree are stale (computed under old rewards).
        # Setting all priorities to max ensures uniform re-sampling until
        # TD errors naturally re-converge under the new reward function.
        for idx in range(len(self._storage)):
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha

        return n_relabeled

    def get_context(self, idx):
        """Return the context at a given index (for debugging)."""
        if idx < len(self._contexts):
            return self._contexts[idx]
        return None
