import numpy as np
import random

import PER.PER_utils as PER_utils


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def append(self, obs_t, action, reward, cost, obs_tp1, done):
        data = (obs_t, action, reward, cost, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, costs, obses_tp1, dones = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, cost, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            costs.append(cost)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(costs), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        cost_batch

        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    """PER replay +（可选）**按来源分池采样**。

    分池采样的动机（`documentation/aifflor_branch_wide_coverage_plan.md` §4.2/§4.4）：
    `aifflor` 的 branch 方案会让同一个块首 state 产生 5 条转移（actor 1 条 + 测试动作 4 条）
    ⇒ 按自然比例采样时 branch 会占 80%，actor 被淹没；而 **actor loss / λ / EMA /
    curriculum / CSV / TB / test 只允许用 actor 数据**。
    因此这里给每条转移挂一组**元数据标签**（`src` / `mode` / `block_id` / `lower_ver`），
    并提供 `sample_pool(pool, n)`：在给定池内做**优先度比例**采样。

    ⚠ 元数据**只用于采样/诊断，绝不进 critic state**。
    ⚠ 不调用 `set_meta` 时（所有既有项目）行为**逐位不变**：
      `_src` 在第一次 `set_meta` 前是 `None`，`pool_idxes()` 退化为"全体"，
      `sample()` 的数学与原实现完全一致（同一个 `_sample_proportional` + 同一个权重公式）。
    """

    #: 元数据字段 → (默认值, dtype)
    META_FIELDS = {
        'src': (0, np.int8),        # 0 = actor，1 = branch（见 aifflor_branch.SRC_*）
        'mode': (0, np.int8),       # 动作模式（见 aifflor_branch.MODE_*）
        'block_id': (-1, np.int32),  # 同一块首的所有 branch 共享
        'lower_ver': (0, np.int32),  # 下层 BESS 版本（branch 标签是否过期的诊断口径）
    }

    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = PER_utils.SumSegmentTree(it_capacity)
        self._it_min = PER_utils.MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        # ---- 元数据（惰性分配：只有 set_meta 被调用才占内存）----
        self._meta = {}             # name -> np.ndarray(maxsize)
        self._occupied = None       # np.bool_ 数组：该槽是否已被写过
        self._src = None            # np.int8 数组：0=actor / 1=branch
        self._n_src = np.zeros(2, dtype=np.int64)   # 当前 **被占用** 槽里各 src 的条数

    def append(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().append(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    # ==================================================================
    # 元数据（来源标签）
    # ==================================================================
    def _meta_arr(self, name):
        arr = self._meta.get(name)
        if arr is None:
            default, dtype = self.META_FIELDS.get(name, (0, np.int32))
            arr = np.full(int(self._maxsize), default, dtype=dtype)
            self._meta[name] = arr
        return arr

    def set_meta(self, idx, src=0, mode=0, block_id=-1, lower_ver=0):
        """给 `idx` 这条转移打标签（`src`/`mode`/`block_id`/`lower_ver`）。

        `src` 的变化同时维护 `n_actor` / `n_branch`（供"池里有没有 branch"判断）。
        """
        idx = int(idx)
        if not (0 <= idx < int(self._maxsize)):
            return
        if self._occupied is None:
            self._occupied = np.zeros(int(self._maxsize), dtype=bool)
            self._src = self._meta_arr('src')
        arr = self._src
        old = int(arr[idx])
        new = int(src)
        if self._occupied[idx]:
            self._n_src[old] -= 1
        else:
            self._occupied[idx] = True
        arr[idx] = new
        self._n_src[min(max(new, 0), 1)] += 1
        self._meta_arr('mode')[idx] = int(mode)
        self._meta_arr('block_id')[idx] = int(block_id)
        self._meta_arr('lower_ver')[idx] = int(lower_ver)

    @property
    def has_meta(self):
        return self._occupied is not None

    @property
    def n_branch(self):
        return int(self._n_src[1]) if self.has_meta else 0

    @property
    def n_actor(self):
        return int(self._n_src[0]) if self.has_meta else 0

    def pool_idxes(self, pool):
        """池内**已占用**槽的下标（无元数据 ⇒ 全体，等价于"只有一个池"）。"""
        L = len(self._storage)
        if L == 0:
            return np.zeros(0, dtype=np.int64)
        if not self.has_meta:
            return np.arange(L, dtype=np.int64)
        return np.flatnonzero(self._src[:L] == int(pool)).astype(np.int64)

    def mode_idxes(self, pool, mode):
        if not self.has_meta or 'mode' not in self._meta:
            return self.pool_idxes(pool)
        L = len(self._storage)
        return np.flatnonzero((self._src[:L] == int(pool))
                              & (self._meta['mode'][:L] == int(mode))).astype(np.int64)

    # ==================================================================
    # 采样
    # ==================================================================
    def _sample_from_pool(self, cand, n):
        """在候选下标 `cand` 内做**优先度比例**采样（含兜底）。

        做法 = `_sample_proportional` 过采样 + 池过滤 + 重抽：
        这样采到的分布仍是"池内 ∝ priority^α"，只是要额外抽几次。
        重抽 `guard` 次仍不够（池占比极小时可能）⇒ 池内**均匀**兜底，
        保证 batch 大小恒等于请求值（不静默缩短 batch）。
        """
        n = int(n)
        cand = np.asarray(cand, dtype=np.int64)
        if n <= 0 or cand.size == 0:
            return []
        allowed = np.zeros(len(self._storage), dtype=bool)
        allowed[cand] = True
        out = []
        guard = 0
        while len(out) < n and guard < 40:
            need = n - len(out)
            draw = self._sample_proportional(min(max(2 * need, 8), 8 * max(need, 1)))
            for i in draw:
                if len(out) >= n:
                    break
                if 0 <= i < allowed.size and allowed[i]:
                    out.append(int(i))
            guard += 1
        while len(out) < n:
            out.append(int(cand[np.random.randint(0, cand.size)]))
        return out

    def sample_pool(self, pool, n, by_mode=False):
        """在 `src == pool` 的池内采样 `n` 条；`by_mode` ⇒ branch 池内部按 mode 均衡。"""
        n = int(n)
        if n <= 0:
            return []
        if not by_mode:
            return self._sample_from_pool(self.pool_idxes(pool), n)
        # ---- mode 均衡：把 n 均分到"池内实际存在的 mode"，余数给前几个 ----
        present = []
        if self.has_meta and 'mode' in self._meta:
            L = len(self._storage)
            m_all = self._meta['mode'][:L][self._src[:L] == int(pool)]
            if m_all.size:
                uniq, cnt = np.unique(m_all, return_counts=True)
                present = [(int(u), int(c)) for u, c in zip(uniq, cnt)]
        if len(present) <= 1:
            return self._sample_from_pool(self.pool_idxes(pool), n)
        out = []
        base = n // len(present)
        rem = n - base * len(present)
        # 按条数从多到少派发余数（稳定的、与 mode id 无关的派发顺序）
        order = sorted(range(len(present)), key=lambda i: (-present[i][1], present[i][0]))
        for rank, i in enumerate(order):
            m = present[i][0]
            k = base + (1 if rank < rem else 0)
            if k <= 0:
                continue
            out += self._sample_from_pool(self.mode_idxes(pool, m), k)
        return out

    def sample_by_idxes(self, idxes, beta):
        """给定下标 ⇒ 标准的 PER 输出（含 importance weight）。权重公式与原 `sample` 逐位相同。"""
        idxes = [int(i) for i in idxes]
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta, beta_increment_per_sampling):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """

        assert beta > 0
        # A2: β 由调用方 (agent.update_policy) 推进并传入。
        # 原实现此处做局部递增但未写回实例，退火永不生效；现已移除。
        # 历史口径（整池一次抽 batch_size）⇒ 交给 sample_by_idxes 统一算权重，
        # **数学与原实现逐位相同**（同一个 _sample_proportional + 同一个权重公式）。
        return self.sample_by_idxes(self._sample_proportional(batch_size), beta)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
