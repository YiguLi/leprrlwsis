# -*- coding: utf-8 -*-
"""`yawflor` 环境层：FLORIS 后端（`FlorisPhysicsYaw`）+ 被控对象（`WindBatteryYaw`）。

分工（`yawflor/README.md` §5）
-----------------------------
* `FlorisPhysicsYaw` —— **纯函数式物理**：`power_deg(u, γ)` / `prefetch_deg(us, γ)`；
  内部做 ①**逐块预取**（块首一次向量化算完该块用到的 5~6 个风速，块内直接查表：
  单条 40 ms → 批量 0.58 ms/条）②**NaN 守卫**（`gauss` 挠曲在**全场同时 ≥45°** 时
  `wake_deflection/gauss.py:189` 的 `sqrt` 出 NaN ⇒ 重算 γ=0 并保证有限值）
  ③可选 `--yaw_symmetrize`（R2 对照臂）。
* `WindBatteryYaw(env.WindBattery)` —— 只改两件事：**物理后端**（FLORIS 取代 WFSim）
  与**标称动作**（无量纲偏航 `n = 0 ⇒ γ = 0°`，取代轴向因子 1/3）。其余（储能、
  电价、波动基准 `P_G_prev`、观测布局、九例接口）**逐字继承**，保证与旧环境同形。

动作语义（单一真源 = `yawflor_turbine`）
----------------------------------------
网络动作 = 6 维 `[n₁…n₅, signal]`，`n ∈ [−1,1]`、`signal ∈ [−0.5,0.5]`；
物理偏航角 `γ = 25° · clip(n, −1, 1)`。轴向因子**固定 a = 1/3**（不是动作）。

执行器现实性（默认全关，见 `yawflor/README.md` §4）
---------------------------------------------------
`YAW_RATE_MAX_DEG_PER_BLOCK` 默认 `None`（不限制）；`YAW_TAU_S` 默认 0（关）；
`YAW_TRANSPORT_DELAY` 默认 `False`（关 —— 开启会使 env **有记忆**，破坏 test9 的
确定性判据 ⇒ 只能作对照臂，且这里直接拒绝而不是静默改变语义）。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))        # yawflor/
_ROOT = os.path.dirname(_HERE)                            # 仓库根
if _ROOT not in sys.path:                                 # 生产模块：env/util 在仓库根
    sys.path.insert(0, _ROOT)

from env import WindBattery                                # noqa: E402  （旧环境基类）
from yawflor import yawflor_turbine as YT                   # noqa: E402  （口径单一真源）


# ===========================================================================
# 1. FLORIS 物理后端
# ===========================================================================
class FlorisPhysicsYaw:
    """FLORIS 后端：逐块预取 + NaN 守卫 + 可开关缓存。

    Parameters
    ----------
    nt : int                    机组台数（主臂 5）
    gamma_max : float           动作带（deg；主臂 25）
    cache_enabled : bool        True（默认）= 启用逐块预取查表；False = **全部直算**
                                （契约测试用它对照"预取路径 ≡ 直算路径"）
    **model_kw : 透传 `yawflor_turbine.build_model`
                                （如 `turbine_type='nrel_5MW'`、`wind_shear=0.0`）

    统计量：`n_batch`（批量调用次数）/ `n_direct`（单条调用次数）/ `n_hit`（查表命中）/
    `n_nonfinite`（触发 NaN 守卫的次数）—— 供 `summary()` 打印，也是"预取真的生效了"
    的证据。
    """

    def __init__(self, nt=YT.NT, gamma_max=YT.GAMMA_MAX, cache_enabled=True, **model_kw):
        self.model, self.info = YT.build_model(nt=nt, **model_kw)
        self.nt = int(nt)
        self.gamma_max = float(gamma_max)
        self.cache_enabled = bool(cache_enabled)
        self._cache = {}
        self.n_batch = 0
        self.n_direct = 0
        self.n_hit = 0
        self.n_nonfinite = 0

    # ---------------- 动作换算 ----------------
    def yaw_deg(self, n):
        """无量纲动作 → 偏航角 [deg]（**夹到 ±gamma_max 带内**，结构性排除 NaN 区）。"""
        arr = np.asarray(n, dtype=float).reshape(-1)[:self.nt]
        if arr.size < self.nt:                                  # 允许标量广播
            arr = np.full(self.nt, float(arr.reshape(-1)[0]) if arr.size else 0.0)
        return np.clip(arr, -1.0, 1.0) * self.gamma_max

    # ---------------- 缓存键 ----------------
    @staticmethod
    def _key(u, gamma):
        return (round(float(u), 12), tuple(np.round(np.asarray(gamma, dtype=float), 12).tolist()))

    # ---------------- 预取 ----------------
    def prefetch_deg(self, us, gamma_deg):
        """**替换**缓存：把 `us` 这批风速在**同一个偏航向量**下的功率一次算完。

        缓存设计成"只服务当前块"（每次预取先清空）⇒ 内存 O(块长)，不会随训练步数增长；
        块内若动作变了（正常不会，H 的动作按块固定）⇒ 缓存未命中，退回单条直算。
        """
        us = np.atleast_1d(np.asarray(us, dtype=float))
        gamma = np.asarray(gamma_deg, dtype=float).reshape(-1)[:self.nt]
        if us.size == 0:
            return
        yaws = np.tile(gamma, (us.size, 1))
        farm, _ = YT.eval_batch(self.model, us, yaws,
                                symmetrize=YT.YAW_SYMMETRIZE)
        self.n_batch += 1
        self._cache = {self._key(us[i], gamma): float(farm[i]) for i in range(us.size)}
        return farm

    # ---------------- 求值 ----------------
    def power_deg(self, u, gamma_deg):
        """风场总出力 [MW]（含 NaN 守卫；命中缓存则直接返回预取值）。

        不变量（`agentutil/test_yaw_env_contract.py` 断言）：
        命中缓存与直算**逐位相同** —— 缓存只是省调用，不改变数值。
        """
        u = float(u)
        gamma = np.asarray(gamma_deg, dtype=float).reshape(-1)[:self.nt]
        key = self._key(u, gamma)
        if self.cache_enabled and key in self._cache:
            self.n_hit += 1
            return float(self._cache[key])
        # 未命中 ⇒ 直算，**不入缓存**：缓存严格等于"当前预取窗口"（≤ K 条），
        # 因此内存 O(1) 且不随训练步数增长（连续动作下 (u,γ) 键几乎不重复）。
        return self._compute(u, gamma, store=False)

    def _compute(self, u, gamma, store=False):
        farm = YT.farm_power(self.model, u, gamma, symmetrize=YT.YAW_SYMMETRIZE)
        self.n_direct += 1
        if not np.isfinite(farm):
            # R3：全场同时大偏航时 gauss 挠曲模型 sqrt 出 NaN。结构性防线是 ±25° 带
            # （`yaw_deg` 已夹带），这里是**兜底**：退回 γ=0（有限值由构造保证）。
            self.n_nonfinite += 1
            farm = YT.farm_power(self.model, u, np.zeros(self.nt))
            if not np.isfinite(farm):                          # pragma: no cover
                farm = 0.0
        farm = float(farm)
        if store:
            self._cache[self._key(u, gamma)] = farm
        return farm

    # ---------------- env 接口 ----------------
    def PowerOutput(self, u0, act_wind):
        """`env.WindBattery` 的后端接口：动作是**无量纲 n**（不带执行器约束）。"""
        return self.power_deg(u0, self.yaw_deg(act_wind))

    # ---------------- 诊断 ----------------
    def summary(self):
        tot = self.n_batch + self.n_direct
        return ('[yawflor-physics] FLORIS {} | 批量调用 {} 次 / 单条调用 {} 次 / 查表命中 {} 次'
                ' | 单条占比 {:.1%} | NaN 守卫触发 {} 次'
                .format(self.info.get('floris_version', '?'), self.n_batch, self.n_direct,
                        self.n_hit, (self.n_direct / tot if tot else 0.0), self.n_nonfinite))


# ===========================================================================
# 2. 被控对象（继承旧环境，只改物理与标称动作）
# ===========================================================================
class WindBatteryYaw(WindBattery):
    """偏航 + FLORIS 环境。

    与 `env.WindBattery` 的差异（**仅此三处**，其余逐字继承）：
    1. `nominal_action_value = 0.0`（无量纲偏航动作；旧环境是轴向因子 0.3333）；
    2. `physics` 必填（`FlorisPhysicsYaw`）⇒ `PowerOutput` 走 FLORIS；
    3. `PowerOutput` 多做一件事：**块首逐块预取**该块用到的风速（观测布局/守卫/
       `P_G_prev` 基准/储能/电价/九例接口全部与父类同形）。
    """

    nominal_action_value = 0.0        # n = 0 ⇒ γ = 0°（单机自损最小）
    nominal_action = None
    BLOCK = 5                         # H 决策周期（步）；与 hsrl_warmup / 主循环一致
    PREFETCH_K = 6                    # 每次预取算 u_idx … u_{idx+K−1}（块内 5 步 + 跨块那一次）

    # ---- 执行器现实性（默认全关 ⇒ 逐位不变；见 README §4）----
    yaw_rate_max_deg_per_block = YT.YAW_RATE_MAX_DEG_PER_BLOCK
    yaw_tau_s = YT.YAW_TAU_S
    yaw_transport_delay = YT.YAW_TRANSPORT_DELAY

    def __init__(self, max_E, num_turbine, upload, engine=None, physics=None):
        if physics is None:
            raise ValueError('WindBatteryYaw 必须给 physics=FlorisPhysicsYaw(...)'
                             '（FLORIS 后端；旧 MATLAB 路径请直接用 env.WindBattery）')
        if YT.YAW_TRANSPORT_DELAY:
            raise NotImplementedError(
                'YAW_TRANSPORT_DELAY 默认必须为 False：开启会让 env **有记忆**，'
                '破坏 test9 的确定性判据（yawflor/README.md §4）。若要作对照臂，'
                '请显式在 WindBatteryYaw 里实现延迟队列并相应放宽该判据。')
        super().__init__(max_E, num_turbine, upload, engine, physics=physics)
        self._wind_mode = None         # 'day' | 'noise' | 'scenario'（序列本体动态读取）
        self._pf_state = None          # (首下标, 末下标, γ 键)：当前预取窗口
        self._gamma_prev_block = None  # 速率限制/一阶延迟的状态（默认关时不使用）
        self.n_prefetch = 0
        if physics.nt != int(num_turbine):
            raise ValueError('physics.nt({}) != num_turbine({})'
                             .format(physics.nt, num_turbine))

    # ------------------------------------------------------------------
    # 风速序列（供预取用）—— 三种模式与父类的取数逻辑一一对应
    # ------------------------------------------------------------------
    def reset(self):
        obs = super().reset()                       # 父类里已设 wind_day/count_wind/u
        self._wind_mode = 'day'
        return obs

    def TestReset(self, day=0):
        # ⚠ 测试路径的**后续风速恒取 `wind_noise` 第 1 列**（`TestWindSpeed` 不看 day）；
        #   `day != 0` 只影响**初始点**（取自 `test_scenarios`）⇒ 预取序列也按 wind_noise，
        #   首点不匹配只会让那次调用退回直算（键是 (u, γ)，绝不会取到错值）。
        self._wind_mode = 'noise'
        return super().TestReset(day)

    def ScenarioReset(self, wind_seq, price_seq):
        self._wind_mode = 'scenario'
        return super().ScenarioReset(wind_seq, price_seq)

    def _wind_series(self):
        """返回 `(序列 ndarray, 回绕方式)`；无法推导时 `(None, None)`。

        ⚠ **每次都读当前属性，不做快照**：评测脚本（`test_five_days.py`）会先
        `env.reset()` 再把 `env.wind_day` 换成抽中的那一天 ⇒ 快照式缓存会用错风速
        序列（数值仍正确 —— 键是 (u, γ)、未命中就直算 —— 只是预取全部落空）。
        """
        mode = self._wind_mode
        if mode == 'scenario':
            seq, wrap = self.scenario_wind, 'clamp'
        elif mode == 'noise':
            seq, wrap = self.wind_noise.iloc[:, 1], 'wrap'
        else:
            seq, wrap = getattr(self, 'wind_day', None), 'wrap'
        if seq is None or len(seq) == 0:
            return None, None
        return np.asarray(seq, dtype=float), wrap

    def _upcoming_wind(self, k):
        """从当前时刻起的 k 个风速（含当前）；无法推导时返回 None（⇒ 不预取）。"""
        seq, wrap = self._wind_series()
        if seq is None:
            return None
        idx = self._current_index()
        n = len(seq)
        if wrap == 'clamp':
            out = [seq[min(idx + i, n - 1)] for i in range(k)]
        else:
            out = [seq[(idx + i) % n] for i in range(k)]
        return np.asarray(out, dtype=float)

    def _is_block_start(self):
        """块首 = H 的决策点（**只用于执行器约束**；预取不依赖它，见 `_need_prefetch`）。"""
        if self._wind_mode == 'scenario':
            return int(self.scenario_idx) % self.BLOCK == 0
        return int(self.count_wind) % self.BLOCK == 0

    def _current_index(self):
        """当前风速在序列里的下标（与 `CalWindSpeed`/`TestWindSpeed`/`ScenarioStep` 同源）。"""
        if self._wind_mode == 'scenario':
            return int(self.scenario_idx)
        return int(self.count_wind)

    def _need_prefetch(self, idx, gamma):
        """是否需要（重新）预取。判据与**调用点无关**，只看"窗口够不够 + 动作变没变"：

        1. 还没预取过 ⇒ 要；
        2. 本块动作变了（γ 键不同）⇒ 要 —— 这是训练路径的主要触发点
           （H 每块给新动作 ⇒ 每块恰好一次批量调用）；
        3. 当前下标**越过了**窗口末端 ⇒ 要（常动作路径下的滚动补充）。

        这样既不会每步都调用（固定开销 ~40 ms/次，实测见
        `agentutil/_scratch/probe_floris_timing.py`），也不会因"块末那一步的
        `count_wind % 5 == 0`"而重复预取同一动作。
        """
        if self._pf_state is None:
            return True
        i0, i1, gkey = self._pf_state
        if gkey != self._gamma_key(gamma):
            return True
        return int(idx) > int(i1)

    @staticmethod
    def _gamma_key(gamma):
        return tuple(np.round(np.asarray(gamma, dtype=float), 12).tolist())

    # ------------------------------------------------------------------
    # 执行器约束（默认关闭；开启时才改变动作）
    # ------------------------------------------------------------------
    def _apply_executor(self, gamma, block_start):
        """块级执行器约束。默认（速率 None / τ=0）⇒ **原样返回**（逐位不变）。

        * 速率限制：`|Δγ| ≤ YAW_RATE_MAX_DEG_PER_BLOCK`（相对**上一块**的实际角度）；
        * 一阶延迟：`γ ← γ_prev + (1−e^{−Δt/τ})·(γ_target − γ_prev)`，Δt = 5 min；
          ⚠ 开启后 env **有状态**（只用于对照臂）。
        """
        g = np.asarray(gamma, dtype=float).copy()
        if self.yaw_rate_max_deg_per_block is None and self.yaw_tau_s <= 0.0:
            return g
        prev = self._gamma_prev_block
        if prev is None:
            prev = np.zeros_like(g)
        if self.yaw_rate_max_deg_per_block is not None:
            g = YT.limit_yaw_rate(prev, g, self.yaw_rate_max_deg_per_block)
        if self.yaw_tau_s > 0.0:
            dt_s = self.BLOCK * 60.0
            alpha = 1.0 - np.exp(-dt_s / float(self.yaw_tau_s))
            g = prev + alpha * (g - prev)
        self._gamma_prev_block = np.asarray(g, dtype=float).copy()
        return g

    # ------------------------------------------------------------------
    # 物理：与父类**同形**（守卫 / 6 维动作截断 / 标量返回），只换后端
    # ------------------------------------------------------------------
    def PowerOutput(self, u0, act_wind):
        if not (3.5 <= float(u0) <= 25):
            return 0.0
        act = np.asarray(act_wind, dtype=float).reshape(-1)[:self.num_turbine]
        gamma = self.physics.yaw_deg(act)           # ±25° 带内（结构性排除 NaN 区）
        if self.yaw_rate_max_deg_per_block is not None or self.yaw_tau_s > 0.0:
            if self._is_block_start():
                gamma = self._apply_executor(gamma, True)
        if self.physics.cache_enabled:
            idx = self._current_index()
            if self._need_prefetch(idx, gamma):
                us = self._upcoming_wind(self.PREFETCH_K)
                if us is not None:
                    self.physics.prefetch_deg(us, gamma)
                    self._pf_state = (idx, idx + len(us) - 1, self._gamma_key(gamma))
                    self.n_prefetch += 1
        return self.physics.power_deg(float(u0), gamma)

    # ------------------------------------------------------------------
    def summary(self):
        return ('[{}] 预取 {} 次 | {}'.format(type(self).__name__, self.n_prefetch,
                                             self.physics.summary()))
