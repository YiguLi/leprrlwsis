# -*- coding: utf-8 -*-
"""`aifflor` 环境层：FLORIS 后端（`AifflorPhysics`）+ 被控对象（`WindBatteryAIF`）。

分工
----
* `AifflorPhysics` —— **纯函数式物理**：`power(u, a)` / `power_deg` 等价物；
  内部做 ①**非有限值守卫**（标定后的窄尾流在极端动作下可能出负速度/NaN）
  ②按需 `prefetch`（把一批 `(u, a)` 一次算完）。
* `WindBatteryAIF(env.WindBattery)` —— **只改一处**：物理后端换成 FLORIS。
  其余（储能、电价、波动基准 `P_G_prev`、观测布局 `[pr, u, E, P_last]`、
  标称动作、九例接口、`3.5≤u≤25` 守卫）**逐字继承** `env.py`。

⚠ 为什么 `WindBatteryAIF` 只改这一处就够
------------------------------------------
`env.py::WindBattery` 的物理出口**唯一**：`PowerOutput(u0, act_wind)`，被
`reset` / `TestReset` / `ScenarioReset` / `step` / `TestStep` / `ScenarioStep`
共 6 处调用；且 `physics is not None` 时分派给后端（`env.py:139-140`）。
⇒ 换引擎 = 换这个后端，**`env.py` 一行都不用动**。

动作语义（单一真源 = `aifflor_turbine`，**与 td3per 逐字相同**）
----------------------------------------------------------------
网络动作 = 6 维 `[a₁…a₅, signal]`，`a ∈ [0, 0.5]`（轴向因子）、`signal ∈ [−0.5,0.5]`；
航向恒为 0（FLORIS 的 `yaw_angles ≡ 0`，相当于 WFSim 的 `Control.phi = 0`）。
⇒ `td3per/td3per_myfunc.py` 的 `CheckAction` / `PENALTY_NN_H_COEF=400` /
`violation_amount` 边界**全部无需修改**（这是与 `yawflor` 最大的不同：
yawflor 把动作带换成了 `[−1,+1]`，所以必须改 myfunc；aifflor 不用）。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))            # aifflor/
_ROOT = os.path.dirname(_HERE)                                # 仓库根
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from env import WindBattery                                   # noqa: E402  （旧环境基类）
from aifflor import aifflor_turbine as AT                     # noqa: E402  （口径单一真源）


# ===========================================================================
# 1. FLORIS 物理后端
# ===========================================================================
class AifflorPhysics:
    """FLORIS 后端（轴向因子 `a` 通道）：非有限值守卫 + 可选预取。

    Parameters
    ----------
    nt : int                机组台数（主臂 5）
    calibrated : bool       `True`（默认）施加标定后的尾流参数
    cache_enabled : bool    `True`（默认）允许 `prefetch` 填充查表缓存
    **model_kw :            透传（保留给将来；主臂不传）

    统计量：`n_batch` / `n_direct` / `n_hit` / `n_nonfinite`，供 `summary()` 打印。
    """

    def __init__(self, nt=AT.NT, calibrated=True, cache_enabled=True, verbose=False):
        self.model, self.info = AT.build_model(nt=nt, calibrated=calibrated,
                                               verbose=verbose)
        self.nt = int(nt)
        self.calibrated = bool(calibrated)
        self.cache_enabled = bool(cache_enabled)
        self._cache = {}
        self.n_batch = 0
        self.n_direct = 0
        self.n_hit = 0
        self.n_nonfinite = 0        # 出口守卫兜底次数（退回标称动作）
        self.n_cache_bad = 0        # 试图进入/已在缓存里的非有限值条数

    # ---------------- 动作换算（学习带 [0,0.5]；物理层再抬到安全下界） ----------------
    def a_clip(self, act):
        """动作 → 轴向因子向量（长度补齐到 `nt`）。

        ⚠ 夹带分两段（**这是本环境唯一的"动作 ≠ 物理"之处**）：
        * 先按**学习带** `[A_LO, A_HI] = [0, 0.5]` 夹（与 `td3per` 同口径）；
        * 再抬到**引擎安全下界** `A_PHYS_LO = 0.025`（`a = 0` 会让高斯尾流 0/0 出
          NaN；见 `aifflor_turbine.A_PHYS_LO` 的论证与实测）。
        ⇒ `a ∈ [0, 0.025)` 这一段在物理上按 `0.025` 计。这是**刻意的、有文档的**映射，
        不是静默兜底：区间宽度 0.025 只占学习带的 5%，且 `a → 0` 本身是停机奇点。
        """
        arr = np.asarray(act, dtype=float).reshape(-1)[:self.nt]
        if arr.size < self.nt:                                  # 允许标量广播
            arr = np.full(self.nt, float(arr.reshape(-1)[0]) if arr.size
                          else AT.A_NOMINAL)
        arr = np.clip(arr, AT.A_LO, AT.A_HI)
        return np.clip(arr, AT.A_PHYS_LO, None)

    # ---------------- 缓存键 ----------------
    @staticmethod
    def _key(u, a):
        return (round(float(u), 12),
                tuple(np.round(np.asarray(a, dtype=float), 12).tolist()))

    # ---------------- 预取 ----------------
    def prefetch(self, us, act):
        """**替换**缓存：把 `us` 这批风速在**同一个动作向量**下一次算完。

        ⚠ 只把**有限值**写进缓存：物理层 `AT.eval_batch` 已做逐条重试 + 标称兜底，
        但这里再挡一道（NaN 绝不能进缓存 —— 历史 bug 就是它绕过了守卫）。
        """
        us = np.atleast_1d(np.asarray(us, dtype=float))
        a = self.a_clip(act)
        if us.size == 0:
            return None
        acts = np.tile(a, (us.size, 1))
        farm, _ = AT.eval_batch(self.model, us, acts)
        self.n_batch += 1
        self._cache = {}
        for i in range(us.size):
            v = float(farm[i])
            if np.isfinite(v):
                self._cache[self._key(us[i], a)] = v
            else:
                self.n_cache_bad += 1
        return farm

    def _compute(self, u, a):
        """直算一次（不做守卫 —— 守卫统一放在 `power()` 的出口）。"""
        farm = AT.farm_power(self.model, u, a)
        self.n_direct += 1
        return float(farm)

    def _fallback(self, u):
        """兜底：退回标称动作 `a = 1/3`（实测该点尾流最宽、必有限）。"""
        self.n_nonfinite += 1
        farm = AT.farm_power(self.model, u, np.full(self.nt, AT.A_IND))
        return float(farm) if np.isfinite(farm) else 0.0

    # ---------------- 求值 ----------------
    def power(self, u, act):
        """风场总出力 [MW]（**出口统一守卫**；命中缓存亦然）。

        ⚠ 为什么守卫必须在**出口**而不是只放在直算分支：
        历史 bug（训练 step≈3386 崩在 `PER_buffer: assert priority > 0`）的根因是
        `prefetch` 把 NaN 存进了缓存，而"缓存命中"直接 return ⇒ **绕过**了直算分支
        里的守卫，NaN 一路变成 `ptur` → 观测 → TD 误差 → PER 优先级。
        现在缓存命中也要过同一道检查，物理层（`AT.eval_batch`）另有逐条重试 + 标称兜底。
        """
        u = float(u)
        a = self.a_clip(act)
        key = self._key(u, a)
        if self.cache_enabled and key in self._cache:
            self.n_hit += 1
            farm = float(self._cache[key])
            if np.isfinite(farm):
                return farm
            # 缓存里是坏值 ⇒ 丢弃并即时重算（下面走直算 + 出口守卫）
            self._cache.pop(key, None)
            self.n_cache_bad += 1
        farm = self._compute(u, a)
        if not np.isfinite(farm):
            farm = self._fallback(u)
        return float(farm)

    # ---------------- env 接口 ----------------
    def PowerOutput(self, u0, act_wind):
        """`env.WindBattery` 的后端接口（动作 = **轴向因子**，与 td3per 相同）。"""
        return self.power(u0, act_wind)

    # ---------------- 诊断 ----------------
    def summary(self):
        tot = self.n_batch + self.n_direct
        fs = AT.stats()
        return ('[aifflor-physics] FLORIS {} | 标定={} | 批量 {} 次 / 单条 {} 次 / '
                '命中 {} 次 | 单条占比 {:.1%} | 出口兜底 {} 次 / 坏缓存 {} 条 || '
                'FLORIS 层: 评估点 {} | 坏点 {} ({:.3f}%) | 重试救回 {} | 退回标称 {}'
                .format(self.info.get('floris_version', '?'), self.calibrated,
                        self.n_batch, self.n_direct, self.n_hit,
                        (self.n_direct / tot if tot else 0.0), self.n_nonfinite,
                        self.n_cache_bad, fs['n_eval'], fs['n_bad_in_batch'],
                        100.0 * fs['n_bad_in_batch'] / max(fs['n_eval'], 1),
                        fs['n_retry_ok'], fs['n_fallback']))


# ===========================================================================
# 2. 被控对象（继承旧环境，**只改物理后端**）
# ===========================================================================
class WindBatteryAIF(WindBattery):
    """轴向因子 + FLORIS 环境。

    与 `env.WindBattery` 的差异（**仅此一处**）：
    `physics` 必填（`AifflorPhysics`）⇒ `PowerOutput` 走 FLORIS。
    ⚠ **标称动作不改**（仍是 `nominal_action_value = 0.3333`），因为动作语义
    与 `td3per` 完全相同 —— 这是 aifflor 与 yawflor 的关键区别。
    """

    # nominal_action_value 继承自 WindBattery = 0.3333（轴向因子 1/3）
    # nominal_action = None（继承）

    BLOCK = 5                         # H 决策周期（步）；与 hsrl_warmup / 主循环一致
    PREFETCH_K = 5                    # 每次预取算 u_idx+1 … u_idx+K

    def __init__(self, max_E, num_turbine, upload, engine=None, physics=None):
        if physics is None:
            raise ValueError('WindBatteryAIF 必须给 physics=AifflorPhysics(...)'
                             '（FLORIS 后端；旧 MATLAB 路径请直接用 env.WindBattery）')
        super().__init__(max_E, num_turbine, upload, engine, physics=physics)
        self._wind_mode = None
        self._pf_state = None
        self.n_prefetch = 0
        if physics.nt != int(num_turbine):
            raise ValueError('physics.nt({}) != num_turbine({})'
                             .format(physics.nt, num_turbine))

    # ------------------------------------------------------------------
    # 风速序列（供预取用）—— 三种模式与父类取数逻辑一一对应
    # ------------------------------------------------------------------
    def reset(self):
        obs = super().reset()
        self._wind_mode = 'day'
        return obs

    def TestReset(self, day=0):
        self._wind_mode = 'noise'
        return super().TestReset(day)

    def ScenarioReset(self, wind_seq, price_seq):
        self._wind_mode = 'scenario'
        return super().ScenarioReset(wind_seq, price_seq)

    def _wind_series(self):
        mode = self._wind_mode
        if mode == 'scenario':
            return (np.asarray(self.scenario_wind, dtype=float)
                    if self.scenario_wind is not None else None), 'clamp'
        if mode == 'noise':
            return np.asarray(self.wind_noise.iloc[:, 1], dtype=float), 'wrap'
        seq = getattr(self, 'wind_day', None)
        return (np.asarray(seq, dtype=float) if seq is not None else None), 'wrap'

    def _upcoming_wind(self, k):
        """从**下一步**起的 k 个风速（当前步由 `self.u` 给定，不在此列）。"""
        seq, wrap = self._wind_series()
        if seq is None:
            return None
        idx = int(self.count_wind)
        n = len(seq)
        if n == 0:
            return None
        if wrap == 'clamp':
            return np.asarray([seq[min(idx + 1 + i, n - 1)] for i in range(k)],
                              dtype=float)
        return np.asarray([seq[(idx + 1 + i) % n] for i in range(k)], dtype=float)

    @staticmethod
    def _a_key(a):
        return tuple(np.round(np.asarray(a, dtype=float), 12).tolist())

    def _need_prefetch(self, a):
        if not self.physics.cache_enabled:
            return False
        if self._pf_state is None:
            return True
        _key, us = self._pf_state
        return _key != self._a_key(a)

    # ------------------------------------------------------------------
    # 物理：与父类**同形**（守卫 / 6 维动作截断 / 标量返回），只换后端
    # ------------------------------------------------------------------
    def PowerOutput(self, u0, act_wind):
        if not (3.5 <= float(u0) <= 25):
            return 0.0
        act = np.asarray(act_wind, dtype=float).reshape(-1)[:self.num_turbine]
        a = self.physics.a_clip(act)
        if self._need_prefetch(a):
            us = self._upcoming_wind(self.PREFETCH_K)
            if us is not None and us.size:
                self.physics.prefetch(us, a)
                self._pf_state = (self._a_key(a), us)
                self.n_prefetch += 1
        return self.physics.power(float(u0), a)

    # ------------------------------------------------------------------
    def summary(self):
        return ('[{}] 预取 {} 次 | {}'.format(type(self).__name__, self.n_prefetch,
                                             self.physics.summary()))
