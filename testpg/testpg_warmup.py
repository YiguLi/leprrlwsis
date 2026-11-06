# -*- coding: utf-8 -*-
"""testPg 的 warmup 上层调度：`[rate, goal]` 二维。

为什么要单独写
--------------
`hsrl_warmup.WarmupSchedule` 的 `action()` 按 `nb_turbine + 1` 维拼接（5 台风机的 a
+ 信号）；testPg 的上层只有 **2 维**，不能直接复用它的输出切片。这里保留它**唯一相关
的语义**（goal 维 = 第 6 维的生成方式）并给 rate 维一个覆盖 [0,1] 的采样：

* `rate ~ U(0,1)`，每块独立（warmup 默认 7200 步 = 1440 块 ⇒ 覆盖足够密）；
* `goal` 的生成方式 = `sig_mode`（**与 `hsrl_warmup.SIG_MODES` 同义**，2026-10-30 起
  默认 **`random` = 纯随机**）：
    - `random`（默认）：每块 `i.i.d. U(sig_lo, sig_hi)`，不看电价、不看上一块；
    - `price_walk`（旧口径）：起点 0，低谷(≤300) 上调 / 平峰(580) 不动 / 高峰(≥980) 下调，
      步幅 `U(--warmup_sig_step_lo, --warmup_sig_step_hi)`，夹在 [sig_lo, sig_hi]
      —— 与 `WarmupSchedule.sig_delta` **逐字同式**（含同一组常量）；
    - `fixed`：恒 `sig_value`（更旧的"信号恒 0"口径）。
  这里不生成 5 个风机维。

`--warmup_sched random` 时 `action()` 返回 None ⇒ 主循环退回
`agent_H.random_action()`（与 aifflor 的对照臂同口径）。
"""
import numpy as np

import hsrl_warmup as hwarm


class PGWarmup:
    """warmup 期 `[rate, goal]` 调度（确定性随机流，不消耗训练流的 RNG）。"""

    def __init__(self, enabled=True, sig_mode=hwarm.SIG_MODE_DEFAULT,
                 sig_step_lo=hwarm.SIG_STEP_LO, sig_step_hi=hwarm.SIG_STEP_HI,
                 sig_lo=hwarm.SIG_LO, sig_hi=hwarm.SIG_HI,
                 sig_value=hwarm.SIG_VALUE, seed=0):
        self.enabled = bool(enabled)
        self.sig_mode = str(sig_mode)
        self.sig_step_lo = float(sig_step_lo)
        self.sig_step_hi = float(sig_step_hi)
        self.sig_lo, self.sig_hi = float(sig_lo), float(sig_hi)
        self.sig_value = float(sig_value)
        # 独立随机流（seed 偏移 ⇒ 与 hsrl_warmup / 训练流互不消耗）
        s = int(seed) if seed is not None and int(seed) > 0 else None
        self.rng_rate = np.random.RandomState(None if s is None else s + 300001)
        self.rng_sig = np.random.RandomState(None if s is None else s + 400001)
        self.reset()

    def reset(self):
        self._sig = self.sig_value
        self._blk = 0
        return self

    @property
    def sig(self):
        return float(self._sig)

    def sig_delta(self, price):
        """与 `WarmupSchedule.sig_delta` 逐字同式（只对 `price_walk` 有意义）。"""
        if self.sig_mode != 'price_walk' or price is None:
            return 0.0
        p = float(price)
        if p <= hwarm.PRICE_LOW + 1e-6:          # 低谷 ⇒ 目标上调（充电方向）
            return float(self.rng_sig.uniform(self.sig_step_lo, self.sig_step_hi))
        if p >= hwarm.PRICE_HIGH - 1e-6:         # 高峰 ⇒ 目标下调（放电方向）
            return -float(self.rng_sig.uniform(self.sig_step_lo, self.sig_step_hi))
        return 0.0                               # 平峰 ⇒ 不动

    def action(self, price=None):
        """返回本块的 `[rate, goal]`；`enabled=False` ⇒ None（对照臂）。"""
        if not self.enabled:
            return None
        if self.sig_mode == 'random':
            # 纯随机（**默认**）：每块 i.i.d. 重抽，不看电价、不看上一块
            self._sig = float(self.rng_sig.uniform(self.sig_lo, self.sig_hi))
        else:
            self._sig = float(np.clip(self._sig + self.sig_delta(price),
                                      self.sig_lo, self.sig_hi))
        rate = float(self.rng_rate.uniform(0.0, 1.0))
        self._blk += 1
        return np.array([rate, self._sig], dtype=float)

    def describe(self):
        if self.sig_mode == 'random':
            goal = ('goal: 纯随机 U({:.2f},{:.2f})/块（不看电价，默认）'
                    .format(self.sig_lo, self.sig_hi))
        elif self.sig_mode == 'fixed':
            goal = 'goal: 恒 {:.2f}'.format(self.sig_value)
        else:
            goal = ('goal: 起点 {:.2f}, 低谷 +U({:.2f},{:.2f}) / 平峰 0 / '
                    '高峰 -U({:.2f},{:.2f}), 夹 [{:.1f},{:.1f}]'
                    .format(self.sig_value, self.sig_step_lo, self.sig_step_hi,
                            self.sig_step_lo, self.sig_step_hi,
                            self.sig_lo, self.sig_hi))
        return '[PGWarmup] rate ~ U(0,1) | ' + goal
