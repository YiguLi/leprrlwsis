# -*- coding: utf-8 -*-
"""testPg 的上层 agent：在 `aifflor_ddpg.SP_TD3` 上只改**动作语义**（6→2 维）。

H 动作 = `[rate, goal]`：

* `rate ∈ [0, 1]` —— 降载率（1 = 满发 Pmax，0 = 停机）；
* `goal ∈ [-0.5, 0.5]` —— SOC 目标信号，映射与 aifflor 完全一致
  （`soc_target_from_signal`）。

本类只覆写 4 处（**TD3/PER/对偶本体一行不动**）：

1. `nb_actions=2`；
2. `action_amp = [0.5, 0.5]`（TD3 target smoothing 的动作幅度；rate/goal 半宽都是 0.5）；
3. 探索噪声：rate 维**恒定**幅度 `--pg_rate_noise`（默认 0.05；与 aifflor 的 WT 噪声
   "恒定不衰减"同口径），goal 维沿用 `--sig_noise_max → --sig_noise_min` 的 ε 衰减；
4. `random_action` / `_clip_box`：rate 的物理带是 [0,1]（**不是** [0,0.5]），cost 塔
   只看 clip 后的 rate（`c1` 通过 `P_F` 依赖实际发功率）。

⚠ 不覆写 `nb_turbine`：它仍 = `args.nb_turbine`（=5），只用于其它模块读取"环境机组数"；
H 的动作维数由 `nb_actions=2` 决定，两者在 testPg 里是**不同的量**（与 singleaif 的
`nb_wt` 处理同思路）。
"""
import numpy as np
import torch

from aifflor.aifflor_ddpg import SP_TD3


class SP_TD3_PG(SP_TD3):
    """TD3（上层 H）—— 动作 = `[rate, goal]`。"""

    def __init__(self, nb_states, args, **kwargs):
        super(SP_TD3_PG, self).__init__(nb_states=nb_states, nb_actions=2,
                                        args=args, **kwargs)
        # ---- 动作幅度（target smoothing 用；rate/goal 的半宽都是 0.5）----
        self.action_amp = np.array([0.5, 0.5], dtype=np.float32)
        # ---- cost 塔的动作裁剪：只对第 1 维 rate 生效（上界 1.0）----
        self.na_box = 1
        # ---- 探索噪声 ----
        r_noise = float(getattr(args, 'pg_rate_noise', 0.05))
        _r_floor = getattr(args, 'pg_rate_noise_floor', None)
        r_floor = r_noise if _r_floor is None else float(_r_floor)
        s_max = float(getattr(args, 'sig_noise_max', 0.2))
        s_min = float(getattr(args, 'sig_noise_min', 0.0))
        self.pg_rate_noise = r_noise
        self.pg_rate_noise_floor = r_floor
        self.explore_amp = np.array([r_noise, s_max], dtype=np.float32)
        self.explore_floor = np.array([r_floor, s_min], dtype=np.float32)

    # ------------------------------------------------------------------
    def random_action(self):
        """warmup 的对照臂：rate ~ U(0,1)、goal ~ U(-0.5,0.5)。"""
        self.a_t = np.array([np.random.uniform(0.0, 1.0),
                             np.random.uniform(-0.5, 0.5)], dtype=float)
        return self.a_t

    def _clip_box(self, a):
        """cost 塔的动作 = `clip(rate, 0, 1)` + goal 原样。

        `c1` 只通过 `P_F`（⇒ `ptur = rate·Pmax`）依赖 rate；goal 在块末才通过 gap 罚
        进入成本，且 gap 罚在 `sum_c1` 里按 `action_H[1]` 另行计算 ⇒ 无需裁剪。
        越界梯度由 profit 塔的 `penalty_NN_H` 提供（与 aifflor 的 cost_clip 同思路）。
        """
        if (not self.cost_clip) or a is None or a.shape[-1] != 2:
            return a
        out = a.clone()
        out[..., 0] = torch.clamp(a[..., 0], 0.0, 1.0)
        return out

    def describe_action_layout(self):
        return ('H 动作: [rate ∈ [0,1], goal ∈ [-0.5,0.5]] | '
                '探索: rate 恒定 ±{:.3f}, goal ±{:.3f}→{:.3f} | '
                'target-smoothing amp = [0.5, 0.5]'
                .format(self.pg_rate_noise, self.explore_amp[1],
                        self.explore_floor[1]))
