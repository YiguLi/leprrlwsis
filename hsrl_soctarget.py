"""
HSRL 下层 SOC 目标语义 —— 单一真源
====================================
本项目有**两种**下层信号语义，本模块是其中"**SOC 目标**"这一支的唯一实现：

    a6 ∈ [-0.5, 0.5]  →  soc_target = clip(3 + 5·a6, 0.5, 5.5)   (块级(5 步)固定)

另一支"影子电价"（`a6·2 ∈ (−1,1)` 作影子价、`r2` 含影子套利项、`c2` 以 SOC 安全带
为参照）留在 `per` / `prrbb` 两个项目里，本模块不涉及。

使用本模块的（下层 RL + SOC 目标）项目
--------------------------------------
* `td3per`    —— 上层 = td3perrb 口径（非分段）
* `td3prrbb`  —— 上层 = TD3 + PRRBB 分段奖励（wT/wB 风价分档保留）
* `prle`      —— 上层 = PRLE_DDPG + 分段 + LLM 阈值 relabel

三者的**下层口径逐位相同**（`test_soctarget_uniform.py` 有对照断言），
差异只在各自的上层奖励分段与观测约定。

口径（三条）
------------
1. **r2 = −1·clip_amount**（只保留 RBB 硬裁剪惩罚）
   信号语义改为 SOC 目标后，影子电价套利项失去依据；且收益已由上层 r1
   （`pr·P_G·dt`）全额记账 ⇒ 下层只负责"别被可行窗口裁剪"。
2. **块末追踪偏好进 r1**：`W_GAP·max(0, |soc_target − SOC_end| − SOC_TOL)`
   （`W_GAP = 10 元/MWh`、`SOC_TOL = 0.1 MWh`，来自 `hsrl_cost`）——
   偏好用固定价、约束才配对偶；这是上层"选一个可达目标"的激励来源。
3. **c2 = 10·slack，slack = "能力内追踪失败量"**
       delta   = soc_target − SOC_prev
       g       = η_c·dt (充电) / dt/η_d (放电)
       pb_best = clip(delta/g, pbrulemin, pbrulemax)   ← 规则 SocTracker 的动作
       d_best  = |delta − g·pb_best|                   ← 该动作后的残差
       slack   = max(0, |soc_target − SOC_next| − d_best)
   性质：块首够不着的目标跳变（结构性不可行）**不计费**（由上层 gap 罚负责）；
   **零成本动作恰好是规则 SocTracker 的 pb = clip(δ/g, 窗口)** ⇒ 规则下层与 RL
   下层的最优策略完全同一，两者对比变成纯粹的可学性对比；窗口不可行时 slack = 0
   （被迫吞吐不可归责于下层）。

L 层数值口径（与 `--value_scale` 无关：下层不做值函数归一化）
-------------------------------------------------------------
`r2`/`c2` 在 2026 年那次统一里**整体 ÷100**（r2 的 clip 系数 100→1、c2 系数
1000→10），因此与之同口径的预算与对偶步长必须同步：

| 常量 | 值 | 含义 |
|---|---|---|
| `L_D_BLOCK` | 0.2 | 每块 `sum_c2` 预算 = 允许浪费 ≈0.02 MWh 的可达量（≈0.4 步能力） |
| `L_D_BLOCK_MIN` | 0.025 | curriculum 地板（≈0.05 步能力） |
| `L_COST_LIMIT` | 0.1 | `agent_L.cost_limit`（仅 non-EMA 分支用） |
| `L_EMA_LAM_LR` | 1.0 | 加性对偶步长 η2；c2 口径 ÷100 ⇒ η2 ×100 才能保持 λ2 轨迹 |

即 `d_block_L / c2系数 = 0.02 MWh`、`η2 × c2系数 = 10` 与 ÷100 之前逐位一致
（`agentutil/test_l_lower_unit.py` 断言）。
"""
import numpy as np

import hsrl_cost as hc

# ---------------------------------------------------------------- 目标映射
SOC_TARGET_CENTER = 3.0
SOC_TARGET_SPAN = 5.0
SOC_TARGET_MIN, SOC_TARGET_MAX = 0.5, 5.5

# ---------------------------------------------------------------- L 层数值口径
L_CLIP_COEF = 1.0        # r2 = −L_CLIP_COEF · clip_amount
L_SOC_COEF = 10.0        # c2 = L_SOC_COEF · slack
L_D_BLOCK = 0.2          # 每块 sum_c2 预算
L_D_BLOCK_MIN = 0.025    # curriculum 地板
L_COST_LIMIT = 0.1       # agent_L.cost_limit
L_EMA_LAM_LR = 1.0       # 加性对偶步长 η2

# ---------------------------------------------------------------- 观测刻度
OBS_SOC_SCALE = 10.0     # SOC 与目标都以 /10 进入下层观测


def soc_target_from_signal(a6):
    """上层信号 → SOC 目标: 3 + 5·a6, 夹在 [0.5, 5.5]（合法目标带）。"""
    return float(np.clip(SOC_TARGET_CENTER + SOC_TARGET_SPAN * float(a6),
                         SOC_TARGET_MIN, SOC_TARGET_MAX))


def obs_target_dim(soc_target, scale=OBS_SOC_SCALE):
    """下层观测里的目标维 = soc_target / 10（与 SOC/10 同刻度，网络可直接比较）。"""
    return float(soc_target) / float(scale)


def gap_penalty(soc_target, SOC_end, tol=None):
    """块末追踪偏好（元，进 r1）: W_GAP·max(0, |soc_target − SOC_end| − SOC_TOL)。"""
    return hc.gap_pref(float(soc_target) - float(SOC_end),
                       hc.SOC_TOL if tol is None else tol)


def soc_track_slack(soc_target, SOC_prev, SOC_next, pbrulemin, pbrulemax,
                    eta_c=0.98, eta_d=0.98, time=1.0 / 60):
    """下层"能力内追踪失败量"(slack, MWh) 及其诊断量。

    动机（线性式 |target − SOC'| 的结构性缺陷）
    ------------------------------------------
    SOC 是纯积分器、目标块内固定，且下层单步能力只有
        A = g·|pb|,  g_c = η_c·dt ≈ 0.0163 / g_d = dt/η_d ≈ 0.0170 MWh/MW，
    窗口（平抑 ∩ 物理）通常只放 ~3 MW → A ≈ 0.05 MWh/步。而目标每块(5 步)才变一次、
    相邻块最多差 5 MWh ⇒ **块首的 |δ| 无论如何都消不掉**。直接以 |δ| 计费 ⇒
    每块固定吃一笔"不可控"惩罚：`sum_c2` 变成"H 给的目标有多远"的代理而不是
    "L 追得多差"，`d_block_L` 恒被违反、`lam2` 迅速顶格。

    本函数把"不可控"减掉，只留 L 能控的部分（"本步能做到的最好"与"实际做到"之差）。

    Returns
    -------
    slack : float  (MWh, ≥0)
    info  : dict   {delta, gain, pb_best, soc_best, d_best, d_next, infeasible_window}
    """
    soc_target = float(soc_target)
    SOC_prev = float(SOC_prev)
    SOC_next = float(SOC_next)
    pbrulemin = float(pbrulemin)
    pbrulemax = float(pbrulemax)

    delta = soc_target - SOC_prev
    g = eta_c * time if delta > 0 else time / eta_d
    d_next = abs(soc_target - SOC_next)

    # 平坦窗口不可行（平抑 ∩ 物理 = ∅）: 本步残差不可归责于 L
    if pbrulemin > pbrulemax:
        return 0.0, {
            'delta': delta, 'gain': g, 'pb_best': 0.0,
            'soc_best': SOC_prev, 'd_best': abs(delta), 'd_next': d_next,
            'infeasible_window': True,
        }

    pb_best = float(np.clip(delta / g, pbrulemin, pbrulemax))
    soc_best = SOC_prev + g * pb_best
    d_best = abs(soc_target - soc_best)
    slack = max(0.0, d_next - d_best)

    return slack, {
        'delta': delta, 'gain': g, 'pb_best': pb_best,
        'soc_best': soc_best, 'd_best': d_best, 'd_next': d_next,
        'infeasible_window': False,
    }


def lower_reward(clip_amount, coef=L_CLIP_COEF):
    """下层 r2 = −coef · clip_amount（clip_amount = |pb_cmd − pbrule|，RBB 硬裁剪量）。"""
    return -float(coef) * float(clip_amount)


def lower_cost_soc_track(soc_target, SOC_prev, SOC_next, pbrulemin, pbrulemax,
                         coef=L_SOC_COEF, return_info=False, **kw):
    """下层 c2 = coef · slack（能力内追踪失败量，进拉格朗日约束口）。

    `coef = L_SOC_COEF = 10` 时: 完全不动而 |δ| ≥ A ⇒ 每步 ~10·A ≈ 0.5、每块 ~2.5；
    最坏（反向满功率）每步 ≤ 10·2A ≈ 3.4。因此 `d_block_L = 0.2` 读作
    "允许每块浪费 ≈0.02 MWh 的可达量（≈0.4 步能力）"。
    """
    slack, info = soc_track_slack(soc_target, SOC_prev, SOC_next,
                                  pbrulemin, pbrulemax, **kw)
    cost = float(coef) * slack
    if return_info:
        return cost, info
    return cost
