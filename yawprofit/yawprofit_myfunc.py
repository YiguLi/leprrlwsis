# -*- coding: utf-8 -*-
"""`yawprofit` 的奖励/动作/观测口径 —— **无储能**的偏航调度（纯发电利润）。

一句话
------
    观测 (state) = [pr/1000, u/10] + 24h 相位特征        ← 无 SOC、无上一分钟功率
    动作 (action) = 5 维无量纲偏航指令 n ∈ [−1, +1]      ← 无 SOC 目标信号（yawflor 的第 6 维）
    奖励 (reward) = pr · P_tur(u, γ) · dt − penalty_NN_H  ← 无 bess_profit / 退化 / 可行域缺口
    成本 (cost)   = **恒 0**                              ← 无波动约束通道（λ ≡ 0）

与 `yawflor_myfunc` 的差异（逐项）
----------------------------------
| 项 | yawflor | yawprofit |
|---|---|---|
| 观测基维 | 4 维 `[pr, u, E, P_G]` | **2 维** `[pr, u]` |
| 动作维 | 6 = 5 偏航 + 1 SOC 目标信号 | **5 = 只偏航** |
| `soc_target_from_signal` / `gap_penalty` | 有（块末 gap 罚） | **无** |
| `CheckAction` 1 维电池分支 | 有（下层用） | **无**（传 6 维直接报错） |
| r1 | `pr·P_G·dt − 490·max(−pb,0)·dt − penalty` | `pr·P_tur·dt − penalty` |
| c1 | `P_F + 20(P_F−3)⁺`（+ 可行域缺口） | **≡ 0** |
| 九例测试 | H + L + RBB | 只 H，并附 **无偏航参考臂** `profit_nominal` |
| 越界乘性罚 `--viol_derate` | 有（默认关） | **不设**（yawflor 默认 0 ⇒ 逐位等价） |

**不复制**的部分（跨项目单一真源）
----------------------------------
电价日程 `day_price_schedule`、24h 相位表 `phase_daily_table` / `phase_features`、
风况/电价分档 `get_wind_level` / `get_price_level`、九例场景"代表性时刻"
`rep_minute_for_price_level`、偏航带 `hsrl_warmup.YAW_*` —— 一律从
`yawflor.yawflor_myfunc` / `hsrl_warmup` **import**，不抄第二份
（口径表已冻结；`agentutil/test_phase_uniform.py` 正是为"五份拷贝必须逐位相同"而立）。

一个共享的记账约定（刻意与 yawflor 保持同口径）
----------------------------------------------
`penalty_NN_H` 是**块级**量（每个块由 `CheckAction` 算一次），但在 yawflor 的
`Cal_Rew_RB` 里它被**块内 5 步各减一次**（块内合计 5×）。本模块照抄这一口径，
以免"动作带罚"的强度在换项目时静默改变；它只在该块**动作越界**时非零，
是 actor 未被 tanh 压缩（`model.py` 的刻意设计）时的回正力。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))        # yawprofit/
_ROOT = os.path.dirname(_HERE)                            # 仓库根
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from util import *                                        # noqa: E402,F401,F403
from yawflor.yawflor_myfunc import (                      # noqa: E402  单一真源（不抄第二份）
    PRICE_HIGH, PRICE_MEDIUM, PRICE_LOW, get_price_level,
    WIND_LOW_MAX, WIND_HIGH_MIN, get_wind_level,
    PHASE_DIM, DAY_MINUTES, day_price_schedule, phase_daily_table,
    phase_features, phase_minute_of_day,
    REP_MINUTE, rep_minute_for_price_level,
)
from yawflor import yawflor_turbine as YT                 # noqa: E402  偏航角换算 (γ = 25°·n)

try:
    from test_scenario import build_9case_scenario
except ImportError:                                       # pragma: no cover
    raise


# =============================================================================
# 1. 越界罚（动作带 [−1,+1]；与 yawflor 的 5 维分支逐位相同）
# =============================================================================
# 满带越界 = 200 元：100 元/单位 × 带宽 2。与旧环境 400 × 0.5 = 200 元**等值**
# （yawflor/README.md §4；`hsrl_warmup.YAW_PENALTY_NN_H_COEF` 是同一数字的单一真源）。
PENALTY_NN_H_COEF = 100.0
ACT_HI, ACT_LO = 1.0, -1.0


def violation_amount(action):
    """x = Σ 各维越出 [−1,+1] 的量（0 = 全在带内）。"""
    a = np.asarray(action, dtype=float).reshape(-1)
    v = np.maximum(np.maximum(a - ACT_HI, ACT_LO - a), 0.0)
    return float(v.sum())


def CheckAction(action):
    """把 5 维偏航指令夹进 [−1,+1]，并给出**不封顶**的线性越界罚（元）。

    ⚠ **只接受偏航维**：长度为 6 的动作（= yawflor 的"5 偏航 + SOC 目标信号"）
    直接报错 —— 信号维属于 BESS 通道，在 yawprofit 里不存在，静默截断会掩盖接线错误。
    越界罚**不封顶**（保持越界区恒定的回正力，见 yawflor/README.md run160 诊断）。
    """
    a = np.asarray(action, dtype=float).reshape(-1)
    if a.size == 6:
        raise ValueError(
            'CheckAction(yawprofit): 收到 6 维动作 —— 第 6 维是 yawflor 的 SOC 目标信号，'
            'yawprofit 的动作空间只有 5 维偏航指令（BESS 已抹去）')
    if a.size == 0:
        raise ValueError('CheckAction(yawprofit): 空动作')
    penalty = PENALTY_NN_H_COEF * violation_amount(a)
    return np.clip(a, ACT_LO, ACT_HI), float(penalty)


# =============================================================================
# 2. 上层奖励（纯发电利润；成本通道恒 0）
# =============================================================================
def Cal_Rew_WT(observation, ptur, penalty_NN_H=0.0, time=1.0 / 60):
    """一个 1 min 环境步的奖励/成本。

        r1 = pr(t) · P_tur(u(t), γ_block) · dt − penalty_NN_H     [元]
        c1 = 0                                                     [恒等]

    * `ptur` = 该分钟、该块偏航动作下的风机总出力 [MW]（由 `env.PowerOutput` 给出）；
    * `pr` 取**当前**分钟的电价（`observation[0]`），与 yawflor 的 `Cal_Rew_RB`
      用 `observation[0]` 的约定一致；
    * `time = 1/60 h` ⇒ `pr[元/MWh]·ptur[MW]·dt[h] = 元`；
    * **没有**电池收益项、退化项、可行域缺口项（`490·max(−pb,0)` 之类全部消失）。

    Returns
    -------
    (r1, c1, details)   —— 三元组形状与 `yawflor.Cal_Rew_RB` 相同，便于主循环同构。
    """
    pr = float(observation[0])
    u = float(observation[1])
    ptur = float(ptur)
    turbine_profit = pr * ptur * time
    r1 = turbine_profit - float(penalty_NN_H)
    details = {
        'pr': pr,
        'u': u,
        'ptur': ptur,
        'turbine_profit': turbine_profit,     # = 本步发电收益（元）
        'revenue': turbine_profit,
        'penalty_NN_H': float(penalty_NN_H),
        'P_F': None,                          # 由主循环从 env.step 的 info 填入（诊断）
        'wind_level': get_wind_level(u),
        'price_level': get_price_level(pr),
        # 下列键**不存在**（保留它们的旧名只会掩盖"电池是否还在奖励里"）：
        #   bess_profit / degradation / soc_deviation / soc_target / unfeasible_gap
    }
    return r1, 0.0, details


# =============================================================================
# 3. 观测归一化（2 维基 + 24h 相位特征）
# =============================================================================
def Normalize(obs):
    """`[pr, u, P_tur]` → `[pr/1000, u/10]`（**丢掉 P_tur**：state 只要电价与风速）。"""
    return np.array([obs[0] / 1000, obs[1] / 10])


def Original(obs):
    """`Normalize` 的逆（诊断用）。"""
    return np.array([obs[0] * 1000, obs[1] * 10])


def Normalize_H(obs, minute=0, mode=1, table=None):
    """上层观测归一化: 2 维基观测 + 相位特征（mode 0 ⇒ 恰好 2 维）。

    ⚠ 与 yawflor 的 `Normalize_H` 只差基维数（4 → 2）: yawflor 的第 3/4 维是
    SOC 与电网功率，两者在无电池时都没有意义（电网功率恒等于风机出力）。
    """
    base = Normalize(obs)
    if mode <= 0:
        return base
    return np.concatenate([base, phase_features(minute, mode, table)])


# =============================================================================
# 4. 九例分段测试（只 H；附"无偏航"参考臂）
# =============================================================================
def nine_case_test(agent_H, env_test, scenario=None, seed=None,
                   phase_mode=0, phase_table=None, physics_ref=None):
    """45 步 9-case 训练中测试（**只跑上层**）。

    Layout: 风 LOW→MEDIUM→HIGH；每段内电价 LOW→MEDIUM→HIGH；每 case = 5 步 = 一个
    上层决策周期。与 `yawflor.nine_case_test` 的差别 = 没有下层/RBB/电池。

    Parameters
    ----------
    physics_ref : FlorisPhysicsYaw or None
        可选的**独立**物理实例，用来算"**不偏航** (n ≡ 0) 的同序列发电收益
        `profit_nominal`。必须与 env_test 用**不同实例**：`PowerOutput` 的逐块预取
        缓存会被"另一个偏航动作"整体替换，共用实例只会让测试变慢（数值仍正确）。
        None ⇒ `profit_nominal` 列记 NaN（不编造参考值）。

    Returns
    -------
    case_metrics : list[dict]（9 条）
        每条 = 一个 case（5 步）：`r1`(每步均值) / `profit`(5 步合计, 元) /
        `profit_nominal`(5 步合计, 元; 无参考臂时 NaN) / `gain`(= profit − nominal) /
        `pf_viol`(P_F>3 MW 的步数) / `P_F_mean` / `yaw_deg_mean`(平均 |γ|, deg) /
        `ptur_mean`(MW) / `penalty_NN_H`(5 步合计)
    overall : dict —— 9 个 case 的均值（`pf_viol` 为总计、`profit*` 为 45 步总和）
    """
    if scenario is None:
        wind_seq, price_seq, case_ids, blocks = build_9case_scenario(seed=seed)
    else:
        wind_seq, price_seq, case_ids, blocks = scenario

    obs = env_test.ScenarioReset(wind_seq, price_seq)
    # 相位: 场景电价每 5 步就变 ⇒ 按 case 的电价档映射到该档"代表性时刻"（与 yawflor 同）
    _pminute = {i: rep_minute_for_price_level(price_seq[i]) for i in range(len(price_seq))}
    norm_obs = Normalize_H(obs, _pminute[0], phase_mode, phase_table)

    # ---- 参考臂: 无偏航 (n ≡ 0) 的同序列发电出力（一次性批量预取, 之后全命中缓存）----
    nominal_power = None
    if physics_ref is not None:
        us = np.asarray(wind_seq, dtype=float)
        gamma0 = np.zeros(int(physics_ref.nt), dtype=float)
        physics_ref.prefetch_deg(us, gamma0)
        nominal_power = [float(physics_ref.power_deg(us[i], gamma0)) for i in range(len(us))]

    time = 1.0 / 60
    agent_H.is_training = False
    step = 0
    num_iterations = len(wind_seq)
    n_per_case = num_iterations // len(blocks)

    per_case = {i: {'r1': 0.0, 'profit': 0.0, 'profit_nominal': 0.0, 'pf_viol': 0,
                    'P_F_sum': 0.0, 'yaw_deg': 0.0, 'ptur': 0.0, 'penalty_NN_H': 0.0}
                for i in range(len(blocks))}

    while step < num_iterations:
        if step % 5 == 0:                       # 块首: 上层决策一次, 块内保持不变
            action_raw = agent_H.select_action(norm_obs)
            action, penalty_NN_H = CheckAction(action_raw)
            ptur = float(env_test.PowerOutput(obs[1], action))
        pr_now = float(obs[0])
        r1, c1, details = Cal_Rew_WT(obs, ptur, penalty_NN_H=penalty_NN_H, time=time)

        pw_tru2, obs2, info = env_test.ScenarioStep(action)
        norm_obs2 = Normalize_H(obs2, _pminute[min(step + 1, num_iterations - 1)],
                                phase_mode, phase_table)

        bi = step // 5
        pc = per_case[bi]
        pc['r1'] += r1
        pc['profit'] += details['turbine_profit']
        pc['penalty_NN_H'] += penalty_NN_H
        pc['P_F_sum'] += float(info['P_F'])
        pc['yaw_deg'] += float(np.abs(YT.GAMMA_MAX * np.asarray(action)).mean())
        pc['ptur'] += ptur
        if info['P_F'] > 3.0 + 1e-4:
            pc['pf_viol'] += 1
        if nominal_power is not None:
            pc['profit_nominal'] += nominal_power[step] * pr_now * time

        norm_obs = norm_obs2
        obs = obs2
        ptur = pw_tru2      # 下一分钟的 P_tur（块内动作不变, 但场景已前进 1 步）
        step += 1

    agent_H.is_training = True

    case_metrics = []
    for i, b in enumerate(blocks):
        pc = per_case[i]
        gain = np.nan
        if nominal_power is not None:
            gain = pc['profit'] - pc['profit_nominal']
        case_metrics.append({
            'wind_level': b['wind_level'],
            'price_level': b['price_level'],
            'r1': pc['r1'] / n_per_case,                  # 每步均值 (与 yawflor 同口径)
            'c1': 0.0,
            'profit': pc['profit'],                       # 5 步合计 (元)
            'profit_nominal': (np.nan if nominal_power is None else pc['profit_nominal']),
            'gain': gain,
            'pf_viol': pc['pf_viol'],
            'P_F_mean': pc['P_F_sum'] / n_per_case,
            'yaw_deg_mean': pc['yaw_deg'] / n_per_case,   # 平均 |γ| [deg]
            'ptur_mean': pc['ptur'] / n_per_case,         # [MW]
            'penalty_NN_H': pc['penalty_NN_H'],           # 5 步合计 (与 r1 同记账口径)
        })

    def _mean(key):
        return float(np.mean([m[key] for m in case_metrics]))

    overall = {
        'r1': _mean('r1'),
        'c1': 0.0,
        'profit': float(sum(m['profit'] for m in case_metrics)),
        'profit_nominal': (np.nan if nominal_power is None
                           else float(sum(m['profit_nominal'] for m in case_metrics))),
        'pf_viol': int(sum(m['pf_viol'] for m in case_metrics)),
        'P_F_mean': _mean('P_F_mean'),
        'yaw_deg_mean': _mean('yaw_deg_mean'),
        'ptur_mean': _mean('ptur_mean'),
        'penalty_NN_H': float(sum(m['penalty_NN_H'] for m in case_metrics)),
    }
    overall['gain'] = (np.nan if nominal_power is None
                       else overall['profit'] - overall['profit_nominal'])
    return case_metrics, overall
