# -*- coding: utf-8 -*-
"""testPg 的上层奖励/动作口径 —— **与 `aifflor_myfunc` 同式，只把 6 维动作换成 2 维**。

动作
----
    H 动作 = [rate, goal]
      rate ∈ [0, 1]   降载率：ptur = rate · Pmax(u)，Pmax = P(u; [0.2,0.2,0.2,0.2,0.33])
      goal ∈ [-0.5,0.5]  SOC 目标信号（映射与 aifflor 逐字相同）

与 aifflor 的逐项对应
----------------------
| 项 | aifflor（6 维） | testPg（2 维） |
|---|---|---|
| 机组动作越界罚 | `10·Σ_j [(a_j−0.5)_+ + (−a_j)_+]`，j=0..4 | `10·[(rate−1)_+ + (−rate)_+]` |
| 信号越界罚 | `10·[(goal−0.5)_+ + (−0.5−goal)_+]` | **同式**（默认开；`PG_GOAL_PEN` 可关） |
| 越界乘性罚 | `x = Σ 前 5 维越界量` | `x = rate` 的越界量 |
| r1 | `pr·P_G·dt − 490·max(−pb,0)·dt − pen` | **同式**（`ptur = rate·Pmax` 由主循环给） |
| c1 | 波动当量 + `W_GAP·unfeasible_gap` | **同式** |
| 块末 gap | `10·max(0, |soc_target − SOC_末| − tol)` | **同式**（`soc_target` 由 goal 映射） |
| L 观测 | `[pg(t−1)/10, ptur/10, SOC/10, target/10, n_rem/5]` | **同式** |
| L r2/c2 | `−1·min(clip_amount,20)` / `+1·trackerr` | **同式**（不复制，直接调 `hsrl_soctarget`） |

`soc_target_from_signal` / `gap_penalty` / `lower_reward` / `lower_cost_soc_track` /
相位特征 等**全部从 `aifflor_myfunc` 原样导入**，避免第二份口径。
"""
import numpy as np
import pandas as pd
import torch

import hsrl_cost as hc
import hsrl_soctarget as hst
from util import *

from aifflor.aifflor_myfunc import (                      # noqa: F401  （原样复用）
    PENALTY_NN_H_COEF, PENALTY_NN_L_COEF,
    W_GAP, C1_RISK_COEF, DEG_COST_DISCHARGE, DEG_COST_CHARGE,
    SOC_TOL, SOC_TARGET_CENTER, SOC_TARGET_SPAN, SOC_TARGET_MIN, SOC_TARGET_MAX,
    get_wind_level, get_price_level, Cal_soc_deviation, Cal_r1_continuous,
    soc_target_from_signal, gap_penalty, lower_reward, lower_cost_soc_track,
    Normalize, Original, Normalize_H,
    PHASE_DIM, DAY_MINUTES, day_price_schedule, phase_daily_table,
    phase_features, phase_minute_of_day, rep_minute_for_price_level,
    SOC_SETTLE_PR, build_9case_scenario,
    CheckAction as _aifflor_CheckAction,          # L 层 1 维沿用 aifflor 口径
)


# =============================================================================
# 1. 动作带
# =============================================================================
PG_RATE_MIN, PG_RATE_MAX = 0.0, 1.0      # rate 物理带
PG_GOAL_MIN, PG_GOAL_MAX = -0.5, 0.5     # SOC 目标信号带（= aifflor 的信号带）
PG_RATE_DIM = 0
PG_GOAL_DIM = 1
#: 是否对 goal 越界计罚。默认 True：aifflor 的 `penalty_NN_H` 同时含信号越界罚，
#: "改语义"只替换**机组维**，信号维保持同式（可用 `--pg_goal_pen 0` 关闭做消融）。
PG_GOAL_PEN = True


def CheckAction(action, charge=0.5):
    """动作检查。

    * 2 维（testPg 上层）: `[rate, goal]` → clip 到 `[0,1]×[-0.5,0.5]`；
      罚 = `PENALTY_NN_H_COEF·(rate 越界量 [+ goal 越界量])`，**不封顶**（与 aifflor 同）。
    * 1 维（下层 BESS）: 原样委托 `aifflor_myfunc.CheckAction`（200/单位，仅诊断）。
    * 5/6 维: 原样委托（testPg 主路径不会出现；保留给旧接口/诊断）。
    """
    a = np.asarray(action, dtype=float).reshape(-1)
    if a.size == 2:
        rate, goal = float(a[0]), float(a[1])
        pen = PENALTY_NN_H_COEF * (max(0.0, rate - PG_RATE_MAX)
                                   + max(0.0, PG_RATE_MIN - rate))
        if PG_GOAL_PEN:
            pen += PENALTY_NN_H_COEF * (max(0.0, goal - PG_GOAL_MAX)
                                        + max(0.0, PG_GOAL_MIN - goal))
        applied = np.array([float(np.clip(rate, PG_RATE_MIN, PG_RATE_MAX)),
                            float(np.clip(goal, PG_GOAL_MIN, PG_GOAL_MAX))])
        return applied, float(pen)
    return _aifflor_CheckAction(a, charge=charge)


# =============================================================================
# 2. 越界乘性罚（rate 维）
# =============================================================================
#: 前 N 维 = 降载率；goal 不计（与 aifflor"信号维不计入 x"同）。
PG_VIOL_DERATE_DIMS = 1
#: 与 aifflor 的默认值一致（0 = 关闭；见 aifflor_myfunc.VIOL_DERATE_COEF 的实测记录）
PG_VIOL_DERATE_COEF = 0.0


def violation_amount_pg(action, n_dims=None):
    """x = rate 偏离 [0,1] 的量（只第 1 维；goal 不计）。"""
    n = PG_VIOL_DERATE_DIMS if n_dims is None else int(n_dims)
    a = np.asarray(action, dtype=float).reshape(-1)
    n = min(n, a.size)
    if n <= 0:
        return 0.0
    v = np.maximum(np.maximum(a[:n] - PG_RATE_MAX, PG_RATE_MIN - a[:n]), 0.0)
    return float(v.sum())


def violation_derate_pg(action, coef=None, n_dims=None):
    """乘性系数 `f = exp(-c·x)`；`c<=0` 或 `x==0` 时恰好 1.0（历史路径逐位不变）。"""
    c = float(PG_VIOL_DERATE_COEF if coef is None else coef)
    if c <= 0.0:
        return 1.0
    x = violation_amount_pg(action, n_dims)
    if x <= 0.0:
        return 1.0
    return float(np.exp(-c * x))


# =============================================================================
# 3. 上层 r1 / c1
# =============================================================================
def cal_rew_rb_pg(observation, observation2, action_H, rbb_info,
                  penalty_NN_H=0.0, time=1.0 / 60, ptur=None,
                  pb_cmd=None, turbine_derate=1.0, pmax=None):
    """上层 (H) 奖励/成本 —— **与 `aifflor_myfunc.Cal_Rew_RB` 同式**，只读 2 维动作。

    Parameters
    ----------
    action_H : array(2)   `[rate, goal]`
    ptur     : float      实际风机出力 [MW] = `rate · Pmax(u)`（由主循环算好传入）
    pmax     : float      本步的 Pmax [MW]（仅日志；None ⇒ 用 ptur/rate 反推）

    Returns
    -------
    r1, c1_step, details      # 不含块末 gap 罚（由 main 在块末累加）
    """
    if ptur is None:
        raise ValueError('cal_rew_rb_pg: ptur must be provided')
    a = np.asarray(action_H, dtype=float).reshape(-1)
    if a.size < 2:
        raise ValueError('cal_rew_rb_pg: 需要 2 维 [rate, goal], 得到 %d' % a.size)
    rate = float(a[PG_RATE_DIM])
    soc_signal = float(a[PG_GOAL_DIM])

    pr_old = float(observation[0])
    u = float(observation[1])
    P_G_new = float(observation2[-1])
    P_G_old = float(observation[-1])
    SOC_new = float(observation2[2])
    SOC_decision = float(observation[2])

    pbrule = float(rbb_info['pbrule'])
    unfeasible_gap = float(rbb_info['unfeasible_gap'])
    clip_amount = float(rbb_info['clip_amount'])
    P_F = abs(P_G_new - P_G_old)

    soc_target = soc_target_from_signal(soc_signal)
    wind_level = get_wind_level(u)
    price_level = get_price_level(pr_old)

    # ==== r1（与 aifflor 同式：风机收益 × 乘性系数 + 储能收益 − 退化 − 越界罚）====
    r1 = Cal_r1_continuous(pr_old, ptur, pbrule, penalty_NN_H,
                           wind_level, price_level, SOC=SOC_decision, time=time,
                           turbine_derate=turbine_derate)

    # ==== c1（与 aifflor 逐位同式）====
    c1_risk_coeff = C1_RISK_COEF
    if P_F > 3.0 + 1e-4:
        c1 = c1_risk_coeff * (10.0 * (P_F - 3.0) + 0.1 * P_F)
    else:
        c1 = c1_risk_coeff * (0.1 * P_F)
    c1 = c1 + W_GAP * unfeasible_gap
    c1 = hc.c1_apply(c1, P_F)

    # ==== details（列名与 aifflor 对齐；新增 rate/pmax 两列）====
    w_wind = 1.0
    w_bess = 1.0
    turbine_profit = w_wind * pr_old * ptur * time
    bess_profit = w_bess * (-pr_old * pbrule * time)
    soc_dev = Cal_soc_deviation(SOC_new, wind_level)
    if pmax is None:
        _pmax = (ptur / rate) if abs(rate) > 1e-12 else np.nan
    else:
        _pmax = float(pmax)

    details = {
        'wind_level': wind_level,
        'price_level': price_level,
        'w_wind': w_wind,
        'w_bess': w_bess,
        'pr': pr_old,
        'u': u,
        'rate': rate,
        'pmax': _pmax,
        'ptur_max': curtail_available(ptur, rate),
        'turbine_profit': turbine_profit,
        'bess_profit': bess_profit,
        'revenue': turbine_profit + bess_profit,
        'degradation': (DEG_COST_DISCHARGE * max(-pbrule, 0.0)
                        + DEG_COST_CHARGE * max(pbrule, 0.0)) * time,
        'soc_signal': soc_signal,
        'soc_target': soc_target,
        'soc_delta': SOC_new - soc_target,
        'soc_tol': SOC_TOL,
        'soc_deviation': soc_dev,
        'ptur': ptur,
        'pbrule': pbrule,
        'pb_cmd': float(pb_cmd) if pb_cmd is not None else np.nan,
        'pb_rbb_clip': (abs(float(pb_cmd) - pbrule) if pb_cmd is not None
                        else 0.0),
        'P_F': P_F,
        'SOC': SOC_new,
        'unfeasible_gap': unfeasible_gap,
        'clip_amount': clip_amount,
        'is_infeasible': rbb_info.get('is_infeasible', False),
        'inevitable_violation': rbb_info.get('inevitable_violation', 0.0),
        'penalty_NN_H': penalty_NN_H,
        'c1_fluct': c1,
    }
    return r1, c1, details


def curtail_available(ptur, rate):
    """`ptur / rate`（rate=0 时不可反推 ⇒ NaN）；仅日志，不参与奖励。"""
    r = float(rate)
    return float(ptur) / r if abs(r) > 1e-12 else float('nan')


# =============================================================================
# 4. 下层 L 观测（2 维动作版本）
# =============================================================================
def build_L_obs_pg(observation, pw_tru, applied_action_H, rbb_info, n_rem=None):
    """L 观测 5 维（与 aifflor 同式）：

        [pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]

    `soc_target = soc_target_from_signal(applied_action_H[1])`；`rbb_info` 仅为签名一致。
    """
    if n_rem is None:
        raise ValueError('build_L_obs_pg: 需要 n_rem (块内剩余步数 = 5 - step mod 5)')
    a = np.asarray(applied_action_H, dtype=float).reshape(-1)
    if a.size < 2:
        raise ValueError('build_L_obs_pg: 需要 2 维 [rate, goal], 得到 %d' % a.size)
    soc_target = soc_target_from_signal(float(a[PG_GOAL_DIM]))
    return hst.build_L_obs(float(observation[-1]), float(pw_tru),
                           float(observation[2]), soc_target, int(n_rem))


# =============================================================================
# 5. 输出层 bias 初始化
# =============================================================================
def init_pg_action_bias(agent, rate_value=1.0, goal_value=0.0, verbose=True):
    """把 actor（与 actor_target）输出层 bias 设为 `[rate_value, goal_value]`。

    * `rate_value = 1.0`（默认）⇒ 初始策略 = **不降载**（ptur = Pmax），与 aifflor 的
      "中性宽带"初始化同思路：从一个物理上可解释、收益非负的起点出发；
    * `goal_value = 0.0` ⇒ 初始 SOC 目标 = `soc_target_from_signal(0)`（容量中点）。

    只改 bias，不改权重初始化（`model.py` 的 `fc4.weight ~ ±init_w`）⇒ 与 "纯默认初始化"
    是单因子对照。
    """
    a = agent.actor
    if not hasattr(a, 'fc4'):
        return False
    with torch.no_grad():
        b = a.fc4.bias.data
        if b.shape[0] != 2:
            raise ValueError('init_pg_action_bias: actor 输出维应为 2, 得到 %d'
                             % b.shape[0])
        b[0] = float(rate_value)
        b[1] = float(goal_value)
        agent.actor_target.fc4.bias.data.copy_(b)
    if verbose:
        prGreen('init_pg_action_bias: rate bias = {:.3f}（初值 = 不降载）; '
                'goal bias = {:.3f}（soc_target = {:.4f}）'
                .format(float(b[0]), float(b[1]),
                        soc_target_from_signal(float(b[1]))))
    return True


# =============================================================================
# 6. 9-case 训练中测试（H 动作 2 维）
# =============================================================================
def nine_case_test_pg(agent_H, agent_L, env_test, rbb_system,
                      scenario=None, test_csv=False, output=None, seed=None,
                      phase_mode=0, phase_table=None, viol_derate_coef=None):
    """45-step 9-case 测试（**逐字继承 aifflor 的协议**，只把 H 动作换成 2 维）。

    与 `aifflor_myfunc.nine_case_test` 的差异只有 3 处：
      ① H 动作 `[rate, goal]`（2 维），`ScenarioStep` 传 2 维 H + 1 维 L；
      ② `soc_target` 从 `action_H[1]` 取；
      ③ 指标/CSV 增加 `rate`、`pmax`、`ptur` 三列。

    ⚠ L 层协议（含"把 L actor 的残余动作当命令"与 `E=5` 起点）**刻意原样保留**，
      以便与 aifflor 的 `df_test9.csv` 逐曲线可比。它是历史口径，不是本实验变量。
    """
    if scenario is None:
        wind_seq, price_seq, case_ids, blocks = build_9case_scenario(seed=seed)
    else:
        wind_seq, price_seq, case_ids, blocks = scenario

    obs = env_test.ScenarioReset(wind_seq, price_seq)
    _pminute = {i: rep_minute_for_price_level(price_seq[i])
                for i in range(len(price_seq))}
    norm_obs = Normalize_H(obs, _pminute[0], phase_mode, phase_table)
    time = 1.0 / 60
    agent_H.is_training = False
    agent_L.is_training = False
    step = 0
    num_iterations = len(wind_seq)
    n_per_case = num_iterations // len(blocks)

    _soc0_of, _soc1_of = {}, {}
    per_case = {i: {'r1': 0.0, 'c1': 0.0, 'r2': 0.0, 'c2': 0.0,
                    'cvm': 0, 'pf_viol': 0, 'profit': 0.0,
                    'gap_pen': 0.0, 'soc_track_err': 0.0,
                    'turbine_profit': 0.0, 'derate_loss': 0.0,
                    'penalty_NN_H': 0.0,
                    'rate': 0.0, 'pmax': 0.0, 'ptur': 0.0}
                for i in range(len(blocks))}
    df = pd.DataFrame()
    soc_target = SOC_TARGET_CENTER

    while step < num_iterations:
        if step % 5 == 0:
            action_H_raw = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H_raw)
            _derate = violation_derate_pg(action_H_raw, viol_derate_coef)
            pw_tru = env_test.PowerOutput(obs[1], action_H)      # 2 维 ⇒ rate·Pmax
            soc_target = soc_target_from_signal(action_H[PG_GOAL_DIM])

        ptur = float(pw_tru)
        _rate_now = float(action_H[PG_RATE_DIM])
        pmax_step = (ptur / _rate_now if abs(_rate_now) > 1e-12
                     else float(env_test.Pmax(float(obs[1]))))
        pg_prev = float(obs[-1])
        E_current = float(obs[2])

        _, rbb_info_obs = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=ptur, pg_prev=pg_prev, E=E_current)
        norm_obs_L = build_L_obs_pg(obs, pw_tru, action_H, rbb_info_obs,
                                    5 - (step % 5))
        if step % 5 == 0:
            agent_L.reset(norm_obs_L)
        original_action_L = agent_L.select_action(norm_obs_L)
        action_L, penalty_NN_L = CheckAction(original_action_L)

        pb_original = float(action_L[0])
        pbrule, rbb_info = rbb_system.compute_pbrule(
            pb_original=pb_original, ptur=ptur,
            pg_prev=pg_prev, E=E_current,
        )

        # env 施加的是 **RBB 裁剪后的 pbrule**（与 aifflor 的 action_rbb[-1] 同口径）；
        # `action_L` 只作为"命令"进入 compute_pbrule。
        pw_tru2, obs2, action_L_env, penalty_L, done, done_L, info = \
            env_test.ScenarioStep(action_H, np.array([pbrule]), pw_tru)

        norm_obs2 = Normalize_H(obs2, _pminute[min(step + 1, num_iterations - 1)],
                                phase_mode, phase_table)

        r1, c1_step, details = cal_rew_rb_pg(
            obs, obs2, action_H, rbb_info,
            penalty_NN_H=penalty_NN_H, time=time,
            ptur=ptur, pb_cmd=pb_original,
            turbine_derate=_derate, pmax=pmax_step,
        )

        gap_pen = 0.0
        if step % 5 == 4:
            gap_pen = gap_penalty(soc_target, float(obs2[2]))
        c1 = c1_step + gap_pen

        r2 = lower_reward(rbb_info['clip_amount'])
        c2, c2_info = lower_cost_soc_track(
            soc_target, E_current, float(obs2[2]),
            rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'],
            5 - (step % 5),
            pbrule=float(rbb_info['pbrule']),
            return_info=True)

        bi = step // 5
        if step % 5 == 0:
            _soc0_of[bi] = float(obs[2])
        _soc1_of[bi] = float(obs2[2])
        pc = per_case[bi]
        pc['r1'] += r1
        pc['c1'] += c1
        pc['r2'] += r2
        pc['c2'] += c2
        pc['gap_pen'] += gap_pen
        pc['turbine_profit'] += details['turbine_profit']
        pc['derate_loss'] += details['turbine_profit'] * (1.0 - _derate)
        pc['penalty_NN_H'] += penalty_NN_H
        pc['rate'] += float(action_H[PG_RATE_DIM])
        pc['pmax'] += pmax_step
        pc['ptur'] += ptur
        if c1_step > 0:
            pc['cvm'] += 1
        if details['P_F'] > 3.0 + 1e-4:
            pc['pf_viol'] += 1
        if step % 5 == 4:
            pc['soc_track_err'] = abs(soc_target - float(obs2[2]))

        profit_step = (obs2[-1] * obs[0] * time
                       - DEG_COST_DISCHARGE * max(-float(pbrule), 0.0) * time)
        pc['profit'] += profit_step

        if test_csv:
            df = pd.concat([df, pd.DataFrame([{
                'step': step,
                'case_wind': case_ids[step][0],
                'case_price': case_ids[step][1],
                'wind_level': details['wind_level'],
                'price_level': details['price_level'],
                'u': details['u'], 'pr': details['pr'],
                'rate': details['rate'], 'pmax': details['pmax'],
                'ptur': details['ptur'],
                'r1': r1, 'c1_step': c1_step, 'gap_pen': gap_pen, 'c1': c1,
                'r2': r2, 'c2': c2,
                'c2_slack': c2_info['slack'],
                'pb_best': c2_info['pb_best'],
                'd_best': c2_info['d_best'], 'd_next': c2_info['d_next'],
                'profit': profit_step,
                'soc_signal': details['soc_signal'],
                'soc_target': details['soc_target'],
                'soc_delta': details['soc_delta'],
                'SOC': details['SOC'],
                'pb_original': pb_original, 'pbrule': rbb_info['pbrule'],
                'clip_amount': rbb_info['clip_amount'],
                'P_F': details['P_F'],
                'action_H': action_H,
            }])], ignore_index=True)

        norm_obs = norm_obs2
        obs = obs2
        pw_tru = pw_tru2
        step += 1

    agent_H.is_training = True
    agent_L.is_training = True
    if test_csv and output:
        df.to_csv(f'{output}/test_s_9case.csv', index=False)

    case_metrics = []
    for i, b in enumerate(blocks):
        pc = per_case[i]
        case_metrics.append({
            'wind_level': b['wind_level'],
            'price_level': b['price_level'],
            'r1': pc['r1'] / n_per_case,
            'c1': pc['c1'] / n_per_case,
            'r2': pc['r2'] / n_per_case,
            'c2': pc['c2'] / n_per_case,
            'gap_pen': pc['gap_pen'] / n_per_case,
            'cvm': pc['cvm'],
            'pf_viol': pc['pf_viol'],
            'profit': pc['profit'],
            'soc_track_err': pc['soc_track_err'],
            'turbine_profit': pc['turbine_profit'],
            'derate_loss': pc['derate_loss'],
            'penalty_NN_H': pc['penalty_NN_H'],
            # testPg 诊断（均按 5 步块求均值）
            'rate': pc['rate'] / n_per_case,
            'pmax': pc['pmax'] / n_per_case,
            'ptur': pc['ptur'] / n_per_case,
        })

    for _i, _m in enumerate(case_metrics):
        _s0, _s1 = _soc0_of[_i], _soc1_of[_i]
        _m['soc_start'], _m['soc_end'] = _s0, _s1
        _m['dsoc'] = _s1 - _s0
        _m['profit_econ'] = float(_m['profit']) + (_s1 - _s0) * SOC_SETTLE_PR

    overall = {
        'r1': float(np.mean([m['r1'] for m in case_metrics])),
        'c1': float(np.mean([m['c1'] for m in case_metrics])),
        'r2': float(np.mean([m['r2'] for m in case_metrics])),
        'c2': float(np.mean([m['c2'] for m in case_metrics])),
        'gap_pen': float(np.mean([m['gap_pen'] for m in case_metrics])),
        'cvm': int(sum(m['cvm'] for m in case_metrics)),
        'pf_viol': int(sum(m['pf_viol'] for m in case_metrics)),
        'profit': float(sum(m['profit'] for m in case_metrics)),
        'soc_track_err': float(np.mean([m['soc_track_err'] for m in case_metrics])),
        'turbine_profit': float(sum(m['turbine_profit'] for m in case_metrics)),
        'derate_loss': float(sum(m['derate_loss'] for m in case_metrics)),
        'penalty_NN_H': float(sum(m['penalty_NN_H'] for m in case_metrics)),
        'rate': float(np.mean([m['rate'] for m in case_metrics])),
        'pmax': float(np.mean([m['pmax'] for m in case_metrics])),
        'ptur': float(np.mean([m['ptur'] for m in case_metrics])),
    }
    _b0, _bN = min(_soc0_of), max(_soc1_of)
    overall['soc_start'], overall['soc_end'] = _soc0_of[_b0], _soc1_of[_bN]
    overall['dsoc'] = _soc1_of[_bN] - _soc0_of[_b0]
    overall['profit_econ'] = (float(overall['profit'])
                              + overall['dsoc'] * SOC_SETTLE_PR)
    return case_metrics, overall
