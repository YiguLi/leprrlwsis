"""
pernewact — 奖励/裁剪/归一化/九例测试（PER 版 + 有界行为适配）
==================================================================
Copied from myfunc.py (PER 项目) with the following change:

CheckAction —— 风机维（前 5）不再裁剪/惩罚：
  pernewact 的 actor 输出经 clamp [a_lo, a_hi] 已保证行为恒在界内
  （straight-through 无越界），因此风机维恒等通过、无 500/单位惩罚，
  消除 Q 曲面边界处的 kink。
  信号维（[-0.5, 0.5]）与电池维（[-10, 10]）保留原裁剪+惩罚逻辑。

其余函数（Cal_Rew / Eval_Rew / Normalize / Original / nine_case_test）
与 myfunc.py 完全一致。
"""

import numpy as np
import matlab
import matlab.engine
engine = matlab.engine.start_matlab()
from util import *
import pandas as pd

import sys, os


def Cal_Rew(observation, observation2, action_all, cost_single_cons = 3.):
    time = 1/60
    pr = observation[0]
    p_g = observation2[-1]
    p_g_last = observation[-1]
    
    P_B = action_all[-1]
    r = (p_g * pr - 4.9 * max(-P_B,0)) * time  # 4.90 # deg:unit：RMB/Mwh
    
    P_F = abs(p_g_last - p_g)
    if P_F > cost_single_cons:
        c = 10 * (P_F - cost_single_cons)
    else:
        c = 0.
    return r, c

def Eval_Rew(observation, observation2, action_all, cost_single_cons = 3.):
    time = 1/60
    pr = observation[0]
    p_g = observation2[-1]
    p_g_last = observation[-1]
    
    P_B = action_all[-1]
    r = (p_g * pr - 4.9 * max(-P_B,0)) * time  # 4.90 # deg:unit：RMB/Mwh
    
    P_F = abs(p_g_last - p_g)
    if P_F > cost_single_cons:
        c = P_F - cost_single_cons
    else:
        c = 0.  
    return r, c

def CheckAction(action, charge = 10.):
    penalty = 0
    # case 1 总的 action (5 WT + 1 signal + 1 BESS)
    # 风机维（前 5）：pernewact clamp 已保证界内 → 恒等通过，无裁剪无惩罚
    if len(action) == 7:
        # 影子价格限定范围(-0.5, 0.5), 越界惩罚计入penalty
        signal_val = action[5]
        signal_ma = max(0.5, signal_val)
        signal_mi = min(-0.5, signal_val)
        penalty += 500 * ((signal_ma - 0.5) + (-0.5 - signal_mi))

        for j in action[6:]:
            ma = max(charge, j)
            mi = min(-charge, j)
            penalty += 500 * ((ma - charge)+ (-charge - mi))

        a_H = np.array(action[:5], dtype=float)
        a_H = np.append(a_H, np.clip(signal_val, -0.5, 0.5))
        a_L = np.clip(action[6:], -charge, charge)
        applied_action = np.concatenate((a_H, a_L))

    # case 2 风机 + 影子价格信号
    if len(action) == 6:
        # 风机维恒等通过
        signal_val = action[5]
        signal_ma = max(0.5, signal_val)
        signal_mi = min(-0.5, signal_val)
        penalty += 500 * ((signal_ma - 0.5) + (-0.5 - signal_mi))
        applied_action = np.array(action[:5], dtype=float)
        applied_action = np.append(applied_action, np.clip(signal_val, -0.5, 0.5))
    if len(action) == 5:
        # 风机维恒等通过（仅风机）
        applied_action = np.array(action, dtype=float)

    if len(action) == 11:
        for j in action[:10]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 500 * ((ma - 0.5)+ (0. - mi))
        applied_action = np.clip(action[:10], 0., 0.5)
        applied_action = np.append(applied_action, action[10])
    if len(action) == 10:
        for j in action[:10]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 500 * ((ma - 0.5)+ (0. - mi))
        applied_action = np.clip(action[:10], 0., 0.5)
    # case 3 电池
    if len(action) == 1:
        for j in action:
            ma = max(charge, j)
            mi = min(-charge, j)
            penalty += 500 * ((ma - charge) + (-charge - mi))
        applied_action = np.clip(action, -charge, charge)     

    return applied_action, penalty

def Normalize(obs, max_E = 6.):
    # [Pr,U,E,P_last]
    pr = obs[0] / 1000 #TODO 300,580,980
    u = obs[1] / 10
    e = obs[2] / 10
    p = obs[3] / 10 # park
    return np.array([pr, u, e, p])

def Original(obs, max_E = 6.):
    pr = obs[0] * 1000 # 300,580,980
    u = obs[1] * 10
    e = obs[2] * 10
    p = obs[3] * 10 # park
    return np.array([pr, u, e, p])


# =============================================================================
# 九例分段测试（训练中每 480 episode step 一次，45 步）— PER 版
# =============================================================================

try:
    from test_scenario import build_9case_scenario
except ImportError:
    import sys as _sys, os as _os
    _HSRL_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _HSRL_ROOT not in _sys.path:
        _sys.path.insert(0, _HSRL_ROOT)
    from test_scenario import build_9case_scenario


def nine_case_test(agent_H, agent_L, env_test, rbb_system,
                   scenario=None, test_csv=False, output=None, seed=None):
    """
    PER 45-step 9-case in-training test (奖励与 S_test 一致，场景化执行)。

    Wind bands: LOW < 7.0, MEDIUM [7.0, 11.0), HIGH >= 11.0 m/s.
    Layout: wind LOW->MEDIUM->HIGH; within each wind segment price
    LOW->MEDIUM->HIGH; each case = 5 steps = one upper-level period.

    Returns
    -------
    case_metrics : list of dict (length 9)
        Per-case 5-step mean r1/c1/r2/c2/r/c plus violation count cvm;
        'profit' = the case's 5-step profit sum
        (profit_step = P_G*pr*dt - 4.9*max(-pbrule,0)*dt).
    overall : dict
        Mean over the 9 case means (cvm = total count);
        'profit' = the 45-step total profit sum.
    """
    if scenario is None:
        wind_seq, price_seq, case_ids, blocks = build_9case_scenario(seed=seed)
    else:
        wind_seq, price_seq, case_ids, blocks = scenario

    obs = env_test.ScenarioReset(wind_seq, price_seq)
    norm_obs = Normalize(obs)
    time = 1 / 60
    agent_H.is_training = agent_L.is_training = False
    step = 0
    num_iterations = len(wind_seq)
    n_per_case = num_iterations // len(blocks)

    per_case = {i: {'r1': 0.0, 'c1': 0.0, 'r2': 0.0, 'c2': 0.0,
                    'r': 0.0, 'c': 0.0, 'cvm': 0, 'profit': 0.0}
                for i in range(len(blocks))}
    df = pd.DataFrame()

    while step < num_iterations:
        if step % 5 == 0:
            action_H = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H)
            pw_tru = env_test.PowerOutput(obs[1], action_H)
            pb_l = list()

        norm_obs_L = np.append(norm_obs, action_H[-1])
        action_L = agent_L.select_action(norm_obs_L)
        action_L, penalty_NN_L = CheckAction(action_L)

        action = np.concatenate((action_H, action_L))

        pw_tru2, obs2, action_L, penalty_L, done, done_L, info = \
            env_test.ScenarioStep(action[:6], action[6:], pw_tru)

        action[6:] = action_L

        norm_obs2 = Normalize(obs2)

        r, c = Eval_Rew(obs, obs2, action)
        r1 = obs[0] * obs2[-1] * time - penalty_NN_H
        c1 = c
        pb_l.append(action[-1])

        # Compute pbrule for clip penalty (PER does not hard-clip pb in test)
        pb_test = float(action[-1])
        pbrule_test, rbb_info_test = rbb_system.compute_pbrule(
            pb_original=pb_test,
            ptur=float(pw_tru),
            pg_prev=float(obs[-1]),
            E=float(obs[2]),
        )
        clip_amount_test = rbb_info_test['clip_amount']
        # r2: 影子套利利润 - 100*clip_amount（pbrule 裁剪惩罚进入 r2）
        shadow_price_test = action[5] * 2  # (-0.5,0.5) → (-1,1)
        r2 = 10.0 * (-shadow_price_test * pbrule_test) - 100.0 * clip_amount_test
        # c2: SOC 越界项进入拉格朗日约束（原系数 1000/MWh，安全带 [1.5, 4.5]）
        soc_test = float(obs2[2])
        soc_dev_test = 0.0
        if soc_test < 1.5:
            soc_dev_test = 1.5 - soc_test
        elif soc_test > 4.5:
            soc_dev_test = soc_test - 4.5
        c2 = 1000.0 * soc_dev_test

        bi = step // 5
        pc = per_case[bi]
        pc['r1'] += r1
        pc['c1'] += c1
        pc['r2'] += r2
        pc['c2'] += c2
        pc['r'] += r
        pc['c'] += c
        if c > 0:
            pc['cvm'] += 1

        # profit: 上层利润 = P_G * pr * dt - 4.9*max(-pb,0)*dt（与训练 episode_profit 同口径）
        profit_step = obs2[-1] * obs[0] * time - 4.9 * max(-float(pbrule_test), 0.0) * time
        pc['profit'] += profit_step

        if test_csv:
            df = pd.concat([df, pd.DataFrame([{
                'step': step,
                'case_wind': case_ids[step][0],
                'case_price': case_ids[step][1],
                'u': obs2[1],
                'pr': obs2[0],
                'r1': r1, 'c1': c1, 'r2': r2, 'c2': c2,
                'r': r, 'c': c,
                'profit': profit_step,
                'pbrule': pbrule_test,
                'clip_amount': clip_amount_test,
                'action': action,
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
            'r': pc['r'] / n_per_case,
            'c': pc['c'] / n_per_case,
            'cvm': pc['cvm'],
            'profit': pc['profit'],  # 该 case 5 步 profit 之和
        })

    overall = {
        'r1': float(np.mean([m['r1'] for m in case_metrics])),
        'c1': float(np.mean([m['c1'] for m in case_metrics])),
        'r2': float(np.mean([m['r2'] for m in case_metrics])),
        'c2': float(np.mean([m['c2'] for m in case_metrics])),
        'r': float(np.mean([m['r'] for m in case_metrics])),
        'c': float(np.mean([m['c'] for m in case_metrics])),
        'cvm': int(sum(m['cvm'] for m in case_metrics)),
        'profit': float(sum(m['profit'] for m in case_metrics)),  # 45 步 profit 之和
    }
    return case_metrics, overall
