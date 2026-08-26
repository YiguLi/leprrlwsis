import numpy as np
import matlab
import matlab.engine
engine = matlab.engine.start_matlab()
from util import *
import pandas as pd

import sys, os
_RBB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rbb')
if _RBB_DIR not in sys.path:
    sys.path.insert(0, _RBB_DIR)
from rbb_rule_bess import RuleBasedBESS

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
    # case 1 总的action (5 WT + 1 signal + 1 BESS)
    if len(action) == 7:
        for j in action[:5]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 500 * ((ma - 0.5)+ (0. - mi))

        # 影子价格限定范围(-0.5, 0.5), 越界惩罚计入penalty
        signal_val = action[5]
        signal_ma = max(0.5, signal_val)
        signal_mi = min(-0.5, signal_val)
        penalty += 500 * ((signal_ma - 0.5) + (-0.5 - signal_mi))

        for j in action[6:]:
            ma = max(charge, j)
            mi = min(-charge, j)
            penalty += 500 * ((ma - charge)+ (-charge - mi))

        a_H = np.clip(action[:5], 0., 0.5)
        a_H = np.append(a_H, np.clip(signal_val, -0.5, 0.5))
        a_L = np.clip(action[6:], -charge, charge)
        applied_action = np.concatenate((a_H, a_L))

    # case 2 风机 + 影子价格信号
    if len(action) == 6:
        for j in action[:5]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 500 * ((ma - 0.5)+ (0. - mi))
        applied_action = np.clip(action[:5], 0., 0.5)
        # 影子价格限定范围(-0.5, 0.5), 越界惩罚计入penalty_NN_H
        signal_val = action[5]
        signal_ma = max(0.5, signal_val)
        signal_mi = min(-0.5, signal_val)
        penalty += 500 * ((signal_ma - 0.5) + (-0.5 - signal_mi))
        applied_action = np.append(applied_action, np.clip(signal_val, -0.5, 0.5))
    if len(action) == 5:
        for j in action[:5]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 500 * ((ma - 0.5)+ (0. - mi))
        applied_action = np.clip(action[:5], 0., 0.5)

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

def S_test(agent_H, agent_L, env_test, rbb_system, num_iterations = 5, test_csv = False, output = None, day = 0):
    obs = env_test.TestReset(day)
    norm_obs = Normalize(obs)
    time = 1/60
    agent_H.is_training = agent_L.is_training = False
    step = 0
    episode_reward = episode_cost = episode_r1 = episode_r2 = episode_c1 = episode_c2 = 0
    episode_action = list()
    df = pd.DataFrame()
    data_list = []
    cvm = 0

    
    while step < num_iterations:
        if step % 10 == 0:
            print(f"step: {step}")
        if step % 5 == 0:
            action_H = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H)
            pw_tru = env_test.PowerOutput(obs[1], action_H)
            pb_l = list()
            sum_r1 = 0
            sum_c1 = 0

        # L 层观测自适应: TD3 系 (nb_states=6) 拼入 ptur/10, 基线 (5) 保持原样
        if agent_L.nb_states == 6:
            norm_obs_L = np.concatenate([norm_obs, [pw_tru / 10.0], [action_H[-1]]])
        else:
            norm_obs_L = np.append(norm_obs, action_H[-1])
        action_L = agent_L.select_action(norm_obs_L)
        action_L, penalty_NN_L = CheckAction(action_L)

        action = np.concatenate((action_H, action_L)) 

        pw_tru2, obs2, action_L, penalty_L, done, done_L, info = env_test.TestStep(action[:6], action[6:], pw_tru)
        # pw_ref = env_test.PowerOutput(obs[1], np.ones(env_test.num_turbine)*1/3)

        action[(env_test.num_turbine+1):] = action_L
        episode_action.append(action)

        norm_obs2 = Normalize(obs2)

        r, c = Eval_Rew(obs, obs2, action)
        if c>0:
            cvm += 1
        r1 = obs[0] * obs2[-1] * time - penalty_NN_H
        c1 = c
        sum_r1 += r1
        sum_c1 += c1
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
        pbclip_test = clip_amount_test  # distance from pboriginal to [pbrulemin, pbrulemax]; PER does NOT clip pb
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
        if test_csv:
            df = pd.concat([df, pd.DataFrame([{
                'step': step,
                'obs': obs,
                'norm_obs_L': norm_obs_L,
                'action': action,
                'action_L': action_L,
                'r1': r1,
                'sum_r1': sum_r1,
                'c1': c1,
                'r2': r2,
                'c2': c2,
                'r': r,
                'c': c,
                'pbclip': pbclip_test,
            }])], ignore_index=True)

            # npy文件
            data = {
                'step': step,
                'obs': obs,
                'obs2': obs2,
                'g': done,
                # 'pw_tru': pw_tru,
                # 'pw_ref': pw_ref,
                'r1': r1,
                'r2': r2,
                'c1': c1,
                'c2': c2,
                'r': r,
                'c': c,
            }
            data_list.append(data)
            if step%50 == 0:
                df.to_csv(f'{output}/test_s.csv')
                np.save(f'{output}/test_s.npy', data_list)
            

        episode_reward += r
        episode_cost += c
        episode_r1 += r1
        episode_r2 += r2
        episode_c1 += c1
        episode_c2 += c2
        # update env
        norm_obs = norm_obs2
        obs = obs2
        pw_tru = pw_tru2
        step += 1

    agent_H.is_training = True
    agent_L.is_training = True
    if test_csv:
        df.to_csv(f'{output}/test_s.csv')
        np.save(f'{output}/test_s.npy', data_list)
    # return episode_reward, episode_cost
    return episode_r1, episode_c1, episode_r2, episode_c2, episode_reward, episode_cost, action, cvm


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

        # L 层观测自适应: TD3 系 (nb_states=6) 拼入 ptur/10, 基线 (5) 保持原样
        if agent_L.nb_states == 6:
            norm_obs_L = np.concatenate([norm_obs, [pw_tru / 10.0], [action_H[-1]]])
        else:
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


def H_test(agent_H, agent_L, env_test, num_iterations = 5, test_csv = False, output = None, day = 0):
    obs = env_test.TestReset(day)
    norm_obs = Normalize(obs)
    time = 1/60
    agent_H.is_training = agent_L.is_training = False
    step = 0
    episode_reward = episode_cost = episode_r1 = episode_r2 = 0
    episode_action = list()
    df = pd.DataFrame()
    data_list = []
    cvm = 0

    while step < num_iterations:
        if step % 10 == 0:
            print(f"step: {step}")
        if step % 5 == 0:
            action_H = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H)
            pw_tru = env_test.PowerOutput(obs[1], action_H)
            pb_l = list()
            sum_r1 = 0

        # L 层观测自适应: TD3 系 (nb_states=6) 拼入 ptur/10, 基线 (5) 保持原样
        if agent_L.nb_states == 6:
            norm_obs_L = np.concatenate([norm_obs, [pw_tru / 10.0], [action_H[-1]]])
        else:
            norm_obs_L = np.append(norm_obs, action_H[-1])
        action_L = agent_L.select_action(norm_obs_L)
        action_L, penalty_NN_L = CheckAction(action_L)

        action = np.concatenate((action_H, action_L)) 

        pw_tru2, obs2, action_L, penalty_L, done, done_L, info = env_test.TestStep(action[:6], action[6:], pw_tru)

        action[6:] = action_L
        episode_action.append(action)

        norm_obs2 = Normalize(obs2)

        r, c = Eval_Rew(obs, obs2, action)
        if c>0:
            cvm += 1
        r1 = obs[0] * obs2[-1] * time - c - penalty_NN_H
        sum_r1 += r1
        pb_l.append(action[-1])
        r2 = 1 * (- abs(sum(pb_l)-action[5]) - c) - 4.9 * max(-action[-1],0) - penalty_L - penalty_NN_L
        if test_csv:
            df = pd.concat([df, pd.DataFrame([{
                'step': step,
                'obs': obs,
                'norm_obs_L': norm_obs_L,
                'action': action,
                'action_L': action_L,
                'r1': r1,
                'sum_r1': sum_r1,
                'r2': r2,
                'r': r,
                'c': c,
            }])], ignore_index=True)
            # npy文件
            data = {
                'step': step,
                'obs': obs,
                'obs2': obs2,
                'g': done,
                'r1': r1,
                'r2': r2,
                'c1': None,
                'c2': None,
                'r': r,
                'c': c,
            }
            data_list.append(data)
            if step%50 == 0:
                df.to_csv(f'{output}/test_h.csv')
                np.save(f'{output}/test_h.npy', data_list)

        episode_reward += r
        episode_cost += c
        episode_r1 += r1
        episode_r2 += r2
        # update env
        norm_obs = norm_obs2
        obs = obs2
        pw_tru = pw_tru2
        step += 1

    agent_H.is_training = True
    agent_L.is_training = True
    if test_csv:
        df.to_csv(f'{output}/test_h.csv')
        np.save(f'{output}/test_h.npy', data_list)
    return episode_r1, episode_r2, episode_reward, episode_cost, episode_action, cvm


def BI_test(agent_H, agent_L, env_test, num_iterations = 5, test_csv = False, output = None, day = 0):
    obs = env_test.TestReset(day)
    norm_obs = Normalize(obs)
    time = 1/60
    agent_H.is_training = agent_L.is_training = False
    step = 0
    episode_reward = episode_cost = episode_r1 = episode_r2 = 0
    episode_action = []
    df = pd.DataFrame()
    data_list = []
    cvm = 0

    while step < num_iterations:
        if step % 10 == 0:
            print(f"step: {step}")
        if step % 5 == 0:
            action_H = agent_H.select_action(norm_obs[1:2])
            action_H, penalty_NN_H = CheckAction(action_H)
            pw_tru = env_test.PowerOutput(obs[1], action_H)

        norm_obs_L = np.append(norm_obs[2:], pw_tru/10)
        action_L = agent_L.select_action(norm_obs_L)
        action_L, penalty_NN_L = CheckAction(action_L)

        action = np.concatenate((action_H, action_L)) 
        action_test, penalty_NN = CheckAction(action, env_test.max_E)

        pw_tru2, obs2, action_L, penalty_L, done, done_L, info = env_test.TestStep(action_test[:5], action_test[5:], pw_tru)
        
        action_test[5:] = action_L
        episode_action.append(action_test)

        norm_obs2 = Normalize(obs2)

        r, c = Eval_Rew(obs, obs2, action_test)
        if c>0:
            cvm += 1
        r1 = (pw_tru * 300.) * time - penalty_NN_H
        r2 = obs2[-1] * 300. * time - c - penalty_L - penalty_NN_L
        if test_csv:
            df = pd.concat([df, pd.DataFrame([{
                'step': step,
                'obs': obs,
                'norm_obs_L': norm_obs_L,
                'action': action,
                'action_L': action_L,
                'r1': r1,
                'sum_r1': None,
                'r2': r2,
                'r': r,
                'c': c,
            }])], ignore_index=True)
            # npy文件
            data = {
                'step': step,
                'obs': obs,
                'obs2': obs2,
                'g': done,
                'r1': r1,
                'r2': r2,
                'c1': None,
                'c2': None,
                'r': r,
                'c': c,
            }
            data_list.append(data)
            if step%50 == 0:
                df.to_csv(f'{output}/test_bi.csv')
                np.save(f'{output}/test_bi.npy', data_list)

        episode_reward += r
        episode_cost += c
        episode_r1 += r1
        episode_r2 += r2
        # update env
        norm_obs = norm_obs2
        obs = obs2
        pw_tru = pw_tru2
        step += 1

    agent_H.is_training = True
    agent_L.is_training = True
    if test_csv:
        df.to_csv(f'{output}/test_bi.csv')
        np.save(f'{output}/test_bi.npy', data_list)

    return episode_r1, episode_r2, episode_reward, episode_cost, episode_action, cvm