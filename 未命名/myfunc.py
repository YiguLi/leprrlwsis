import numpy as np
import matlab
import matlab.engine
engine = matlab.engine.start_matlab()
from util import *
import pandas as pd

def Cal_Rew(observation, observation2, action_all, cost_single_cons = 3.):
    time = 1/60
    pr = observation[0]
    p_g = observation2[-1]
    p_g_last = observation[-1]
    
    P_B = action_all[-1]
    r = (p_g * pr - 490 * max(-P_B,0)) * time  # 4900 # deg:unit：RMB/Mwh
    
    P_F = abs(p_g_last - p_g)
    if P_F > cost_single_cons:
        c = 10 * (P_F - cost_single_cons) + 0.1 * cost_single_cons
        # c = 10 * P_F
        # c = 10 * (P_F - cost_single_cons) + 1 * cost_single_cons
    else:
        c = 0.1 * P_F   
    return r, c

def Eval_Rew(observation, observation2, action_all, cost_single_cons = 3.):
    time = 1/60
    pr = observation[0]
    p_g = observation2[-1]
    p_g_last = observation[-1]
    
    P_B = action_all[-1]
    r = (p_g * pr - 490. * max(-P_B,0)) * time  # 4900 # deg:unit：RMB/Mwh
    
    P_F = abs(p_g_last - p_g)
    if P_F > cost_single_cons:
        c = P_F - cost_single_cons
    else:
        c = 0.  
    return r, c

def CheckAction(action, charge = 10.):
    penalty = 0
    # case 1 总的action
    if len(action) == 7:
        for j in action[:5]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 10 * ((ma - 0.5)+ (0. - mi))

        for j in action[6:]:
            ma = max(charge, j)
            mi = min(-charge, j)
            penalty += 10 * ((ma - charge)+ (-charge - mi))

        a_H = np.clip(action[:5], 0., 0.5)
        a_H = np.append(a_H, action[5])
        a_L = np.clip(action[6:], -charge, charge)
        applied_action = np.concatenate((a_H, a_L))

    # case 2 风机
    if len(action) == 6: 
        for j in action[:5]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 10 * ((ma - 0.5)+ (0. - mi))
        applied_action = np.clip(action[:5], 0., 0.5)
        applied_action = np.append(applied_action, action[5])
    if len(action) == 5:
        for j in action[:5]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 10 * ((ma - 0.5)+ (0. - mi))
        applied_action = np.clip(action[:5], 0., 0.5)

    if len(action) == 11: 
        for j in action[:10]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 10 * ((ma - 0.5)+ (0. - mi))
        applied_action = np.clip(action[:10], 0., 0.5)
        applied_action = np.append(applied_action, action[10])
    if len(action) == 10:
        for j in action[:10]:
            ma = max(0.5, j)
            mi = min(0, j)
            penalty += 10 * ((ma - 0.5)+ (0. - mi))
        applied_action = np.clip(action[:10], 0., 0.5)
    # case 3 电池
    if len(action) == 1:
        for j in action:
            ma = max(charge, j)
            mi = min(-charge, j)
            penalty += 10 * ((ma - charge) + (-charge - mi))
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

def S_test(agent_H, agent_L, env_test, num_iterations = 5, test_csv = False, output = None, day = 0):
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

        norm_obs_L = np.append(norm_obs,action_H[-1])
        action_L = agent_L.select_action(norm_obs_L)
        action_L, penalty_NN_L = CheckAction(action_L)

        action = np.concatenate((action_H, action_L)) 

        pw_tru2, obs2, action_L, penalty_L, done, info = env_test.TestStep(action[:6], action[6:], pw_tru)
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
        r2 = - 490. * max(-action[-1],0) - penalty_L - penalty_NN_L
        c2 = 0.1 * abs(sum(pb_l)-action[env_test.num_turbine]) + c #todo
        if test_csv:
            df = df.append(
                {
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
                },
                ignore_index=True)
            
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

        norm_obs_L = np.append(norm_obs,action_H[-1])
        action_L = agent_L.select_action(norm_obs_L)
        action_L, penalty_NN_L = CheckAction(action_L)

        action = np.concatenate((action_H, action_L)) 

        pw_tru2, obs2, action_L, penalty_L, done, info = env_test.TestStep(action[:6], action[6:], pw_tru)
        
        action[6:] = action_L
        episode_action.append(action)

        norm_obs2 = Normalize(obs2)

        r, c = Eval_Rew(obs, obs2, action)
        if c>0:
            cvm += 1
        r1 = obs[0] * obs2[-1] * time - c - penalty_NN_H
        sum_r1 += r1
        pb_l.append(action[-1])
        r2 = 1 * (- abs(sum(pb_l)-action[5]) - c) - 490. * max(-action[-1],0) - penalty_L - penalty_NN_L
        if test_csv:
            df = df.append(
                {
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
                },
                ignore_index=True)
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

        pw_tru2, obs2, action_L, penalty_L, done, info = env_test.TestStep(action_test[:5], action_test[5:], pw_tru)
        
        action_test[5:] = action_L
        episode_action.append(action_test)

        norm_obs2 = Normalize(obs2)

        r, c = Eval_Rew(obs, obs2, action_test)
        if c>0:
            cvm += 1
        r1 = (pw_tru * 300.) * time - penalty_NN_H
        r2 = obs2[-1] * 300. * time - c - penalty_L - penalty_NN_L
        if test_csv:
            df = df.append(
                {
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
                },
                ignore_index=True)
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