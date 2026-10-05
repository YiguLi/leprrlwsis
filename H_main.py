#!/usr/bin/env python3
import os
from pickle import FALSE
import numpy as np
import argparse
from copy import deepcopy
import torch

from env import WindBattery
from H_ddpg import (DDPG, DDPG_L)
from util import *
from myfunc import *

import pandas as pd
from memory import RingBuffer

from torch.utils.tensorboard import SummaryWriter

# 不显示warning
import warnings
warnings.filterwarnings("ignore")

import matlab
import matlab.engine
engine = matlab.engine.start_matlab()

def train(num_iterations_H,
          agent_H,
          agent_L,
          env,
          env_test,
          output,
          max_episode_length=None,
          debug=False,
          upload=False,
          use_csv=False):

    agent_H.is_training = True
    agent_L.is_training = True
    step = episode = episode_steps = 0
    episode_reward = episode_cost = 0.
    observation = None
    time = 1/60
    # sum_r = 0
    if use_csv:
        df = pd.DataFrame()
        df_episode = pd.DataFrame()
        df_test = pd.DataFrame()

    while step < num_iterations_H:
        # start of episode 重置环境
        prYellow('########### step:{} ###########'.format(step))
        if observation is None:
            observation = deepcopy(env.reset())  # [Pr,U,E,P_last]
            norm_observation = Normalize(observation)
            agent_H.reset(norm_observation)
        
        # agent pick action 上层选择动作 
        if episode_steps % 5 == 0:   
            sum_r1 = 0      
            if step <= args.warmup:
                applied_action_H = original_action_H = agent_H.random_action() 
                penalty_NN_H = 0
            else:
                original_action_H = agent_H.select_action(norm_observation)
                applied_action_H, penalty_NN_H = CheckAction(original_action_H)
            norm_observation_H = norm_observation
            pw_tru = env.PowerOutput(observation[1], applied_action_H)
        prGreen('--H-- ori a_t = {}'.format(round_list(original_action_H, 3)))
        
        
        norm_observation_L = np.append(norm_observation, applied_action_H[-1]) 
        prPurple('--L-- obs = {}.'.format(round_list(norm_observation_L,3)))
        # 下层每一个5-step都是一个episode
        if episode_steps % 5 == 0:    
            agent_L.reset(norm_observation_L)
            pb_l = list()
        
        # agent pick action 下层选择动作
        if step <= args.warmup:    
            applied_action_L = original_action_L = agent_L.random_action()
            penalty_NN_L = 0
        else:
            original_action_L = agent_L.select_action(norm_observation_L) 
            applied_action_L, penalty_NN_L = CheckAction(original_action_L)
        prGreen('--L-- ori a_t = {}'.format(round_list(original_action_L, 3)))
        
        action = np.concatenate((applied_action_H, applied_action_L))
        # prGreen('action = {}'.format(round_list(action,3)))
        
        # env response with next_observation, reward, terminate_info
        pw_tru2, observation2, action_L, penalty_L, done, done_L = env.step(action[:6], action[6:], pw_tru)
        action[6] = action_L 
        prGreen('Applied action = {}.'.format(round_list(action,3)))
        norm_observation2 = Normalize(observation2)
        norm_observation2_L = np.append(norm_observation2, applied_action_H[-1])
        
        r, c = Cal_Rew(observation, observation2, action)
        r1 = observation[0] * observation2[-1] * time - c - penalty_NN_H
        sum_r1 += r1
        prGreen('r1 = {}'.format(round(r1, 3)))
        pb_l.append(action[-1])
        r2 = 10 * (- abs(sum(pb_l)-action[5]) - c) - 49. * max(-action[-1],0) - penalty_L - penalty_NN_L
        prGreen('r2 = {}'.format(round(r2, 3)))
        
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True 
        if episode_steps % 5 == 4:
            agent_H.observe_jump(norm_observation_H, original_action_H, sum_r1, norm_observation2, done)
            done_L = True
        agent_L.observe_jump(norm_observation_L, original_action_L, r2, norm_observation2_L, done_L)
        
        if step > args.warmup:
            if episode_steps % 5 == 4:
                agent_H.update_policy()
                agent_L.update_policy()
            
            # Test
            if episode_steps % 10 == 0:
                r1_test, r2_test, r_test, c_test, action_test, cvm_test = H_test(agent_H, agent_L, env_test)
                if use_csv:
                    df_test = df_test.append({'r1': r1_test, 
                                              'r2': r2_test, 
                                              'reward': r_test, 
                                              'cost': c_test, 
                                              'action': action_test
                                              }, 
                                              ignore_index=True)
                    if step % 50 == 0:
                        df_test.to_csv('{}/df_test.csv'.format(output))
                if upload:
                    writer.add_scalar('test_r1', r1_test, step)
                    writer.add_scalar('test_r2', r2_test, step)
                    writer.add_scalar('test_r', r_test, step)
                    writer.add_scalar('test_c', c_test, step)

        

        # 上层保存到data
        if use_csv:
            df = df.append(
                {
                    'step': step,
                    'observation': observation,
                    'norm_observation_L': norm_observation_L,
                    'original_action_H': original_action_H,
                    'original_action_L': original_action_L,
                    'action': action,
                    'r1': r1,
                    'r2': r2,
                    'reward': r,
                    'cost': c,
                },
                ignore_index=True)
            if step % 50 == 0:
                df.to_csv('{}/df.csv'.format(output))
        if upload:
            writer.add_scalar('r1', r1, step)
            writer.add_scalar('r2', r2, step)
            writer.add_scalar('wind_speed', env.u, step)
            writer.add_scalar('reward', r, step)
            writer.add_scalar('cost', c, step)
            if episode_steps % 5 == 4:
                writer.add_scalar('sum_r1', sum_r1, step)

        # update
        step += 1
        episode_steps += 1
        episode_reward += r
        episode_cost += c
        # update env
        # Oh_observation = deepcopy(Oh_observation2)
        norm_observation = deepcopy(norm_observation2)
        observation = deepcopy(observation2)
        pw_tru = pw_tru2

        #-------------------------------------episode-----------------------------
        if done:
            # save intermideate model
            agent_H.save_model(output)
            agent_L.save_model(output)
            prGreen('##########{}: episode_reward:{} step:{}'.format(episode, round(episode_reward,2), step))

            if upload:
                writer.add_scalar('episode_reward', episode_reward, episode)
                writer.add_scalar('episode_cost', episode_cost, episode)
                writer.add_scalar('wind_day', env.j, episode)

            if use_csv:
                df_episode = df_episode.append(
                    {
                        'episode': episode,
                        'episode_reward': episode_reward,
                        'episode_cost': episode_cost,
                        'wind_day': env.j,
                    },
                    ignore_index=True)
                df_episode.to_csv('{}/df_episode.csv'.format(output))

            # end of episode
            # reset
            observation = None
            episode_steps = 0
            episode_reward = episode_cost = 0.
            episode += 1
    if use_csv:
        df.to_csv('{}/df.csv'.format(output))
        df_episode.to_csv('{}/df_episode.csv'.format(output))
        df_test.to_csv('{}/df_test.csv'.format(output))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode',
                        default='train',
                        type=str,
                        help='support option: train/test')
    parser.add_argument('--env',
                        default='H',
                        type=str,
                        help='open-ai gym environment')  #'Pendulum-v1'
    parser.add_argument('--hidden1',
                        default=16,
                        type=int,
                        help='hidden num of first fully connect layer')  # 400
    parser.add_argument('--hidden2',
                        default=100,
                        type=int,
                        help='hidden num of second fully connect layer')  # 300
    parser.add_argument('--rate',
                        default=0.001,
                        type=float,
                        help='learning rate')
    parser.add_argument('--prate',
                        default=0.0001,
                        type=float,
                        help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup',
                        default=100,
                        type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.98, type=float, help='')
    # 0.99
    parser.add_argument('--bsize', default=64, type=int,
                        help='minibatch size')  #64
    parser.add_argument('--rmsize',
                        default=40000,
                        type=int,
                        help='memory size') # 6000000
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau',
                        default=0.001,
                        type=float,
                        help='moving average for target network')
    parser.add_argument('--ou_theta',
                        default=0.15,
                        type=float,
                        help='noise theta')
    parser.add_argument('--ou_sigma',
                        default=0.2,
                        type=float,
                        help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes',
                        default=20,
                        type=int,
                        help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps',
                        default=2000,
                        type=int,
                        help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter_H',
                        default=200000,
                        type=int,
                        help='train iters each timestep')
    parser.add_argument('--epsilon',
                        default=50000,
                        type=int,
                        help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume',
                        default='default',
                        type=str,
                        help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--lam', default=0.1, type=float, help='safe constraint')
    parser.add_argument('--lr_lam', default=10e-3, type=float, help='learning rate for lambda')
    parser.add_argument('--cost_single_limit', default=3, type=float, help='cost_bound for single step')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    #TODO 修改的命令ppyp
    max_E = 6.
    nb_turbine = 5
    upload = True
    use_csv = True
    args.max_episode_length = 500  # 1d*24h*60m/5m = 288, 2d:576
    args.train_iter_H = 1000 * args.max_episode_length
    # 只在**未显式指定**（--seed 默认 -1）时随机化。显式 --seed >= 0 必须被尊重，
    # 否则"同 seed 配对 A/B"会被随机种子混淆（历史 bug：本行曾无条件覆盖用户传入的 --seed）。
    if args.seed is None or int(args.seed) < 0:
        args.seed = np.random.randint(0, 10000)

    # args.warmup = 5

    if upload:
        writer = SummaryWriter(args.output)
        writer.add_scalar('seed', args.seed)
    
    env = WindBattery(max_E, nb_turbine, upload, engine)
    env_test = WindBattery(max_E, nb_turbine, upload, engine)

    if args.seed > 0:
        np.random.seed(args.seed)

    agent_H = DDPG(nb_states = 4, nb_actions = 6, args = args)
    agent_L = DDPG_L(nb_states = 5, nb_actions = 1, args = args)
    #TODO 是否预训练
    # agent_H.load_weights('output/WB3-run296')
    # agent_L.load_weights('output/WB3-run296')

    if args.mode == 'train':
        train(args.train_iter_H,
              agent_H,
              agent_L,
              env,
              env_test,
              args.output,
              max_episode_length=args.max_episode_length,
              debug=args.debug,
              upload=upload,
              use_csv=use_csv)
    if upload:
        writer.close()

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
