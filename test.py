#!/usr/bin/env python3
import os
from pickle import FALSE
import numpy as np
import argparse
from copy import deepcopy
import torch

from env import WindBattery
from sac import *
from sac_S import *
from sac_SP import *

from H_ddpg import *
from S_ddpg import *
from PER_ddpg import *

from ppo import *


from util import *
from myfunc import *

import pandas as pd

# 不显示warning
import warnings
warnings.filterwarnings("ignore")

import matlab
import matlab.engine
engine = matlab.engine.start_matlab()



def test(test_folder,env_test,args, num_iterations,alg,day):    
    
    
    if alg=='DDPG-BI':
        agent_H = DDPG(nb_states = 1, nb_actions = 5, args = args)
        agent_L = DDPG_L(nb_states = 3, nb_actions = 1, args = args)
        agent_H.load_weights(test_folder)
        agent_L.load_weights(test_folder)
        episode_r1, episode_r2, episode_reward, episode_cost, episode_action, cvm = BI_test(agent_H, agent_L, env_test,num_iterations, True, args.output,day)

    if alg=='SAC-H':
        agent_H = SAC(nb_states=4, nb_actions=6, args=args)
        agent_L = SAC(nb_states=5, nb_actions=1, args=args)
        agent_H.load_weights(test_folder)
        agent_L.load_weights(test_folder)
        episode_r1, episode_r2, episode_reward, episode_cost, episode_action, cvm = H_test(agent_H, agent_L, env_test, num_iterations, True, args.output,day)
    
    if alg=='PPO':
        agent_H = PPO(state_dim=4, 
                    action_dim=6, 
                    lr_actor=args.lr_actor, 
                    lr_critic=args.lr_critic, 
                    gamma=args.gamma, 
                    K_epochs=args.K_epochs, 
                    eps_clip=args.eps_clip, 
                    has_continuous_action_space=args.has_continuous_action_space, 
                    action_std_init=args.action_std)
    # agent.load_weights('output/WindBattery-run4')
        agent_L = PPO(state_dim=5, 
                    action_dim=1, 
                    lr_actor=args.lr_actor, 
                    lr_critic=args.lr_critic, 
                    gamma=args.gamma, 
                    K_epochs=args.K_epochs, 
                    eps_clip=args.eps_clip, 
                    has_continuous_action_space=args.has_continuous_action_space, 
                    action_std_init=args.action_std)
        agent_H.load_weights(test_folder)
        agent_L.load_weights(test_folder)
        episode_r1, episode_r2, episode_reward, episode_cost, episode_action, cvm = H_test(agent_H, agent_L, env_test, num_iterations,True,args.output,day)
    


    if alg=='DDPG-S':
        agent_H = PDO_DDPG(nb_states = 4, nb_actions = 6, args = args)
        agent_L = PDO_DDPG(nb_states = 5, nb_actions = 1, args = args)
        agent_H.load_weights(test_folder)
        agent_L.load_weights(test_folder)
        episode_r1, episode_c1, episode_r2, episode_c2, episode_reward, episode_cost, action, cvm = S_test(agent_H, agent_L, env_test, num_iterations,True,args.output,day)    

    if alg=='SAC-S':
        agent_H = SAC_S(nb_states = 4, nb_actions = 6, args = args)
        agent_L = SAC_S(nb_states = 5, nb_actions = 1, args = args)
        agent_H.load_weights(test_folder)
        agent_L.load_weights(test_folder)
        episode_r1, episode_c1, episode_r2, episode_c2, episode_reward, episode_cost, action, cvm = S_test(agent_H, agent_L, env_test, num_iterations,True,args.output,day)
    
    
    return episode_reward, episode_cost, cvm



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode',
                        default='test',
                        type=str,
                        help='support option: train/test')
    parser.add_argument('--env',
                        default='TEST',
                        type=str,
                        help='open-ai gym environment')
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
                        default=200,
                        type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.98, type=float, help='')
    # 0.99
    parser.add_argument('--bsize', default=32, type=int,
                        help='minibatch size')  #64
    parser.add_argument('--rmsize',
                        default=20000,
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
    parser.add_argument('--cuda', default=True, type=bool, help='Whether to use CUDA')
    parser.add_argument('--lam', default=0.1, type=float, help='safe constraint')
    parser.add_argument('--lr_lam', default=10e-3, type=float, help='learning rate for lambda')
    parser.add_argument('--cost_single_limit', default=3, type=float, help='cost_bound for single step')

    parser.add_argument('--PER_alpha', default=0.6, type=float, 
                        help='[0~1] convert the importance of TD error to priority. 0 - no prioritization, 1 - full prioritization')
    parser.add_argument('--PER_beta', default=0.4, type=float, 
                        help='importance-sampling, from initial value increasing to 1. 0 - no corrections, 1 - full correction')
    parser.add_argument('--PER_epsilon', default=0.01, type=float, 
                        help='small amount to avoid zero priority')
    parser.add_argument('--beta_increment_per_sampling', default=0., type=float, 
                        help='')
    
    # SAC specific parameters
    parser.add_argument('--hidden_width', default=256, type=int, help='The number of neurons in hidden layers')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for training')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--adaptive_alpha', default=True, type=bool, help='Whether to automatically learn the temperature alpha')

    # PPO specific parameters
    parser.add_argument('--has_continuous_action_space', default=True, type=bool) 

    parser.add_argument('--max_training_timesteps', default=int(3e6), type=int)

    parser.add_argument('--action_std', default=0.6, type=float,
                        help='starting std for action distribution (Multivariate Normal)')
    parser.add_argument('--action_std_decay_rate', default=0.05 , type=float,
                        help='linearly decay action_std (action_std = action_std - action_std_decay_rate)')
    parser.add_argument('--min_action_std', default=0.1 , type=float,
                        help='minimum action_std (stop decay after action_std <= min_action_std)')
    parser.add_argument('--action_std_decay_freq', default=int(2.5e5) , type=int,
                        help='action_std decay frequency (in num timesteps)')
    
    ################ PPO hyperparameters ################
    parser.add_argument('--update_timestep', default=10, type=int,
                        help='update policy every n timesteps')
    parser.add_argument('--K_epochs', default=80 , type=int,
                        help='update policy for K epochs in one PPO update')
    
    parser.add_argument('--eps_clip', default=0.2 , type=float,
                        help='clip parameter for PPO')

    parser.add_argument('--lr_actor', default=0.0003 , type=float)
    parser.add_argument('--lr_critic', default=0.001 , type=float)
    

    args = parser.parse_args()

    #TODO 修改的命令ppyp
    max_E = 0.5
    nb_turbine = 5
    upload = True
    use_csv = True
    args.max_episode_length = 500  # 1d*24h*60m/5m = 288, 2d:576
    args.train_iter_H = 1000 * args.max_episode_length
    args.seed = np.random.randint(0, 10000)
    
    

    # Set device and random seeds
    if (args.cuda and torch.cuda.is_available()):
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        print("Using GPU:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Set random seeds
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子

    
    env_test = WindBattery(max_E, nb_turbine, upload, engine)

    if args.seed > 0:
        np.random.seed(args.seed)

    #FIXME:day=1-4
    day = 4

    alg = 'DDPG-S'
    test_folder = 'output/PER-run394'


    args.env = 'TEST-' + alg + test_folder.split('-')[-1][3:]
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)
   
    num_iterations = 140
    

    episode_reward, episode_cost, cvm = test(test_folder,env_test,args, num_iterations,alg,day)
    prLightPurple(f"episode_reward: {episode_reward}, episode_cost: {episode_cost}, cvm: {cvm}")


