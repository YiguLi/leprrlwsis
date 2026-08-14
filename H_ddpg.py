import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic, Actor_L, Critic_L)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()


class DDPG(object):
    def __init__(self, nb_states, nb_actions, args):

        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic_p = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target_p = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim_p = Adam(self.critic_p.parameters(), lr=args.rate)

        hard_update(self.actor_target,self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target_p, self.critic_p)

        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize,
                                       window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions,
                                                       theta=args.ou_theta,
                                                       mu=args.ou_mu,
                                                       sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        #
        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True
        # self.p_loss = 0.

        #
        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, profit_batch, cost_batch,\
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        # print('-------sample------')
        # print(state_batch, action_batch, profit_batch, next_state_batch, terminal_batch)
        
        # Prepare for the target q batch -- profit
        next_q_values_p = self.critic_target_p([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values_p.volatile = False

        target_q_batch_p = to_tensor(profit_batch) + \
            self.discount * to_tensor(terminal_batch.astype(float)) * next_q_values_p

        # Critic update
        self.critic_p.zero_grad()

        q_batch_p = self.critic_p(
            [to_tensor(state_batch),
             to_tensor(action_batch)])

        value_loss_p = criterion(q_batch_p, target_q_batch_p)
        value_loss_p.backward()
        #Gradient Clipping
        nn.utils.clip_grad_norm_(self.critic_p.parameters(), max_norm=2.0, norm_type=2)
        # nn.utils.clip_grad_value_(self.critic_p.parameters(), clip_value=1.0)
        self.critic_optim_p.step()

        # Actor update
        self.actor.zero_grad()
        policy_loss = -(self.critic_p([to_tensor(state_batch),self.actor(to_tensor(state_batch))])) #TODO

        policy_loss = policy_loss.mean()
        # self.p_loss = policy_loss.item()
        # prGreen('*** {} policy loss: {}'.format(self.nb_actions, self.p_loss))
        policy_loss.backward(retain_graph=True)
        
        #Gradient Clipping
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optim.step()

        # check gradient
        # for name, parms in self.actor.named_parameters():	
        #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)
        # print('value_loss: {:.4f}'.format(value_loss_p.item()))
        # print('policy_loss: {:.4f}'.format(policy_loss.item()))

        
        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target_p, self.critic_p, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()

        self.critic_p.eval()
        self.critic_target_p.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic_p.cuda()
        self.critic_target_p.cuda()
    
    def observe_jump(self, s_t, a_t, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, None, s_t1, done)


    def random_action(self):
        charge = 10.
        if self.nb_actions == 5:
            action = np.random.uniform(0., 0.5, self.nb_actions)
        if self.nb_actions == 1:
            action = np.random.uniform(-charge, charge, self.nb_actions)
        if self.nb_actions == 6:
            action_H = np.random.uniform(0., 0.5, 5)
            action_L = np.random.uniform(-charge, charge, 1)
            action = np.concatenate((action_H, action_L))
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)
        # action += self.is_training * max(self.epsilon,0) * self.random_process.sample()
        action += self.is_training * max(self.epsilon,0) * ((np.random.random(10)-0.5)/10)[0] # noise [0.0, 1.0)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(torch.load('{}/actor_{}.pkl'.format(output, self.nb_actions)))
        self.critic_p.load_state_dict(torch.load('{}/critic_p_{}.pkl'.format(output, self.nb_actions)))

        hard_update(self.actor_target,self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target_p, self.critic_p)

    def save_model(self, output):
        torch.save(self.actor.state_dict(), '{}/actor_{}.pkl'.format(output, self.nb_actions))
        torch.save(self.critic_p.state_dict(), '{}/critic_p_{}.pkl'.format(output, self.nb_actions))

    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)

class DDPG_L(DDPG):
    def __init__(self,nb_states, nb_actions, args):
        super().__init__(nb_states, nb_actions, args)

        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }
        self.actor = Actor_L(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor_L(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic_p = Critic_L(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target_p = Critic_L(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim_p = Adam(self.critic_p.parameters(), lr=args.rate)

        hard_update(self.actor_target,self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target_p, self.critic_p)

        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize,
                                       window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions,
                                                       theta=args.ou_theta,
                                                       mu=args.ou_mu,
                                                       sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        #
        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True
        # self.p_loss = 0.

        if USE_CUDA: self.cuda()

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)
        # action += self.is_training * max(self.epsilon,0) * self.random_process.sample()
        action += self.is_training * max(self.epsilon,0) * ((np.random.random(10)-0.5)/10)[0] # noise [0.0, 1.0)
        if decay_epsilon:
            self.epsilon -= self.depsilon
        self.a_t = action
        return action