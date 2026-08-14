"""
PRLE Extended SP-DDPG Agent
============================
Extends SP_DDPG from PER_ddpg.py to use PRLEBuffer instead of
PrioritizedReplayBuffer. The observe_jump method is extended to
accept an optional context parameter for reward relabeling.

All other methods (update_policy, select_action, random_action, reset,
save_model, save_checkpoint, load_weights, seed) are inherited unchanged.
"""

import numpy as np
import torch

from model import (Actor, Critic)
from prle.prle_buffer import PRLEBuffer
from random_process import OrnsteinUhlenbeckProcess
from util import *

criterion = torch.nn.MSELoss()


class PRLE_DDPG(object):
    """
    SP-DDPG agent variant that uses PRLEBuffer for reward relabeling support.

    The only difference from PER_ddpg.SP_DDPG:
      1. Uses PRLEBuffer instead of PrioritizedReplayBuffer
      2. observe_jump() accepts an optional 'context' parameter
      3. observe() accepts an optional 'context' parameter
    """

    def __init__(self, nb_states, nb_actions, args, lam_max=None):

        # Determine device first
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        print(f"Using device: {self.device}")

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
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.prate)

        self.critic_p = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target_p = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim_p = torch.optim.Adam(self.critic_p.parameters(), lr=args.rate)

        self.critic_c = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target_c = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim_c = torch.optim.Adam(self.critic_c.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target_p, self.critic_p)
        hard_update(self.critic_target_c, self.critic_c)

        # Create PRLE replay buffer (block_size=5 for H, =1 for L)
        block_size = 5 if nb_actions >= 6 else 1
        self.memory = PRLEBuffer(size=args.rmsize,
                                  alpha=args.PER_alpha,
                                  block_size=block_size)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions,
                                                       theta=args.ou_theta,
                                                       mu=args.ou_mu,
                                                       sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True

        # Ensure all models and tensors are on the same device
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic_p.to(self.device)
        self.critic_target_p.to(self.device)
        self.critic_c.to(self.device)
        self.critic_target_c.to(self.device)

        # Dual variable
        self.lam = args.lam
        self.lr_lam = args.lr_lam
        self.lam_max = lam_max  # None means no upper bound
        self.cost_limit = args.cost_single_limit
        self.lam_loss = 0.0  # critic_c - cost_limit, updated in update_policy()

        self.PER_beta = args.PER_beta
        self.PER_epsilon = args.PER_epsilon
        self.beta_increment_per_sampling = args.beta_increment_per_sampling
        self.nb_turbine = args.nb_turbine

    def update_policy(self):
        """Sample batch, compute TD errors, update actor/critic/dual. Identical to SP_DDPG."""
        # Sample batch
        state_batch, action_batch, profit_batch, cost_batch,\
        next_state_batch, terminal_batch, weights, batch_idxes \
        = self.memory.sample(self.batch_size, self.PER_beta, self.beta_increment_per_sampling)

        # reshape data
        profit_batch = profit_batch.reshape(-1, 1)
        cost_batch = cost_batch.reshape(-1, 1)
        terminal_batch = terminal_batch.reshape(-1, 1)

        # Ensure all tensors are on the correct device
        state_batch = to_tensor(state_batch).to(self.device)
        action_batch = to_tensor(action_batch).to(self.device)
        profit_batch = to_tensor(profit_batch).to(self.device)
        cost_batch = to_tensor(cost_batch).to(self.device)
        next_state_batch = to_tensor(next_state_batch).to(self.device)
        terminal_batch = to_tensor(terminal_batch).to(self.device)
        weights = torch.Tensor(weights).to(self.device)

        # Prepare for the target q batch -- profit
        next_q_values_p = self.critic_target_p([
            next_state_batch,
            self.actor_target(next_state_batch),
        ])

        # Prepare for the target q batch -- cost
        next_q_values_c = self.critic_target_c([
            next_state_batch,
            self.actor_target(next_state_batch),
        ])

        target_q_batch_p = profit_batch + \
            self.discount * terminal_batch * next_q_values_p
        target_q_batch_c = cost_batch + \
            self.discount * terminal_batch * next_q_values_c

        # Critic update
        self.critic_p.zero_grad()
        self.critic_c.zero_grad()

        q_batch_p = self.critic_p([
            state_batch,
            action_batch
        ])
        q_batch_c = self.critic_c([
            state_batch,
            action_batch
        ])

        TD_errors_p = abs(target_q_batch_p - q_batch_p)
        TD_errors_c = abs(target_q_batch_c - q_batch_c)

        # Compute the value loss
        value_loss_p = torch.mean(torch.mean(weights * (q_batch_p - target_q_batch_p) ** 2, dim=1))
        value_loss_p.backward()
        nn = torch.nn
        nn.utils.clip_grad_norm_(self.critic_p.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optim_p.step()

        # Compute the value loss for cost
        value_loss_c = torch.mean(torch.mean(weights * (q_batch_c - target_q_batch_c) ** 2, dim=1))
        value_loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic_c.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optim_c.step()

        # Actor update
        self.actor.zero_grad()
        policy_loss = -(self.critic_p([state_batch, self.actor(state_batch)]) -
            self.lam * self.critic_c([state_batch, self.actor(state_batch)]))

        policy_loss = policy_loss.mean()
        policy_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)

        self.actor_optim.step()

        # Update dual variable
        lam_loss = (self.critic_c([state_batch, self.actor(state_batch)]) - self.cost_limit)
        lam_loss = lam_loss.mean().item()
        self.lam_loss = lam_loss
        self.lam += lam_loss * self.lr_lam
        self.lam = max(self.lam, 0.1)
        if self.lam_max is not None:
            self.lam = min(self.lam, self.lam_max)

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target_p, self.critic_p, self.tau)
        soft_update(self.critic_target_c, self.critic_c, self.tau)

        # For prioritized experience replay
        td_errors = (TD_errors_p + self.lam * TD_errors_c).detach().cpu().numpy()
        new_priorities = td_errors + self.PER_epsilon
        self.memory.update_priorities(batch_idxes, new_priorities)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic_p.eval()
        self.critic_target_p.eval()
        self.critic_c.eval()
        self.critic_target_c.eval()

    def observe(self, r_t, c_t, s_t1, done, context=None):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, c_t, s_t1, done, context=context)
            self.s_t = s_t1

    def observe_jump(self, s_t, a_t, r_t, c_t, s_t1, done, context=None):
        """Store transition with optional context for reward relabeling."""
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, c_t, s_t1, done, context=context)

    def random_action(self):
        charge = 10.
        if self.nb_actions == self.nb_turbine:
            action = np.random.uniform(0., 0.5, self.nb_actions)
        if self.nb_actions == 1:
            action = np.random.uniform(-charge, charge, self.nb_actions)
        if self.nb_actions == self.nb_turbine + 1:
            action_H = np.random.uniform(0., 0.5, self.nb_turbine)
            action_L = np.random.uniform(-charge, charge, 1)
            action = np.concatenate((action_H, action_L))
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)
        action += self.is_training * max(self.epsilon, 0) * ((np.random.random(self.nb_actions) - 0.5) / 10)[0]

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None:
            return
        import torch
        self.actor.load_state_dict(torch.load('{}/actor_{}.pkl'.format(output, self.nb_actions)))
        self.critic_p.load_state_dict(torch.load('{}/critic_p_{}.pkl'.format(output, self.nb_actions)))
        self.critic_c.load_state_dict(torch.load('{}/critic_c_{}.pkl'.format(output, self.nb_actions)))
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target_p, self.critic_p)
        hard_update(self.critic_target_c, self.critic_c)

    def save_model(self, output):
        import torch
        torch.save(self.actor.state_dict(), '{}/actor_{}.pkl'.format(output, self.nb_actions))
        torch.save(self.critic_p.state_dict(), '{}/critic_p_{}.pkl'.format(output, self.nb_actions))
        torch.save(self.critic_c.state_dict(), '{}/critic_c_{}.pkl'.format(output, self.nb_actions))

    def save_checkpoint(self, output, step):
        import os
        ckpt_dir = os.path.join(output, 'checkpoint_{}'.format(step))
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.actor.state_dict(),
                   '{}/actor_{}.pkl'.format(ckpt_dir, self.nb_actions))
        torch.save(self.critic_p.state_dict(),
                   '{}/critic_p_{}.pkl'.format(ckpt_dir, self.nb_actions))
        torch.save(self.critic_c.state_dict(),
                   '{}/critic_c_{}.pkl'.format(ckpt_dir, self.nb_actions))

    def seed(self, s):
        torch.manual_seed(s)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(s)
