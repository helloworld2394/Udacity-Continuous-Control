# DDPG agent class
# Copyright 2018 hojun yoon. All rights reserved.

import torch
import numpy as np
import collections as col
import random


class DDPGbuilder:
    def __init__(self, state_size, action_size, actor_nn, critic_nn, lr, batch_size, actor_l2, critic_l2,
                 discount_factor, update_tau, frames, replay_size, ou_theta, ou_sigma):

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(0)

        self._observation_size = state_size
        self._state_size = state_size * frames
        self._action_size = action_size
        self._update_tau = update_tau
        self._discount_factor = discount_factor
        self._frames = frames
        self._batch_size = batch_size

        self.actor = actor_nn.cuda()
        self.target_actor = actor_nn.cuda()
        self.critic = critic_nn.cuda()
        self.target_critic = critic_nn.cuda()

        self.critic_loss = torch.nn.MSELoss()

        self.replay_memory = col.deque(maxlen=replay_size)
        self.noise = self._OUnoise(ou_theta, ou_sigma, action_size)

        self._actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=actor_l2)
        self._critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, weight_decay=critic_l2)

        self._copy_network()

    def _copy_network(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    class _OUnoise:
        def __init__(self, theta, sigma, dim, dt=0.1):
            self.theta = theta
            self.sigma = sigma
            self.dim = dim
            self.dt = dt
            self.noise = np.zeros([dim])

        def get_noise(self):
            self.noise += self.theta * - self.noise * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal([self.dim])
            return torch.from_numpy(self.noise.astype(np.float32)).cuda()

    def _smooth_update(self):
        # refer to udacity dqn repo;
        for target, current in zip(self.target_actor.parameters(), self.actor.parameters()):
            target.data.copy_(self._update_tau * current.data + (1 - self._update_tau) * target.data)
        for target, current in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.copy_(self._update_tau * current.data + (1 - self._update_tau) * target.data)

    def get_action(self, state, add_noise=True):
        action = self.actor(torch.tensor(state))
        noise = self.noise.get_noise() * add_noise
        action += noise
        return action

    def append_memory(self, s, a, r, s_next, done):
        for i in range(s.shape[0]):
            self.replay_memory.append([s[i], a[i], r[i], s_next[i], done[i]])

    def get_batch(self, batch_size):
        batch = random.sample(self.replay_memory, batch_size)
        state = np.empty([batch_size, self._state_size])
        next_state = np.empty([batch_size, self._state_size])
        action, reward, done = np.empty([batch_size, self._action_size]), np.empty([batch_size]), np.empty([batch_size])

        for i in range(batch_size):
            state[i] = batch[i][0]
            action[i] = batch[i][1]
            reward[i] = batch[i][2]
            next_state[i] = batch[i][3]
            done[i] = batch[i][4]
        return state.astype(np.float32), action.astype(np.float32), reward.astype(np.float32), next_state.astype(np.float32), done.astype(np.float32)

    def train(self):
        s, a, r, s_n, done = self.get_batch(self._batch_size)
        
        s = torch.tensor(s)
        a = torch.tensor(a)
        r = torch.tensor(r)
        s_n = torch.tensor(s_n)
        done = torch.tensor(done)
        target = r + ((self._discount_factor * self.target_critic(s_n, self.target_actor(s_n))) * (1 - done))

        target = torch.tensor(target)

        # Updace critic network
        self._critic_opt.zero_grad()
        current = torch.tensor(self.critic(s, a), requires_grad=True)
        crit_loss = self.critic_loss(current, target)
        crit_loss.backward()
        self._critic_opt.step()

        # Update actor network
        self._actor_opt.zero_grad()
        actor_loss = -torch.mean(self.critic(s, self.actor(s)))
        actor_loss.backward()
        self._actor_opt.step()
        
        self._smooth_update()

    def save_model(self, path):
        torch.save(self.critic.state_dict, path + "_critic")
        torch.save(self.target_critic.state_dict, path + "_target_critic")
        torch.save(self.actor.state_dict, path + "_actor")
        torch.save(self.target_actor.state_dict, path + "_target_critic")

    def restore_model(self, path):
        self.critic.load_state_dict = torch.load(path + "_critic")
        self.target_critic.load_state_dict = torch.load(path + "_target_critic")
        self.actor.load_state_dict = torch.load(path + "_actor")
        self.target_actor.load_state_dict = torch.load(path + "_target_critic")