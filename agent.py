import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from memory_buffer import MemoryBuffer
from dqn import DQN

class Agent(object):
    def __init__(self, action_space, observation_space, memory_size, lr=0.001, eps=1, gamma=0.990 ,batch_size=32, update_target_cntr=50, name='MoonLander-policy'):
      self.action_space = action_space.n
      self.observation_space = observation_space.shape[0]
      self.update_target_cntr = update_target_cntr

      self.memory = MemoryBuffer(observation_space, mem_size=memory_size)

      self.policy = DQN(action_space, observation_space, lr=lr)
      self.target = DQN(action_space, observation_space, lr=lr)
      self.target.load_state_dict(self.policy.state_dict())
      self.batch_size = batch_size

      self.eps = eps
      self.gamma = gamma
      self.eps_max = eps
      self.eps_min = 0.01
      self.eps_decay_rate = 0.001

      self.episode = 0
      self.name = name
      self.best_score = float('-inf')

    def store_transition(self, state, next_state, action ,reward, done):
        self.memory.insert(state=state, next_state=next_state, action=action, reward=reward, flag=done)

    def act(self, state):
        if np.random.random() > self.eps:
            state = torch.tensor(state).float()
            q_vals = self.policy.forward_pass(state)
            return torch.argmax(q_vals).item()
        else:
          return np.random.choice(self.action_space)
    
    def eval(self, model_name):
        self.eps = 0
        self.policy.load_state_dict(torch.load(model_name))

    def learn(self):
      batch = self.memory.sample(self.batch_size)
      if batch:
        state_batch, next_state_batch, action_batch, reward_batch, flag_batch = batch


        state_batch = torch.from_numpy(state_batch)
        next_state_batch = torch.from_numpy(next_state_batch)
        reward_batch = torch.from_numpy(reward_batch)
        flag_batch = torch.from_numpy(flag_batch)

        index = np.arange(self.batch_size)

        self.policy.optim.zero_grad()

        q_vals = self.policy.forward_pass(state_batch)[index, action_batch]

        q_vals_next = self.target.forward_pass(next_state_batch)
        q_vals_next = torch.max(q_vals_next, dim=1)[0]

        y = reward_batch + (1 - flag_batch) * self.gamma * q_vals_next

        loss = self.policy.loss_func(q_vals, y)
        loss.backward()
        self.policy.optim.step()



    def update(self, episode, score):
      self.eps = self.eps_min + (self.eps_max - self.eps_min)*np.exp(-episode*self.eps_decay_rate)

      if episode % self.update_target_cntr == 0 and episode != 0:
           self.target.load_state_dict(self.policy.state_dict())

      if self.eps <= 0.15 and score >= self.best_score:
          torch.save(self.policy.state_dict(), self.name)
          self.best_score = score
