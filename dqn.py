import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, action_range, observation_range, lr):
      super(DQN, self).__init__()
      self.action_range = action_range.n
      self.observation_range = observation_range.shape[0]

      self.net = nn.Sequential(
          nn.Linear(self.observation_range, 128),
          nn.ReLU(),
          nn.Linear(128, 32),
          nn.ReLU(),
          nn.Linear(32, self.action_range)
      )

      self.lr = lr

      self.optim = optim.Adam(self.net.parameters(), lr=self.lr)
      self.loss_func = nn.SmoothL1Loss()

    def forward_pass(self, x):
      return self.net(x)