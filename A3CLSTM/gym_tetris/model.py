from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init

class agentNET(torch.nn.Module):
    def __init__(self, num_inputs = 1, num_outputs = 5):
        super(agentNET, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 16, 4, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 8, 3, stride=1, padding=0)

        self.lstm = nn.LSTMCell(312, 78)

        self.fc1 = nn.Linear(78, 20)

        self.critic_linear = nn.Linear(20, 1)
        self.actor_linear = nn.Linear(20, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.fc1.weight.data = norm_col_init(
            self.fc1.weight.data, 1.0)
        self.fc1.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))

        x = F.relu(self.fc1(hx))

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
