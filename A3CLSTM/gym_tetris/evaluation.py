from __future__ import division
import torch
import torch.nn.functional as F
from utils import setup_logger
from model import agentNET
from torch.autograd import Variable
import time
import logging
import random
import copy
from environment import *

def evaluation(args):
    env = Tetris(100)
    reward_sum = 0
    
    model = agentNET()
    model.eval()
    saved_state = torch.load('./tetris.dat')
    model.load_state_dict(model.state_dict())

    for _ in range(1):
        cx = Variable(torch.zeros(1, 78))
        hx = Variable(torch.zeros(1, 78))

        state = env.reset()#50 100 3
        state = torch.from_numpy(state).float()

        while(1):
            value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)),(hx, cx)))

            prob = F.softmax(logit)
            action = prob.max(1)[1].data

            state, done, reward, clean = env.step(action.numpy()[0])
            state = torch.from_numpy(state).float()
            reward_sum += reward
            time.sleep(0.2)

            if done:
                print('reward: ', reward_sum)
                reward_sum = 0
                print(clean['1'])
                break

if __name__ == '__main__':
    evaluation(1)
