from __future__ import division
import torch
import torch.nn.functional as F
from utils import setup_logger, setup_logger_add, reverse
from model import agentNET
from torch.autograd import Variable
import time
import logging
import random
import numpy as np
import sys
from environment import *
import copy

def test(args, shared_model):
    log = {}
    logger = setup_logger("test_log",  "./logs/test_log")

    torch.manual_seed(args.seed)
    env = Tetris(50)

    model = agentNET()
    model.eval()

    test_time = 0
    reward_num = 0
    clean_sum = 0
    max_reward = -1

    while(1):
        model.load_state_dict(shared_model.state_dict())
        if args.gpu:
            model = model.cuda()
            cx = Variable(torch.zeros(1, 78).cuda())
            hx = Variable(torch.zeros(1, 78).cuda())
        else:
            cx = Variable(torch.zeros(1, 78))
            hx = Variable(torch.zeros(1, 78))

        state = env.reset()#50 100 3
        state = torch.from_numpy(state).float()

        while(1):
            if args.gpu:
                value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)).cuda(), (hx, cx)))
            else:
                value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)),(hx, cx)))

            prob = F.softmax(logit)
            if args.gpu:
                action = prob.max(1)[1].data.cpu()
            else:
                action = prob.max(1)[1].data

            state, done, reward, clean = env.step(action.numpy()[0])
            state = torch.from_numpy(state).float()
            reward_num += reward
            clean_sum += clean.get('1', -1000)

            if done:
                #print('dead', test_time)
                test_time += 1
                break

        if test_time % 50 == 0:
            if reward_num > max_reward:
                if args.gpu:
                    model = model.cpu()
                state_to_save = model.state_dict()
                torch.save(state_to_save, "./tetris.dat")
                logger.info('save')
                max_reward = reward_num
                if args.gpu:
                    model = model.cuda()
            logger.info('reward = ' + str(reward_num / test_time))
            logger.info('cleaned = ' + str(clean_sum / test_time))
            test_time = 0
            reward_num = 0
            clean_sum = 0