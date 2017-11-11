from __future__ import division
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import ensure_shared_grads, setup_logger, setup_logger_add, reverse
from model import agentNET
from torch.autograd import Variable
from shared_optim import SharedRMSprop, SharedAdam
import torch.nn as nn
import time
import random
import numpy as np
import torch.nn as nn
import copy
from utils import setup_logger, setup_logger_add
from environment import *

def train(rank, args, shared_model, optimizer):
    torch.manual_seed(args.seed + rank)
    #logger = setup_logger("worker_%d_log" % rank,  "/home/shin/SFG-IRL-map-master/logs/worker_%d_log" % rank)

    env = Tetris(rank)
    model = agentNET()
    model.train()

    done = True
    episode_length = 0
    uploadtime = 0

    while True:
        model.load_state_dict(shared_model.state_dict())

        if args.gpu:
            model = model.cuda()
            cx = Variable(torch.zeros(1, 78).cuda())
            hx = Variable(torch.zeros(1,78).cuda())
        else:
            cx = Variable(torch.zeros(1, 78))
            hx = Variable(torch.zeros(1, 78))

        state = env.reset()
        state = torch.from_numpy(state).float()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        while(1):
            if args.gpu:
                value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)).cuda(), (hx, cx)))
            else:
                value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)),(hx, cx)))

            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            if args.gpu:
                action = prob.multinomial().data.cpu()
                action.view(-1,1)
                log_prob = log_prob.gather(1, Variable(action.cuda()))
            else:
                action = prob.multinomial().data
                action.view(-1,1)
                log_prob = log_prob.gather(1, Variable(action))

            state, done, reward, _ = env.step(action.numpy()[0][0])
            state = torch.from_numpy(state).float()

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            if args.gpu:
                value, _, _ = model((Variable(state.unsqueeze(0).cuda()), (hx, cx)))
            else:
                value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        if args.gpu:
            values.append(Variable(R.cuda()))
            R = Variable(R.cuda())
        else:
            values.append(Variable(R))
            R = Variable(R)

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            if args.gpu:
                delta_t = rewards[i] + args.gamma * \
                    values[i + 1].data.cpu() - values[i].data.cpu()
            else:
                delta_t = rewards[i] + args.gamma * \
                    values[i + 1].data - values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            if args.gpu:
                policy_loss = policy_loss - \
                    log_probs[i] * Variable(gae.cuda()) - 0.01 * entropies[i]
            else:
                policy_loss = policy_loss - \
                    log_probs[i] * Variable(gae) - 0.01 * entropies[i]


        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()

        if args.gpu:
            model = model.cpu()

        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        optimizer.step()
        
        uploadtime += 1
        if uploadtime % 1000 == 0 and rank == 1:
            print('---> after {0} steps <---'.format(uploadtime * args.workers))