from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import sys
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Lock, Value
from model import agentNET
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
from evaluation import evaluation
import time
import copy
from utils import setup_logger

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--train',
    default=True,
    metavar='L',
    help='train model')
parser.add_argument(
    '--gpu',
    default=True,
    metavar='L',
    help='gpu model')
parser.add_argument(
    '--agentload',
    default=False,
    metavar='L',
    help='load trained agent models')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE')
parser.add_argument(
    '--seed',
    type=int,
    default=233,
    metavar='S',
    help='random seed')
parser.add_argument(
    '--workers',
    type=int,
    default=10,
    metavar='W',
    help='how many training processes to use')

if __name__ == '__main__':
    args = parser.parse_args()
    mp.set_start_method('spawn')
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(args.seed)

    if args.train:
        shared_model = agentNET()
        if args.agentload:
           saved_state = torch.load("./tetris.dat")
           shared_model.load_state_dict(saved_state)
        shared_model.share_memory()
        optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

        processes = []
        p = Process(target=test, args=(args, shared_model))
        p.start()
        processes.append(p)
        time.sleep(0.1)
        for rank in range(0, args.workers):
            p = Process(target=train, args=(rank, args, shared_model, optimizer))
            p.start()
            processes.append(p)
            time.sleep(0.1)
        for rank in range(0, args.workers):
            processes[rank].join()
    else:
        evaluation(args)
