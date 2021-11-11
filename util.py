'''''''''''''''''''''
// Implementation of XPipe based on the framework of Pytorch
// Author: Lei Guan
// Email: guanleics@gmail.com
// Time: Jan.25, 2019
// Copyright 2019 All rights reserved.
// Redistribution and use the source code for commercial purpose are forbidden.
// Redistribution of the source code must be under permission of the author.
// Redistribution of the source code must retain the above copyright notice.
'''''''''''''''''''''
import time
import torch
import torch.optim as optim
import torch.distributed as dist
import random

def set_optimizer(model, args):
    if args.optimizer == 'SGD':
        model.optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSProp':
        model.optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9,
                                        eps=1e-8, weight_decay=0, momentum=0, centered=False)
    elif args.optimizer == 'Adam':
        model.optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                     betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'AdaGrad':
        model.optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)

