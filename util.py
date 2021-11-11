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

def get_optimizer(module, args):
    optimizer = None
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(module.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSProp':
        optimizer = optim.RMSprop(module.parameters(), lr=args.lr, alpha=0.9,
                                        eps=1e-8, weight_decay=0, momentum=0, centered=False)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(module.parameters(), lr=args.lr,
                                     betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'AdaGrad':
        optimizer = torch.optim.Adagrad(module.parameters(), lr=args.lr)

    return optimizer

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

