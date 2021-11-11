# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from collections import OrderedDict
import importlib
import json
import os
import shutil
import sys
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.append("..")
from util import *
import gpipe_runtime
from data_loader import imagenet_data_loader, cifar10_dataset_loader

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', type=str,
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='cifar10', metavar='DATA',
                    help='dataset used')
parser.add_argument('--distributed_backend', type=str,
                    help='distributed backend to use (gloo|nccl)')
parser.add_argument('--module', '-m', required=True,
                    help='name of module that contains model and tensor_shapes definition')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--eval-batch-size', default=100, type=int,
                    help='eval mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_policy', default='step', type=str,
                    help='policy for controlling learning rate')
parser.add_argument('--lr_warmup', action='store_true',
                    help='Warmup learning rate first 5 epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--fp16', action='store_true',
                    help='train model in fp16 precision')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='static loss scale, positive power of 2 to improve fp16 convergence')
parser.add_argument('--master_addr', default=None, type=str,
                    help="IP address of master (machine with rank 0)")
parser.add_argument('--config_path', default=None, type=str,
                    help="Path of configuration file")
parser.add_argument('--no_input_pipelining', action='store_true',
                    help="No pipelining of inputs")
parser.add_argument('--rank', default=None, type=int,
                    help="Rank of worker")
parser.add_argument('--local_rank', default=0, type=int,
                    help="Local rank of worker")
parser.add_argument('--forward_only', action='store_true',
                    help="Run forward pass only")
parser.add_argument('--num_minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                    help='path to directory to save checkpoints')
parser.add_argument('--checkpoint_dir_not_nfs', action='store_true',
                    help='checkpoint dir is not on a shared NFS server')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                    help="Log verbose information")
parser.add_argument('--num_ranks_in_server', default=1, type=int,
                    help="number of gpus per machine")
# Recompute tensors from forward pass, instead of saving them.
parser.add_argument('--recompute', action='store_true',
                    help='Recompute tensors in backward pass')
# Macrobatching reduces the number of weight versions to save,
# by not applying updates every minibatch.
parser.add_argument('--macrobatch', action='store_true',
                    help='Macrobatch updates to save memory')

parser.add_argument('--partitions', type=int, default=4, metavar='P',
                    help='number of partitions (default: 4)')
parser.add_argument('--optimizer', type=str, default='SGD', metavar='O',
                    help='optimizer method (default: SGD)')
parser.add_argument('--log-interval', type=int, default=200, metavar='L',
                    help='how many batches to wait before logging training status')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    assert args.batch_size % args.partitions == 0
    args.micro_batch_size = int(args.batch_size / args.partitions)

    torch.cuda.set_device(0)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # create stages of the model
    module = importlib.import_module(args.module)
    args.arch = module.arch()
    model = module.model(criterion)

    # determine shapes of all tensors in passed-in model
    if args.arch == 'inception_v3':
        input_size = [args.batch_size, 3, 299, 299]
    else:
        input_size = [args.batch_size, 3, 224, 224]
    training_tensor_shapes = {"input0": input_size, "target": [args.batch_size]}
    dtypes = {"input0": torch.int64, "target": torch.int64}
    inputs_module_destinations = {"input": 0}
    target_tensor_names = {"target"}
    for (stage, inputs, outputs) in model[:-1]:  # Skip last layer (loss).
        input_tensors = []
        for input in inputs:
            input_tensor = torch.zeros(tuple(training_tensor_shapes[input]),
                                       dtype=torch.float32)
            input_tensors.append(input_tensor)
        with torch.no_grad():
            output_tensors = stage(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            training_tensor_shapes[output] = list(output_tensor.size())
            dtypes[output] = output_tensor.dtype

    # get the tensor shapes for evaluation
    eval_tensor_shapes = {}
    for key in training_tensor_shapes:
        eval_tensor_shapes[key] = tuple(
            [args.eval_batch_size] + training_tensor_shapes[key][1:])
        training_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])

    #### 
    print("training_tensor_shapes", training_tensor_shapes)
    #training_tensor_shapes
    ## #{'input0': (64, 3, 224, 224), 'target': (64,), 'out0': (64, 64, 224, 224),
    # 'out1': (64, 256, 56, 56), 'out2': (64, 512, 28, 28), 'out3': (64, 1000)}

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)

    is_micro_batch = True
    if args.dataset == 'cifar10':
        train_loader, val_loader = cifar10_dataset_loader(args, is_micro_batch)
    elif args.dataset == 'imagenet':
        train_loader, val_loader = imagenet_data_loader(args, is_micro_batch)

    ### obtain input_names, output_names, send_names, recv_names

    r0 = gpipe_runtime.StageRuntime(model, 0, inputs_module_destinations, configuration_maps,
                                    model_type=gpipe_runtime.IMAGE_CLASSIFICATION)
    r1 = gpipe_runtime.StageRuntime(model, 1, inputs_module_destinations, configuration_maps,
                                    model_type=gpipe_runtime.IMAGE_CLASSIFICATION)
    r2 = gpipe_runtime.StageRuntime(model, 2, inputs_module_destinations, configuration_maps,
                                    model_type=gpipe_runtime.IMAGE_CLASSIFICATION)
    r3 = gpipe_runtime.StageRuntime(model, 3, inputs_module_destinations, configuration_maps,
                                    model_type=gpipe_runtime.IMAGE_CLASSIFICATION)

    set_optimizer(r0.sub_module, args)
    set_optimizer(r1.sub_module, args)
    set_optimizer(r2.sub_module, args)
    set_optimizer(r3.sub_module, args)











if __name__ == '__main__':
    main()
