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


# Synthetic Dataset class.
class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length

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

    '''
    is_micro_batch = True
    if args.dataset == 'cifar10':
        train_loader, val_loader = cifar10_dataset_loader(args, is_micro_batch)
    elif args.dataset == 'imagenet':
        train_loader, val_loader = imagenet_data_loader(args, is_micro_batch)
    '''
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.arch == 'inception_v3':
        if args.synthetic_data:
            train_dataset = SyntheticDataset((3, 299, 299), 10000)
        else:
            traindir = os.path.join(args.data_dir, 'train')
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(299),
                    transforms.ToTensor(),
                    normalize,
                ])
            )
    else:
        if args.synthetic_data:
            train_dataset = SyntheticDataset((3, 224, 224), 1000000)
        else:
            traindir = os.path.join(args.data_dir, 'train')
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

    if args.synthetic_data:
        val_dataset = SyntheticDataset((3, 224, 224), 10000)
    else:
        valdir = os.path.join(args.data_dir, 'val')
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    distributed_sampler = False
    train_sampler = None
    val_sampler = None
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
        if num_ranks_in_first_stage > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank)
            distributed_sampler = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)

    ### obtain input_names, output_names, send_names, recv_names
    world_size = 4

    r0 = gpipe_runtime.StageRuntime(model, 0, world_size, inputs_module_destinations, configuration_maps,
                                    target_tensor_names,
                                    model_type=gpipe_runtime.IMAGE_CLASSIFICATION)
    r1 = gpipe_runtime.StageRuntime(model, 1, world_size, inputs_module_destinations, configuration_maps,
                                    target_tensor_names,
                                    model_type=gpipe_runtime.IMAGE_CLASSIFICATION)
    r2 = gpipe_runtime.StageRuntime(model, 2, world_size, inputs_module_destinations, configuration_maps,
                                    target_tensor_names,
                                    model_type=gpipe_runtime.IMAGE_CLASSIFICATION)
    r3 = gpipe_runtime.StageRuntime(model, 3, world_size, inputs_module_destinations, configuration_maps,
                                    target_tensor_names,
                                    model_type=gpipe_runtime.IMAGE_CLASSIFICATION)

    r0.optimizer = get_optimizer(r0.sub_module, args)
    r1.optimizer = get_optimizer(r1.sub_module, args)
    r2.optimizer = get_optimizer(r2.sub_module, args)
    r3.optimizer = get_optimizer(r3.sub_module, args)

    train(train_loader, r0, r1, r2, r3)


def train(train_loader, r0, r1, r2, r3):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    r0.train()
    r1.train()
    r2.train()
    r3.train()

    r0.set_loader(train_loader)
    r1.set_loader(None)
    r2.set_loader(None)
    r3.set_loader(None)
    print("***begin train")

    end = time.time()
    epoch_start_time = time.time()

    for i in range(len(train_loader)):
        r0.run_forward()
        for output_name in r0.send_ranks:
            r1.tensors[output_name] = r0.tensors[output_name]
        r1.run_forward()

        for output_name in r1.send_ranks:
            r2.tensors[output_name] = r1.tensors[output_name]
        r2.run_forward()

        for output_name in r2.send_ranks:
            r3.tensors[output_name] = r2.tensors[output_name]
        for name in r3.target_tensor_names:
            r3.tensors[name] = r0.tensors[name]
        r3.run_forward()

        if 3 == 3:
            # measure accuracy and record loss
            output, target, loss = r3.output, r3.tensors["target"], r3.loss
            # print("this is proc3", output.shape, target.shape, loss)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), output.size(0))
            top1.update(prec1[0], output.size(0))
            top5.update(prec5[0], output.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            epoch_time = (end - epoch_start_time) / 3600.0
            full_epoch_time = (epoch_time / float(i + 1)) * float(len(train_loader))

            epoch = 1

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'
                      'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    epoch_time=epoch_time, full_epoch_time=full_epoch_time,
                    loss=losses, top1=top1, top5=top5,
                    memory=(float(torch.cuda.memory_allocated()) / 10 ** 9),
                    cached_memory=(float(torch.cuda.memory_cached()) / 10 ** 9)))
                import sys;
                sys.stdout.flush()
        else:
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\tMemory: {memory:.3f} ({cached_memory:.3f})'.format(
                    epoch, i, n, memory=(float(torch.cuda.memory_allocated()) / 10 ** 9),
                    cached_memory=(float(torch.cuda.memory_cached()) / 10 ** 9)))
                import sys;
                sys.stdout.flush()

        ## perform backward
        r0.optimizer.zero_grad()
        r1.optimizer.zero_grad()
        r2.optimizer.zero_grad()
        r3.optimizer.zero_grad()

        r3.run_backward(r3.criterion_input_names, r3.criterion_output_names)

        r0.optimizer.step()
        r1.optimizer.step()
        r2.optimizer.step()
        r3.optimizer.step()
        # r3.run_backward(r3.input_names, r3.output_names)
        # break














if __name__ == '__main__':
    main()
