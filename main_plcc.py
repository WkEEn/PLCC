#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import os
import random
import time
import warnings

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

torch.autograd.set_detect_anomaly(True)

# from loader import *
import plcc.builder
import plcc.queue
from loader import *
from utils import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')

# moco specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=256, type=int,
                    help='hidden dimension of the predictor (default: 256)')
parser.add_argument('--num-classes', default=3, type=int,
                    help='number of classes (default: 5)')

# cdc specific configs:
parser.add_argument('--max-size', default=2048, type=int,
                    help='max size of each cluster queue (default: 2048)')
parser.add_argument('--threshold-neg', default=0.8, type=float,
                    help='threshold for negative sample updating')
parser.add_argument('--threshold-pos', default=0.8, type=float,
                    help='threshold for positive sample updating')
parser.add_argument('--temperature', default=0.05, type=float,
                    help='name list of classes')
parser.add_argument('--resume-cluster', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint of cluster queue (default: none)')


parser.add_argument('--topk', default=10,
                    help='Path where save the model checkpoint')
parser.add_argument('--dataset', default='egfr',
                    help='Path where save the model checkpoint')
parser.add_argument('--checkpoint', default='./egfr',
                    help='Path where save the model checkpoint')
parser.add_argument('--checkpoint-model', default='checkpoint-model',
                    help='Path where save the confusion matrix')
parser.add_argument('--checkpoint-cluster', default='checkpoint-cluster',
                    help='Path where save the confusion matrix')


def main():
    args = parser.parse_args()

    # base_dir_path = 'neg{}-pos{}'.format(args.threshold_neg, args.threshold_pos)
    # args.checkpoint = os.path.join(base_dir_path, args.checkpoint)
    args.checkpoint_model = os.path.join(args.checkpoint, args.checkpoint_model)
    args.checkpoint_cluster = os.path.join(args.checkpoint, args.checkpoint_cluster)
    args.checkpoint_plot = os.path.join(args.checkpoint, 'checkpoint-plot')

    if args.checkpoint_model:
        os.makedirs(args.checkpoint_model, exist_ok=True)
    if args.checkpoint_cluster:
        os.makedirs(args.checkpoint_cluster, exist_ok=True)
    if args.checkpoint_plot:
        os.makedirs(args.checkpoint_plot, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # slurmd settings
    args.rank = int(os.environ["SLURM_PROCID"])
    args.world_size = int(os.environ["SLURM_NPROCS"])

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # suppress printing if not master
    if args.multiprocessing_distributed and args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = plcc.builder.PLCC(
        models.__dict__[args.arch],
        args.num_classes, args.dim, args.pred_dim)


    print("=> creating cluster queue")
    queue = plcc.queue.ClusterQueue(
        args.num_classes, args.max_size, args.pred_dim,
        args.threshold_neg, args.threshold_pos, tmp=args.temperature)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.pretrained, map_location=loc)
            state_dict = checkpoint['state_dict']

            # rename moco pre-trained keys
            for k in list(state_dict.keys()):
                if k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            print("=> missing_keys\n", msg.missing_keys)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            queue.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            queue.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        queue.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            queue.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            queue.cuda()
    print(model)

    # define loss function (criterion) and optimizer
    criterion1 = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion2 = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_file = os.path.join(args.data, 'train.txt')
    valid_file = os.path.join(args.data, 'val.txt')

    train_dataset = CategoryDataset(
        train_file, args.dataset, args.num_classes,
        transforms.Compose([
            transforms.ToTensor(),
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # initializing cluster queue
    print("=> staring initializing cluster")
    if args.resume_cluster and os.path.exists(args.resume_cluster):
        if args.gpu is None:
            queue.load_cluster(args.resume_cluster)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            queue.load_cluster(args.resume_cluster, loc)
    else:
        print("=> no cluster checkpoint found in {}".format(args.resume_cluster))
        init_cluster(train_loader, model, queue, args)
        queue.save_cluster('{}/checkpoint_init.pth.tar'.format(args.checkpoint_cluster))
    if not os.path.exists(os.path.join(args.checkpoint_plot, 'epoch-init')):
        queue.visualize_cluster(train_dataset.images, 'init', os.path.join(args.checkpoint_plot, 'epoch-init'))

    print("=> staring training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, queue, criterion1, criterion2, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.checkpoint_model, epoch))

            queue.save_cluster('{}/checkpoint_{:04d}.pth.tar'.format(args.checkpoint_cluster, epoch))
            queue.visualize_cluster(train_dataset.images, epoch, os.path.join(args.checkpoint_plot, 'epoch{}'.format(epoch)))

def init_cluster(data_loader, model, queue, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time],
        prefix="Initializing cluster queue")

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, labels, _, inds) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)
                inds = inds.cuda(args.gpu, non_blocking=True)

            _, _, z = model(images)
            queue.init_cluster(z, labels, inds)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    print('Finish initializing cluster queue')


def train(train_loader, model, queue, criterion1, criterion2, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    losses_cl = AverageMeter('Loss_CL', ':.4f')
    losses_ce = AverageMeter('Loss_CE', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.train()
    queue.reset_update_info()

    end = time.time()
    for i, (images, labels, masks, inds) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
            masks = masks.cuda(args.gpu, non_blocking=True)
            inds = inds.cuda(args.gpu, non_blocking=True)

        # compute output
        output, p, z = model(images)
        # output = queue.get_soft_label(p)
        cluster_labels = queue.update(z, labels, masks, inds)

        loss_ce = criterion1(output, cluster_labels)
        loss_cl = 1 - criterion2(p, z).mean()
        print(loss_ce, loss_cl)
        loss = loss_ce + loss_cl * 0.1

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))
        losses_cl.update(loss_cl.item(), images.size(0))
        losses_ce.update(loss_ce.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            queue.print_update_info()


if __name__ == '__main__':
    main()
