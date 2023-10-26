#!/usr/bin/env python
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
# import torch.optim
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from transformers import AutoTokenizer
from utils import save_checkpoint, AverageMeter, ProgressMeter, MultiVarMeter, accuracy, concat_all_gather, \
    concat_output
from model import EMMA, EMMA_CTL, EMMA_GMM
from dataset import LanguageDataset, LanguagePairDataset, LineCacheDataset
from criterion import EMContrastive


parser = argparse.ArgumentParser(description='PyTorch EMMA-X Cross-lingual Sentence Representation Learning')
parser.add_argument('--data', default='data_small/', metavar='DIR',
                    help='path to dataset')
# parser.add_argument('--lmdb-data', default='data_lmdb/', metavar='DIR',
#                     help='path to LMDB dataset')
parser.add_argument('--pretrained-model-path',
                    default='pretrained_model/xlm-roberta-base',
                    metavar='DIR', help='path to pretrained model')

parser.add_argument('-a', '--arch', metavar='ARCH', default='emma-ctl',
                    choices=['emma-ctl', 'emma-gmm', 'emma-itr'], help='network architecture')
parser.add_argument('--ctl-ckpt', default='emma-ctl-01.pth.tar', type=str, metavar='PATH',
                    help='path to the latest checkpoint from the ctl step (default: none)')
parser.add_argument('--gmm-ckpt', default='emma-gmm-01.pth.tar', type=str, metavar='PATH',
                    help='path to the latest checkpoint from the gmm step (default: none)')

parser.add_argument('--use-parallel-data', action='store_true', default=True,
                    help='Whether to use parallel data, must set to True in the first two steps.')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--do-mlm', action='store_true', default=True,
                    help='Whether to use MLM pretraining ')
parser.add_argument('-b', '--batch-size', default=6, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--K', default=768, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--lr', '--learning-rate', default=4e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
# emma specific configs
parser.add_argument('--m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--low-t', default=0.05, type=float,
                    help='temperature for the highest rank (default: 0.05)')
parser.add_argument('--update-mean', default=0.001, type=float,
                    help='momentum of updating rank mean (default: 0.001)')
parser.add_argument('--high-t', default=0.1, type=float,
                    help='temperature for the lowest rank (default: 0.1)')
parser.add_argument('--rank-num', default=4, type=int,
                    help='how many ranks we want to distinguish (default: 4)')
parser.add_argument('--warm-up', default=8000, type=int,
                    help='warm up rate (default: 8000)')
parser.add_argument('--label-smoothing', default=0.1, type=float,
                    help='label smoothing (default: 0.1)')
parser.add_argument('--rank-confidence', default=0.3, type=float,
                    help='rank confidence (default: 0.3)')
parser.add_argument('--mask-probability', default=0.1, type=float,
                    help='mask probability (default: 0.3)')
parser.add_argument('--loss-balance', default=1, type=float,
                    help='loss balance (default: 0.5)')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--adam-eps', default=1e-6, type=float, metavar='M',
                    help='momentum of SGD solver')

parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=True,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    args = parser.parse_args()
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        # warnings.warn('You have chosen to seed training. '
        #               'This will turn on the CUDNN deterministic setting, '
        #               'which can slow down your training considerably! '
        #               'You may see unexpected behavior when restarting '
        #               'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

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

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

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
    # create model
    if args.arch == 'emma-ctl':
        model = EMMA_CTL(args)
    elif args.arch == 'emma-gmm':
        model = EMMA_GMM(args)
    elif args.arch == 'emma-itr':
        model = EMMA(args)
    else:
        raise RuntimeError("No Model with Name {:s}".format(args.arch))
    print("=> creating model '{}'".format(args.arch))
    # print(model)
    # print(model.mu.requires_grad)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


    cudnn.benchmark = True

    # Data loading code
    train_dataset = LanguagePairDataset(args)
    # train_dataset = LMDBDataset(args)
    # train_dataset = LineCacheDataset(args)
    criterion = EMContrastive(args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    batch_size = args.batch_size // 2

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), collate_fn=train_dataset.collate_fn,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up,
                                                num_training_steps=args.epochs * len(train_loader))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    # For second and third step, initilized parameters from previous steps.
    if args.arch == 'emma-gmm':
        if os.path.isfile(args.ctl_ckpt):
            print("=> loading checkpoint '{}'".format(args.ctl_ckpt))
            checkpoint = torch.load(args.ctl_ckpt, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
            if len(missing_keys) > 0:
                print('Missing keys in state_dict: {}'.format(', '.join('"{}"'.format(k) for k in missing_keys)))
            if len(unexpected_keys) > 0:
                print('Unexpected keys in state_dict: {}'.format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
        else:
            print("=> no checkpoint found at '{}'".format(args.ctl_ckpt))
        # for n, p in model.named_parameters():
        #     print(n, p.requires_grad)

    elif args.arch == 'emma-itr':
        if os.path.isfile(args.ctl_ckpt) and os.path.isfile(args.gmm_ckpt):
            print("=> loading checkpoint '{}' and '{}'".format(args.ctl_ckpt, args.gmm_ckpt))
            ctl_checkpoint = torch.load(args.ctl_ckpt, map_location='cpu')['state_dict']
            gmm_checkpoint = torch.load(args.gmm_ckpt, map_location='cpu')['state_dict']
            # for n, p in ctl_checkpoint.items():
            #     if "encoder_q" in n: 
            #         if "lm_head" not in n:
            #             assert (p == gmm_checkpoint[n]).all()
            #     if "encoder_k" in n:
            #         if "lm_head" not in n:
            #             assert (p == gmm_checkpoint[n]).all()
            ctl_checkpoint.update(gmm_checkpoint)
            # for n, p in ctl_checkpoint.items():
            #     if "module.mu" in n:
            #         assert (p == gmm_checkpoint[n]).all()
            #     if "module.lg_sigma2" in n:
            #         assert (p == gmm_checkpoint[n]).all()
            #     if "module.pi" in n:
            #         assert (p == gmm_checkpoint[n]).all()
            missing_keys, unexpected_keys = model.load_state_dict(ctl_checkpoint, strict=False)
            if len(missing_keys) > 0:
                print('Missing keys in state_dict: {}'.format(', '.join('"{}"'.format(k) for k in missing_keys)))
            if len(unexpected_keys) > 0:
                print('Unexpected keys in state_dict: {}'.format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
        else:
            print("=> no checkpoint found at '{} or {}'".format(args.ctl_ckpt, args.gmm_ckpt))
        # for n, p in model.named_parameters():
        #     print(n, p.requires_grad)


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        start = time.time()
        train(train_loader, model, criterion, optimizer, epoch, args, scheduler)
        end = time.time()
        print("Epoch: ", epoch)
        print("Cost time: ", time.strftime("%H:%M:%S", time.gmtime(end - start)))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best=False, filename='{:s}-{:02d}.pth.tar'.format(args.arch, epoch))


def train(train_loader, model, criterion, optimizer, epoch, args, scheduler):
    losses = AverageMeter('ALL:', ':.4f')
    mlm_losses = AverageMeter('MLM:', ':.4f')
    gmm_losses = AverageMeter('GMM:', ':.4f')
    cont_losses = AverageMeter('CTL:', ':.4f')
    gmm_rankes = MultiVarMeter('GMM-Rank:', args.rank_num, "int")
    cts_rankes = MultiVarMeter('CTL-Rank:', args.rank_num, "int")
    erm = MultiVarMeter("ERM:", args.rank_num, "float")
    ctl_dist = MultiVarMeter("CTL-Dist:", args.rank_num, "int")
    gmm_pi = MultiVarMeter("GMM-Pi:", args.rank_num, "float")
    acc1 = AverageMeter("CTL-A@1:", ":.4f")
    acc5 = AverageMeter("CTL-A@5:", ":.4f")
    gmm_top1 = AverageMeter('GMM-A@1:', ':.4f')
    consistency = AverageMeter('CST:', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        # [losses, gmm_losses, cont_losses, mlm_losses, gmm_rankes, gmm_top1, cts_rankes, consistency, acc1, acc5, erm, ctl_dist, gmm_pi],
        [losses, cont_losses, ctl_dist, cts_rankes, acc1, acc5, gmm_losses, gmm_rankes, gmm_top1, gmm_pi, consistency,
         mlm_losses, erm],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    with torch.autograd.set_detect_anomaly(True):
        model.train()

        # end = time.time()
        for i, inputs in enumerate(train_loader):
            # measure data loading time
            # data_time.update(time.time() - end)

            # if args.gpu is not None:
            #     images[0] = images[0].cuda(args.gpu, non_blocking=True)
            #     images[1] = images[1].cuda(args.gpu, non_blocking=True)

            # # compute output
            is_pair = args.use_parallel_data
            # inpl = True
            # do_gmm = False
            # do_ctl = True
            # if epoch >= args.inpl_end_epoch:
            #     inpl = False
            # if epoch >= args.gmm_start_epoch:
            #     do_gmm = True
            # if args.inpl_end_epoch > epoch >= args.gmm_start_epoch:
            #     do_ctl = False

            logits = model(inputs["inputs"], is_pair=is_pair)
            loss = criterion(logits, mlm_labels=inputs["labels"])

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss["total"].backward()
            optimizer.step()
            scheduler.step()

            cat_loss = {k: concat_all_gather(v.unsqueeze(0)).mean() for k, v in loss.items()}
            losses.update(loss["total"].item())
            if cat_loss.get("mlm", None) is not None:
                mlm_losses.update(loss["mlm"].item())
            if cat_loss.get("gmm", None) is not None:
                gmm_losses.update(loss["gmm"].item())
            if cat_loss.get("cts", None) is not None:
                cont_losses.update(loss["cts"].item())

            if logits.get("acc", None) is not None:
                acc = concat_output(logits["acc"])
                top1, top5 = accuracy(acc["pred"], acc["label"], topk=(1, 5))
                acc1.update(top1[0], acc["label"].size(0))
                acc5.update(top5[0], acc["label"].size(0))

            if logits.get("gmm", None) is not None:
                gmm = concat_output(logits["gmm"])

                gmm_acc = gmm["pred"].eq(gmm["label"]).sum() / float(gmm["pred"].numel())
                gmm_top1.update(gmm_acc.item())

                if logits.get("cts", None) is not None:
                    cts = concat_output(logits["cts"])
                    consist = gmm["pred"].eq(cts["pred"]).sum() / float(gmm["pred"].numel())
                    consistency.update(consist.item())

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            if i % args.print_freq == 0:
                if logits.get("cts", None) is not None:
                    cts = concat_output(logits["cts"])
                    cts_pred_rank = []
                    cts_dist = []
                    for j in range(args.rank_num):
                        cts_pred_rank.append(cts["pred"].eq(j).sum().item())
                        cts_dist.append(cts["dist"].eq(j).sum().item())
                    cts_rankes.update(cts_pred_rank)
                    ctl_dist.update(cts_dist)

                if logits.get("gmm", None) is not None:
                    gmm_pred_rank = []
                    for j in range(args.rank_num):
                        gmm_pred_rank.append(gmm["pred"].eq(j).sum().item())
                    gmm_rankes.update(gmm_pred_rank)

                erm.update(model.module.rank_mean)
                gmm_pi.update(model.module.pi)

                progress.display(i)


if __name__ == '__main__':
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))