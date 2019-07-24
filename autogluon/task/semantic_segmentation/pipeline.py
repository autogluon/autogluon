
import argparse
import numpy as np
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.loss import *
from gluoncv.utils import LRScheduler
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.utils.parallel import *
from gluoncv.data import get_segmentation_dataset

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')
    # model and dataset 
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone name (default: resnet50)')
    parser.add_argument('--dataset', type=str, default='pascal_aug',
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=16,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default= False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.5,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, \
                        and beta/gamma for batchnorm layers.')
    # cuda and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs (default: 4)')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='default',
                        help='set the checkpoint name')
    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default= False,
                        help='using Synchronized Cross-GPU BatchNorm')
    # the parser
    args = parser.parse_args()
    # handle contexts
    if args.no_cuda:
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        print('Number of GPUs:', args.ngpus)
        args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    print(args)
    return args

def train_segmentation(args, reporter):
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                   'crop_size': args.crop_size}
    trainset = get_segmentation_dataset(
        args.dataset, split=args.train_split, mode='train', **data_kwargs)
    valset = get_segmentation_dataset(
        args.dataset, split='val', mode='val', **data_kwargs)
    train_data = gluon.data.DataLoader(
        trainset, args.batch_size, shuffle=True, last_batch='rollover',
        num_workers=args.workers)
    eval_data = gluon.data.DataLoader(valset, args.test_batch_size,
        last_batch='rollover', num_workers=args.workers)
    # create network
    #model = get_model(args.model_zoo, pretrained=True)
    model = args.model
    model.cast(args.dtype)
    print(model)
    net = DataParallelModel(model, args.ctx, args.syncbn)
    evaluator = DataParallelModel(SegEvalModel(model), args.ctx)
    # resume checkpoint if needed
    if args.resume is not None:
        if os.path.isfile(args.resume):
            model.load_parameters(args.resume, ctx=args.ctx)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'" \
                .format(args.resume))
    # create criterion
    criterion = MixSoftmaxCrossEntropyLoss(args.aux, aux_weight=args.aux_weight)
    criterion = DataParallelCriterion(criterion, args.ctx, args.syncbn)
    # optimizer and lr scheduling
    lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr,
                               nepochs=args.epochs,
                               iters_per_epoch=len(train_data),
                               power=0.9)
    kv = mx.kv.create(args.kvstore)
    optimizer_params = {'lr_scheduler': lr_scheduler,
                        'wd':args.weight_decay,
                        'momentum': args.momentum,
                        'learning_rate': args.lr
                       }
    if args.dtype == 'float16':
        optimizer_params['multi_precision'] = True

    if args.no_wd:
        for k, v in net.module.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    optimizer = gluon.Trainer(net.module.collect_params(), 'sgd',
                                   optimizer_params, kvstore = kv)
    # evaluation metrics
    metric = gluoncv.utils.metrics.SegmentationMetric(trainset.num_class)

    def training(epoch):
        tbar = tqdm(train_data)
        train_loss = 0.0
        alpha = 0.2
        for i, (data, target) in enumerate(tbar):
            with autograd.record(True):
                outputs = net(data.astype(args.dtype, copy=False))
                losses = criterion(outputs, target)
                mx.nd.waitall()
                autograd.backward(losses)
            optimizer.step(args.batch_size)
            for loss in losses:
                train_loss += np.mean(loss.asnumpy()) / len(losses)
            mx.nd.waitall()

    def validation(epoch):
        #total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        metric.reset()
        tbar = tqdm(eval_data)
        for i, (data, target) in enumerate(tbar):
            outputs = evaluator(data.astype(args.dtype, copy=False))
            outputs = [x[0] for x in outputs]
            targets = mx.gluon.utils.split_and_load(target, args.ctx, even_split=False)
            metric.update(targets, outputs)
            pixAcc, mIoU = metric.get()
            #tbar.set_description('Epoch %d, validation pixAcc: %.3f, mIoU: %.3f'%\
            #    (epoch, pixAcc, mIoU))
            reporter(epoch=epoch, pixAcc=pixAcc, mIoU=mIoU)
            mx.nd.waitall()

    for epoch in range(args.start_epoch, args.epochs):
        training(epoch)
        validation(epoch)
