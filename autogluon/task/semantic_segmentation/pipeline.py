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
    print(args)
    return args

@autogluon_register_args(
    model=ListSpace(),
    dataset=,
    workers=16,
    base_size=520,
    crop_size=480,
    aux=True,
    aux_weight=LinearSpace(0.2, 0,8)
    epochs=50,
    batch_size=16,
    lr=LogLinearSpace(1e-3, 1e-1),
    momentum=0.9,
    weight_decay=1e-4,
    no_wd=True,
    kvstore='device',
    norm_layer=mx.gluon.contrib.nn.SyncBatchNorm,
    norm_kwargs={'num_devices': args.ngpus},
    )
def train_semantic_segmentation(args, reporter):
    args.ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                   'crop_size': args.crop_size}
    trainset = get_segmentation_dataset(
        args.dataset, split='train', mode='train', **data_kwargs)
    valset = get_segmentation_dataset(
        args.dataset, split='val', mode='val', **data_kwargs)
    train_data = gluon.data.DataLoader(
        trainset, args.batch_size, shuffle=True, last_batch='rollover',
        num_workers=args.workers)
    eval_data = gluon.data.DataLoader(valset, args.batch_size,
        last_batch='rollover', num_workers=args.workers)
    # create network
    if isinstance(args.model, mx.gluon.Block):
        model = args.model
    else:
        assert isinstance(args.model, str)
        model = get_segmentation_model(model=args.model, dataset=args.dataset,
                                       backbone=args.backbone, norm_layer=args.norm_layer,
                                       norm_kwargs=args.norm_kwargs, aux=args.aux,
                                       crop_size=args.crop_size)
    net = DataParallelModel(model, args.ctx)
    evaluator = DataParallelModel(SegEvalModel(model), args.ctx)
    # create criterion
    criterion = MixSoftmaxCrossEntropyLoss(args.aux, aux_weight=args.aux_weight)
    criterion = DataParallelCriterion(criterion, args.ctx)
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
        metric.reset()
        tbar = tqdm(eval_data)
        for i, (data, target) in enumerate(tbar):
            outputs = evaluator(data.astype(args.dtype, copy=False))
            outputs = [x[0] for x in outputs]
            targets = mx.gluon.utils.split_and_load(target, args.ctx, even_split=False)
            metric.update(targets, outputs)
            pixAcc, mIoU = metric.get()
            reporter(epoch=epoch, pixAcc=pixAcc, mIoU=mIoU)
            mx.nd.waitall()

    for epoch in range(args.epochs):
        training(epoch)
        validation(epoch)
