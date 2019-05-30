import os
import numpy as np

import argparse, time, logging
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler
from gluoncv.data import transforms as gcv_transforms

import autogluon as ag
from autogluon import autogluon_method

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='number of gpus to use.')
    parser.add_argument('--model', type=str, default='cifar_resnet20_v1',
                        help='model to use. options are resnet and wrn. default is cifar_resnet20_v1.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--epochs', default=20, type=int,
                        help='number of epochs')
    parser.add_argument('--num-trials', default=10, type=int,
                        help='number of trail tasks')
    parser.add_argument('--scheduler', type=str, default='fifo',
                        help='scheduler name (default: fifo)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/cifar1.ag',
                        help='checkpoint path (default: None)')
    parser.add_argument('--resume', action='store_true', default= False,
                        help='resume from the checkpoint if needed')
    parser.add_argument('--debug', action='store_true', default= False,
                        help='debug if needed')
    args = parser.parse_args()
    return args


@autogluon_method
def train_cifar(args, reporter):
    batch_size = args.batch_size

    num_gpus = args.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = args.num_workers

    model_name = args.model
    net = get_model(model_name, classes=10)

    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    def test(ctx, val_data):
        metric = mx.metric.Accuracy()
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
        return metric.get()

    def train(epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        net.initialize(mx.init.Xavier(), ctx=ctx)

        train_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        lr_scheduler = LRScheduler(mode='cosine', base_lr=args.lr,
                                   nepochs=args.epochs,
                                   iters_per_epoch=len(train_data))
        trainer = gluon.Trainer(net.collect_params(), 'sgd',
                                {'lr_scheduler': lr_scheduler, 'wd': args.wd, 'momentum': args.momentum})
        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

        iteration = 0
        best_val_score = 0

        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(train_data)
            alpha = 1

            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

                with autograd.record():
                    output = [net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
                for l in loss:
                    l.backward()
                trainer.step(batch_size)
                train_loss += sum([l.sum().asscalar() for l in loss])

                train_metric.update(label, output)
                name, acc = train_metric.get()
                iteration += 1

            train_loss /= batch_size * num_batch
            name, acc = train_metric.get()
            name, val_acc = test(ctx, val_data)
            reporter(epoch=epoch, accuracy=val_acc)

    train(args.epochs, context)

if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # creating hyperparameters
    cs = CS.ConfigurationSpace()
    lr = CSH.UniformFloatHyperparameter('lr', lower=1e-3, upper=1e-1, log=True)
    wd = CSH.UniformFloatHyperparameter('wd', lower=1e-4, upper=5e-4, log=False)
    cs.add_hyperparameters([lr, wd])

    # create searcher and scheduler
    searcher = ag.searcher.RandomSampling(cs)
    if args.scheduler == 'hyperband':
        myscheduler = ag.scheduler.Hyperband_Scheduler(train_cifar, args,
                                                       {'num_cpus': 2, 'num_gpus': args.num_gpus}, searcher,
                                                       num_trials=args.num_trials,
                                                       checkpoint=args.checkpoint,
                                                       resume = args.resume,
                                                       time_attr='epoch', reward_attr="accuracy",
                                                       max_t=args.epochs, grace_period=args.epochs//4)
    else:
        myscheduler = ag.scheduler.FIFO_Scheduler(train_cifar, args,
                                                  {'num_cpus': 2, 'num_gpus': args.num_gpus}, searcher,
                                                  num_trials=args.num_trials,
                                                  checkpoint=args.checkpoint,
                                                  resume = args.resume,
                                                  reward_attr="accuracy")

    myscheduler.run()
    myscheduler.join_tasks()
    myscheduler.get_training_curves('{}.png'.format(os.path.splitext(args.checkpoint)[0]))

    print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                                   myscheduler.get_best_reward()))
