import os

import argparse, time, logging
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import LRScheduler
from gluoncv.data import transforms as gcv_transforms

import autogluon.core as ag


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='number of gpus to use.')
    parser.add_argument('--num-trials', default=10, type=int,
                        help='number of trail tasks')
    parser.add_argument('--epochs', default=20, type=int,
                        help='number of epochs')
    parser.add_argument('--scheduler', type=str, default='fifo',
                        help='scheduler name (default: fifo)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/cifar1.ag',
                        help='checkpoint path (default: None)')
    parser.add_argument('--debug', action='store_true', default= False,
                        help='debug if needed')
    args = parser.parse_args()
    return args


@ag.args(
    batch_size=64,
    num_workers=2,
    num_gpus=1,
    model='cifar_resnet20_v1',
    j=4,
    lr=ag.space.Real(1e-2, 1e-1, log=True),
    momentum=0.9,
    wd=ag.space.Real(1e-5, 1e-3, log=True),
    epochs=20,
)
def train_cifar(args, reporter):
    print('args', args)
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

        start_epoch = 0

        for epoch in range(start_epoch, epochs):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(train_data)
            alpha = 1

            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

                with mx.autograd.record():
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
            reporter(epoch=epoch+1, accuracy=val_acc)

    train(args.epochs, context)

def cifar_evaluate(net, args):
    batch_size = args.batch_size
    batch_size *= max(1, args.num_gpus)

    ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
    net.collect_params().reset_ctx(ctx)
    metric = mx.metric.Accuracy()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    val_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()[1]

if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    train_cifar.update(epochs=args.epochs)
    # create searcher and scheduler
    extra_node_ips = []
    if args.scheduler == 'hyperband':
        myscheduler = ag.scheduler.HyperbandScheduler(train_cifar,
                                                      resource={'num_cpus': 2, 'num_gpus': args.num_gpus},
                                                      num_trials=args.num_trials,
                                                      checkpoint=args.checkpoint,
                                                      time_attr='epoch', reward_attr="accuracy",
                                                      max_t=args.epochs, grace_period=args.epochs//4,
                                                      dist_ip_addrs=extra_node_ips)
    elif args.scheduler == 'fifo':
        myscheduler = ag.scheduler.FIFOScheduler(train_cifar,
                                                 resource={'num_cpus': 2, 'num_gpus': args.num_gpus},
                                                 num_trials=args.num_trials,
                                                 checkpoint=args.checkpoint,
                                                 reward_attr="accuracy",
                                                 dist_ip_addrs=extra_node_ips)
    else:
        raise RuntimeError('Unsuported Scheduler!')

    print(myscheduler)
    myscheduler.run()
    myscheduler.join_jobs()
    myscheduler.get_training_curves('{}.png'.format(os.path.splitext(args.checkpoint)[0]))
    print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                                   myscheduler.get_best_reward()))
