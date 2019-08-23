import warnings
import logging

import mxnet as mx

from mxnet import gluon, init, autograd, nd
from mxnet.gluon import nn
from gluoncv.model_zoo import get_model
from mxnet.gluon.data.vision import transforms


#from .losses import get_loss_instance
from .metrics import get_metric_instance

from ...optim import SGD, NAG
from ...basic import autogluon_register_args
from ...network import get_finetune_network
from ...dataset import get_built_in_dataset
from ...basic.space import *

__all__ = ['train_image_classification']

logger = logging.getLogger(__name__)


@autogluon_register_args(
    train_dataset=get_built_in_dataset('cifar', train=True, crop_size=ListSpace(32, 48)),
    val_dataset=get_built_in_dataset('cifar', train=True, crop_size=224),
    net=ListSpace('resnet50v1b', 'ResNet50_v1d'),
    optimizer=ListSpace(
        SGD(learning_rate=LogLinearSpace(1e-4, 1e-2),
            momentum=0.9,
            wd=LogLinearSpace(1e-5, 1e-3)),
        NAG(learning_rate=LogLinearSpace(1e-4, 1e-2),
            momentum=0.9,
            wd=LogLinearSpace(1e-5, 1e-3)),
        ),
    metric='Accuracy',
    num_workers=4,
    num_gpus=0,
    batch_size=64,
    epochs=20,
    )
def train_image_classification(args, reporter):
    batch_size = args.batch_size * max(args.num_gpus, 1)
    ctx = [mx.gpu(i)
           for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
    if type(net) == str:
        net = get_finetune_network(args.net, ctx)
    else:
        net = args.net

    train_data = gluon.data.DataLoader(
        args.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        last_batch="rollover",
        num_workers=args.num_workers)
    val_data = gluon.data.DataLoader(
        args.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    trainer = gluon.Trainer(net.collect_params(), args.optimizer)

    def _print_debug_info(args):
        logger.debug('Print debug info:')
        for k, v in vars(args).items():
            logger.debug('%s:%s' % (k, v))

    _print_debug_info(args)

    # TODO (cgraywang): update with search space
    L = get_loss_instance(args.loss)
    metric = get_metric_instance(args.metric)

    def train(epoch):
        #TODO (cgraywang): change to lr scheduler
        if hasattr(args, 'lr_step') and hasattr(args, 'lr_factor'):
            if epoch % args.lr_step == 0:
                trainer.set_learning_rate(trainer.learning_rate * args.lr_factor)

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            with autograd.record():
                outputs = [net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(batch_size)

        mx.nd.waitall()

    def test(epoch):
        test_loss = 0
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            outputs = [net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]

            test_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            metric.update(label, outputs)

        _, test_acc = metric.get()
        test_loss /= len(val_data)
        reporter(epoch=epoch, accuracy=test_acc, loss=test_loss)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
