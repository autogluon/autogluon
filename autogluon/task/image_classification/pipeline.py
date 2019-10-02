import warnings
import logging
<<<<<<< HEAD
import numpy as np

import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon, init, autograd, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model

from .metrics import get_metric_instance
from ...core.optimizer import SGD, NAG
from ...core import *
from .nets import get_built_in_network
from .dataset import get_built_in_dataset
=======
import os
import numpy as np
import mxnet as mx

from mxnet import gluon, init, autograd, nd
from mxnet.gluon import nn
from gluoncv.model_zoo import get_model
from mxnet.gluon.data.vision import transforms


from ...basic import autogluon_method
from .losses import get_loss_instance
from .metrics import get_metric_instance
import time
>>>>>>> awslabs/master

__all__ = ['train_image_classification']

logger = logging.getLogger(__name__)

<<<<<<< HEAD
=======

>>>>>>> awslabs/master
lr_schedulers = {
    'poly': mx.lr_scheduler.PolyScheduler,
    'cosine': mx.lr_scheduler.CosineScheduler
}

<<<<<<< HEAD
@autogluon_register_args()
def train_image_classification(args, reporter):
    logger.debug('pipeline args: {}'.format(args))

    batch_size = args.batch_size * max(args.num_gpus, 1)
    ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
    if type(args.net) == str:
        net = get_built_in_network(args.net, args.dataset.num_classes, ctx)._lazy_init()
    else:
        net = args.net
        net.initialize(ctx=ctx)

    if isinstance(args.dataset, str):
        train_dataset = get_built_in_dataset(args.dataset)._lazy_init()
        val_dataset = get_built_in_dataset(args.dataset, train=False)._lazy_init()
    else:
        train_dataset = args.dataset.train
        val_dataset = args.dataset.val
    if val_dataset is None:
        split = 2 if not args.final_fit else 0
        train_dataset, val_dataset = _train_val_split(train_dataset, split)

    train_data = gluon.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        last_batch="rollover",
        num_workers=args.num_workers)
    if not args.final_fit:
=======
@autogluon_method
def train_image_classification(args, reporter):
    """The training script for image classification.

     Args:
        args: the argument parser.
        reporter: the reporter (StatusReporter)
    """
    # Set Hyper-params
    def _init_hparams():
        batch_size = args.data.batch_size * max(args.num_gpus, 1)
        ctx = [mx.gpu(i)
               for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
        return batch_size, ctx

    batch_size, ctx = _init_hparams()

    def _train_val_split(train_dataset):
        split = args.data.split
        if split == 0:
            return train_dataset, None
        split_len = int(len(train_dataset) / 10)
        if split == 1:
            data = [train_dataset[i][0].expand_dims(0) for i in
                    range(split * split_len, len(train_dataset))]
            label = [np.array([train_dataset[i][1]]) for i in
                     range(split * split_len, len(train_dataset))]
        else:
            data = [train_dataset[i][0].expand_dims(0) for i in
                    range((split - 1) * split_len)] + \
                   [train_dataset[i][0].expand_dims(0) for i in
                    range(split * split_len, len(train_dataset))]
            label = [np.array([train_dataset[i][1]]) for i in range((split - 1) * split_len)] + \
                    [np.array([train_dataset[i][1]]) for i in
                     range(split * split_len, len(train_dataset))]
        train = gluon.data.dataset.ArrayDataset(
            nd.concat(*data, dim=0),
            np.concatenate(tuple(label), axis=0))
        val_data = [train_dataset[i][0].expand_dims(0) for i in
                    range((split - 1) * split_len, split * split_len)]
        val_label = [np.array([train_dataset[i][1]]) for i in
                     range((split - 1) * split_len, split * split_len)]
        val = gluon.data.dataset.ArrayDataset(
            nd.concat(*val_data, dim=0),
            np.concatenate(tuple(val_label), axis=0))
        return train, val

    # Define DataLoader
    def _get_dataloader():
        def _init_dataset(dataset, transform_fn, transform_list):
            if dataset is None:
                return dataset
            if transform_fn is not None:
                dataset = dataset.transform(transform_fn)
            if transform_list is not None:
                dataset = dataset.transform_first(transforms.Compose(transform_list))
            return dataset
        # args.data._read_dataset(**args.data._kwargs)
        # if reporter == None:
        #     args.data.split = 0
        train_dataset = _init_dataset(args.data.train, args.data.transform_train_fn,
                                      args.data.transform_train_list)
        val_dataset = _init_dataset(args.data.val, args.data.transform_val_fn,
                                    args.data.transform_val_list)
        if val_dataset is None:
            train_dataset, val_dataset = _train_val_split(train_dataset)
        train_data = gluon.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            last_batch="discard",
            num_workers=args.data.num_workers)
        if val_dataset is None:
            return train_data, None
>>>>>>> awslabs/master
        val_data = gluon.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
<<<<<<< HEAD
            num_workers=args.num_workers)

    if isinstance(args.lr_scheduler, str):
        lr_scheduler = lr_schedulers[args.lr_scheduler](len(train_data) * args.epochs, base_lr=args.optimizer.lr)
    else:
        lr_scheduler = args.lr_scheduler
    args.optimizer.lr_scheduler = lr_scheduler
    trainer = gluon.Trainer(net.collect_params(), args.optimizer)

    L = args.loss
    metric = get_metric_instance(args.metric)

    def train(epoch):
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
=======
            num_workers=args.data.num_workers)
        return train_data, val_data
    train_data, val_data = _get_dataloader()

    # Define Network
    net = get_model(args.model, pretrained=args.pretrained)
    with net.name_scope():
        num_classes = args.data.num_classes
        if hasattr(args, 'classes'):
            warnings.warn('Warning: '
                          'number of class of labels can be inferred.')
            num_classes = args.classes
        if hasattr(net, 'output'):
            net.output = nn.Dense(num_classes)
        else:
            net.fc = nn.Dense(num_classes) #TODO (cgraywang): deal with resnet v1/2 diff
    if not args.pretrained:
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    else:
        # TODO (cgraywang): hparams for initializer
        if hasattr(net, 'output'):
            net.output.initialize(init.Xavier(), ctx=ctx)
        else:
            net.fc.initialize(init.Xavier(), ctx=ctx)
        net.collect_params().reset_ctx(ctx)
    net.hybridize()

    # Define trainer
    def _set_optimizer_params(args):
        # TODO (cgraywang): a better way?
        if args.optimizer == 'sgd' or args.optimizer == 'nag':
            if 'lr_scheduler' not in vars(args):
                optimizer_params = {
                    'learning_rate': args.lr,
                    'momentum': args.momentum,
                    'wd': args.wd
                }
            else:
                optimizer_params = {
                    'lr_scheduler': lr_schedulers[args.lr_scheduler](len(train_data), base_lr=args.lr),
                    'momentum': args.momentum,
                    'wd': args.wd
                }
        elif args.optimizer == 'adam':
            if 'lr_scheduler' not in vars(args):
                optimizer_params = {
                    'learning_rate': args.lr,
                    'wd': args.wd
                }
            else:
                optimizer_params = {
                    'lr_scheduler': lr_schedulers[args.lr_scheduler](len(train_data), base_lr=args.lr),
                    'wd': args.wd
                }
        else:
            raise NotImplementedError
        return optimizer_params

    optimizer_params = _set_optimizer_params(args)
    trainer = gluon.Trainer(net.collect_params(),
                            args.optimizer,
                            optimizer_params)

    def _print_debug_info(args):
        logger.debug('Print debug info:')
        for k, v in vars(args).items():
            logger.debug('%s:%s' % (k, v))

    _print_debug_info(args)

    # TODO (cgraywang): update with search space
    L = get_loss_instance(args.loss)
    metric = get_metric_instance(args.metric)

    def _demo_early_stopping(batch_id):
        if 'demo' in vars(args):
            if args.demo and batch_id == 3:
                return True
        return False

    def train(epoch):
        #TODO (cgraywang): change to lr scheduler
        if hasattr(args, 'lr_step') and hasattr(args, 'lr_factor'):
            if epoch % args.lr_step == 0:
                trainer.set_learning_rate(trainer.learning_rate * args.lr_factor)

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0],
                                              ctx_list=ctx,
                                              batch_axis=0,
                                              even_split=False)
            label = gluon.utils.split_and_load(batch[1],
                                               ctx_list=ctx,
                                               batch_axis=0,
                                               even_split=False)
>>>>>>> awslabs/master
            with autograd.record():
                outputs = [net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(batch_size)
<<<<<<< HEAD

        mx.nd.waitall()
=======
            if _demo_early_stopping(i):
                break
        mx.nd.waitall()
        #TODO: fix mutli gpu bug
        # if reporter is not None:
        #     reporter.save_dict(epoch=epoch, params=net.collect_params())
>>>>>>> awslabs/master

    def test(epoch):
        test_loss = 0
        for i, batch in enumerate(val_data):
<<<<<<< HEAD
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
=======
            data = gluon.utils.split_and_load(batch[0],
                                              ctx_list=ctx,
                                              batch_axis=0,
                                              even_split=False)
            label = gluon.utils.split_and_load(batch[1],
                                               ctx_list=ctx,
                                               batch_axis=0,
                                               even_split=False)
>>>>>>> awslabs/master
            outputs = [net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]

            test_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            metric.update(label, outputs)
<<<<<<< HEAD

        _, reward = metric.get()
        test_loss /= len(val_data)
        if reporter:
            reporter(epoch=epoch, reward=reward, loss=test_loss)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if not args.final_fit:
            test(epoch)

    if args.final_fit:
        return net

def _train_val_split(train_dataset, split=1):
    # temporary solution, need to change using batchify function
    if split == 0:
        return train_dataset, None
    split_len = len(train_dataset) // 10
    if split == 1:
        data = [train_dataset[i][0].expand_dims(0) for i in
                range(split * split_len, len(train_dataset))]
        label = [np.array([train_dataset[i][1]]) for i in
                 range(split * split_len, len(train_dataset))]
    else:
        data = [train_dataset[i][0].expand_dims(0) for i in
                range((split - 1) * split_len)] + \
               [train_dataset[i][0].expand_dims(0) for i in
                range(split * split_len, len(train_dataset))]
        label = [np.array([train_dataset[i][1]]) for i in range((split - 1) * split_len)] + \
                [np.array([train_dataset[i][1]]) for i in
                 range(split * split_len, len(train_dataset))]
    train = gluon.data.dataset.ArrayDataset(
        nd.concat(*data, dim=0),
        np.concatenate(tuple(label), axis=0))
    val_data = [train_dataset[i][0].expand_dims(0) for i in
                range((split - 1) * split_len, split * split_len)]
    val_label = [np.array([train_dataset[i][1]]) for i in
                 range((split - 1) * split_len, split * split_len)]
    val = gluon.data.dataset.ArrayDataset(
        nd.concat(*val_data, dim=0),
        np.concatenate(tuple(val_label), axis=0))
    return train, val
=======
            if _demo_early_stopping(i):
                break
        _, test_acc = metric.get()
        test_loss /= len(val_data)
        # TODO (cgraywang): add ray
        reporter(epoch=epoch, accuracy=test_acc, loss=test_loss)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if reporter is not None:
            test(epoch)
    if reporter is None:
        net_path = os.path.join(os.path.splitext(args.savedir)[0], 'net.params')
        net.save_parameters(net_path)
>>>>>>> awslabs/master
