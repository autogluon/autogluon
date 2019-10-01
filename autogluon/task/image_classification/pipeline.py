import warnings
import logging
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

__all__ = ['train_image_classification']

logger = logging.getLogger(__name__)

lr_schedulers = {
    'poly': mx.lr_scheduler.PolyScheduler,
    'cosine': mx.lr_scheduler.CosineScheduler
}

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
        val_data = gluon.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
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
