import numpy as np
import mxnet as mx
from mxnet import gluon
import gluoncv as gcv
from .nets import *
from .dataset import *

__all__ = ['get_data_loader', 'get_network', 'imagenet_batch_fn',
           'default_batch_fn', 'default_val_fn', 'default_train_fn']

def get_data_loader(dataset, input_size, batch_size, num_workers, final_fit):
    if isinstance(dataset, AutoGluonObject):
        dataset = dataset.init()
    if isinstance(dataset, str):
        train_dataset = get_built_in_dataset(dataset, train=True,
                                             input_size=input_size,
                                             batch_size=batch_size,
                                             num_workers=num_workers).init()
        val_dataset = get_built_in_dataset(dataset, train=False,
                                           input_size=input_size,
                                           batch_size=batch_size,
                                           num_workers=num_workers).init()
    else:
        train_dataset = dataset.train
        val_dataset = dataset.val
    if val_dataset is None and not final_fit:
        split = 2
        train_dataset, val_dataset = _train_val_split(train_dataset, split)

    if isinstance(dataset, str) and dataset.lower() == 'imagenet':
        train_data = train_dataset
        val_data = val_dataset
        batch_fn = imagenet_batch_fn
        imagenet_samples = 1281167
        num_batches = imagenet_samples // batch_size
    else:
        train_data = gluon.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            last_batch="rollover", num_workers=num_workers)
        val_data = None
        if not final_fit:
            val_data = gluon.data.DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers)
        batch_fn = default_batch_fn
        num_batches = len(train_data)
    return train_data, val_data, batch_fn, num_batches


def get_network(net, num_classes, ctx):
    if type(net) == str:
        net = get_built_in_network(net, num_classes, ctx=ctx)
    else:
        net.initialize(ctx=ctx)
    return net

def imagenet_batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
    return data, label

def default_batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    return data, label

def default_val_fn(net, batch, batch_fn, metric, ctx):
    with mx.autograd.pause(train_mode=False):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X) for X in data]
    metric.update(label, outputs)

def default_train_fn(net, batch, batch_size, criterion, trainer, batch_fn, ctx):
    data, label = batch_fn(batch, ctx)
    with mx.autograd.record():
        outputs = [net(X) for X in data]
        loss = [criterion(yhat, y) for yhat, y in zip(outputs, label)]
    for l in loss:
        l.backward()
    trainer.step(batch_size, ignore_stale_grad=True)

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
        mx.nd.concat(*data, dim=0),
        np.concatenate(tuple(label), axis=0))
    val_data = [train_dataset[i][0].expand_dims(0) for i in
                range((split - 1) * split_len, split * split_len)]
    val_label = [np.array([train_dataset[i][1]]) for i in
                 range((split - 1) * split_len, split * split_len)]
    val = gluon.data.dataset.ArrayDataset(
        mx.nd.concat(*val_data, dim=0),
        np.concatenate(tuple(val_label), axis=0))
    return train, val
