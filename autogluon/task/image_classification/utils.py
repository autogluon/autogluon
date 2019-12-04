import numpy as np
import mxnet as mx
from mxnet import gluon
import gluoncv as gcv
from .nets import *
from .dataset import *
from ...core import AutoGluonObject
from ...utils import get_split_samplers, SampledDataset, DataLoader

__all__ = ['get_data_loader', 'imagenet_batch_fn',
           'default_batch_fn', 'default_val_fn', 'default_train_fn']

def get_data_loader(dataset, input_size, batch_size, num_workers, final_fit, split_ratio):
    if isinstance(dataset, AutoGluonObject):
        dataset = dataset.init()
    if isinstance(dataset, str):
        dataset = get_built_in_dataset(dataset, train=True,
                                       input_size=input_size,
                                       batch_size=batch_size,
                                       num_workers=num_workers)
    if final_fit:
        train_dataset, val_dataset = dataset, None
    else:
        train_dataset, val_dataset = _train_val_split(dataset, split_ratio)

    if isinstance(dataset, str) and dataset.lower() == 'imagenet':
        train_data = train_dataset
        val_data = val_dataset
        batch_fn = imagenet_batch_fn
        imagenet_samples = 1281167
        num_batches = imagenet_samples // batch_size
    else:
        num_workers = 0
        train_data = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            last_batch="discard", num_workers=num_workers)
        val_data = None
        if not final_fit:
            val_data = DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers)
        batch_fn = default_batch_fn
        num_batches = len(train_data)
    return train_data, val_data, batch_fn, num_batches

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
    outputs = [net(X) for X in data]
    with mx.autograd.record():
        outputs = [net(X) for X in data]
        loss = [criterion(yhat, y) for yhat, y in zip(outputs, label)]
    for l in loss:
        l.backward()
    trainer.step(batch_size, ignore_stale_grad=True)

def _train_val_split(train_dataset, split_ratio=0.2):
    train_sampler, val_sampler = get_split_samplers(train_dataset, split_ratio)
    return SampledDataset(train_dataset, train_sampler), SampledDataset(train_dataset, val_sampler)
