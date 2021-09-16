import mxnet as mx
import numpy as np
from mxnet import gluon, nd

from .dataset import *
from autogluon.core import AutoGluonObject
from ..utils import get_split_samplers, SampledDataset, DataLoader

__all__ = ['get_data_loader', 'imagenet_batch_fn',
           'default_batch_fn', 'default_val_fn', 'default_train_fn']


def get_data_loader(dataset, input_size, batch_size, num_workers, final_fit, split_ratio):
    if isinstance(dataset, AutoGluonObject):
        dataset = dataset.init()
    elif isinstance(dataset, str):
        dataset = get_built_in_dataset(dataset,
                                       train=True,
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
        num_workers = 0  # ?
        train_data = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            last_batch="discard", num_workers=num_workers
        )
        val_data = None
        if not final_fit:
            val_data = DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers
            )
        batch_fn = default_batch_fn
        num_batches = len(train_data)
    return train_data, val_data, batch_fn, num_batches


def imagenet_batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
    return data, label


def default_batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
    return data, label


def default_val_fn(net, batch, batch_fn, metric, ctx, dtype='float32'):
    with mx.autograd.pause(train_mode=False):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X.astype(dtype, copy=False)) for X in data]
    metric.update(label, outputs)


def mixup_transform(label, classes, lam=1, eta=0.0):
    if isinstance(label, nd.NDArray):
        label = [label]
    res = []
    for l in label:
        y1 = l.one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
        y2 = l[::-1].one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
        res.append(lam * y1 + (1 - lam) * y2)
    return res


def smooth(label, classes, eta=0.1):
    if isinstance(label, nd.NDArray):
        label = [label]
    smoothed = [
        l.one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
        for l in label
    ]
    return smoothed


def default_train_fn(epoch, num_epochs, net, batch, batch_size, criterion, trainer, batch_fn, ctx,
                     mixup=False, label_smoothing=False, distillation=False,
                     mixup_alpha=0.2, mixup_off_epoch=0, classes=1000,
                     dtype='float32', metric=None, teacher_prob=None):
    data, label = batch_fn(batch, ctx)
    if mixup:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        if epoch >= num_epochs - mixup_off_epoch:
            lam = 1
        data = [lam * X + (1 - lam) * X[::-1] for X in data]
        if label_smoothing:
            eta = 0.1
        else:
            eta = 0.0
        label = mixup_transform(label, classes, lam, eta)
    elif label_smoothing:
        hard_label = label
        label = smooth(label, classes)

    with mx.autograd.record():
        outputs = [net(X.astype(dtype, copy=False)) for X in data]
        if distillation:
            loss = [
                criterion(
                    yhat.astype('float', copy=False),
                    y.astype('float', copy=False),
                    p.astype('float', copy=False)
                )
                for yhat, y, p in zip(outputs, label, teacher_prob(data))
            ]
        else:
            loss = [criterion(yhat, y.astype(dtype, copy=False)) for yhat, y in zip(outputs, label)]

    for l in loss:
        l.backward()
    trainer.step(batch_size, ignore_stale_grad=True)

    if metric:
        if mixup:
            output_softmax = [
                nd.SoftmaxActivation(out.astype('float32', copy=False))
                for out in outputs
            ]
            metric.update(label, output_softmax)
        else:
            if label_smoothing:
                metric.update(hard_label, outputs)
            else:
                metric.update(label, outputs)
        return metric
    else:
        return


def _train_val_split(train_dataset, split_ratio=0.2):
    train_sampler, val_sampler = get_split_samplers(train_dataset, split_ratio)
    return SampledDataset(train_dataset, train_sampler), SampledDataset(train_dataset, val_sampler)
