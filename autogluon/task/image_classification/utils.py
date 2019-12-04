import os
from mxnet import optimizer as optim
import autogluon as ag

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
import gluoncv as gcv

from .nets import *
# from .nets import get_built_in_network

# from .dataset import *
from autogluon.core import AutoGluonObject

from .dataset import get_built_in_dataset
# from ...utils.dataset import get_split_samplers, SampledDataset
from ...utils import get_split_samplers, SampledDataset

__all__ = ['get_data_loader', 'get_network', 'imagenet_batch_fn',
           'default_batch_fn', 'default_val_fn', 'default_train_fn',
           'config_choice',
           'get_network_origin']

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

def get_network_origin(net, num_classes, ctx):
    if type(net) == str:
        net = get_built_in_network_origin(net, num_classes, ctx=ctx)
    else:
        net.initialize(ctx=ctx)
    return net

def get_network(net, **kwargs):
    if type(net) == str:
        net = get_built_in_network(net, **kwargs)
    else:
        net.initialize(ctx=kwargs['ctx'])
    return net

def imagenet_batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
    return data, label

def default_batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    return data, label

def default_val_fn(net, batch, batch_fn, metric, ctx, dtype):
    with mx.autograd.pause(train_mode=False):
        data, label = batch_fn(batch, ctx)
        # outputs = [net(X) for X in data]
        outputs = [net(X.astype(dtype, copy=False)) for X in data]

    metric.update(label, outputs)



def mixup_transform(label, classes, lam=1, eta=0.0):
    if isinstance(label, nd.NDArray):
        label = [label]
    res = []
    for l in label:
        y1 = l.one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
        y2 = l[::-1].one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
        res.append(lam*y1 + (1-lam)*y2)
    return res

def smooth(label, classes, eta=0.1):
    if isinstance(label, nd.NDArray):
        label = [label]
    smoothed = []
    for l in label:
        res = l.one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
        smoothed.append(res)
    return smoothed

def default_train_fn(epoch, num_epochs, net, batch, batch_size, criterion, trainer, batch_fn, ctx,
                     mixup, label_smoothing, distillation,
                     mixup_alpha, mixup_off_epoch, classes,
                     dtype, metric, teacher_prob):
    data, label = batch_fn(batch, ctx)
    #outputs = [net(X) for X in data]
    if mixup:
        #
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
    # with ag.record():
        outputs = [net(X.astype(dtype, copy=False)) for X in data]
        if distillation:
            loss = [criterion(yhat.astype('float', copy=False),
                      y.astype('float', copy=False),
                      p.astype('float', copy=False)) for yhat, y, p in zip(outputs, label, teacher_prob(data))]
        else:
            loss = [criterion(yhat, y.astype(dtype, copy=False)) for yhat, y in zip(outputs, label)]
    for l in loss:
        l.backward()
    trainer.step(batch_size, ignore_stale_grad=True)
    # trainer.step(batch_size)
    if mixup:
        output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False)) \
                          for out in outputs]
        metric.update(label, output_softmax)
    else:
        if label_smoothing:
            metric.update(hard_label, outputs)
        else:
            metric.update(label, outputs)
    return metric

def _train_val_split(train_dataset, split_ratio=0.2):
    train_sampler, val_sampler = get_split_samplers(train_dataset, split_ratio)
    return SampledDataset(train_dataset, train_sampler), SampledDataset(train_dataset, val_sampler)

def config_choice(dataset, root):
    ## data
    dataset_path = os.path.join(root, dataset, 'train')

    # tricks
    tricks = {'distillation': True,
              'teacher_name': None,
              'hard_weight': 0.5,
              'temperature': 20.0,
              'mixup': False, # bug
              'mixup_alpha': 0.2,
              'mixup_off_epoch': 0,
              'label_smoothing': True,
              'no_wd': True,
              'use_pretrained': True,
              'use_gn': False,
              'last_gamma': True,
              'batch_norm': False,
              'use_se': False
              }

    lr_config = {'lr_decay_epoch':'10,20,30',
                 'lr_decay': 0.1,
                 'lr_decay_period': 0,
                 'warmup_epochs': 5,
                 'warmup_lr': 0.0}
    lr_scheduler = ag.space.Categorical('poly', 'cosine')

    ## optimizer_params
    @ag.obj(
        learning_rate=ag.space.Real(1e-4, 1e-2, log=True),
        momentum=ag.space.Real(0.85, 0.95),
        wd=ag.space.Real(1e-6, 1e-2, log=True),
        # multi_precision = False # True fix

    )
    class NAG(optim.NAG):
        pass
    optimizer = NAG()

    ## net
    # net_test = ag.space.Categorical('resnet50_v1b')
    net_18 = ag.space.Categorical('resnet18_v1', 'resnet50_v1b', 'resnet101_v1c')
    net_50 = ag.space.Categorical('resnet50_v1b', 'resnet101_v1c', 'resnext101_64x4d')


    if dataset == 'dogs-vs-cats-redux-kernels-edition/':
        kaggle_choice = {'classes': 2, 'net': net_18, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler,
                         'dataset': dataset_path,
                         'batch_size': 64,
                         'epochs': 20,
                         'ngpus_per_trial': 4,
                         'num_trials': 5}

    elif dataset == 'aerial-cactus-identification/':
        kaggle_choice = {'classes': 2, 'net': net_18, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler,
                         'dataset': dataset_path,
                         'batch_size': 64,
                         'epochs': 30,
                         'ngpus_per_trial': 4,
                         'num_trials': 20}

    elif dataset == 'plant-seedlings-classification/':
        kaggle_choice = {'classes': 12, 'net': net_18, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler,
                         'dataset': dataset_path,
                         'batch_size': 64,
                         'epochs': 20,
                         'ngpus_per_trial': 4,
                         'num_trials': 10}

    elif dataset == 'fisheries_Monitoring/':
        kaggle_choice = {'classes': 8, 'net': net_18, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler,
                         'dataset': dataset_path,
                         'batch_size': 32,
                         'epochs': 40,
                         'ngpus_per_trial': 4,
                         'num_trials': 10}

    elif dataset == 'dog-breed-identification/':
        kaggle_choice = {'classes': 120, 'net': net_50, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler,
                         'dataset': dataset_path,
                         'batch_size': 32,
                         'epochs': 80,
                         'ngpus_per_trial': 4,
                         'num_trials': 15}

    kaggle_choice['tricks'] = tricks
    kaggle_choice['lr_config'] = lr_config
    return kaggle_choice