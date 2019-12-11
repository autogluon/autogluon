import os
from mxnet import optimizer as optim
import autogluon as ag
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from .nets import *
from .dataset import *
from ...core import AutoGluonObject
from ...utils import get_split_samplers, SampledDataset, DataLoader

__all__ = ['get_data_loader', 'get_network', 'imagenet_batch_fn',
           'default_batch_fn', 'default_val_fn', 'default_train_fn',
           'config_choice']

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
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0 ,even_split=False)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0 ,even_split=False)
    return data, label

def default_val_fn(net, batch, batch_fn, metric, ctx, dtype = 'float32'):
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

def config_choice(dataset, data_path):
    if dataset == 'dogs-vs-cats-redux-kernels-edition':
        dataset_path = os.path.join(data_path, dataset, 'train')
        net_cat = ag.space.Categorical('resnet50_v1b')
        @ag.obj(
            learning_rate=ag.space.Real(1e-3, 1e-2, log=True),
            momentum=ag.space.Real(0.88, 0.95),
            wd=ag.space.Real(1e-6, 1e-3, log=True),
            multi_precision=False
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        kaggle_choice = {'classes': 2, 'net': net_cat, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 8,
                         'epochs': 60,
                         'ngpus_per_trial': 2,
                         'num_trials': 1}
    elif dataset == 'aerial-cactus-identification':
        dataset_path = os.path.join(data_path, dataset, 'train')
        net_aeri = ag.space.Categorical('resnet18_v1')
        @ag.obj(
            learning_rate=ag.space.Real(1e-4, 1e-3, log=True),
            momentum=ag.space.Real(0.88, 0.95),
            wd=ag.space.Real(1e-6, 1e-5, log=True),
            multi_precision=False
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        kaggle_choice = {'classes': 2, 'net': net_aeri, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 16,
                         'epochs': 30,
                         'ngpus_per_trial': 2,
                         'num_trials': 1}
    elif dataset == 'plant-seedlings-classification':
        dataset_path = os.path.join(data_path, dataset, 'train')
        net_test = ag.space.Categorical('resnet50_v1b')
        @ag.obj(
            learning_rate=ag.space.Real(1e-4, 1e-3, log=True),
            momentum=ag.space.Real(0.93, 0.95),
            wd=ag.space.Real(1e-6, 1e-4, log=True),
            multi_precision=False
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        kaggle_choice = {'classes': 12, 'net': net_test, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 16,
                         'epochs': 40,
                         'ngpus_per_trial': 2,
                         'num_trials': 1}
    elif dataset == 'fisheries_Monitoring':
        dataset_path = os.path.join(data_path, dataset, 'train')
        net_fish = ag.space.Categorical('resnet18_v1')
        @ag.obj(
            learning_rate=ag.space.Real(1e-3, 1e-2, log=True),
            momentum=ag.space.Real(0.85, 0.90),
            wd=ag.space.Real(1e-6, 1e-4, log=True),
            multi_precision=False
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        kaggle_choice = {'classes': 8, 'net': net_fish, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 16,
                         'epochs': 40,
                         'ngpus_per_trial': 2,
                         'num_trials': 1}
    elif dataset == 'dog-breed-identification':
        dataset_path = os.path.join(data_path, dataset, 'train')
        net_dog = ag.space.Categorical('resnext101_64x4d')
        @ag.obj(
            learning_rate=ag.space.Real(1e-4, 1e-3, log=True),
            momentum=ag.space.Real(0.90, 0.95),
            wd=ag.space.Real(1e-6, 1e-4, log=True),
            multi_precision=False  # True fix
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        kaggle_choice = {'classes': 120, 'net': net_dog, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 16,
                         'epochs': 120,
                         'ngpus_per_trial': 2,
                         'num_trials': 1}
    elif dataset == 'shopee-iet-machine-learning-competition':
        dataset_path = os.path.join(data_path, dataset, 'train')
        net_shopee = ag.space.Categorical('resnet101_v1c')
        net_18 = ag.space.Categorical('resnet18_v1', 'resnet50_v1b', 'resnet101_v1c')
        net_50 = ag.space.Categorical('resnet50_v1b', 'resnet101_v1c', 'resnext101_64x4d')
        @ag.obj(
            learning_rate=ag.space.Real(1e-3, 1e-2, log=True),
            momentum=ag.space.Real(0.90, 0.95),
            wd=ag.space.Real(1e-3, 1e-2, log=True),
            multi_precision=False
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        kaggle_choice = {'classes': 18, 'net': net_shopee, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 32,
                         'epochs': 120,
                         'ngpus_per_trial': 2,
                         'num_trials': 1}
    elif dataset == 'shopee-iet':
        dataset_path = os.path.join(data_path, dataset, 'train')
        net_shopee = ag.space.Categorical('resnet101_v1c')
        @ag.obj(
            learning_rate=ag.space.Real(1e-4, 1e-2, log=True),
            momentum=ag.space.Real(0.85, 0.95),
            wd=ag.space.Real(1e-6, 1e-2, log=True),
            multi_precision=False
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        kaggle_choice = {'classes': 4, 'net': net_shopee, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 64,
                         'epochs': 1,
                         'ngpus_per_trial': 1,
                         'num_trials': 1}

    return kaggle_choice

