import mxnet as mx
from mxnet import gluon
import gluoncv as gcv

from ...scheduler.resource import get_gpu_count

def default_reward_fn(metric, net):
    reward = metric * ((net.avg_latency / net.latency) ** 0.07)
    return reward

def default_val_fn(net, batch, batch_fn, ctx):
    metric = mx.metric.Accuracy()
    with mx.autograd.pause(train_mode=False):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X) for X in data]
    metric.update(label, outputs)
    _, acc = metric.get()
    return acc

def default_train_fn(net, batch, batch_size, criterion, trainer, batch_fn, ctx):
    data, label = batch_fn(batch, ctx)
    #net.collect_params().zero_grad()
    with mx.autograd.record():
        outputs = [net(X) for X in data]
        loss = [criterion(yhat, y) for yhat, y in zip(outputs, label)]
    for l in loss:
        l.backward()
    trainer.step(batch_size, ignore_stale_grad=True)
    mx.nd.waitall()

def init_default_train_args(batch_size, net, epochs, iters_per_epoch):
    train_args = {}
    lr_scheduler = gcv.utils.LRScheduler('cosine', base_lr=0.1, target_lr=0.0001,
                                         nepochs=epochs, iters_per_epoch=iters_per_epoch)
    optimizer_params = {'wd': 1e-4, 'momentum': 0.9, 'lr_scheduler': lr_scheduler}
    train_args['trainer'] = gluon.Trainer(net.collect_params(), 'sgd', optimizer_params)
    train_args['batch_size'] = batch_size
    train_args['criterion'] = gluon.loss.SoftmaxCrossEntropyLoss()
    return train_args

def imagenet_batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
    return data, label

def default_batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    return data, label
