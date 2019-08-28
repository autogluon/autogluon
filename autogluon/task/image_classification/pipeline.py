import warnings
import logging

import mxnet as mx

from mxnet import gluon, init, autograd, nd
from mxnet.gluon import nn
from gluoncv.model_zoo import get_model
from mxnet.gluon.data.vision import transforms


from .metrics import get_metric_instance

from ...optim import SGD, NAG
from ...basic import autogluon_register_args
from ...network import get_built_in_network
from .dataset import get_built_in_dataset
from ...basic.space import *

__all__ = ['train_image_classification']

logger = logging.getLogger(__name__)

# Flexible:
#    - Even do need to tell which parameters are searchable
#    - Reusing script from gluoncv or gluonnlp
@autogluon_register_args()
def train_image_classification(args, reporter):
    print('pipeline args:', args)

    batch_size = args.batch_size * max(args.num_gpus, 1)
    ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
    if type(args.net) == str:
        net = get_built_in_network(args.net, ctx)._lazy_init()
    else:
        net = args.net
        net.initialize(ctx=ctx)

    if isinstance(args.train_dataset, str):
        print('Using built-in datasets')
        args.train_dataset = get_built_in_dataset(args.train_dataset)._lazy_init()
        args.val_dataset = get_built_in_dataset(args.val_dataset, train=False)._lazy_init()

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

    L = args.loss
    metric = get_metric_instance(args.metric)

    def train(epoch):
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
        if reporter:
            reporter(epoch=epoch, accuracy=test_acc, loss=test_loss)
        print('epoch: {epoch}, acc: {accuracy}, loss: {loss}'. \
            format(epoch=epoch, accuracy=test_acc, loss=test_loss))

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
