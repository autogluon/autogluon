import warnings
import logging
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

__all__ = ['train_image_classification']

logger = logging.getLogger(__name__)


@autogluon_method
def train_image_classification(args, reporter):
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
        val_data = gluon.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
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
            optimizer_params = {
                'learning_rate': args.lr,
                'momentum': args.momentum,
                'wd': args.wd
            }
        elif args.optimizer == 'adam':
            optimizer_params = {
                'learning_rate': args.lr,
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
            with autograd.record():
                outputs = [net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(batch_size)
            if _demo_early_stopping(i):
                break
        mx.nd.waitall()
        #TODO: fix mutli gpu bug
        # if reporter is not None:
        #     reporter.save_dict(epoch=epoch, params=net.collect_params())

    def test(epoch):
        test_loss = 0
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0],
                                              ctx_list=ctx,
                                              batch_axis=0,
                                              even_split=False)
            label = gluon.utils.split_and_load(batch[1],
                                               ctx_list=ctx,
                                               batch_axis=0,
                                               even_split=False)
            outputs = [net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]

            test_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            metric.update(label, outputs)
            if _demo_early_stopping(i):
                break
        _, test_acc = metric.get()
        test_loss /= len(val_data)
        # TODO (cgraywang): add ray
        reporter(epoch=epoch, accuracy=test_acc, loss=test_loss)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if val_data is not None:
            test(epoch)
    if val_data is None:
        net_path = os.path.join(os.path.splitext(args.savedir)[0], 'net.params')
        net.save_parameters(net_path)
