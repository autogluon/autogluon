import warnings
import logging

import random
import mxnet as mx
import numpy as np

from mxnet import gluon, init, autograd, nd
from mxnet.gluon import nn
import gluoncv
from gluoncv.model_zoo import get_model
from mxnet.gluon.data.vision import transforms


from ...basic import autogluon_method
from .dataset import get_transform_fn
from .losses import get_loss_instance
from .metrics import get_metric_instance
from .model_zoo import get_norm_layer

__all__ = ['train_object_detection']

logger = logging.getLogger(__name__)

@autogluon_method
def train_object_detection(args, reporter):
    # Set Hyper-params
    def _init_hparams():
        ctx = [mx.gpu(i)
               for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
        return ctx
    ctx = _init_hparams()

    # Define Network
    def _get_net_async_net(ctx):
        if args.norm_layer == 'SyncBatchNorm' and len(ctx) > 1:
            net = get_model(args.model,
                            pretrained=args.pretrained,
                            pretrained_base=args.pretrained_base,
                            norm_layer=get_norm_layer(args.norm_layer),
                            norm_kwargs={'num_devices': len(ctx)})
            async_net = get_model(args.model, pretrained=args.pretrained, pretrained_base=False)
        else:
            net = get_model(args.model,
                            pretrained=args.pretrained,
                            pretrained_base=args.pretrained_base,
                            norm_layer=get_norm_layer(args.norm_layer))
            async_net = get_model(args.model, pretrained=args.pretrained, pretrained_base=False)
        return net, async_net
    net, async_net = _get_net_async_net(ctx)
    net.reset_class(classes=args.data.train.classes)
    if not args.pretrained:
        net.collect_params().initialize(init.Xavier(), ctx=ctx)
    else:
        net.collect_params().reset_ctx(ctx)

    # Define DataLoader
    def _get_dataloader():
        def _init_dataset(dataset, transform_fn, transform_list):
            if transform_fn is not None:
                dataset = dataset.transform(transform_fn)
            if transform_list is not None:
                dataset = dataset.transform_first(transforms.Compose(transform_list))
            return dataset
        def _get_width_height_achors():
            data_shape = int(args.model.split('_')[1])
            width, height = data_shape, data_shape
            with autograd.train_mode():
                _, _, anchors = async_net(mx.nd.zeros((1, 3, height, width)))
            return width, height, anchors
        width, height, anchors = _get_width_height_achors()
        train_dataset = _init_dataset(args.data.train,
                                      get_transform_fn(args.data.transform_train_fn,
                                                       width, height, anchors),
                                      args.data.transform_train_list)
        val_dataset = _init_dataset(args.data.val,
                                    get_transform_fn(args.data.transform_val_fn,
                                                     width, height),
                                    args.data.transform_val_list)
        train_data = gluon.data.DataLoader(
            train_dataset,
            batch_size=args.data.batch_size,
            shuffle=True,
            batchify_fn=args.data.batchify_train_fn,
            last_batch='rollover',
            num_workers=args.data.num_workers)
        val_data = gluon.data.DataLoader(
            val_dataset,
            batch_size=args.data.batch_size,
            shuffle=False,
            batchify_fn=args.data.batchify_val_fn,
            last_batch='keep',
            num_workers=args.data.num_workers)
        return train_data, val_data
    train_data, val_data = _get_dataloader()

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
    metric = get_metric_instance(args.metric, 0.5, args.data.val.classes)

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
        net.hybridize(static_alloc=True, static_shape=True)
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = L(
                    cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)
            trainer.step(1)

            if _demo_early_stopping(i):
                break
        mx.nd.waitall()
<<<<<<< HEAD
=======
        reporter.save_dict(epoch=epoch, params=net.collect_params())
>>>>>>> awslabs/master

    def test(epoch):
        metric.reset()
        net.set_nms(nms_thresh=0.45, nms_topk=400)
        net.hybridize(static_alloc=True, static_shape=True)
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0,
                                              even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0,
                                               even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(
                    y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
            metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
            if _demo_early_stopping(i):
                break
        map_name, mean_ap = metric.get()
        # TODO (cgraywang): add ray
        reporter(epoch=epoch, map=float(mean_ap[-1]))

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if hasattr(args, 'val_interval'):
            if epoch % args.val_interval == 0:
                test(epoch)
        else:
            test(epoch)
