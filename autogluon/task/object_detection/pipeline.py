import warnings
import logging
import os
import time

import numpy as np
import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon, init, autograd, nd
from mxnet.gluon.data.vision import transforms
import gluoncv as gcv
from gluoncv.model_zoo import get_model
from gluoncv import utils as gutils

from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils import LRScheduler, LRSequential

#TODO: move it to general util.py
from ..image_classification.utils import _train_val_split 



from ...core import *
from ...utils.mxutils import collect_params
from ...utils import tqdm

def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, args):
    """Get dataloader."""
    # when it is not in final_fit stage and val_dataset is not provided, we randomly sample (1 - args.split_ratio) data as our val_dataset
    if (not args.final_fit) and (not val_dataset):
        train_dataset, val_dataset = _train_val_split(train_dataset, args.split_ratio)

    width, height = data_shape, data_shape
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))  # stack image, all targets generated
    if args.no_random_shape:
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(YOLO3DefaultTrainTransform(width, height, net, mixup=args.mixup)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    else:
        transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, net, mixup=args.mixup) for x in range(10, 20)]
        train_loader = RandomTransformDataLoader(
            transform_fns, train_dataset, batch_size=batch_size, interval=10, last_batch='rollover',
            shuffle=True, batchify_fn=batchify_fn, num_workers=num_workers)

    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = None
    if val_dataset:
        val_loader = gluon.data.DataLoader(
            val_dataset.transform(YOLO3DefaultValTransform(width, height)),
            batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader

def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()


def train(net, train_data, val_data, eval_metric, ctx, args, reporter, final_fit):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    if args.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    if args.label_smooth:
        net._target_generator._label_smooth = True

    if args.lr_decay_period > 0:
        lr_decay_epoch = list(range(args.lr_decay_period, args.epochs, args.lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
    lr_decay_epoch = [e - args.warmup_epochs for e in lr_decay_epoch]
    num_batches = args.num_samples // args.batch_size
    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=args.lr,
                    nepochs=args.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(args.lr_mode, base_lr=args.lr,
                    nepochs=args.epochs - args.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=args.lr_decay, power=2),
    ])

    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler},
        kvstore='local')

    # targets
    sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    l1_loss = gluon.loss.L1Loss()

    # metrics
    obj_metrics = mx.metric.Loss('ObjLoss')
    center_metrics = mx.metric.Loss('BoxCenterLoss')
    scale_metrics = mx.metric.Loss('BoxScaleLoss')
    cls_metrics = mx.metric.Loss('ClassLoss')

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    #logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]

    pre_current_map = 0
    for epoch in range(args.start_epoch, args.epochs):
        #tbar2.next()
        if args.mixup:
            # TODO(zhreshold): more elegant way to control mixup during runtime
            try:
                train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
            except AttributeError:
                train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
            if epoch >= args.epochs - args.no_mixup_epochs:
                try:
                    train_data._dataset.set_mixup(None)
                except AttributeError:
                    train_data._dataset._data.set_mixup(None)

        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        net.hybridize()
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            # objectness, center_targets, scale_targets, weights, class_targets
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
            gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
            sum_losses = []
            obj_losses = []
            center_losses = []
            scale_losses = []
            cls_losses = []
            with autograd.record():
                for ix, x in enumerate(data):
                    obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                    sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                    obj_losses.append(obj_loss)
                    center_losses.append(center_loss)
                    scale_losses.append(scale_loss)
                    cls_losses.append(cls_loss)
                autograd.backward(sum_losses)
            trainer.step(batch_size)
            obj_metrics.update(0, obj_losses)
            center_metrics.update(0, center_losses)
            scale_metrics.update(0, scale_losses)
            cls_metrics.update(0, cls_losses)
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                name4, loss4 = cls_metrics.get()
                logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, trainer.learning_rate, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
            btic = time.time()

        name1, loss1 = obj_metrics.get()
        name2, loss2 = center_metrics.get()
        name3, loss3 = scale_metrics.get()
        name4, loss4 = cls_metrics.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
        if (not (epoch + 1) % args.val_interval) and not final_fit:
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = ' '.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            #$tbar.set_description('[Epoch {}] Validation: {}'.format(epoch, val_msg))
            logger.info('[Epoch {}] Validation: {}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
            pre_current_map = current_map
        else:
            current_map = pre_current_map
        reporter(epoch=epoch, map_reward=current_map)

@args()
def train_object_detection(args, reporter):
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]

    net_name = '_'.join(('yolo3', args.net, 'custom'))
    args.save_prefix += net_name

    # use sync bn if specified
    if args.syncbn and len(ctx) > 1:
        net = gcv.model_zoo.get_model(net_name, 
                                    classes=args.dataset.get_classes(),
                                    pretrained_base=False, 
                                    transfer=args.transfer,
                                    norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                    norm_kwargs={'num_devices': len(ctx)})
        if not args.reuse_pred_weights:
            net.reset_class(args.dataset.get_classes(), reuse_weights=None)

        async_net = gcv.model_zoo.get_model(net_name, 
                                            classes=args.dataset.get_classes(),
                                            pretrained_base=False, 
                                            transfer=args.transfer)
        if not args.reuse_pred_weights:
            async_net.reset_class(args.dataset.get_classes(), reuse_weights=None)
    else:
        net = gcv.model_zoo.get_model(net_name, 
                                    classes=args.dataset.get_classes(),
                                    pretrained_base=False, 
                                    transfer=args.transfer)
        if not args.reuse_pred_weights:
            net.reset_class(args.dataset.get_classes(), reuse_weights=None)
        async_net = net

    if args.resume.strip():
        net.load_parameters(args.resume.strip())
        async_net.load_parameters(args.resume.strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()


    # training data
    train_dataset, eval_metric = args.dataset.get_dataset_and_metric()
    train_data, val_data = get_dataloader(
        async_net, train_dataset, None, args.data_shape, args.batch_size, args.num_workers, args)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args, reporter, args.final_fit)

    if args.final_fit:
        return {'model_params': collect_params(net)}
