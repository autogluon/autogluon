import logging
import os
import time
import warnings

import gluoncv as gcv
import mxnet as mx
import numpy as np
from gluoncv import utils as gutils
from gluoncv.data.batchify import FasterRCNNTrainBatchify, Tuple, Append
from gluoncv.data.batchify import Stack, Pad
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform, \
    FasterRCNNDefaultValTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform, \
    YOLO3DefaultValTransform
from gluoncv.utils.parallel import Parallel
from mxnet import gluon, autograd

from .data_parallel import ForwardBackwardTask
from .utils import get_lr_scheduler, get_yolo3_metrics, get_faster_rcnn_metrics, get_rcnn_losses, \
    rcnn_split_and_load
# TODO: move it to general util.py
from ..image_classification.utils import _train_val_split
from ...core import args
from ...utils.mxutils import collect_params


def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, args):
    """Get dataloader."""
    # when it is not in final_fit stage and val_dataset is not provided, we randomly
    # sample (1 - args.split_ratio) data as our val_dataset
    if (not args.final_fit) and (not val_dataset):
        train_dataset, val_dataset = _train_val_split(train_dataset, args.split_ratio)

    width, height = data_shape, data_shape
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(
        1)]))  # stack image, all targets generated
    if args.no_random_shape:
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(
                YOLO3DefaultTrainTransform(width, height, net, mixup=args.mixup)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover',
            num_workers=num_workers)
    else:
        transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, net, mixup=args.mixup) for x in
                         range(10, 20)]
        train_loader = RandomTransformDataLoader(
            transform_fns, train_dataset, batch_size=batch_size, interval=10, last_batch='rollover',
            shuffle=True, batchify_fn=batchify_fn, num_workers=num_workers)

    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = None
    if val_dataset:
        val_loader = gluon.data.DataLoader(
            val_dataset.transform(YOLO3DefaultValTransform(width, height)),
            batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep',
            num_workers=num_workers)
    return train_loader, val_loader


def get_faster_rcnn_dataloader(net, train_dataset, val_dataset, train_transform, val_transform,
                               batch_size, num_shards, args):
    """Get faster rcnn dataloader."""
    if (not args.final_fit) and (not val_dataset):
        train_dataset, val_dataset = _train_val_split(train_dataset, args.split_ratio)

    train_bfn = FasterRCNNTrainBatchify(net, num_shards)
    if hasattr(train_dataset, 'get_im_aspect_ratio'):
        im_aspect_ratio = train_dataset.get_im_aspect_ratio()
    else:
        im_aspect_ratio = [1.] * len(train_dataset)
    train_sampler = \
        gcv.nn.sampler.SplitSortedBucketSampler(im_aspect_ratio, batch_size, num_parts=1,
                                                part_index=0, shuffle=True)
    train_loader = gluon.data.DataLoader(train_dataset.transform(
        train_transform(net.short, net.max_size, net, ashape=net.ashape, multi_stage=True)),
        batch_sampler=train_sampler, batchify_fn=train_bfn, num_workers=args.num_workers)
    val_bfn = Tuple(*[Append() for _ in range(3)])
    short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
    # validation use 1 sample per device
    val_loader = None
    if val_dataset:
        val_loader = gluon.data.DataLoader(
            val_dataset.transform(val_transform(short, net.max_size)), num_shards, False,
            batchify_fn=val_bfn, last_batch='keep', num_workers=args.num_workers)
    args.num_samples = len(train_dataset)
    return train_loader, val_loader


def validate(net, val_data, ctx, eval_metric, args):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    if args.meta_arch == 'yolo3':
        net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    for batch in val_data:
        if args.meta_arch == 'yolo3':
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0,
                                              even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0,
                                               even_split=False)
            split_batch = data, label
        elif args.meta_arch == 'faster_rcnn':
            split_batch = rcnn_split_and_load(batch, ctx_list=ctx)
            clipper = gcv.nn.bbox.BBoxClipToImage()
        else:
            raise NotImplementedError('%s not implemented.' % args.meta_arch)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for data in zip(*split_batch):
            if args.meta_arch == 'yolo3':
                x, y = data
            elif args.meta_arch == 'faster_rcnn':
                x, y, im_scale = data
            else:
                raise NotImplementedError('%s not implemented.' % args.meta_arch)
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            if args.meta_arch == 'yolo3':
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            elif args.meta_arch == 'faster_rcnn':
                det_bboxes.append(clipper(bboxes, x))
                # rescale to original resolution
                im_scale = im_scale.reshape((-1)).asscalar()
                det_bboxes[-1] *= im_scale
            else:
                raise NotImplementedError('%s not implemented.' % args.meta_arch)
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            if args.meta_arch == 'faster_rcnn':
                gt_bboxes[-1] *= im_scale
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        if args.meta_arch == 'yolo3':
            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        elif args.meta_arch == 'faster_rcnn':
            for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids,
                                                                            det_scores, gt_bboxes,
                                                                            gt_ids, gt_difficults):
                eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
        else:
            raise NotImplementedError('%s not implemented.' % args.meta_arch)
    return eval_metric.get()


def train(net, train_data, val_data, eval_metric, ctx, args, reporter, final_fit):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    if args.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0
    if args.meta_arch == 'faster_rcnn':
        net.collect_params().setattr('grad_req', 'null')
        net.collect_train_params().setattr('grad_req', 'write')

    if args.label_smooth:
        net._target_generator._label_smooth = True

    lr_scheduler = get_lr_scheduler(args)

    trainer = gluon.Trainer(
        net.collect_train_params() if args.meta_arch == 'faster_rcnn' else net.collect_params(),
        'sgd', {'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler},
        kvstore='nccl')

    # metrics & losses
    if args.meta_arch == 'yolo3':
        raw_metrics, metrics = get_yolo3_metrics()
    elif args.meta_arch == 'faster_rcnn':
        raw_metrics, metrics = get_faster_rcnn_metrics()
        rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss = get_rcnn_losses(args)
    else:
        raise NotImplementedError('%s not implemented.' % args.meta_arch)

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
    # logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]

    if args.meta_arch == 'faster_rcnn':
        rcnn_task = ForwardBackwardTask(net, trainer, rpn_cls_loss, rpn_box_loss, rcnn_cls_loss,
                                        rcnn_box_loss, mix_ratio=1.0, enable_amp=False)
        executor = Parallel(args.num_gpus // 2, rcnn_task)

    pre_current_map = 0
    for epoch in range(args.start_epoch, args.epochs):
        mix_ratio = 1.0
        net.hybridize()
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
        for metric in raw_metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        for i, batch in enumerate(train_data):
            batch_size = args.batch_size
            raw_metrics_values = [[] for _ in raw_metrics]
            metrics_values = [[] for _ in metrics]
            if args.meta_arch == 'yolo3':
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                # objectness, center_targets, scale_targets, weights, class_targets
                fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0)
                                 for it in range(1, 6)]
                gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
                sum_losses = []
                with autograd.record():
                    for ix, x in enumerate(data):
                        obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix],
                                                                          *[ft[ix] for ft in
                                                                            fixed_targets])
                        sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                        result = obj_loss, center_loss, scale_loss, cls_loss
                        for k in range(len(raw_metrics_values)):
                            raw_metrics_values[k].append(result[k])
                    autograd.backward(sum_losses)
            elif args.meta_arch == 'faster_rcnn':
                batch = rcnn_split_and_load(batch, ctx_list=ctx)
                if executor is not None:
                    for data in zip(*batch):
                        executor.put(data)
                for j in range(len(ctx)):
                    if executor is not None:
                        result = executor.get()
                    else:
                        result = rcnn_task.forward_backward(list(zip(*batch))[0])
                    for k in range(len(raw_metrics_values)):
                        raw_metrics_values[k].append(result[k])
                    for k in range(len(metrics_values)):
                        metrics_values[k].append(result[len(metrics_values) + k])
            else:
                raise NotImplementedError('%s not implemented.' % args.meta_arch)
            trainer.step(batch_size)

            for metric, record in zip(raw_metrics, raw_metrics_values):
                metric.update(0, record)
            for metric, records in zip(metrics, metrics_values):
                for pred in records:
                    metric.update(pred[0], pred[1])
            if args.log_interval and not (i + 1) % args.log_interval:
                msg = ','.join(
                    ['{}={:.3f}'.format(*metric.get()) for metric in raw_metrics + metrics])
                logger.info(
                    '[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}'.format(
                        epoch, i, trainer.learning_rate, batch_size / (time.time() - btic), msg))
            btic = time.time()

        msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in raw_metrics])
        logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
            epoch, (time.time() - tic), msg))
        if (not (epoch + 1) % args.val_interval) and not final_fit:
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric, args)
            val_msg = ' '.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            # $tbar.set_description('[Epoch {}] Validation: {}'.format(epoch, val_msg))
            logger.info('[Epoch {}] Validation: {}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
            pre_current_map = current_map
        else:
            current_map = pre_current_map
        # Note: epoch reported back must start with 1, not 0
        reporter(epoch=epoch+1, map_reward=current_map)


@args()
def train_object_detection(args, reporter):
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
    if args.meta_arch == 'yolo3':
        net_name = '_'.join((args.meta_arch, args.net, 'custom'))
        kwargs = {}
    elif args.meta_arch == 'faster_rcnn':
        net_name = '_'.join(('custom', args.meta_arch, 'fpn'))
        kwargs = {'base_network_name': args.net, 'short': args.data_shape, 'max_size': 1000,
                  'nms_thresh': 0.5, 'nms_topk': -1, 'min_stage': 2, 'max_stage': 6, 'post_nms': -1,
                  'roi_mode': 'align', 'roi_size': (7, 7), 'strides': (4, 8, 16, 32, 64),
                  'clip': 4.14, 'rpn_channel': 256, 'base_size': 16, 'scales': (2, 4, 8, 16, 32),
                  'ratios': (0.5, 1, 2), 'alloc_size': (384, 384), 'rpn_nms_thresh': 0.7,
                  'rpn_train_pre_nms': 12000, 'rpn_train_post_nms': 2000, 'rpn_test_pre_nms': 6000,
                  'rpn_test_post_nms': 1000, 'rpn_min_size': 1,
                  'per_device_batch_size': args.batch_size // args.num_gpus, 'num_sample': 512,
                  'pos_iou_thresh': 0.5, 'pos_ratio': 0.25, 'max_num_gt': 100}
    else:
        raise NotImplementedError(args.meta_arch, 'is not implemented.')
    args.save_prefix += net_name

    # use sync bn if specified
    if args.syncbn and len(ctx) > 1:
        net = gcv.model_zoo.get_model(net_name,
                                      classes=args.dataset.get_classes(),
                                      pretrained_base=True,
                                      transfer=args.transfer,
                                      norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                      norm_kwargs={'num_devices': len(ctx)}, **kwargs)
        if not args.reuse_pred_weights:
            net.reset_class(args.dataset.get_classes(), reuse_weights=None)
        if args.meta_arch == 'yolo3':
            async_net = gcv.model_zoo.get_model(net_name,
                                                classes=args.dataset.get_classes(),
                                                pretrained_base=True,
                                                transfer=args.transfer, **kwargs)
            if not args.reuse_pred_weights:
                async_net.reset_class(args.dataset.get_classes(), reuse_weights=None)
    else:
        net = gcv.model_zoo.get_model(net_name,
                                      classes=args.dataset.get_classes(),
                                      pretrained_base=True,
                                      transfer=args.transfer, **kwargs)
        if not args.reuse_pred_weights:
            net.reset_class(args.dataset.get_classes(), reuse_weights=None)
        async_net = net

    if args.resume.strip():
        net.load_parameters(args.resume.strip())
        if args.meta_arch == 'yolo3':
            async_net.load_parameters(args.resume.strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            if args.meta_arch == 'yolo3':
                async_net.initialize()

    # training data
    train_dataset, eval_metric = args.dataset.get_dataset_and_metric()
    if args.meta_arch == 'yolo3':
        train_data, val_data = get_dataloader(
            async_net, train_dataset, None, args.data_shape, args.batch_size, args.num_workers,
            args)
    elif args.meta_arch == 'faster_rcnn':
        train_data, val_data = get_faster_rcnn_dataloader(
            net, train_dataset, None, FasterRCNNDefaultTrainTransform,
            FasterRCNNDefaultValTransform, args.batch_size, args.num_gpus, args)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args, reporter, args.final_fit)

    if args.final_fit:
        return {'model_params': collect_params(net)}
