import logging

import mxnet as mx
from gluoncv.loss import DistillationSoftmaxCrossEntropyLoss
from mxnet import gluon, nd

from autogluon.mxnet.task.metrics import get_metric_instance
from autogluon.mxnet.task.utils import *
from autogluon.core.utils import tqdm
from autogluon.mxnet.utils import collect_params
import autogluon.core as ag
from .processing_params import Sample_params, Getmodel_kwargs
from ..utils.learning_rate import LR_params

__all__ = ['train_image_classification']


@ag.args()
def train_image_classification(args, reporter):
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    if args.verbose:
        logger.setLevel(logging.INFO)
        logger.info(args)

    target_params = Sample_params(args.batch_size, args.num_gpus, args.num_workers)
    batch_size = target_params.get_batchsize
    ctx = target_params.get_context
    classes = args.dataset.num_classes if hasattr(args.dataset, 'num_classes') else None
    target_kwargs = Getmodel_kwargs(ctx,
                                    classes,
                                    args.net,
                                    args.tricks.teacher_name,
                                    args.tricks.hard_weight,
                                    args.hybridize,
                                    args.optimizer.multi_precision,
                                    args.tricks.use_pretrained,
                                    args.tricks.use_gn,
                                    args.tricks.last_gamma,
                                    args.tricks.batch_norm,
                                    args.tricks.use_se)
    distillation = target_kwargs.distillation
    net = target_kwargs.get_net
    input_size = net.input_size if hasattr(net, 'input_size') else args.input_size

    if args.tricks.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    if args.tricks.label_smoothing or args.tricks.mixup:
        sparse_label_loss = False
    else:
        sparse_label_loss = True

    if distillation:
        teacher = target_kwargs.get_teacher

        def teacher_prob(data):
            return [
                nd.softmax(teacher(X.astype(target_kwargs.dtype, copy=False)) / args.tricks.temperature)
                for X in data
            ]

        L = DistillationSoftmaxCrossEntropyLoss(temperature=args.tricks.temperature,
                                                hard_weight=args.tricks.hard_weight,
                                                sparse_label=sparse_label_loss)
    else:
        L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)
        teacher_prob = None
    if args.tricks.mixup:
        metric = get_metric_instance('rmse')
    else:
        metric = get_metric_instance(args.metric)

    train_data, val_data, batch_fn, num_batches = get_data_loader(
        args.dataset, input_size, batch_size, args.num_workers, args.final_fit, args.split_ratio
    )

    if isinstance(args.lr_config.lr_mode, str):  # fix
        target_lr = LR_params(args.optimizer.lr, args.lr_config.lr_mode, args.epochs, num_batches,
                              args.lr_config.lr_decay_epoch,
                              args.lr_config.lr_decay,
                              args.lr_config.lr_decay_period,
                              args.lr_config.warmup_epochs,
                              args.lr_config.warmup_lr)
        lr_scheduler = target_lr.get_lr_scheduler
    else:
        lr_scheduler = args.lr_config.lr_mode
    args.optimizer.lr_scheduler = lr_scheduler

    trainer = gluon.Trainer(net.collect_params(), args.optimizer)

    def train(epoch, num_epochs, metric):
        for i, batch in enumerate(train_data):
            metric = default_train_fn(epoch, num_epochs, net, batch, batch_size, L, trainer,
                                      batch_fn, ctx, args.tricks.mixup, args.tricks.label_smoothing,
                                      distillation, args.tricks.mixup_alpha, args.tricks.mixup_off_epoch,
                                      classes, target_kwargs.dtype, metric, teacher_prob)
            mx.nd.waitall()
        return metric

    def test(epoch):
        metric.reset()
        for i, batch in enumerate(val_data):
            default_val_fn(net, batch, batch_fn, metric, ctx, target_kwargs.dtype)
        _, reward = metric.get()
        reporter(epoch=epoch, classification_reward=reward)
        return reward

    # Note: epoch must start with 1, not 0
    tbar = tqdm(range(1, args.epochs + 1))
    for epoch in tbar:
        metric = train(epoch, args.epochs, metric)
        train_metric_name, train_metric_score = metric.get()
        tbar.set_description(f'[Epoch {epoch}] training: {train_metric_name}={train_metric_score :.3f}')
        if not args.final_fit:
            reward = test(epoch)
            tbar.set_description(f'[Epoch {epoch}] Validation: {reward :.3f}')

    if args.final_fit:
        return {'model_params': collect_params(net), 'num_classes': classes}
