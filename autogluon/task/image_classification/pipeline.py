import warnings
import logging

import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon, init, autograd, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model

from .metrics import get_metric_instance
from ...core.optimizer import SGD, NAG
from ...core import *
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ...utils import tqdm
from ...utils.mxutils import collect_params
from .nets import get_network
from .utils import *

__all__ = ['train_image_classification']


lr_schedulers = {
    'poly': mx.lr_scheduler.PolyScheduler,
    'cosine': mx.lr_scheduler.CosineScheduler
}

@args()
def train_image_classification(args, reporter):
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    if args.verbose:
        logger.setLevel(logging.INFO)
        logger.info(args)
    batch_size = args.batch_size * max(args.num_gpus, 1)
    ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]

    num_classes = args.dataset.num_classes if hasattr(args.dataset, 'num_classes') else None
    net = get_network(args.net, num_classes, ctx)
    if args.hybridize:
        net.hybridize(static_alloc=True, static_shape=True)

    input_size = net.input_size if hasattr(net, 'input_size') else args.input_size
    train_data, val_data, batch_fn, num_batches = get_data_loader(
            args.dataset, input_size, batch_size, args.num_workers, args.final_fit,
            args.split_ratio)
 
    if isinstance(args.lr_scheduler, str):
        lr_scheduler = lr_schedulers[args.lr_scheduler](num_batches * args.epochs,
                                                        base_lr=args.optimizer.lr)
    else:
        lr_scheduler = args.lr_scheduler
    args.optimizer.lr_scheduler = lr_scheduler
    trainer = gluon.Trainer(net.collect_params(), args.optimizer)

    metric = get_metric_instance(args.metric)
    def train(epoch):
        for i, batch in enumerate(train_data):
            default_train_fn(net, batch, batch_size, args.loss, trainer, batch_fn, ctx)
            mx.nd.waitall()

    def test(epoch):
        metric.reset()
        for i, batch in enumerate(val_data):
            default_val_fn(net, batch, batch_fn, metric, ctx)
        _, reward = metric.get()
        reporter(epoch=epoch, classification_reward=reward)
        return reward

    tbar = tqdm(range(1, args.epochs + 1))
    for epoch in tbar:
        train(epoch)
        if not args.final_fit:
            reward = test(epoch)
            tbar.set_description('[Epoch {}] Validation: {:.3f}'.format(epoch, reward))

    if args.final_fit:
        return {'model_params': collect_params(net),
                'num_classes': num_classes}
