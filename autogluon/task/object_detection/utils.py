import mxnet as mx
from gluoncv.utils import LRScheduler, LRSequential
from gluoncv.utils.metrics.rcnn import RPNAccMetric, RPNL1LossMetric, RCNNAccMetric, \
    RCNNL1LossMetric

from .nets import get_built_in_network


def get_network(meta_arch, net, transfer_classes, transfer=None, ctx=mx.cpu(), syncbn=False):
    if type(net) == str:
        net = get_built_in_network(meta_arch, net, transfer_classes, transfer, ctx=ctx,
                                   syncbn=syncbn)
    else:
        net.initialize(ctx=ctx)
    return net


def get_lr_scheduler(args):
    if args.lr_decay_period > 0:
        lr_decay_epoch = list(range(args.lr_decay_period, args.epochs, args.lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
    num_batches = args.num_samples // args.batch_size
    if args.meta_arch == 'yolo3':
        return LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=args.lr,
                        nepochs=args.warmup_epochs, iters_per_epoch=num_batches),
            LRScheduler(args.lr_mode, base_lr=args.lr, nepochs=args.epochs - args.warmup_epochs,
                        iters_per_epoch=num_batches, step_epoch=lr_decay_epoch,
                        step_factor=args.lr_decay, power=2)])
    elif args.meta_arch == 'faster_rcnn':
        return LRSequential([
            LRScheduler('linear', base_lr=args.lr * args.warmup_factor, target_lr=args.lr,
                        niters=args.warmup_iters, iters_per_epoch=num_batches),
            LRScheduler(args.lr_mode, base_lr=args.lr, nepochs=args.epochs,
                        iters_per_epoch=num_batches, step_epoch=lr_decay_epoch,
                        step_factor=args.lr_decay, power=2)])


def get_faster_rcnn_metrics():
    return [mx.metric.Loss('RPN_Conf'), mx.metric.Loss('RPN_SmoothL1'),
            mx.metric.Loss('RCNN_CrossEntropy'), mx.metric.Loss('RCNN_SmoothL1')], \
           [RPNAccMetric(), RPNL1LossMetric(), RCNNAccMetric(), RCNNL1LossMetric()]


def get_yolo3_metrics():
    return [mx.metric.Loss('ObjLoss'), mx.metric.Loss('BoxCenterLoss'),
            mx.metric.Loss('BoxScaleLoss'), mx.metric.Loss('ClassLoss')], \
           []


def get_rcnn_losses(args):
    return mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False), \
           mx.gluon.loss.HuberLoss(rho=0.001), \
           mx.gluon.loss.SoftmaxCrossEntropyLoss(), \
           mx.gluon.loss.HuberLoss(rho=0.01)


def rcnn_split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    new_batch = []
    for i, data in enumerate(batch):
        if isinstance(data, (list, tuple)):
            new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        else:
            new_data = [data.as_in_context(ctx_list[0])]
        new_batch.append(new_data)
    return new_batch
