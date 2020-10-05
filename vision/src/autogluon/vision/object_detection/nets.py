import gluoncv as gcv
import mxnet as mx
from mxnet import gluon


def get_built_in_network(meta_arch, name, *args, **kwargs):
    def _get_network(meta_arch, name, transfer_classes, transfer, ctx=mx.cpu(), syncbn=False):
        name = name.lower()
        if meta_arch == 'yolo3':
            net_name = '_'.join((meta_arch, name, 'custom'))
            net = gcv.model_zoo.get_model(net_name,
                                          classes=transfer_classes,
                                          pretrained_base=False,
                                          transfer=None)
            net.initialize(ctx=ctx, force_reinit=True)
        elif meta_arch == 'faster_rcnn':
            net_name = '_'.join(('custom', meta_arch, 'fpn'))
            kwargs = {'base_network_name': name, 'short': 600, 'max_size': 1000,
                      'nms_thresh': 0.5, 'nms_topk': -1, 'min_stage': 2, 'max_stage': 6,
                      'post_nms': -1, 'roi_mode': 'align', 'roi_size': (7, 7),
                      'strides': (4, 8, 16, 32, 64), 'clip': 4.14, 'rpn_channel': 256,
                      'base_size': 16, 'scales': (2, 4, 8, 16, 32), 'ratios': (0.5, 1, 2),
                      'alloc_size': (384, 384), 'rpn_nms_thresh': 0.7, 'rpn_train_pre_nms': 12000,
                      'rpn_train_post_nms': 2000, 'rpn_test_pre_nms': 6000,
                      'rpn_test_post_nms': 1000, 'rpn_min_size': 1, 'per_device_batch_size': 1,
                      'num_sample': 512, 'pos_iou_thresh': 0.5, 'pos_ratio': 0.25,
                      'max_num_gt': 100}
            if syncbn and len(ctx) > 1:
                net = gcv.model_zoo.get_model(net_name,
                                              classes=transfer_classes,
                                              pretrained_base=True, transfer=transfer,
                                              norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                              norm_kwargs={'num_devices': len(ctx)}, **kwargs)
            else:
                net = gcv.model_zoo.get_model(net_name, classes=transfer_classes,
                                              pretrained_base=False,
                                              transfer=transfer, **kwargs)
            net.initialize(ctx=ctx, force_reinit=True)
        else:
            raise NotImplementedError('%s not implemented.' % meta_arch)
        return net

    name = name.lower()
    return _get_network(meta_arch, name, *args, **kwargs)
