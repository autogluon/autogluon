import mxnet as mx
from mxnet import gluon, init
from gluoncv.model_zoo import get_model
import gluoncv as gcv

from ...core import *

def get_built_in_network(name, *args, **kwargs):

    def _get_network(name, transfer_classes, ctx=mx.cpu()):
        name = name.lower()
        net_name = '_'.join(('yolo3', name, 'custom'))
        net = gcv.model_zoo.get_model(net_name, 
                                  classes=transfer_classes,
                                  pretrained_base=False, 
                                  transfer='coco')
        net.initialize(ctx=ctx, force_reinit=True)
        return net

    name = name.lower()
    return _get_network(name, *args, **kwargs)