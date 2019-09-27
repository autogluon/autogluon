import ConfigSpace as CS
import mxnet as mx
from gluoncv.model_zoo import get_model

from ..core import *
from ..basic import *

from ..basic.decorators import autogluon_function

@autogluon_function()
def get_built_in_network(name, *args, **kwargs):
    def _get_finetune_network(model_name, num_classes, ctx):
        finetune_net = get_model(model_name, pretrained=pretrained)
        # change the last fully connected layer to match the number of classes
        with finetune_net.name_scope():
            finetune_net.output = gluon.nn.Dense(num_classes)
        # initialize and context
        finetune_net.output.initialize(init.Xavier(), ctx=ctx)
        finetune_net.collect_params().reset_ctx(ctx)
        finetune_net.hybridize()
        return finetune_net

    def _get_cifar_netwrok(name, ctx=mx.cpu(), *args, **kwargs):
        name = name.lower()
        assert 'cifar' in name
        net = get_model(name, *args, **kwargs)
        net.initialize(ctx=ctx)
        return net

    name = name.lower()
    if 'cifar' in name:
        return _get_cifar_netwrok(name, *args, **kwargs)
    else:
        return _get_finetune_network(name, *args, **kwargs)
