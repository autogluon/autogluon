import ConfigSpace as CS
import mxnet as mx
from mxnet import gluon, init
from gluoncv.model_zoo import get_model

from ...core import *

@autogluon_function()
def get_built_in_network(name, *args, **kwargs):
    def _get_finetune_network(model_name, num_classes, ctx):
        finetune_net = get_model(model_name, pretrained=True)
        # change the last fully connected layer to match the number of classes
        with finetune_net.name_scope():
            if hasattr(finetune_net, 'output'):
                finetune_net.output = gluon.nn.Dense(num_classes)
                finetune_net.output.initialize(init.Xavier(), ctx=ctx)
            else:
                assert hasattr(finetune_net, 'fc')
                finetune_net.fc = gluon.nn.Dense(num_classes)
                finetune_net.fc.initialize(init.Xavier(), ctx=ctx)
        # initialize and context
        finetune_net.collect_params().reset_ctx(ctx)
        finetune_net.hybridize()
        return finetune_net

    def _get_cifar_network(name, num_classes, ctx=mx.cpu(), *args, **kwargs):
        name = name.lower()
        assert 'cifar' in name
        net = get_model(name, *args, **kwargs)
        net.initialize(ctx=ctx)
        return net

    name = name.lower()
    if 'cifar' in name:
        return _get_cifar_network(name, *args, **kwargs)
    else:
        return _get_finetune_network(name, *args, **kwargs)
