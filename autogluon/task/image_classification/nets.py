import ConfigSpace as CS
import mxnet as mx
<<<<<<< HEAD
from mxnet import gluon, init
=======
>>>>>>> c8b325866201574caeb688c623d02b23799a65fc
from gluoncv.model_zoo import get_model

from ...core import *

@autogluon_function()
def get_built_in_network(name, *args, **kwargs):
    def _get_finetune_network(model_name, num_classes, ctx):
<<<<<<< HEAD
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
=======
        finetune_net = get_model(model_name, pretrained=pretrained)
        # change the last fully connected layer to match the number of classes
        with finetune_net.name_scope():
            finetune_net.output = gluon.nn.Dense(num_classes)
        # initialize and context
        finetune_net.output.initialize(init.Xavier(), ctx=ctx)
>>>>>>> c8b325866201574caeb688c623d02b23799a65fc
        finetune_net.collect_params().reset_ctx(ctx)
        finetune_net.hybridize()
        return finetune_net

<<<<<<< HEAD
    def _get_cifar_netwrok(name, num_classes, ctx=mx.cpu(), *args, **kwargs):
=======
    def _get_cifar_netwrok(name, ctx=mx.cpu(), *args, **kwargs):
>>>>>>> c8b325866201574caeb688c623d02b23799a65fc
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
