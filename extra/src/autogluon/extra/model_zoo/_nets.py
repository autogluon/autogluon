import logging
import mxnet as mx
from mxnet import gluon, init

from .model_zoo import get_model

logger = logging.getLogger(__name__)

__all__ = ['get_network', 'get_built_in_network']


# TODO: Consider deleting this file, it is only used in a single tutorial and shouldn't be a part the codebase
def get_network(net, **kwargs):
    if type(net) == str:
        net = get_built_in_network(net, **kwargs)
    else:
        net.initialize(ctx=kwargs['ctx'])
    return net


def get_built_in_network(name, *args, **kwargs):
    def _get_finetune_network(model_name, num_classes, ctx, **kwargs):
        kwargs['pretrained'] = True
        finetune_net = get_model(model_name, **kwargs)
        # change the last fully connected layer to match the number of classes
        with finetune_net.name_scope():
            if hasattr(finetune_net, 'output'):
                finetune_net.output = gluon.nn.Dense(num_classes)
                finetune_net.output.initialize(init.Xavier(), ctx=ctx)
            elif hasattr(finetune_net, '_fc'):
                finetune_net._fc = gluon.nn.Dense(num_classes)
                finetune_net._fc.initialize(init.Xavier(), ctx=ctx)
            else:
                assert hasattr(finetune_net, 'fc')
                finetune_net.fc = gluon.nn.Dense(num_classes)
                finetune_net.fc.initialize(init.Xavier(), ctx=ctx)
        # initialize and context
        finetune_net.collect_params().reset_ctx(ctx)
        # finetune_net.load_parameters(opt.resume_params, ctx=context, cast_dtype=True)
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
