import logging
import mxnet as mx
from mxnet import gluon, init
from mxnet.gluon import nn

from ...model_zoo.model_zoo import get_model
from ...core import *

logger = logging.getLogger(__name__)

__all__ = ['Ensemble', 'get_network', 'auto_suggest_network']

class Ensemble(object):
    def __init__(self, model_list):
        self.model_list = model_list

    def __call__(self, *inputs):
        outputs = [model(*inputs) for model in self.model_list]
        output = outputs[0].exp()
        for i in range(1, len(outputs)):
            output += outputs[i].exp()

        output /= len(outputs)
        return output.log()

class Identity(mx.gluon.HybridBlock):
    def hybrid_forward(self, F, x):
        return x

class ConvBNReLU(mx.gluon.HybridBlock):
    def __init__(self, in_channels, channels, kernel, stride):
        super().__init__()
        padding = (kernel - 1) // 2
        self.conv = nn.Conv2D(channels, kernel, stride, padding, in_channels=in_channels)
        self.bn = nn.BatchNorm(in_channels=channels)
        self.relu = nn.Activation('relu')
    def hybrid_forward(self, F, x):
        return self.relu(self.bn(self.conv(x)))

class ResUnit(mx.gluon.HybridBlock):
    def __init__(self, in_channels, channels, hidden_channels, kernel, stride):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, hidden_channels, kernel, stride)
        self.conv2 = ConvBNReLU(hidden_channels, channels, kernel, 1)
        if in_channels == channels and stride == 1:
            self.shortcut = Identity()
        else:
            self.shortcut = nn.Conv2D(channels, 1, stride, in_channels=in_channels)
    def hybrid_forward(self, F, x):
        return self.conv2(self.conv1(x)) + self.shortcut(x)

@func()
def mnist_net():
    mnist_net = gluon.nn.Sequential()
    mnist_net.add(ResUnit(1, 8, hidden_channels=8, kernel=3, stride=2))
    mnist_net.add(ResUnit(8, 8, hidden_channels=8, kernel=5, stride=2))
    mnist_net.add(ResUnit(8, 16, hidden_channels=8, kernel=3, stride=2))
    mnist_net.add(nn.GlobalAvgPool2D())
    mnist_net.add(nn.Flatten())
    mnist_net.add(nn.Activation('relu'))
    mnist_net.add(nn.Dense(10, in_units=16))
    return mnist_net

def auto_suggest_network(dataset, net):
    if isinstance(dataset, str):
        dataset_name = dataset
    elif isinstance(dataset, AutoGluonObject):
        if 'name' in dataset.kwargs and dataset.kwargs['name'] is not None:
            dataset_name = dataset.kwargs['name']
        else:
            return net
    else:
        return net
    dataset_name = dataset_name.lower()
    if 'mnist' in dataset_name:
        if isinstance(net, str) or isinstance(net, Categorical):
            net = mnist_net()
            logger.info('Auto suggesting network net for dataset {}'.format(net, dataset_name))
            return net
    elif 'cifar' in dataset_name:
        if isinstance(net, str):
            if 'cifar' not in net:
                net = 'cifar_resnet20_v1'
        elif isinstance(net, Categorical):
            newdata = []
            for x in net.data:
                if 'cifar' in x:
                    newdata.append(x)
            net.data = newdata if len(newdata) > 0 else ['cifar_resnet20_v1', 'cifar_resnet56_v1']
        logger.info('Auto suggesting network net for dataset {}'.format(net, dataset_name))
        return net

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
