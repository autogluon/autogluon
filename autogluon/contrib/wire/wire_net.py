import os
import collections
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock

from ...nas.models.utils import *
from ...nas.models.mbconv import *
from .wire import Wire_Stage, Wire_Sequential

__all__ = ['WireNet']

def gcd(x, y):
    while y != 0:
        (x, y) = (y, x % y)
    return x

class ReLUConvBN(mx.gluon.HybridBlock):
    def __init__(self, in_channels, channels, kernel, stride=1, groups=None):
        super().__init__()
        groups = groups if groups else gcd(in_channels, channels)
        padding = (kernel - 1) // 2
        self.relu = nn.Activation('relu')
        self.conv = nn.Conv2D(channels, kernel, stride, padding, in_channels=in_channels, groups=groups)
        self.bn = nn.BatchNorm(in_channels=channels)

    def hybrid_forward(self, F, x):
        return self.bn(self.conv(self.relu(x)))

class ConvBNReLU(ReLUConvBN):
    def hybrid_forward(self, F, x):
        return self.relu(self.bn(self.conv(x)))

class WireNet(mx.gluon.HybridBlock):
    def __init__(self, blocks_args=None, num_classes=1000, dropout_rate=0.2,
                 **kwargs):
        r"""WireNet model

        Args:
            blocks_args (list of autogluon.Dict)
        """
        super(WireNet, self).__init__(**kwargs)
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self.conv1 = ConvBNReLU(3, 32, 3, 2, groups=1)

        stages = []
        for block_arg in blocks_args:
            stage = []
            num_repeat = block_arg.pop('num_repeat')
            # add the first module
            stage.append(ReLUConvBN(**block_arg))

            block_arg.update(stride=1, in_channels=block_arg.channels)
            for _ in range(num_repeat - 1):
                stage.append(ReLUConvBN(**block_arg))
            stages.append(Wire_Stage(stage))
            out_channels = block_arg.channels

        self.blocks = Wire_Sequential(stages)
        self.conv6 = ReLUConvBN(out_channels, 1280, 1, 1, groups=1)
        self._dropout = dropout_rate
        self.fc = nn.Dense(num_classes, use_bias=True, in_units=1280)

    @property
    def graph(self):
        return self.blocks.graph

    @property
    def sample(self, **config):
        return self.blocks.sample(**config)

    @property
    def kwspaces(self):
        return self.blocks.kwspaces

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv6(x)

        x = F.contrib.AdaptiveAvgPooling2D(x, 1)
        if self._dropout:
            x = F.Dropout(x, self._dropout)
        x = self.fc(x)
        return x
