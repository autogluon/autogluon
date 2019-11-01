import os
import collections
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock

from ...nas.models.utils import *
from ...nas.models.mbconv import *
from ...nas.models.efficientnet import get_efficientnet_blockargs
from .wire import Wire_Stage, Wire_Sequential

__all__ = ['WireEfficientNet', 'get_wire_efficientnet_b0', 'get_wire_efficientnet_b1',
           'get_wire_efficientnet_b2']


class WireEfficientNet(HybridBlock):
    def __init__(self, blocks_args=None, dropout_rate=0.2, num_classes=1000, width_coefficient=1.0,
                 depth_coefficient=1.0, depth_divisor=8, min_depth=None, drop_connect_rate=0.2,
                 input_size=224, **kwargs):
        r"""WireEfficientNet model

            Parameters
            ----------
            blocks_args: nametuple, it concludes the hyperparameters of the MBConvBlock block.
            dropout_rate: float, rate of hidden units to drop.
            num_classes: int, number of output classes.
            width_coefficient:float, coefficient of the filters used for
            expanding or reducing the channels.
            depth_coefficient:float, it is used for repeat the EfficientNet Blocks.
            depth_divisor:int , it is used for reducing the number of filters.
            min_depth: int, used for deciding the minimum depth of the filters.
            drop_connect_rate: used for dropout.
        """
        super(WireEfficientNet, self).__init__(**kwargs)
        if blocks_args is None:
            blocks_args = get_efficientnet_blockargs()
        else:
            assert isinstance(blocks_args, list), 'blocks_args should be a list'
            assert len(blocks_args) > 0, 'block args must be greater than 0'
        self.input_size = input_size
        with self.name_scope():
            self.features = nn.HybridSequential()
            with self.features.name_scope():
                out_channels = round_filters(32, width_coefficient,
                                             depth_divisor, min_depth)
                _add_conv(self.features, out_channels, kernel=3,
                    stride=2, activation='swish', batchnorm=True,
                    input_size=input_size, in_channels=3)

            input_size = _update_input_size(input_size, 2)
            stages = []
            for block_arg in blocks_args:
                stage = []
                block_arg.update(
                    in_channels=out_channels,
                    channels=round_filters(block_arg.channels, width_coefficient,
                                           depth_divisor, min_depth),
                    num_repeat=round_repeats(
                        block_arg.num_repeat, depth_coefficient),
                        input_size=input_size)
                stage.append(MBConvBlock(**block_arg))

                out_channels=block_arg.channels
                input_size = _update_input_size(input_size, block_arg.stride)

                if block_arg.num_repeat > 1:
                    block_arg.update(
                        in_channels=out_channels, stride=1,
                        input_size=input_size)

                for _ in range(block_arg.num_repeat - 1):
                    stage.append(MBConvBlock(**block_arg))
                stages.append(Wire_Stage(stage))

            self._blocks = Wire_Sequential(stages)
            # Head
            hidden_channels = round_filters(1280, width_coefficient,
                                         depth_divisor, min_depth)
            self._conv_head = nn.HybridSequential()
            with self._conv_head.name_scope():
                _add_conv(self._conv_head, hidden_channels,
                          activation='swish', batchnorm=True, input_size=input_size,
                          in_channels=out_channels)
            out_channels = hidden_channels
            # Final linear layer
            self._dropout = dropout_rate
            self._fc = nn.Dense(num_classes, use_bias=True, in_units=out_channels)

    def sample(self, **config):
        return self._blocks.sample(**config)

    @property
    def kwspaces(self):
        return self._blocks.kwspaces

    @property
    def graph(self):
        return self._blocks.graph

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self._blocks(x)
        x = self._conv_head(x)
        x = F.contrib.AdaptiveAvgPooling2D(x, 1)
        if self._dropout:
            x = F.Dropout(x, self._dropout)
        x = self._fc(x)
        return x

def get_wire_efficientnet_b0(width_coefficient=1.0, depth_coefficient=1.0, input_size=224, **kwargs):
    model = WireEfficientNet(width_coefficient=width_coefficient,
                             depth_coefficient=depth_coefficient,
                             input_size=input_size, **kwargs)
    return model
def get_wire_efficientnet_b1(width_coefficient=1.0, depth_coefficient=1.1, input_size=240, **kwargs):
    model = WireEfficientNet(width_coefficient=width_coefficient,
                             depth_coefficient=depth_coefficient,
                             input_size=input_size, **kwargs)
    return model
def get_wire_efficientnet_b2(width_coefficient=1.1, depth_coefficient=1.2, input_size=260, **kwargs):
    model = WireEfficientNet(width_coefficient=width_coefficient,
                             depth_coefficient=depth_coefficient,
                             input_size=input_size, **kwargs)
    return model

