import os
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock

from ...utils import EasyDict
from .mbconv import *
from .utils import *

__all__ = ['EfficientNet', 'get_efficientnet_blockargs', 'get_efficientnet',
           'get_efficientnet_b0', 'get_efficientnet_b1', 'get_efficientnet_b2',
           'get_efficientnet_b3', 'get_efficientnet_b4', 'get_efficientnet_b5',
           'get_efficientnet_b6', 'get_efficientnet_b7']

class EfficientNet(HybridBlock):
    def __init__(self, blocks_args=None, dropout_rate=0.2, num_classes=1000, width_coefficient=1.0,
                 depth_coefficient=1.0, depth_divisor=8, min_depth=None, drop_connect_rate=0.2,
                 input_size=224, **kwargs):
        r"""EfficientNet model from the
            `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
            <https://arxiv.org/abs/1905.11946>`_ paper.

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
        super(EfficientNet, self).__init__(**kwargs)
        if blocks_args is None:
            blocks_args = get_efficientnet_blockargs()
        else:
            assert isinstance(blocks_args, list), 'blocks_args should be a list'
            assert len(blocks_args) > 0, 'block args must be greater than 0'
        self.input_size = input_size
        with self.name_scope():
            self.features = nn.HybridSequential()
            with self.features.name_scope():
                # stem conv
                out_channels = round_filters(32, width_coefficient,
                                             depth_divisor, min_depth)
                _add_conv(self.features, out_channels, kernel=3,
                    stride=2, activation='swish', batchnorm=True, input_size=input_size)

            input_size = _update_input_size(input_size, 2)

            self._blocks = nn.HybridSequential()
            with self._blocks.name_scope():
                for block_arg in blocks_args:
                    block_arg.update(
                        in_channels=out_channels,
                        channels=round_filters(block_arg.channels, width_coefficient,
                                               depth_divisor, min_depth),
                        num_repeat=round_repeats(
                            block_arg.num_repeat, depth_coefficient),
                            input_size=input_size)
                    self._blocks.add(MBConvBlock(**block_arg))

                    out_channels=block_arg.channels
                    input_size = _update_input_size(input_size, block_arg.stride)

                    if block_arg.num_repeat > 1:
                        block_arg.update(
                            in_channels=out_channels, stride=1,
                            input_size=input_size)

                    for _ in range(block_arg.num_repeat - 1):
                        self._blocks.add(MBConvBlock(**block_arg))

            # Head
            out_channels = round_filters(1280, width_coefficient,
                                         depth_divisor, min_depth)
            self._conv_head = nn.HybridSequential()
            with self._conv_head.name_scope():
                _add_conv(self._conv_head, out_channels,
                          activation='swish', batchnorm=True, input_size=input_size)
            # Final linear layer
            self._dropout = dropout_rate
            self._fc = nn.Dense(num_classes, use_bias=True)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self._blocks(x)
        x = self._conv_head(x)
        x = F.contrib.AdaptiveAvgPooling2D(x, 1)
        if self._dropout:
            x = F.Dropout(x, self._dropout)
        x = self._fc(x)
        return x


def get_efficientnet_blockargs():
    """ Creates a predefined efficientnet model, which searched by original paper. """
    blocks_args = [
        EasyDict(kernel=3, num_repeat=1, channels=16, expand_ratio=1, stride=1, se_ratio=0.25, in_channels=32),
        EasyDict(kernel=3, num_repeat=2, channels=24, expand_ratio=6, stride=2, se_ratio=0.25, in_channels=16),
        EasyDict(kernel=5, num_repeat=2, channels=40, expand_ratio=6, stride=2, se_ratio=0.25, in_channels=24),
        EasyDict(kernel=3, num_repeat=3, channels=80, expand_ratio=6, stride=2, se_ratio=0.25, in_channels=40),
        EasyDict(kernel=5, num_repeat=3, channels=112, expand_ratio=6, stride=1, se_ratio=0.25, in_channels=80),
        EasyDict(kernel=5, num_repeat=4, channels=192, expand_ratio=6, stride=2, se_ratio=0.25, in_channels=112),
        EasyDict(kernel=3, num_repeat=1, channels=320, expand_ratio=6, stride=1, se_ratio=0.25, in_channels=192),
    ]
    return blocks_args

def get_efficientnet(dropout_rate=None, num_classes=None, width_coefficient=None, depth_coefficient=None,
                     depth_divisor=None, min_depth=None, drop_connect_rate=None, input_size=224, **kwargs):

    blocks_args = get_efficientnet_blockargs()
    model = EfficientNet(blocks_args, dropout_rate, num_classes, width_coefficient,
                         depth_coefficient, depth_divisor, min_depth, drop_connect_rate,
                         input_size, **kwargs)
    return model

def get_efficientnet_b0(pretrained=False, dropout_rate=0.2, classes=1000, width_coefficient=1.0, depth_coefficient=1.0,
                        depth_divisor=8, min_depth=None, drop_connect_rate=0.2, input_size=224, ctx=mx.cpu(),
                        root=os.path.join('~', '.autogluon', 'models')):
    model = get_efficientnet(dropout_rate, classes, width_coefficient, depth_coefficient,
                             depth_divisor, min_depth, drop_connect_rate,
                             input_size=input_size)
    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('efficientnet_b0', root=root), ctx=ctx)
    else:
        model.collect_params().initialize(ctx=ctx)
    return model

def get_efficientnet_b1(pretrained=False, dropout_rate=0.2, classes=1000, width_coefficient=1.0, depth_coefficient=1.1,
                        depth_divisor=8, min_depth=None, drop_connect_rate=0.2, input_size=240, ctx=mx.cpu(),
                        root=os.path.join('~', '.autogluon', 'models')):
    model = get_efficientnet(dropout_rate, classes, width_coefficient, depth_coefficient,
                             depth_divisor, min_depth, drop_connect_rate,
                             input_size=input_size)
    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('efficientnet_b1', root=root), ctx=ctx)
    else:
        model.collect_params().initialize(ctx=ctx)
    return model

def get_efficientnet_b2(pretrained=False, dropout_rate=0.2, classes=1000, width_coefficient=1.1, depth_coefficient=1.2,
                        depth_divisor=8, min_depth=None, drop_connect_rate=0.2, input_size=260, ctx=mx.cpu(),
                        root=os.path.join('~', '.autogluon', 'models')):
    model = get_efficientnet(dropout_rate, classes, width_coefficient, depth_coefficient,
                             depth_divisor, min_depth, drop_connect_rate,
                             input_size=input_size)
    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('efficientnet_b2', root=root), ctx=ctx)
    else:
        model.collect_params().initialize(ctx=ctx)
    return model

def get_efficientnet_b3(pretrained=False, dropout_rate=0.2, classes=1000, width_coefficient=1.2, depth_coefficient=1.4,
                        depth_divisor=8, min_depth=None, drop_connect_rate=0.2, input_size=300, ctx=mx.cpu(),
                        root=os.path.join('~', '.autogluon', 'models')):
    model = get_efficientnet(dropout_rate, classes, width_coefficient, depth_coefficient,
                             depth_divisor, min_depth, drop_connect_rate,
                             input_size=input_size)
    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('efficientnet_b3', root=root), ctx=ctx)
    else:
        model.collect_params().initialize(ctx=ctx)
    return model

def get_efficientnet_b4(pretrained=False, dropout_rate=0.2, classes=1000, width_coefficient=1.4, depth_coefficient=1.8,
                        depth_divisor=8, min_depth=None, drop_connect_rate=0.2, input_size=380, ctx=mx.cpu(),
                        root=os.path.join('~', '.autogluon', 'models')):
    model = get_efficientnet(dropout_rate, classes, width_coefficient, depth_coefficient,
                             depth_divisor, min_depth, drop_connect_rate,
                             input_size=input_size)
    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('efficientnet_b4', root=root), ctx=ctx)
    else:
        model.collect_params().initialize(ctx=ctx)
    return model

def get_efficientnet_b5(pretrained=False, dropout_rate=0.2, classes=1000, width_coefficient=1.6, depth_coefficient=2.2,
                        depth_divisor=8, min_depth=None, drop_connect_rate=0.2, input_size=456, ctx=mx.cpu(),
                        root=os.path.join('~', '.autogluon', 'models')):
    model = get_efficientnet(dropout_rate, classes, width_coefficient, depth_coefficient,
                             depth_divisor, min_depth, drop_connect_rate,
                             input_size=input_size)
    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('efficientnet_b5', root=root), ctx=ctx)
    else:
        model.collect_params().initialize(ctx=ctx)
    return model

def get_efficientnet_b6(pretrained=False, dropout_rate=0.2, classes=1000, width_coefficient=1.8, depth_coefficient=2.6,
                        depth_divisor=8, min_depth=None, drop_connect_rate=0.2, input_size=528, ctx=mx.cpu(),
                        root=os.path.join('~', '.autogluon', 'models')):
    model = get_efficientnet(dropout_rate, classes, width_coefficient, depth_coefficient,
                             depth_divisor, min_depth, drop_connect_rate,
                             input_size=input_size)
    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('efficientnet_b6', root=root), ctx=ctx)
    else:
        model.collect_params().initialize(ctx=ctx)
    return model

def get_efficientnet_b7(pretrained=False, dropout_rate=0.2, classes=1000, width_coefficient=2.0, depth_coefficient=3.1,
                        depth_divisor=8, min_depth=None, drop_connect_rate=0.2, input_size=600, ctx=mx.cpu(),
                        root=os.path.join('~', '.autogluon', 'models')):
    model = get_efficientnet(dropout_rate, classes, width_coefficient, depth_coefficient,
                             depth_divisor, min_depth, drop_connect_rate,
                             input_size=input_size)
    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('efficientnet_b7', root=root), ctx=ctx)
    else:
        model.collect_params().initialize(ctx=ctx)
    return model
