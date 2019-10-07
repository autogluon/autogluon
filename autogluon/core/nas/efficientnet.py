import math
import collections
import re

import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
#import mxnet.ndarray as F

from ..basic import *

__all__ = ['mobilenet_block_args', 'EfficientNet', 'MBConvBlockArgs', 'MBConvBlock']

MBConvBlockArgs = collections.namedtuple('MBConvBlockArgs', [
    'kernel', 'num_repeat', 'channels', 'expand_ratio',
    'stride', 'se_ratio', 'in_channels'])

@autogluon_function()
def mobilenet_block_args(kernel, num_repeat, channels, expand_ratio, stride, se_ratio=0.25, in_channels=0):
    return MBConvBlockArgs(kernel=kernel, num_repeat=num_repeat,
                          channels=channels, expand_ratio=expand_ratio,
                          stride=stride, se_ratio=se_ratio, in_channels=in_channels)

class MBConvBlock(HybridBlock):
    def __init__(self, in_channels, channels, expand_ratio, kernel, stride, se_ratio=0,
                 drop_connect_rate=0, input_size=None, num_repeat=None, **kwargs):
        r"""MobileNet V3 Block
            Parameters
            ----------
            int_channels: int, input channels.
            channels: int, output channels.
            t: int, the expand ratio used for increasing channels.
            kernel: int, filter size.
            stride: int, stride of the convolution.
            se_ratio: float, ratio of the squeeze layer and excitation layer.
            drop_connect_rate: float, drop rate of drop out.
        """
        super(MBConvBlock, self).__init__(**kwargs)
        assert input_size
        if isinstance(input_size, int):
            input_size = (input_size,) * 2
        self.use_shortcut = stride == 1 and in_channels == channels
        self.se_ratio = se_ratio
        self.drop_connect_rate = drop_connect_rate
        #with self.name_scope():

        self.depth_conv = nn.HybridSequential(prefix='depth_conv_')
        #with self.depth_conv.name_scope():
        if expand_ratio != 1:
            _add_conv(self.depth_conv, in_channels * expand_ratio,
                      active=True, batchnorm=True, input_size=input_size)
        _add_conv(self.depth_conv, in_channels * expand_ratio,
            kernel=kernel, stride=stride,
            num_group=in_channels * expand_ratio,
            active=True, batchnorm=True, input_size=input_size)
        input_size = _update_input_size(input_size, stride)

        if se_ratio:
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se_module = nn.HybridSequential(prefix='se_module_')
            #with self.se_module.name_scope():
            _add_conv(self.se_module, num_squeezed_channels,
                      active=True, batchnorm=False, input_size=input_size)
            _add_conv(self.se_module, in_channels * expand_ratio,
                      active=False, batchnorm=False, input_size=input_size)
        self.project_conv = nn.HybridSequential(prefix='preject_conv_')
        #with self.project_conv.name_scope():
        _add_conv(self.project_conv, channels,
                  active=False, batchnorm=True, input_size=input_size)
        if drop_connect_rate:
            self.drop_out = nn.Dropout(drop_connect_rate)

    def hybrid_forward(self, F, inputs):
        x = inputs
        x = self.depth_conv(x)
        if self.se_ratio:
            out = F.contrib.AdaptiveAvgPooling2D(x, 1)
            out = self.se_module(out)
            out = F.broadcast_mul(F.sigmoid(out), x)
        out = self.project_conv(out)
        if self.use_shortcut:
            if self.drop_connect_rate:
                out = self.drop_out(out)
            out = F.elemwise_add(out, inputs)
        return out

def round_repeats(repeats, depth_coefficient=None):
    """ Round number of filters based on depth multiplier. """
    multiplier = depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

def round_filters(filters, width_coefficient=None, depth_divisor=None, min_depth=None):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = width_coefficient
    if not multiplier:
        return filters
    divisor = depth_divisor
    min_depth = min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(
        min_depth, int(
            filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)

class SamePadding(HybridBlock):
    def __init__(self, kernel_size, stride, dilation, input_size, **kwargs):
        super(SamePadding, self).__init__(**kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(input_size, int):
            input_size = (input_size,) * 2

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def hybrid_forward(self, F, x):
        ih, iw = self.input_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] +
                    (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] +
                    (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, mode='constant', pad_width=(0, 0, 0, 0, pad_w//2, pad_w -pad_w//2,
                                                     pad_h//2, pad_h - pad_h//2))
            return x
        return x

class Swish(HybridBlock):
    def __init__(self, beta=1.0, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self._beta = beta


    def hybrid_forward(self, F, x):
        return x * F.sigmoid(self._beta * x)

def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, batchnorm=True, input_size=None):
    out.add(SamePadding(kernel, stride, dilation=(1, 1), input_size=input_size))
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    if batchnorm:
        out.add(nn.BatchNorm(scale=True, momentum=0.99, epsilon=1e-3))
    if active:
        out.add(Swish())

def _update_input_size(input_size, stride):
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ih, iw = (input_size, input_size) if isinstance(input_size, int) else input_size
    oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
    input_size = (oh, ow)
    return input_size

class EfficientNet(HybridBlock):
    def __init__(self, blocks_args, dropout_rate, num_classes, width_coefficient,
                 depth_cofficient, depth_divisor, min_depth, drop_connect_rate,
                 input_size, **kwargs):
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
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._blocks_args = blocks_args
        self.input_size = input_size
        with self.name_scope():
            self.features = nn.HybridSequential()
            with self.features.name_scope():
                # stem conv
                out_channels = round_filters(32, width_coefficient,
                                             depth_divisor, min_depth)
                _add_conv(self.features, out_channels, kernel=3,
                    stride=2, active=True, batchnorm=True, input_size=input_size)

            input_size = _update_input_size(input_size, 2)

            self._blocks = nn.HybridSequential()
            with self._blocks.name_scope():
                for block_arg in self._blocks_args:
                    # Update block input and output filters based on depth
                    # multiplier.
                    block_arg = block_arg._replace(
                        in_channels=out_channels,
                        channels=round_filters(block_arg.channels, width_coefficient,
                                               depth_divisor, min_depth),
                        num_repeat=round_repeats(
                            block_arg.num_repeat, depth_cofficient))

                    out_channels=block_arg.channels

                    arg_dict = block_arg._asdict()
                    arg_dict['input_size'] = input_size
                    self._blocks.add(MBConvBlock(**arg_dict))

                    input_size = _update_input_size(input_size, block_arg.stride)
                    if block_arg.num_repeat > 1:
                        block_arg = block_arg._replace(
                            in_channels=block_arg.channels, stride=1)

                    arg_dict = block_arg._asdict()
                    arg_dict['input_size'] = input_size
                    for _ in range(block_arg.num_repeat - 1):
                        self._blocks.add(MBConvBlock(**arg_dict))

            # Head
            out_channels = round_filters(1280, width_coefficient,
                                         depth_divisor, min_depth)
            self._conv_head = nn.HybridSequential()
            with self._conv_head.name_scope():
                _add_conv(self._conv_head, out_channels,
                          active=True, batchnorm=True, input_size=input_size)
            # Final linear layer
            self._dropout = dropout_rate
            self._fc = nn.Dense(num_classes, use_bias=False)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        for block in self._blocks:
            x = block(x)
        x = self._conv_head(x)
        x = F.contrib.AdaptiveAvgPooling2D(x, 1)
        #x = F.squeeze(F.squeeze(x, axis=-1), axis=-1)
        if self._dropout:
            x = F.Dropout(x, self._dropout)
        x = self._fc(x)
        return x
