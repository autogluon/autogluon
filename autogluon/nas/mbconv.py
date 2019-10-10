import collections

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

from ..core import autogluon_function
from .utils import *

__all__ = ['mobilenet_block_args', 'MBConvBlockArgs', 'MBConvBlock']

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

        self.depth_conv = nn.HybridSequential(prefix='depth_conv_')
        if expand_ratio != 1:
            _add_conv(self.depth_conv, in_channels * expand_ratio,
                      activation='swish', batchnorm=True, input_size=input_size)
        _add_conv(self.depth_conv, in_channels * expand_ratio,
            kernel=kernel, stride=stride,
            num_group=in_channels * expand_ratio,
            activation='swish', batchnorm=True, input_size=input_size)
        input_size = _update_input_size(input_size, stride)

        if se_ratio:
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se_module = nn.HybridSequential(prefix='se_module_')
            self.se_module.add(nn.Conv2D(num_squeezed_channels, 1, 1, 0, use_bias=True))
            self.se_module.add(Swish())
            self.se_module.add(nn.Conv2D(in_channels * expand_ratio, 1, 1, 0, use_bias=True))
        self.project_conv = nn.HybridSequential(prefix='preject_conv_')
        _add_conv(self.project_conv, channels,
                  activation=None, batchnorm=True, input_size=input_size)
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

