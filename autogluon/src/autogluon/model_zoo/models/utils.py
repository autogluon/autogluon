import math
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock

__all__ = ['round_repeats', 'round_filters', 'SamePadding', 'Swish',
           '_add_conv', '_update_input_size']

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

        ih, iw = self.input_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)

        self.pad_h = max((oh - 1) * self.stride[0] +
                    (kh - 1) * self.dilation[0] + 1 - ih, 0)
        self.pad_w = max((ow - 1) * self.stride[1] +
                    (kw - 1) * self.dilation[1] + 1 - iw, 0)

    def hybrid_forward(self, F, x):
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, mode='constant', pad_width=(0, 0, 0, 0, self.pad_h//2, self.pad_h - self.pad_h//2,
                                                     self.pad_w//2, self.pad_w -self.pad_w//2))
            return x
        return x

    def __repr__(self):
        s = '{}({}, {}, {}, {})'
        return s.format(self.__class__.__name__,
                        self.pad_h//2, self.pad_h - self.pad_h//2,
                        self.pad_w//2, self.pad_w -self.pad_w//2)
    

#class swish(mx.autograd.Function):
#    def __init__(self, beta):
#        super().__init__()
#        self.beta = beta
#
#    def forward(self, x):
#        y = x * self.beta
#        y1 = y.sigmoid()
#        y = x * y1
#        self.save_for_backward(x, y1)
#        return y
#
#    def backward(self, dy):
#        x, y1 = self.saved_tensors
#        return dy * (- self.beta * x * y1 * (1 - y1) + y1)
#
#    def __repr__(self):
#        return '{} (beta={})'.format(self.__class__.__name__, self.beta)
#

class Swish(HybridBlock):
    def __init__(self, beta=1.0, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self._beta = beta

    def hybrid_forward(self, F, x):
        return x * F.sigmoid(self._beta * x)
        #return swish(self._beta)(x)

    def __repr__(self):
        return '{} (beta={})'.format(self.__class__.__name__, self._beta)

def _add_conv(out, channels=1, kernel=1, stride=1, pad=0, num_group=1,
              activation='swish', batchnorm=True, input_size=None,
              in_channels=0):
    out.add(SamePadding(kernel, stride, dilation=(1, 1), input_size=input_size))
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group,
                      use_bias=False, in_channels=in_channels))
    if batchnorm:
        out.add(nn.BatchNorm(in_channels=channels, scale=True,
                             momentum=0.99, epsilon=1e-3))
    if activation == 'relu':
        out.add(nn.Activation('relu'))
    elif activation == 'swish':
        out.add(Swish())

def _update_input_size(input_size, stride):
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ih, iw = (input_size, input_size) if isinstance(input_size, int) else input_size
    oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
    input_size = (oh, ow)
    return input_size

