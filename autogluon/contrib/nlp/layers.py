# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Layers."""
__all__ = ['MultiHeadDense', 'PositionalEmbedding', 'SinusoidalPositionalEmbedding',
           'LearnedPositionalEmbedding', 'BucketPositionalEmbedding', 'AdaptiveEmbedding',
           'PositionwiseFFN', 'ProjectedAdaptiveLogSoftmaxWithLoss',
           'get_norm_layer', 'get_activation']

import math
import re
import numpy as np
from collections import OrderedDict
import mxnet as mx
from mxnet import use_np
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn
from typing import Union, Optional, List
from .op import relative_position_bucket


InitializerType = Optional[Union[mx.init.Initializer, str]]


@use_np
def get_norm_layer(normalization: str = 'layer_norm',
                   axis: int = -1,
                   epsilon: float = 1e-5,
                   in_channels: int = 0, **kwargs):
    """
    Get the normalization layer based on the type

    Parameters
    ----------
    normalization
        The type of the layer normalization from ['layer_norm', 'no_norm', 'batch_norm']
    axis
        The axis to normalize the
    epsilon
        The epsilon of the normalization layer
    in_channels
        Input channel

    Returns
    -------
    norm_layer
        The layer normalization layer
    """
    if isinstance(normalization, str):
        if normalization == 'layer_norm':
            norm_layer = nn.LayerNorm(axis=axis, epsilon=epsilon, in_channels=in_channels,
                                      **kwargs)
        elif normalization == 'no_norm':
            norm_layer = NoNorm(in_channels=in_channels, **kwargs)
        elif normalization == 'identity':
            norm_layer = IdentityActivation()
        elif normalization == 'batch_norm':
            norm_layer = nn.BatchNorm(axis=axis, epsilon=epsilon, in_channels=in_channels, **kwargs)
        else:
            raise NotImplementedError('normalization={} is not supported'.format(normalization))
        return norm_layer
    else:
        raise NotImplementedError('The type of normalization must be str')


@use_np
class NoNorm(HybridBlock):
    r"""
    Apply an element-wise linear transformation to the n-dimensional input array.
    replacing the layer normalization.

    .. math::
        out = \gmmma \circ data + \beta

    Parameters
    ----------
    in_channels : int
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called

    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.

    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.

    References
    ----------
        `MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices
        <https://arxiv.org/pdf/2004.02984.pdf>`_

    Examples
    --------
    >>> # Input of shape (2, 5)
    >>> x = mx.np.array([[1, 2, 3, 4, 5], [1, 1, 2, 2, 2]])
    >>> # Layer normalization is calculated with the above formula
    >>> layer = NoNorm(in_channels=5)
    >>> layer.initialize(ctx=mx.cpu(0))
    >>> layer(x)
    array([[1., 2., 3., 4., 5.],
       [1., 1., 2., 2., 2.]])
    """
    def __init__(self, in_channels, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 **kwargs):
        super(NoNorm, self).__init__(**kwargs)
        self._kwargs = {'center': center, 'scale': scale}
        self._in_channels = in_channels
        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True)

    def hybrid_forward(self, F, data, gamma, beta):
        return data * gamma + beta

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))


def _fmt_and_check_cutoffs(cutoffs, vocab_size):
    """Parse and get the cutoffs used in adaptive embedding + adaptive softmax

    Parameters
    ----------
    cutoffs
        The cutoffs of the
    vocab_size
        Size of the vocabulary

    Returns
    -------
    cutoffs
        The parsed cutoffs, will be [0, c0, c1, ..., c_{k-1}, V]
        If the original cutoffs is empty or is None, return None
    """
    # Sanity checks
    if cutoffs is None:
        return None
    if isinstance(cutoffs, int):
        cutoffs = [cutoffs]
    else:
        cutoffs = list(cutoffs)
        if len(cutoffs) == 0:
            return None
    if cutoffs != sorted(cutoffs):
        raise ValueError('cutoffs must be a sorted list of cutoff values. '
                         'Got {}, but expected {}'.format(cutoffs, sorted(cutoffs)))
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError('cutoffs cannot contain duplicates! cutoffs={}'.format(cutoffs))
    if not cutoffs:
        raise ValueError('cutoffs must not be empty. Got {}'.format(cutoffs))
    if cutoffs[0] <= 0:
        raise ValueError('The first cutoff value ({}) must be greater 0.'.format(cutoffs[0]))
    if cutoffs[-1] >= vocab_size:
        raise ValueError(
            'The last cutoff value ({}) must be smaller than vocab_size ({}).'.format(
                cutoffs[-1], vocab_size))
    return cutoffs


def _gen_repr_with_kwargs(kwargs, cls_name):
    s = '{name}(\n'.format(name=cls_name)
    for i, (k, v) in enumerate(kwargs.items()):
        if i != len(kwargs.items()) - 1:
            s += '\t{}={},\n'.format(k, v)
        else:
            s += '\t{}={}\n'.format(k, v)
    s += ')'
    return s


@use_np
def get_activation(act: Optional[Union[str, HybridBlock]]) -> HybridBlock:
    """Get the activation based on the string

    Parameters
    ----------
    act
        The activation

    Returns
    -------
    ret
        The activation layer

    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act.startswith('leaky'):
            # TODO(sxjscience) Add regex matching here to parse `leaky(0.1)`
            default_alpha = 0.1
            match_ret = re.match('leaky\((\d+.\d*)\)', act)
            if match_ret is not None:
                alpha = float(match_ret.groups()[0])
                return nn.LeakyReLU(alpha)
            else:
                return nn.LeakyReLU(default_alpha)
        elif act == 'prelu':
            return nn.PReLU()
        elif act == 'identity':
            return IdentityActivation()
        elif act == 'elu':
            return ELU()
        elif act == 'gelu':
            return GELU(mode='erf')
        elif act == 'gelu(tanh)':
            return GELU(mode='tanh')
        elif act == 'gelu(sigmoid)':
            return GELU(mode='sigmoid')
        elif act in ['relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']:
            return nn.Activation(act)
        else:
            raise NotImplementedError('act={} is not supported'.format(act))
    else:
        return act


@use_np
class MultiHeadDense(HybridBlock):
    def __init__(self, units, num_heads, use_bias=True, dtype='float32',
                 weight_initializer=None, bias_initializer=None,
                 prefix=None, params=None):
        """Multiple Dense with different parameters and the same number of units
        The inner shapes of the weight and bias are
            weight: (self._parallel_num[0] * ... * self._parallel_num[k] * units, in_units)
            bias: (self._parallel_num[0] * ... * self._parallel_num[k],)
        Parameters
        ----------
        units : int
            The basic units.
        num_heads : int or tuple
        use_bias : bool, default True
        dtype : str, default 'float32'
            The data type
        weight_initializer : None or initialzer, default None
        bias_initializer : None or initializer, default None
        prefix : str or None
        params : None
        """
        super().__init__(prefix=prefix, params=params)
        if not isinstance(num_heads, (list, tuple)):
            num_heads = (int(num_heads),)
        else:
            num_heads = tuple(num_heads)
        self._num_heads = num_heads
        self._use_bias = use_bias
        for ele in self._num_heads:
            if ele <= 0:
                raise ValueError('Invalid number of heads, all numbers need to be larger than 0.'
                                 ' num_heads={}'.format(num_heads))
        self._units = units
        self._mult = np.prod(num_heads)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(self._mult * units, 0),
                                          init=weight_initializer, dtype=dtype,
                                          allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get('bias', shape=(self._mult * units,),
                                            init=bias_initializer, dtype=dtype,
                                            allow_deferred_init=True)
            else:
                self.bias = None

    def hybrid_forward(self, F, data, weight, bias=None):
        """
        Parameters
        ----------
        F
        data : Symbol or NDArray
            Shape (B, ..., C_in)
        Returns
        -------
        ret : Symbol or NDArray
            Shape (B,) + num_heads + (, ..., C_out)
        """
        ret = F.npx.fully_connected(data, weight, bias, no_bias=bias is None,
                                    num_hidden=self._mult * self._units, flatten=False, name='fwd')
        ret = F.npx.reshape(ret, newshape=(-4, self._mult, -1, -6), reverse=True)
        ret = F.np.moveaxis(ret, -2, 1)
        for i in range(len(self._num_heads) - 1, 0, -1):
            ret = F.npx.reshape(ret, newshape=(-2, -6, -1, self._num_heads[i], -4))
        return ret

    def __repr__(self):
        s = '{name}(' \
            'units={units},' \
            ' num_heads={num_heads},' \
            ' use_bias={use_bias},' \
            ' weight={weight}' \
            ')'.format(name=self.__class__.__name__,
                       units=self._units,
                       num_heads=self._num_heads,
                       use_bias=self._use_bias,
                       weight=self.weight.shape)
        return s


@use_np
class IdentityActivation(HybridBlock):
    def hybrid_forward(self, F, x):
        return x


@use_np
class GELU(HybridBlock):
    r"""Gaussian Error Linear Unit.

    This is a smoother version of the RELU. See https://arxiv.org/abs/1606.08415 for more details.

    The original formula is `x gaussian_cdf(x)`.
    Here, we provide three different ways to calculate/approximate GELU.

        - mode = 'erf'

            y = 0.5 x (1 + erf(\frac{x}{\sqrt{2}}))

        - mode = 'tanh'

            y =  0.5 x (1 + tanh[\sqrt(2/\pi) * (x + 0.044715 x^3)])

        - mode = 'sigmoid'

            y = x \sigma(1.702x)


    Parameters
    ----------
    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, mode='erf', prefix=None, params=None):
        """

        Parameters
        ----------
        mode
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        if mode not in ['erf', 'tanh', 'sigmoid']:
            raise ValueError('Unsupported mode, only support "erf", "tanh", or "sigmoid". '
                             'Received mode={}'.format(mode))
        self._mode = mode

    def hybrid_forward(self, F, x):
        if self._mode == 'erf':
            return x * 0.5 * (1.0 + F.npx.erf(x / math.sqrt(2.0)))
        elif self._mode == 'tanh':
            return 0.5 * x * (1.0 + F.np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))
        elif self._mode == 'sigmoid':
            return x * F.npx.sigmoid(1.702 * x)
        else:
            raise NotImplementedError

    def __repr__(self):
        s = '{name}(mode={mode})'
        return s.format(name=self.__class__.__name__, mode=self._mode)


@use_np
class ELU(HybridBlock):
    r"""
    Exponential Linear Unit (ELU)
        "Fast and Accurate Deep Network Learning by Exponential Linear Units", Clevert et al, 2016
        https://arxiv.org/abs/1511.07289
        Published as a conference paper at ICLR 2016
    Parameters
    ----------
    alpha : float
        The alpha parameter as described by Clevert et al, 2016
    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha

    def hybrid_forward(self, F, x):
        return - self._alpha * F.npx.relu(1.0 - F.np.exp(x)) + F.npx.relu(x)

    def __repr__(self):
        s = '{name}(alpha={alpha})'
        return s.format(name=self.__class__.__name__, alpha=self._alpha)


@use_np
class PositionalEmbedding(HybridBlock):
    def __init__(self, units, max_length=None, method='sinusoidal',
                 dtype='float32', prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._units = units
        self._max_length = max_length
        self._method = method
        self._dtype = dtype
        with self.name_scope():
            if method == 'sinusoidal':
                self._embed = SinusoidalPositionalEmbedding(units=units,
                                                            dtype=dtype,
                                                            prefix='embed_')
            elif method == 'learned':
                self._embed = LearnedPositionalEmbedding(units=units,
                                                         max_length=max_length,
                                                         dtype=dtype,
                                                         prefix='embed_')
            else:
                raise NotImplementedError

    def hybrid_forward(self, F, positions):
        """

        Parameters
        ----------
        F
        positions : mx.numpy.ndarray or mx.numpy.Symbol
            Shape (..., )

        Returns
        -------
        ret :
            Shape (..., units)
        """
        return self._embed(positions)


@use_np
class SinusoidalPositionalEmbedding(HybridBlock):
    def __init__(self, units: int, dtype: Union[str, type] = 'float32', prefix=None, params=None):
        """Use a geometric sequence of timescales.

        Parameters
        ----------
        units
            The number of units for positional embedding
        dtype
            The dtype of the inner positional embeddings
        """
        super().__init__(prefix=prefix, params=params)

        def _init_sinusodial_base(units):
            half_units = units // 2
            val = np.log(10000) / (half_units - 1)
            val = np.exp(np.arange(half_units, dtype=np.float32) * -val)
            return val

        self._units = units
        self._dtype = dtype
        self.base_mult = self.params.get_constant('base_mult', _init_sinusodial_base(units))

    def hybrid_forward(self, F, positions, base_mult):
        """

        Parameters
        ----------
        F
        positions : NDArray
            Shape (..., )

        Returns
        -------
        ret :
            Shape (..., units)
        """
        emb = F.np.expand_dims(positions.astype(self._dtype), axis=-1) * base_mult
        sin_emb = F.np.sin(emb)
        cos_emb = F.np.cos(emb)
        if self._units % 2 == 0:
            return F.np.concatenate([sin_emb, cos_emb], axis=-1)
        else:
            return F.np.concatenate(
                [sin_emb, cos_emb, F.np.expand_dims(F.np.zeros_like(positions).astype(self._dtype),
                                                    axis=-1)], axis=-1)

    def __repr__(self):
        s = '{name}(units={units}, dtype={dtype})'
        return s.format(name=self.__class__.__name__,
                        units=self._units,
                        dtype=self._dtype)


@use_np
class LearnedPositionalEmbedding(HybridBlock):
    def __init__(self, units, max_length, mode='clip',
                 dtype='float32', weight_initializer=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._units = units
        self._dtype = dtype
        self._max_length = max_length
        self._mode = mode

        with self.name_scope():
            self.weight = self.params.get('weight', shape=(max_length, units),
                                          init=weight_initializer, dtype=dtype,
                                          allow_deferred_init=True)

    def __repr__(self):
        s = '{name}(units={units}, max_length={max_length}, mode={mode}, dtype={dtype})'
        return s.format(name=self.__class__.__name__,
                        units=self._units,
                        max_length=self._max_length,
                        mode=self._mode,
                        dtype=self._dtype)

    def hybrid_forward(self, F, positions, weight):
        return F.np.take(weight, positions, axis=0, mode=self._mode)


@use_np
class BucketPositionalEmbedding(HybridBlock):
    """Divide the positional space into buckets and assign the relative positions within each
    bucket to the same value. For positions that are out-of-the-boundary, they are treated as
    falling into one bucket.

    This is used in the T5 paper:
    "[Arxiv2019] Exploring the limits of transfer learning with a unified text-to-text transformer",

    Here, the first half of the buckets handles the small shifts and the second half
    of the buckets handles the large shifts (mapping them in logarithmically separated bins).
    """
    def __init__(self, units, bidirectional=True, num_buckets=32, max_distance=128,
                 dtype='float32', embed_initializer=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._units = units
        self._bidirectional = bidirectional
        self._num_buckets = num_buckets
        self._max_distance = max_distance
        self._dtype = dtype
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(num_buckets, units),
                                          init=embed_initializer, dtype=dtype,
                                          allow_deferred_init=True)

    def __repr__(self):
        s = '{name}(units={units}, bidirectional={bidirectional}, num_buckets={num_buckets},' \
            ' max_distance={max_distance}, dtype={dtype})'
        return s.format(name=self.__class__.__name__,
                        units=self._units,
                        bidirectional=self._bidirectional,
                        num_buckets=self._num_buckets,
                        max_distance=self._max_distance,
                        dtype=self._dtype)

    def hybrid_forward(self, F, relative_positions, weight):
        buckets = relative_position_bucket(F, relative_positions,
                                           bidirectional=self._bidirectional,
                                           num_buckets=self._num_buckets,
                                           max_distance=self._max_distance)
        return F.np.take(weight, buckets, axis=0)


@use_np
class PositionwiseFFN(HybridBlock):
    """The Position-wise FFN layer used in Transformer-like architectures

    If pre_norm is True:
        norm(data) -> fc1 -> act -> act_dropout -> fc2 -> dropout -> res(+data)
    Else:
        data -> fc1 -> act -> act_dropout -> fc2 -> dropout -> norm(res(+data))
    """
    def __init__(self,
                 units: int = 512,
                 hidden_size: int = 2048,
                 activation_dropout: float = 0.0,
                 dropout: float = 0.1,
                 weight_initializer=None,
                 bias_initializer='zeros',
                 activation='relu',
                 normalization: str = 'layer_norm',
                 layer_norm_eps: float = 1E-5,
                 pre_norm: bool = False,
                 dtype='float32',
                 prefix=None, params=None):
        """

        Parameters
        ----------
        units
        hidden_size
        activation_dropout
        dropout
        weight_initializer
        bias_initializer
        activation
        normalization
            layer_norm or no_norm
        layer_norm_eps
        pre_norm
            Pre-layer normalization as proposed in the paper:
            "[ACL2018] The Best of Both Worlds: Combining Recent Advances in
             Neural Machine Translation"
            This will stabilize the training of Transformers.
            You may also refer to
            "[Arxiv2020] Understanding the Difficulty of Training Transformers"
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        self._dtype = dtype
        self._pre_norm = pre_norm
        self._kwargs = OrderedDict([
            ('units', units),
            ('hidden_size', hidden_size),
            ('activation_dropout', activation_dropout),
            ('activation', activation),
            ('dropout', dropout),
            ('normalization', normalization),
            ('layer_norm_eps', layer_norm_eps),
            ('pre_norm', pre_norm),
            ('dtype', self._dtype)
        ])
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.activation_dropout_layer = nn.Dropout(activation_dropout)
            self.ffn_1 = nn.Dense(units=hidden_size,
                                  in_units=units,
                                  flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  dtype=dtype,
                                  prefix='ffn1_')
            self.activation = get_activation(activation)
            self.ffn_2 = nn.Dense(units=units,
                                  in_units=hidden_size,
                                  flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  dtype=dtype,
                                  prefix='ffn2_')
            # TODO(sxjscience) We may need to set the dtype flag in LayerNorm, need to double check
            self.layer_norm = get_norm_layer(normalization=normalization,
                                             in_channels=units,
                                             epsilon=layer_norm_eps,
                                             prefix='ln_')

    def hybrid_forward(self, F, data):
        """

        Parameters
        ----------
        F
        data :
            Shape (B, seq_length, C_in)

        Returns
        -------
        out :
            Shape (B, seq_length, C_out)
        """
        if self._pre_norm:
            data = self.layer_norm(data)
        out = self.activation(self.ffn_1(data))
        out = self.activation_dropout_layer(out)
        out = self.ffn_2(out)
        out = self.dropout_layer(out)
        out = out + data
        if not self._pre_norm:
            out = self.layer_norm(out)
        return out

    def __repr__(self):
        return _gen_repr_with_kwargs(self._kwargs, self.__class__.__name__)


@use_np
class AdaptiveEmbedding(HybridBlock):
    """Adaptive Embedding.

    It uses larger embedding units for tokens with higher frequencies. This helps reduce the risk
    of overfitting to rare words.

    Baevski, Alexei, and Michael Auli.
     "Adaptive input representations for neural language modeling." ICLR 2019.

    From input = (..., ) --> embedding (..., units)
    """
    def __init__(self, vocab_size: int,
                 embed_size: int,
                 units: int,
                 cutoffs: Optional[Union[int, List]] = None,
                 div_val: float = 1.0,
                 dtype='float32',
                 scaled=True,
                 embedding_initializer: InitializerType = None,
                 weight_initializer: InitializerType = None,
                 prefix=None,
                 params=None):
        """

        Parameters
        ----------
        vocab_size
            The size of the vocabulary
        embed_size
            The base size of the embedding vectors. The embedding size of each cluster will be
            [embed_size / div_val**0, embed_size / div_val**1, embed_size / div_val**2, ...]
        units
            The number of units after the mapping
        cutoffs
            The cutoffs to slice the vocab to multiple clusters. It should be a sorted list. Each
            value should be between 1 --> vocab_size - 1.
        div_val
            The base denominator for computing the size of the embedding vector in each cluster.
        dtype
            The data type of layer
        scaled
            Whether to scale the embedding by sqrt(units)
        embedding_initializer
            Initializer of the embedding vectors
        weight_initializer
            Initializer of projection layers
        bias_initializer
            Initializer of the bias
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        cutoffs = _fmt_and_check_cutoffs(cutoffs, vocab_size)
        if cutoffs is None:
            assert div_val == 1.0
        self._dtype = dtype
        self._kwargs = OrderedDict([
            ('cutoffs', cutoffs),
            ('vocab_size', vocab_size),
            ('embed_size', embed_size),
            ('units', units),
            ('div_val', div_val),
            ('dtype', dtype),
            ('scaled', scaled)
        ])
        self._vocab_size = vocab_size
        self._cutoffs = cutoffs
        self._units = units
        self._embed_size = embed_size
        self._div_val = div_val
        self._scaled = scaled
        if self._scaled:
            self._emb_scale = units**0.5
        with self.name_scope():
            if div_val == 1.0:
                setattr(self, 'embed0_weight',
                        self.params.get('embed0_weight',
                                        shape=(vocab_size, embed_size),
                                        init=embedding_initializer,
                                        allow_deferred_init=True))

                if units != embed_size:
                    setattr(self, 'inter_proj0_weight',
                            self.params.get('inter_proj0_weight',
                                            shape=(embed_size, units),
                                            init=weight_initializer,
                                            allow_deferred_init=True))
                else:
                    self.proj_layers = None
            else:
                self.proj_layers = nn.HybridSequential(prefix='inter_proj')
                for i, (l_idx, r_idx) in enumerate(zip([0] + cutoffs, cutoffs + [vocab_size])):
                    inner_embed_size = int(embed_size / div_val**i)
                    if inner_embed_size == 0:
                        raise ValueError('div_val = {} is too large for the layer. Currently, the '
                                         'cutoffs are {} and the embed_size is {}. Using the '
                                         'div_val = {} will cause some clusters to have '
                                         'embed_size=0.'.format(div_val, cutoffs, embed_size,
                                                                div_val))
                    setattr(
                        self, 'embed{}_weight'.format(i),
                        self.params.get('embed{}_weight'.format(i),
                                        shape=(r_idx - l_idx, inner_embed_size),
                                        init=embedding_initializer,
                                        allow_deferred_init=True))
                    setattr(self, 'inter_proj{}_weight'.format(i),
                            self.params.get('inter_proj{}_weight'.format(i),
                                            shape=(inner_embed_size, units),
                                            init=weight_initializer,
                                            allow_deferred_init=True))

    def hybrid_forward(self, F, inp, **params):  # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        F
        inp
            Shape (...,)
        params

        Returns
        -------
        out
            Shape (..., units)
        """
        if self._div_val == 1.0:
            emb = F.np.take(params['embed0_weight'], inp, axis=0)
            if self._units != self._embed_size:
                emb = F.np.dot(emb, params['inter_proj0_weight'])
        else:
            emb = None
            # TODO(?) We can refactor the code using
            #  F.np._internal.nonzero() + F.npx.index_update
            for i, (l_idx, r_idx) in enumerate(zip([0] + self._cutoffs,
                                                   self._cutoffs + [self._vocab_size])):
                emb_i = F.np.take(params['embed{}_weight'.format(i)],
                                  inp - l_idx, axis=0,
                                  mode='clip')
                emb_i = F.np.dot(emb_i, params['inter_proj{}_weight'.format(i)])
                if emb is None:
                    emb = emb_i
                else:
                    emb = F.np.where(F.np.expand_dims((inp >= l_idx) * (inp < r_idx), axis=-1),
                                     emb_i, emb)
        if self._scaled:
            emb = emb * self._emb_scale
        return emb

    def __repr__(self):
        return _gen_repr_with_kwargs(self._kwargs, self.__class__.__name__)


@use_np
class ProjectedAdaptiveLogSoftmaxWithLoss(HybridBlock):
    r"""Projected Adaptive LogSoftmax Loss.

    Projected Adaptive LogSoftmax is a practical way to accelerate the computation of log-softmax.
    We divide the words into multiple clusters based on the cutoffs:
    For example, if the cutoffs are [c0, c1] and there are N words, we can divide these N words into
    three clusters:

    Cluster-1: [V_0, V_1, ..., V_{c0}],
    Cluster-2: [V_{c0 + 1}, V_{c0 + 2}, ... V_{c1}]
    Cluster-3: [V_{c1 + 1}, V_{c1 + 2}, ... V_{N - 1}]

    Usually, the cutoffs are chosen based on the frequency of the words. The
    top clusters will contain more common words and the bottom ones contain less frequent
    words.

    Based on this property, Adaptive Softmax calculate the logits step-by-step.
    We first calculate the probability for all words in the first cluster +
    additional probability values for the situations that the word belongs to the other
    clusters.

    For the example above, we will have two additional virtual words: T2, and T3, meaning that the
    correct word should be at the 2nd or 3rd cluster

    prob1 = \softmax([V_0, V_1, ..., V_{c0}, T2, T3])
    prob2 = p(T2) * \softmax([V_{c0 + 1}, V_{c0 + 2}, ... V_{c1}])
    prob3 = p(T3) * softmax([V_{c1 + 1}, V_{c1 + 2}, ... V_{N - 1}])


    Converting to log-probability, we have
    lprob1 = log-softmax([V_0, V_1, ..., V_{c0}, T2, T3])
    lprob2 = lprob1[T2] + log-softmax([V_{c0 + 1}, V_{c0 + 2}, ... V_{c1}])
    lprob3 = lprob2[T3] + log-softmax([V_{c1 + 1}, V_{c1 + 2}, ... V_{N - 1}])


    @inproceedings{grave2017efficient,
      title={Efficient softmax approximation for GPUs},
      author={Grave, Edouard and Joulin, Armand and Ciss{\'e}, Moustapha and J{\'e}gou, Herv{\'e} and others},
      booktitle={Proceedings of the 34th International Conference on Machine Learning-Volume 70},
      pages={1302--1310},
      year={2017},
      organization={JMLR. org}
    }
    """
    def __init__(self, vocab_size: int, embed_size: int, in_units: int,
                 cutoffs: Optional[Union[int, List]] = None,
                 div_val: float = 1.0,
                 dtype='float32',
                 use_bias=True,
                 weight_initializer: InitializerType = None,
                 bias_initializer: InitializerType = None,
                 prefix=None,
                 params=None):
        """

        Parameters
        ----------
        vocab_size
            Size of the vocabulary
        embed_size
            Base embedding size. The hidden will be first projected to
            embed_size and then project to vocab_size
        in_units
            The number of input units
        cutoffs
            The cutoff values
        div_val
            The base denominator for computing the size of the embedding vector in each cluster.
        dtype
            Data type
        use_bias
            Whether to use bias when computing the scores for the tokens
        weight_initializer
        bias_initializer
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        cutoffs = _fmt_and_check_cutoffs(cutoffs, vocab_size)
        if cutoffs is None:
            assert div_val == 1.0
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._in_units = in_units
        self._cutoffs = cutoffs
        self._div_val = div_val
        if cutoffs is not None:
            self._num_tail_clusters = len(self._cutoffs)
        self._dtype = dtype
        self._kwargs = OrderedDict([
            ('cutoffs', cutoffs),
            ('vocab_size', vocab_size),
            ('embed_size', embed_size),
            ('in_units', in_units),
            ('div_val', div_val),
            ('dtype', dtype),
            ('use_bias', use_bias)
        ])
        with self.name_scope():
            if cutoffs is not None:
                self.tail_cluster_score_proj = nn.Dense(units=self._num_tail_clusters,
                                                        in_units=embed_size,
                                                        flatten=False,
                                                        use_bias=use_bias,
                                                        weight_initializer=weight_initializer,
                                                        bias_initializer=bias_initializer,
                                                        prefix='tail_cluster_score_proj_')
            self.inter_proj_l = nn.HybridSequential(prefix='inter_proj')
            self.out_proj_l = nn.HybridSequential(prefix='embed')
            if div_val == 1.0:
                if in_units != embed_size:
                    with self.inter_proj_l.name_scope():
                        self.inter_proj_l.add(nn.Dense(in_units=in_units,
                                                       units=embed_size,
                                                       flatten=False,
                                                       use_bias=False,
                                                       prefix='0_',
                                                       weight_initializer=weight_initializer,
                                                       bias_initializer=bias_initializer))
                with self.out_proj_l.name_scope():
                    self.out_proj_l.add(nn.Dense(in_units=embed_size,
                                                 units=vocab_size,
                                                 flatten=False,
                                                 use_bias=use_bias,
                                                 prefix='0_',
                                                 weight_initializer=weight_initializer,
                                                 bias_initializer=bias_initializer))
            else:
                for i, (l_idx, r_idx) in enumerate(zip([0] + self._cutoffs,
                                                       self._cutoffs + [vocab_size])):
                    ele_embed_size = int(embed_size / (div_val ** i))
                    with self.inter_proj_l.name_scope():
                        self.inter_proj_l.add(nn.Dense(in_units=in_units,
                                                       units=ele_embed_size,
                                                       flatten=False,
                                                       use_bias=False,
                                                       prefix='{}_'.format(i),
                                                       weight_initializer=weight_initializer,
                                                       bias_initializer=bias_initializer))
                    with self.out_proj_l.name_scope():
                        self.out_proj_l.add(nn.Dense(in_units=ele_embed_size,
                                                     units=r_idx - l_idx,
                                                     flatten=False,
                                                     use_bias=use_bias,
                                                     prefix='{}_'.format(i),
                                                     weight_initializer=weight_initializer,
                                                     bias_initializer=bias_initializer))

    def get_logits(self, F, hidden):
        """Get all the logits.

        Parameters
        ----------
        F
        hidden
            The hidden representation
            Shape (..., in_units)

        Returns
        -------
        logits
            Shape (..., |V|)

        """
        if self._cutoffs is None:
            if self._in_units != self._embed_size:
                hidden = self.inter_proj_l[0](hidden)
            logits = self.out_proj_l[0](hidden)
            return logits
        else:
            all_logits = []
            if self._div_val == 1.0:
                if self._in_units == self._embed_size:
                    all_scores = self.out_proj_l[0](hidden)
                    tail_cluster_scores = self.tail_cluster_score_proj(hidden)
                else:
                    inter_hidden = self.inter_proj_l[0](hidden)
                    all_scores = self.out_proj_l[0](inter_hidden)
                    tail_cluster_scores = self.tail_cluster_score_proj(inter_hidden)
                all_scores_l = F.np.split(all_scores, self._cutoffs, axis=-1)
                head_scores = all_scores_l[0]
            else:
                inter_hidden = self.inter_proj_l[0](hidden)
                head_scores = self.out_proj_l[0](inter_hidden)
                tail_cluster_scores = self.tail_cluster_score_proj(inter_hidden)
            head_tail_cluster_logits = \
                F.npx.log_softmax(F.np.concatenate([head_scores, tail_cluster_scores],
                                                   axis=-1), axis=-1)
            head_logits, tail_cluster_logits = \
                F.np.split(head_tail_cluster_logits, [self._cutoffs[0]], axis=-1)
            tail_cluster_logits = F.np.split(tail_cluster_logits, self._num_tail_clusters, axis=-1)
            all_logits.append(head_logits)
            for i in range(1, len(self._cutoffs) + 1):
                if self._div_val == 1.0:
                    ele_scores = all_scores_l[i]
                else:
                    ele_scores = self.out_proj_l[i](self.inter_proj_l[i](hidden))
                ele_logits = F.npx.log_softmax(ele_scores, axis=-1)
                ele_logits = tail_cluster_logits[-i] + ele_logits
                all_logits.append(ele_logits)
            return F.np.concatenate(all_logits, axis=-1)

    def hybrid_forward(self, F, hidden, target):
        """

        Parameters
        ----------
        F
        hidden
            The hidden representation
            Shape (..., in_units)
        target
            The target representation
            Shape (...,)

        Returns
        -------
        sel_logits
            The log probability that each hidden has when label == target
        """
        # TODO(sxjscience) The computation here can be greatly accelerated! Due to the
        #  missing feature of index_update, we are not able to do this here.
        logits = self.get_logits(F, hidden)
        sel_logits = F.npx.pick(logits, target, axis=-1)
        return sel_logits

    def __repr__(self):
        return _gen_repr_with_kwargs(self._kwargs, self.__class__.__name__)


@use_np
class SelfAttentiveSpanExtractor(HybridBlock):
    def __init__(self, in_units, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        pass

    def hybrid_forward(self, F, sequence_data, sequence_valid_length,
                       span_indices, span_valid_length):
        pass


@use_np
class EndpointSpanExtractor(HybridBlock):
    def __init__(self, in_units, combination, num_width_embeddings, span_width_embedding_dim,
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        pass

    def hybrid_forward(self, F, sequence_data, sequence_valid_length,
                       span_indices, span_valid_length):
        """

        Parameters
        ----------
        F
        sequence_data
            The token embeddings. Shape (B, T, C_in)
        sequence_valid_length
            Shape (B,)
        span_indices
            Shape (B, num_span, 2)
        span_valid_length
            Shape (B,)

        Returns
        -------
        span_features
            Shape (B, num_span, C_out)
        """
        pass
