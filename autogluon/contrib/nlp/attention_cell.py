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
"""Attention cells."""
import math
import numpy as np
import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from .layers import SinusoidalPositionalEmbedding,\
                    BucketPositionalEmbedding,\
                    LearnedPositionalEmbedding
from typing import Optional


# TODO(sxjscience)
#  We can optimize the whole function by writing a custom-op,
#  or automatically fuse these operators.
def gen_self_attn_mask(F, data,
                       valid_length=None,
                       dtype: type = np.float32,
                       attn_type: str = 'full'):
    """Generate the mask used for the encoder, i.e, self-attention.

    In our implementation, 1 --> not masked, 0 --> masked

    Let's consider the data with two samples:

    data =
        [['I',   'can', 'now',   'use', 'numpy', 'in',  'Gluon@@', 'NLP'  ],
         ['May', 'the', 'force', 'be',  'with',  'you', '<PAD>',   '<PAD>']]
    valid_length =
        [8, 6]

    - attn_type = 'causal'
        Each token will attend to itself + the tokens before.
        It will not attend to tokens in the future.

        For our example, the mask of the first sample is
                   ['I', 'can', 'now', 'use', 'numpy', 'in', 'Gluon@@', 'NLP']
        'I':         1,    0,     0,     0,      0,     0,      0,      0
        'can':       1,    1,     0,     0,      0,     0,      0,      0
        'now':       1,    1,     1,     0,      0,     0,      0,      0
        'use':       1,    1,     1,     1,      0,     0,      0,      0
        'numpy':     1,    1,     1,     1,      1,     0,      0,      0
        'in':        1,    1,     1,     1,      1,     1,      0,      0
        'Gluon@@':   1,    1,     1,     1,      1,     1,      1,      0
        'NLP':       1,    1,     1,     1,      1,     1,      1,      1

        The mask of the second sample is
                   ['May', 'the', 'force', 'be', 'with', 'you', '<PAD>', '<PAD>']
        'May':        1,    0,     0,     0,      0,     0,      0,      0
        'the':        1,    1,     0,     0,      0,     0,      0,      0
        'force':      1,    1,     1,     0,      0,     0,      0,      0
        'be':         1,    1,     1,     1,      0,     0,      0,      0
        'with':       1,    1,     1,     1,      1,     0,      0,      0
        'you':        1,    1,     1,     1,      1,     1,      0,      0
        '<PAD>':      0,    0,     0,     0,      0,     0,      0,      0
        '<PAD>':      0,    0,     0,     0,      0,     0,      0,      0


    - attn_type = 'full'
        Each token will attend to both the tokens before and in the future

        For our example, the mask of the first sample is
                   ['I', 'can', 'now', 'use', 'numpy', 'in', 'Gluon@@', 'NLP']
        'I':         1,    1,     1,     1,      1,     1,      1,      1
        'can':       1,    1,     1,     1,      1,     1,      1,      1
        'now':       1,    1,     1,     1,      1,     1,      1,      1
        'use':       1,    1,     1,     1,      1,     1,      1,      1
        'numpy':     1,    1,     1,     1,      1,     1,      1,      1
        'in':        1,    1,     1,     1,      1,     1,      1,      1
        'Gluon@@':   1,    1,     1,     1,      1,     1,      1,      1
        'NLP':       1,    1,     1,     1,      1,     1,      1,      1

        The mask of the second sample is
                   ['May', 'the', 'force', 'be', 'with', 'you', '<PAD>', '<PAD>']
        'May':        1,    1,     1,     1,      1,     1,      0,      0
        'the':        1,    1,     1,     1,      1,     1,      0,      0
        'force':      1,    1,     1,     1,      1,     1,      0,      0
        'be':         1,    1,     1,     1,      1,     1,      0,      0
        'with':       1,    1,     1,     1,      1,     1,      0,      0
        'you':        1,    1,     1,     1,      1,     1,      0,      0
        '<PAD>':      0,    0,     0,     0,      0,     0,      0,      0
        '<PAD>':      0,    0,     0,     0,      0,     0,      0,      0

    Parameters
    ----------
    F :
    data :
        The data. Shape (batch_size, seq_length, C)
    valid_length :
        Shape (batch_size,)
    dtype
        Data type of the mask
    attn_type : str
        Can be 'full' or 'causal'

    Returns
    -------
    mask
        Shape (batch_size, seq_length, seq_length)
    """
    if attn_type == 'full':
        if valid_length is not None:
            valid_length = valid_length.astype(dtype)
            steps = F.npx.arange_like(data, axis=1)  # (seq_length,)
            mask1 = (F.npx.reshape(steps, (1, 1, -1))
                     < F.npx.reshape(valid_length, (-2, 1, 1)))
            mask2 = (F.npx.reshape(steps, (1, -1, 1))
                     < F.npx.reshape(valid_length, (-2, 1, 1)))
            mask = mask1 * mask2
        else:
            # TODO(sxjscience) optimize
            seq_len_ones = F.np.ones_like(F.npx.arange_like(data, axis=1))  # (seq_length,)
            batch_ones = F.np.ones_like(F.npx.arange_like(data, axis=0))    # (batch_size,)
            mask = batch_ones.reshape((-1, 1, 1)) * seq_len_ones.reshape((1, -1, 1))\
                   * seq_len_ones.reshape((1, 1, -1))
    elif attn_type == 'causal':
        steps = F.npx.arange_like(data, axis=1)
        # mask: (seq_length, seq_length)
        # batch_mask: (batch_size, seq_length)
        mask = (F.np.expand_dims(steps, axis=0) <= F.np.expand_dims(steps, axis=1)).astype(dtype)
        if valid_length is not None:
            valid_length = valid_length.astype(dtype)
            batch_mask = (F.np.expand_dims(steps, axis=0) < F.np.expand_dims(valid_length, axis=-1)).astype(dtype)
            mask = mask * F.np.expand_dims(batch_mask, axis=-1)
        else:
            batch_ones = F.np.ones_like(F.npx.arange_like(data, axis=0), dtype=np.float32)  # (batch_size,)
            mask = mask * batch_ones.reshape((-1, 1, 1))
    else:
        raise NotImplementedError
    mask = mask.astype(dtype)
    return mask


def gen_mem_attn_mask(F, mem, mem_valid_length, data, data_valid_length=None, dtype=np.float32):
    """Generate the mask used for the decoder. All query slots are attended to the memory slots.

    In our implementation, 1 --> not masked, 0 --> masked

    Let's consider the data + mem with a batch of two samples:

    mem = [['I',   'can', 'now',   'use'],
           ['May', 'the', 'force', '<PAD>']]
    mem_valid_length =
        [4, 3]
    data =
        [['numpy', 'in',    'Gluon@@', 'NLP'  ],
         ['be',    'with',  'you',     '<PAD>']]
    data_valid_length =
        [4, 3]

    For our example, the mask of the first sample is
                   ['I', 'can', 'now', 'use']
        'numpy':     1,    1,     1,     1
        'in':        1,    1,     1,     1
        'Gluon@@':   1,    1,     1,     1
        'NLP':       1,    1,     1,     1

    The mask of the second sample is
                   ['be', 'with', 'you', '<PAD>']
        'May':        1,    1,     1,     0
        'the':        1,    1,     1,     0
        'force':      1,    1,     1,     0
        '<PAD>':      0,    0,     0,     0


    Parameters
    ----------
    F :
    mem :
        Shape (batch_size, mem_length, C_mem)
    mem_valid_length :
        Shape (batch_size,)
    data :
        Shape (batch_size, query_length, C_data)
    data_valid_length :
        Shape (batch_size,)
    dtype : type
        Data type of the mask

    Returns
    -------
    mask :
        Shape (batch_size, query_length, mem_length)
    """
    mem_valid_length = mem_valid_length.astype(dtype)
    mem_steps = F.npx.arange_like(mem, axis=1)  # (mem_length,)
    mem_mask = (F.npx.reshape(mem_steps, (1, 1, -1))
                < F.npx.reshape(mem_valid_length, (-2, 1, 1))).astype(dtype)  # (B, 1, mem_length)
    if data_valid_length is not None:
        data_valid_length = data_valid_length.astype(dtype)
        data_steps = F.npx.arange_like(data, axis=1)  # (query_length,)
        data_mask = (F.npx.reshape(data_steps, (1, -1, 1))
                     < F.npx.reshape(data_valid_length, (-2, 1, 1))).astype(dtype)  # (B, query_length, 1)
        mask = mem_mask * data_mask
    else:
        query_length_ones = F.np.ones_like(F.npx.arange_like(data, axis=1))  # (query_length,)
        mask = query_length_ones.reshape((1, -1, 1)) * mem_mask
    return mask


# TODO(sxjscience) Directly implement a kernel for masked softmax
def masked_softmax(F, att_score, mask, dtype=np.float32, axis: int = -1):
    """Ignore the masked elements when calculating the softmax. The mask can be broadcastable.

    Parameters
    ----------
    F : symbol or ndarray
    att_score : Symborl or NDArray
        Shape (..., length, ...)
    mask : Symbol or NDArray or None
        Shape (..., length, ...)
        1 --> The element is not masked
        0 --> The element is masked
    dtype
        data type
    axis
        The axis to calculate the softmax. att_score.shape[axis] must be the same as mask.shape[axis]
    Returns
    -------
    att_weights : Symborl or NDArray
        Shape (..., length, ...)
    """
    if mask is not None:
        # Fill in the masked scores with a very small value
        neg = -1e18
        if np.dtype(dtype) == np.float16:
            neg = -1e4
        else:
            try:
                # if AMP (automatic mixed precision) is enabled, -1e18 will cause NaN.
                from mxnet.contrib import amp
                if amp.amp._amp_initialized:
                    neg = -1e4
            except ImportError:
                pass

        att_score = F.np.where(mask, att_score, neg)
        logits = F.npx.softmax(att_score, axis=axis) * mask
    else:
        logits = F.npx.softmax(att_score, axis=axis)
    return logits


# TODO(sxjscience) Directly implement a kernel for masked logsoftmax
def masked_logsoftmax(F, att_score, mask, dtype=np.float32, axis: int = -1):
    """Ignore the masked elements when calculating the softmax. The mask can be broadcastable.

    Parameters
    ----------
    F : symbol or ndarray
    att_score : Symborl or NDArray
        Shape (..., length, ...)
    mask : Symbol or NDArray or None
        Shape (..., length, ...)
        mask = 1 --> not masked
        mask = 0 --> masked
    dtype
        data type
    axis
        The axis to calculate the softmax. att_score.shape[axis] must be the same as mask.shape[axis]
    Returns
    -------
    logits : Symborl or NDArray
        Shape (..., length, ...)
        The masked values will be all zero
    """
    if mask is not None:
        # Fill in the masked scores with a very small value
        neg = -1e18
        if np.dtype(dtype) == np.float16:
            neg = -1e4
        else:
            try:
                # if AMP (automatic mixed precision) is enabled, -1e18 will cause NaN.
                from mxnet.contrib import amp
                if amp.amp._amp_initialized:
                    neg = -1e4
            except ImportError:
                pass
        att_score = F.np.where(mask, att_score, neg)
        logits = F.np.where(mask, F.npx.log_softmax(att_score, axis=axis), -np.inf)
    else:
        logits = F.npx.log_softmax(att_score, axis=axis)
    return logits


def l2_normalize(F, data, axis=-1, eps=1E-6):
    """Normalize the data by L2 normalization.

    Parameters
    ----------
    F : mx.sym or mx.nd
    data : symbol or ndarray
    axis : int, default -1
    eps : float, default 1E-6

    Returns
    -------
    ret : mx.sym or mx.nd
    """
    ret = data / (F.np.linalg.norm(data, axis=axis, keepdims=True) + eps)
    return ret


# TODO(sxjscience) Default to einsum. Current it is not the default because
#   1) einsum is super-slow: https://github.com/apache/incubator-mxnet/issues/18043
def dot_attn_score(F, query, key, scaled=True, normalized=False, eps=1E-6,
                   layout='NT'):
    """The inner function call to calculate the score used in dot-product attention.

    We support multiple leading batch dimensions.

    scaled is True:
        D(h_q, h_k) = <h_q, h_k> / sqrt(dim_q)

    normalized is True:
            D(h_q, h_k) = <h_q / ||h_q||, h_k / ||h_k||>

    both scaled and normalized:
            D(h_q, h_k) = <h_q / ||h_q||, h_k / ||h_k||> / sqrt(dim_q)

    Parameters
    ----------
    F : mx.sym or mx.nd
    query : symbol or ndarray
        - layout is 'NT'
            (B0, ..., BN, query_length, query_dim)
        - layout is 'TN'
            (query_length, B0, ..., BN, query_dim)
    key : symbol or ndarray
        - layout is 'NT'
            (B0, ..., BN, key_length, key_dim)
        - layout is 'TN'
            (key_length, B0, ..., BN, key_dim)
    scaled : bool
        Whether to divide the query by the square-root of the query_dim
        If True: D(h_q, h_k) = <h_q, h_k> / sqrt(dim_q)
    normalized : bool
        Whether to normalize the query and the key embeddings
        If True: D(h_q, h_k) = <h_q / ||h_q||, h_k / ||h_k||>
    eps : float
        The epsilon used in the normalization
    layout
        The layout of the layer. Can be 'TN' or 'NT'.

    Returns
    -------
    scores : symbol or ndarray
        (B0, ..., BN, query_length, key_length)
    """
    if normalized:
        query = l2_normalize(F, query, -1, eps=eps)
        key = l2_normalize(F, key, -1, eps=eps)
    if scaled:
        query_shape = F.npx.shape_array(query)
        # TODO(sxjscience) Remove .astype(np.float32).
        #  Wait for https://github.com/apache/incubator-mxnet/issues/18084
        query_units = query_shape[-1].astype(np.float32)
        query = query / F.np.sqrt(query_units)
    if layout == 'NT':
        scores = F.npx.batch_dot(query, key, transpose_b=True)
    else:
        raise NotImplementedError('layout={} is not supported.'
                                  ' Currently, only layout = "NT" is implemented!'.format(layout))
    return scores


def multi_head_dot_attn(F, query, key, value,
                        mask=None,
                        edge_scores=None,
                        dropout: float = 0.0,
                        scaled: bool = True, normalized: bool = False,
                        eps: float = 1E-6, query_head_units: Optional[int] = None,
                        layout: str = 'NKT',
                        use_einsum: bool = False):
    """Multihead dot product attention between the query, key, value.

    scaled is False, normalized is False:
        D(h_q, h_k) = <h_q, h_k>
    scaled is True, normalized is False:
        D(h_q, h_k) = <h_q, h_k> / sqrt(dim_q)
    scaled is False, normalized is True:
        D(h_q, h_k) = <h_q / ||h_q||, h_k / ||h_k||>
    scaled is True, normalized is True:
        D(h_q, h_k) = <h_q / ||h_q||, h_k / ||h_k||> / sqrt(dim_q)

    If edge_scores is provided, we will calcualte the attention as
        scores = D(h_q, h_k) + EdgeScore_{q, k}

    Parameters
    ----------
    F
    query
        Query. The shape depends on the layout
        - layout is 'NKT'
            Shape (batch_size, num_heads, query_length, key_dim)
        - layout is 'NTK'
            Shape (batch_size, query_length, num_heads, key_dim)
        - layout is 'TNK'
            Shape (query_length, batch_size, num_heads, key_dim)
    key
        Key. The shape depends on the layout
        - layout is 'NKT'
            Shape (batch_size, num_heads, mem_length, key_dim)
        - layout is 'NTK'
            Shape (batch_size, mem_length, num_heads, key_dim)
        - layout is 'TNK'
            Shape (mem_length, batch_size, num_heads, key_dim)
    value
        Value. The shape depends on the layout
        - layout is 'NKT'
            Shape (batch_size, num_heads, mem_length, value_dim)
        - layout is 'NTK'
            Shape (batch_size, mem_length, num_heads, value_dim)
        - layout is 'TNK'
            Shape (mem_length, batch_size, num_heads, value_dim)
    mask
        Mask between query and memory. Shape (batch_size, query_length, mem_length)
    edge_scores
        The edge attention score. Shape can be any shape that is broadcastable to
        (batch_size, num_heads, query_length, mem_length)
    dropout
        Dropout rate
    scaled
        Whether to divide the attention weights by the sqrt of the query dimension.
        This is first proposed in "[NIPS2017] Attention is all you need."::

            score = <h_q, h_k> / sqrt(dim_q)

    normalized
        If turned on, the cosine distance is used, i.e::

            score = <h_q / ||h_q||, h_k / ||h_k||>

    eps
        The epsilon value used in L2 normalization
    query_head_units
        The units of each query head. If it's empty, we will estimate it via the
        shape_array of the query.
    layout
        This stands for the layout of the attention cell. The shape of the input/output will depend
        on the layout. Currently, we support 'NKT', 'NTK' and 'TNK' in which
        'N' means the batch_size, 'K' means the head, and 'T' means the length dimension.
    use_einsum
        Whether to use einsum for the computation

    Returns
    -------
    context_vec
        - layout is 'NKT' or 'NTK'
            Shape (batch_size, query_length, num_heads * value_units)
        - layout is 'TNK'
            Shape (query_length, batch_size, num_heads * value_units)
    additional_info
        scores:
            Shape (batch_size, num_head, query_length, mem_length)
        attn_weight:
            Shape (batch_size, num_head, query_length, mem_length)
    """
    # TODO(sxjscience) Profile layout
    if normalized:
        query = l2_normalize(F, query, axis=-1, eps=eps)
        key = l2_normalize(F, key, axis=-1, eps=eps)
    if scaled:
        if query_head_units is None:
            query_shape = F.npx.shape_array(query)
            scale = F.np.sqrt(query_shape[-1])
        else:
            scale = math.sqrt(query_head_units)
    else:
        scale = None
    if layout == 'NKT':
        # 1. Expand the dimension of the mask:
        #   (B, L_query, L_mem) --> (B, 1, L_query, L_mem)
        if mask is not None:
            mask = F.np.expand_dims(mask, axis=1)
        # 2. Calculate the attention weights
        #   Score: (B, N, L_query, C_Q) X (B, N, L_mem, C_Q) --> (B, N, L_query, L_mem)
        scores = F.npx.batch_dot(query, key, transpose_b=True)
        if edge_scores is not None:
            scores = scores + edge_scores
        if scaled:
            scores = scores / scale
        attn_weights = masked_softmax(F, scores, mask, axis=-1)
        attn_weights = F.npx.dropout(attn_weights, p=dropout)
        # 3. Calculate the context vector
        # (B, N, L_query, L_mem) X (B, N, L_mem, C_V) --> (B, L_query, N * C_V)
        if use_einsum:
            context_vec = F.np.einsum('bnij,bnjc->binc', attn_weights, value)
        else:
            context_vec = F.npx.batch_dot(attn_weights, value).transpose((0, 2, 1, 3))
        context_vec = F.npx.reshape(context_vec, (-2, -2, -1))
    elif layout == 'NTK':
        # 1. Expand the dimension of the mask:
        #   (B, L_query, L_mem) --> (B, 1, L_query, L_mem)
        if mask is not None:
            mask = F.np.expand_dims(mask, axis=1)
        # 2. Calculate the attention weights
        #   Score: (B, L_query, N, C_Q) X (B, L_mem, N, C_Q) --> (B, N, L_query, L_mem)
        if use_einsum:
            scores = F.np.einsum('binc,bjnc->bnij', query, key)
        else:
            scores = F.npx.batch_dot(F.np.swapaxes(query, 1, 2), F.np.swapaxes(key, 1, 2),
                                     transpose_b=True)
        if edge_scores is not None:
            scores = scores + edge_scores
        if scaled:
            scores = scores / scale
        attn_weights = masked_softmax(F, scores, mask)
        attn_weights = F.npx.dropout(attn_weights, p=dropout)
        # 3. Calculate the context vector
        # (B, N, L_query, L_mem) X (B, L_mem, N, C_V) --> (B, L_query, N * C_V)
        if use_einsum:
            context_vec = F.np.einsum('bnij,bjnc->binc', attn_weights, value)
        else:
            context_vec = F.npx.batch_dot(attn_weights,
                                          F.np.swapaxes(value, 1, 2)).transpose((0, 2, 1, 3))
        context_vec = F.npx.reshape(context_vec, (-2, -2, -1))
    elif layout == 'TNK':
        # 1. Expand the dimension of the mask:
        #   (B, L_query, L_mem) --> (B, 1, L_query, L_mem)
        if mask is not None:
            mask = F.np.expand_dims(mask, axis=1)
        # 2. Calculate the attention weights
        #   Score: (L_query, B, N, C_Q) X (L_mem, B, N, C_Q) --> (B, N, L_query, L_mem)
        #   This layout structure can be implemented very efficiently because B, N are consecutive
        #   to each other. To have a clear picture of what's happening, we may consider the
        #   (i, j)th element of the output
        #       out[i, j, :, :] = query[:, i, j, :] X key[:, i, j, :].T, which is just one GEMM call
        #   We can thus implement the whole kernel via a single call of batched GEMM with stride.
        if use_einsum:
            scores = F.np.einsum('ibnc,jbnc->bnij', query, key)
        else:
            scores = F.npx.batch_dot(query.transpose((1, 2, 0, 3)),
                                     key.transpose((1, 2, 3, 0)))
        if edge_scores is not None:
            scores = scores + edge_scores
        if scaled:
            scores = scores / scale
        attn_weights = masked_softmax(F, scores, mask)
        attn_weights = F.npx.dropout(attn_weights, p=dropout)
        # 3. Calculate the context vector
        # (B, N, L_query, L_mem) X (L_mem, B, N, C_V) --> (L_query, B, N * C_V)
        # Again, we can implement it via a single call to batched GEMM with stride.

        # Shape (B, N, L_query, C_V)
        if use_einsum:
            context_vec = F.np.einsum('bnij,jbnc->ibnc', attn_weights, value)
        else:
            context_vec = F.npx.batch_dot(attn_weights,
                                          value.transpose((1, 2, 0, 3))).transpose((2, 0, 1, 3))
        context_vec = F.npx.reshape(context_vec, (-2, -2, -1))
    else:
        raise NotImplementedError('layout="{}" is not supported! '
                                  'We only support layout = "NKT", "NTK", and "TNK".'
                                  .format(layout))
    return context_vec, [scores, attn_weights]


class MultiHeadAttentionCell(HybridBlock):
    """The multi-head attention

    out = softmax(<Q_i, K_j> + R_{i, j}) V

    We support multiple layouts

    Let's denote batch_size as B, num_heads as K,
     query_length as L_q, mem_length as L_m, key_dim as C_k, value_dim as C_v

    - layout="NKT"
        query: (B, K, L_q, C_k)
        key: (B, K, L_m, C_k)
        value: (B, K, L_m, C_v)
        out: (B, L_q, K * C_v)
    - layout="NTK"
        query: (B, L_q, K, C_k)
        key: (B, L_m, K, C_k)
        value: (B, L_m, K, C_v)
        out: (B, L_q, K * C_v)
    - layout="TNK"
        query: (L_q, B, K, C_k)
        key: (L_m, B, K, C_k)
        value: (L_m, B, K, C_v)
        out: (L_q, B, K * C_v)
    """
    def __init__(self, query_units=None, num_heads=None, attention_dropout=0.0,
                 scaled: bool = True, normalized: bool = False, eps: float = 1E-6,
                 dtype='float32', layout='NTK', use_einsum=False,
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._query_units = query_units
        self._num_heads = num_heads
        self._attention_dropout = attention_dropout
        self._scaled = scaled
        self._normalized = normalized
        self._eps = eps
        self._dtype = dtype
        self._layout = layout
        self._use_einsum = use_einsum
        if self._query_units is not None:
            assert self._num_heads is not None
            assert self._query_units % self._num_heads == 0,\
                'The units must be divisible by the number of heads.'
            self._query_head_units = self._query_units // self._num_heads
        else:
            self._query_head_units = None

    def hybrid_forward(self, F, query, key, value, mask=None, edge_scores=None):
        return multi_head_dot_attn(F, query=query, key=key, value=value,
                                   mask=mask, edge_scores=edge_scores,
                                   dropout=self._attention_dropout,
                                   scaled=self._scaled, normalized=self._normalized,
                                   eps=self._eps,
                                   query_head_units=self._query_head_units,
                                   layout=self._layout, use_einsum=self._use_einsum)

    def __repr__(self):
        s = '{name}(\n' \
            '   query_units={query_units},\n' \
            '   num_heads={num_heads},\n' \
            '   attention_dropout={attention_dropout},\n' \
            '   scaled={scaled},\n' \
            '   normalized={normalized},\n' \
            '   layout="{layout}",\n' \
            '   use_einsum={use_einsum},\n' \
            '   dtype={dtype}\n' \
            ')'
        return s.format(name=self.__class__.__name__,
                        query_units=self._query_units,
                        num_heads=self._num_heads,
                        attention_dropout=self._attention_dropout,
                        scaled=self._scaled,
                        normalized=self._normalized,
                        layout=self._layout,
                        use_einsum=self._use_einsum,
                        dtype=self._dtype)


class RelAttentionScoreCell(HybridBlock):
    r"""Get the score based on the query and relative position index. This is used for implementing
     relative attention.

    For the multi-head attention with relative positional encoding, we have the formula:

    out = softmax(\frac(Q K^T + R}{\sqrt(d)}) V

    Here, R is the relative positional encoding matrix. Usually, R_{i, j} is calculate based on the
    relative positional difference $i - j$.

    This function aims at generating the R matrix given the query and the relative positions.
    We support the following methods:

    - method = 'transformer_xl'
        R_{i, j} = <Q, W S_{i - j}>, in which S_{i, j} is the sinusoidal embedding and
        W is a Dense layer that maps S_{i - j} to the same dimension as the query.
        This is proposed in paper:

        - [ACL2019] Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

    - method = 'shaw'
        R_{i, j} = < Q, E_{i - j}>, in which E_{i - j} is the learned positional embedding
        This is proposed in paper:

        - [NAACL2018] Self-Attention with Relative Position Representations
    - method = 't5'
        R_{i, j} = E_{i - j}, in which E_{i - j} is the bucket positional embedding.
        This is proposed in paper:

        - [Arxiv2019] Exploring the Limits of Transfer Learning with a Unified
        Text-to-Text Transformer

    Like in MultiHeadAttentionCell, we support different layouts to cope with the query matrix.

    - layout="NKT"
        query: (B, K, L_q, C_k)
    - layout="NTK"
        query: (B, L_q, K, C_k)
    - layout="TNK"
        query: (L_q, B, K, C_k)
    """
    def __init__(self, query_units,
                 num_heads,
                 pos_embed_units: Optional[int] = None,
                 max_distance=None,
                 bidirectional=False,
                 num_buckets=None,
                 method='transformer_xl',
                 dropout: float = 0.0,
                 dtype='float32',
                 layout='NTK',
                 use_einsum=False,
                 prefix=None, params=None):
        """

        Parameters
        ----------
        query_units
        num_heads
        pos_embed_units
        max_distance
        bidirectional
        num_buckets
        method
        dropout
        attention_dropout
        query_add_bias
            Add additional bias term to the query
        scaled
        dtype
        layout
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        self._dropout = dropout
        self._method = method
        self._query_units = query_units
        self._num_heads = num_heads
        self._bidirectional = bidirectional
        self._num_buckets = num_buckets
        assert query_units % num_heads == 0, 'The units must be divisible by the number of heads.'
        self._head_query_units = query_units // num_heads
        self._max_distance = max_distance
        self._pos_embed_units = pos_embed_units
        self._dtype = dtype
        self._use_einsum = use_einsum
        self._layout = layout
        if self._layout not in ['NKT', 'NTK', 'TNK']:
            raise ValueError('layout="{}" is not supported'.format(self._layout))
        with self.name_scope():
            if method == 'transformer_xl':
                if pos_embed_units is None:
                    pos_embed_units = self._num_heads * self._head_query_units
                self._rel_pos_embed = SinusoidalPositionalEmbedding(units=pos_embed_units,
                                                                    prefix='rel_pos_embed_',
                                                                    dtype=self._dtype)
                self._rel_proj = nn.Dense(units=query_units,
                                          in_units=pos_embed_units,
                                          flatten=False,
                                          use_bias=False,
                                          prefix='rel_proj_',
                                          dtype=self._dtype)
                self._dropout_layer = nn.Dropout(dropout)
            elif method == 'shaw':
                assert self._max_distance is not None, 'Must set max_distance when method="shaw".'
                if self._bidirectional:
                    vocab_size = self._max_distance * 2 + 1
                else:
                    vocab_size = self._max_distance + 1
                self._rel_pos_embed = LearnedPositionalEmbedding(
                    units=self._num_heads * self._head_query_units,
                    max_length=vocab_size,
                    weight_initializer=mx.init.Xavier(rnd_type="gaussian",
                                                      factor_type="in",
                                                      magnitude=1),
                    prefix='rel_pos_embed_',
                    mode='wrap' if self._bidirectional else 'raise',
                    dtype=self._dtype)
            elif method == 't5':
                if self._num_buckets is None:
                    self._num_buckets = 32
                if self._max_distance is None:
                    self._max_distance = 128
                self._rel_pos_embed = BucketPositionalEmbedding(
                    units=num_heads,
                    num_buckets=self._num_buckets,
                    max_distance=self._max_distance,
                    bidirectional=self._bidirectional,
                    prefix='rel_pos_embed_',
                    dtype=self._dtype)
            else:
                raise NotImplementedError('method="{}" is currently not supported!'.format(method))

    def hybrid_forward(self, F, rel_positions, query=None):
        """

        Parameters
        ----------
        F
        rel_positions
            The relative shifts. Shape (query_length, mem_length)
            Each element represents the shift between the i-th element of query and the j-th
            element of memory.
        query
            The query for computing the relative scores. The shape depends on the layout.
            If we use T5 attention, the query won't be used

        Returns
        -------
        rel_scores
            The relative attention scores
            Can have shape (batch_size, num_heads, query_length, mem_length)
             or (num_heads, query_length, mem_length)
        """
        if self._method == 'transformer_xl' or self._method == 'shaw':
            assert query is not None, 'Must specify query if method={}'.format(self._method)
            if self._bidirectional:
                if self._max_distance is not None:
                    rel_positions = F.np.clip(rel_positions,
                                              a_min=-self._max_distance, a_max=self._max_distance)
            else:
                if self._max_distance is not None:
                    rel_positions = F.np.clip(rel_positions,
                                              a_min=0, a_max=self._max_distance)
            # uniq_rel.shape = (#uniq,), rev_index.shape = (L_q, L_m)
            uniq_rel, rev_index = F.np.unique(rel_positions, return_inverse=True)

            uniq_rel_pos_embed = self._rel_pos_embed(uniq_rel)
            if self._method == 'transformer_xl':
                uniq_rel_pos_embed = self._rel_proj(self._dropout_layer(uniq_rel_pos_embed))
            # Shape (#uniq, K, C_q)
            uniq_rel_pos_embed = F.npx.reshape(uniq_rel_pos_embed,
                                               (-2, self._num_heads, self._head_query_units))
            # Calculate the dot-product between query and the relative positional embeddings.
            # After the calculation, rel_score.shape = (L_q, #uniq, N, K)
            if self._layout == 'NKT':
                # query_for_rel: (N, K, L_q, C_q)
                if self._use_einsum:
                    rel_score = F.np.einsum('bnid,jnd->ijbn', query, uniq_rel_pos_embed)
                else:
                    rel_score = F.np.transpose(
                        F.np.matmul(query,
                                    F.np.transpose(uniq_rel_pos_embed, (1, 2, 0))),
                        (2, 3, 0, 1)
                    )
            elif self._layout == 'NTK':
                # query_for_rel: (N, L_q, K, C_q)
                if self._use_einsum:
                    rel_score = F.np.einsum('bind,jnd->ijbn', query, uniq_rel_pos_embed)
                else:
                    rel_score = F.np.transpose(
                        F.np.matmul(F.np.swapaxes(query, 1, 2),
                                    F.np.transpose(uniq_rel_pos_embed, (1, 2, 0))),
                        (2, 3, 0, 1)
                    )
            elif self._layout == 'TNK':
                # query_for_rel: (L_q, N, K, C_q)
                if self._use_einsum:
                    rel_score = F.np.einsum('ibnd,jnd->ijbn', query, uniq_rel_pos_embed)
                else:
                    rel_score = F.np.transpose(
                        F.np.matmul(F.np.transpose(query, (1, 2, 0, 3)),
                                    F.np.transpose(uniq_rel_pos_embed, (1, 2, 0))),
                        (2, 3, 0, 1)
                    )
            else:
                raise NotImplementedError
            # We use gather_nd to select the elements
            # TODO(sxjscience) Use advanced indexing once available
            rev_index = F.npx.reshape_like(rev_index, rel_positions).astype(np.int32)
            query_idx = F.np.expand_dims(F.npx.arange_like(rel_positions, axis=0).astype(np.int32),
                                         axis=-1) + F.np.zeros_like(rev_index)
            rel_score = F.npx.gather_nd(rel_score, F.np.stack([query_idx, rev_index]))
            rel_score = F.np.transpose(rel_score, (2, 3, 0, 1))
        elif self._method == 't5':
            # shape is (K, L_q, L_m)
            rel_score = self._rel_pos_embed(rel_positions).transpose((2, 0, 1))
        else:
            raise NotImplementedError
        return rel_score
