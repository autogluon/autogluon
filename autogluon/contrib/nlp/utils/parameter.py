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
"""Utility functions for trainer and parameters."""
__all__ = ['grad_global_norm', 'clip_grad_global_norm']


import warnings

import numpy as np
import mxnet as mx
from collections import defaultdict
from mxnet.gluon import Parameter
from mxnet.util import use_np
from typing import Iterable, Optional, Tuple


@use_np
def grad_global_norm(parameters: Iterable[Parameter]) -> float:
    """Calculate the 2-norm of gradients of parameters, and how much they should be scaled down
    such that their 2-norm does not exceed `max_norm`, if `max_norm` if provided.
    If gradients exist for more than one context for a parameter, user needs to explicitly call
    ``trainer.allreduce_grads`` so that the gradients are summed first before calculating
    the 2-norm.

    .. note::
        This function is only for use when `update_on_kvstore` is set to False in trainer.

    Example::
        trainer = Trainer(net.collect_params(), update_on_kvstore=False, ...)
        for x, y in mx.gluon.utils.split_and_load(X, [mx.gpu(0), mx.gpu(1)]):
            with mx.autograd.record():
                y = net(x)
                loss = loss_fn(y, label)
            loss.backward()
        trainer.allreduce_grads()
        norm = grad_global_norm(net.collect_params().values())
        ...

    Parameters
    ----------
    parameters
        The list of Parameters

    Returns
    -------
    total_norm
        Total norm. It's a numpy scalar.
    """
    # Distribute gradients among contexts,
    # For example, assume there are 8 weights and four GPUs, we can ask each GPU to
    # compute the squared sum of two weights and then add the results together
    idx = 0
    arrays = defaultdict(list)
    sum_norms = []
    num_ctx = None
    for p in parameters:
        if p.grad_req != 'null':
            p_grads = p.list_grad()
            if num_ctx is None:
                num_ctx = len(p_grads)
            else:
                assert num_ctx == len(p_grads)
            arrays[idx % num_ctx].append(p_grads[idx % num_ctx])
            idx += 1
    assert len(arrays) > 0, 'No parameter found available for gradient norm.'

    # TODO(sxjscience)
    #  Investigate the float16 case.
    #  The inner computation accumulative type of norm should be float32.
    ctx = arrays[0][0].context
    for idx, arr_l in enumerate(arrays.values()):
        sum_norm = mx.np.linalg.norm(mx.np.concatenate([mx.np.ravel(ele) for ele in arr_l]))
        sum_norms.append(sum_norm.as_in_ctx(ctx))

    # Reduce over ctx
    if num_ctx == 1:
        total_norm = sum_norms[0]
    else:
        total_norm = mx.np.linalg.norm(mx.np.concatenate(sum_norms, axis=None))
    total_norm = float(total_norm)
    return total_norm


@use_np
def clip_grad_global_norm(parameters: Iterable[Parameter],
                          max_norm: float,
                          check_isfinite: bool = True) -> Tuple[float, float, bool]:
    """Rescales gradients of parameters so that the sum of their 2-norm is smaller than `max_norm`.
    If gradients exist for more than one context for a parameter, user needs to explicitly call
    ``trainer.allreduce_grads`` so that the gradients are summed first before calculating
    the 2-norm.

    .. note::
        This function is only for use when `update_on_kvstore` is set to False in trainer.
        In cases where training happens on multiple contexts, this method should be used in
        conjunction with ``trainer.allreduce_grads()`` and ``trainer.update()``.
        (**not** ``trainer.step()``)

    Example::
        trainer = Trainer(net.collect_params(), update_on_kvstore=False, ...)
        for x, y in mx.gluon.utils.split_and_load(X, [mx.gpu(0), mx.gpu(1)]):
            with mx.autograd.record():
                y = net(x)
                loss = loss_fn(y, label)
            loss.backward()
        trainer.allreduce_grads()
        nlp.utils.clip_grad_global_norm(net.collect_params().values(), max_norm)
        trainer.update(batch_size)
        ...

    Parameters
    ----------
    parameters
        The list of parameters to calculate the norm
    max_norm
        If the gradient norm is larger than max_norm, it will be clipped to have max_norm
    check_isfinite
         If True, check whether the total_norm is finite (not nan or inf).

    Returns
    -------
    total_norm
        The total norm
    ratio
        The expected clipping ratio: grad = grad / ratio
        It will be calculated as max(total_norm / max_norm, 1)
    is_finite
        Whether the total norm is finite
    """
    total_norm = grad_global_norm(parameters)
    is_finite = bool(np.isfinite(total_norm))
    ratio = np.maximum(1, total_norm / max_norm)
    if check_isfinite and not is_finite:
        warnings.warn(
            UserWarning('nan or inf is detected. Clipping results will be undefined.'
                        ' Thus, skip clipping'),
            stacklevel=2)
        return total_norm, ratio, is_finite
    scale = 1 / ratio
    for p in parameters:
        if p.grad_req != 'null':
            for arr in p.list_grad():
                arr *= scale
    return total_norm, ratio, is_finite


@use_np
def move_to_ctx(arr, ctx):
    """Move a nested structure of array to the given context

    Parameters
    ----------
    arr
        The input array
    ctx
        Context

    Returns
    -------
    new_arr
        The array that has been moved to context
    """
    if isinstance(arr, tuple):
        return tuple(move_to_ctx(ele, ctx) for ele in arr)
    elif isinstance(arr, list):
        return [move_to_ctx(ele, ctx) for ele in arr]
    else:
        return None if arr is None else arr.as_in_ctx(ctx)
