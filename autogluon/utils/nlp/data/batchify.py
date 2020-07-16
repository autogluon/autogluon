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
"""Batchify functions. They can be used in Gluon data loader to help combine individual samples
into batches for fast processing."""
__all__ = ['Stack', 'Pad', 'Tuple', 'List', 'NamedTuple', 'Dict']

import warnings
import math
from typing import Dict as t_Dict, Callable as t_Callable,\
    NamedTuple as t_NamedTuple, List as t_List, Tuple as t_Tuple, AnyStr,\
    Union as t_Union

import numpy as np
import mxnet as mx
from mxnet.util import is_np_array


def _pad_arrs_to_max_length(arrs, pad_axis, pad_val, use_shared_mem, dtype, round_to=None):
    """Inner Implementation of the Pad batchify

    Parameters
    ----------
    arrs : list
    pad_axis : int
    pad_val : number
    use_shared_mem : bool, default False
    dtype :
    round_to : int

    Returns
    -------
    ret : NDArray
    original_length : NDArray
    """
    if isinstance(arrs[0], mx.nd.NDArray):
        dtype = dtype or arrs[0].dtype
        arrs = [arr.asnumpy() for arr in arrs]
    elif not isinstance(arrs[0], np.ndarray):
        arrs = [np.asarray(ele) for ele in arrs]
    else:
        dtype = dtype or arrs[0].dtype

    original_length = [ele.shape[pad_axis] for ele in arrs]
    max_size = max(original_length)
    if round_to is not None:
        max_size = round_to * math.ceil(max_size / round_to)

    ret_shape = list(arrs[0].shape)
    ret_shape[pad_axis] = max_size
    ret_shape = (len(arrs), ) + tuple(ret_shape)

    ret = np.full(shape=ret_shape, fill_value=pad_val, dtype=dtype)

    for i, arr in enumerate(arrs):
        if arr.shape[pad_axis] == max_size:
            ret[i] = arr
        else:
            slices = [slice(None) for _ in range(arr.ndim)]
            slices[pad_axis] = slice(0, arr.shape[pad_axis])
            if slices[pad_axis].start != slices[pad_axis].stop:
                slices = [slice(i, i + 1)] + slices
                ret[tuple(slices)] = arr

    ctx = mx.Context('cpu', 0) if use_shared_mem else mx.cpu()
    if is_np_array():
        ret = mx.np.array(ret, ctx=ctx, dtype=dtype)
    else:
        ret = mx.nd.array(ret, ctx=ctx, dtype=dtype)
    return ret


def _stack_arrs(arrs, use_shared_mem, dtype):
    if isinstance(arrs[0], mx.nd.NDArray):
        dtype = dtype or arrs[0].dtype
        if use_shared_mem:
            if is_np_array():
                out = mx.np.empty((len(arrs),) + arrs[0].shape, dtype=dtype,
                                  ctx=mx.Context('cpu_shared', 0))
                return mx.np.stack(arrs, out=out)
            else:
                out = mx.nd.empty((len(arrs),) + arrs[0].shape, dtype=dtype,
                                  ctx=mx.Context('cpu_shared', 0))
                return mx.nd.stack(*arrs, out=out)
        else:
            if is_np_array():
                return mx.np.stack(arrs)
            else:
                return mx.nd.stack(*arrs)
    else:
        out = np.asarray(arrs)
        dtype = dtype or out.dtype
        if use_shared_mem:
            if is_np_array():
                return mx.np.array(out, ctx=mx.Context('cpu_shared', 0), dtype=dtype)
            else:
                return mx.nd.array(out, ctx=mx.Context('cpu_shared', 0), dtype=dtype)
        else:
            if is_np_array():
                return mx.np.array(out, dtype=dtype)
            else:
                return mx.nd.array(out, dtype=dtype)


class Stack:
    r"""Stack the input data samples to construct the batch.

    The N input samples must have the same shape/length and will be stacked to construct a batch.

    Parameters
    ----------
    dtype : str or numpy.dtype, default None
        The value type of the output. If it is set to None, the input data type is used.

    Examples
    --------
    >>> import gluonnlp.data.batchify as bf
    >>> # Stack multiple lists
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6, 8]
    >>> c = [8, 9, 1, 2]
    >>> bf.Stack()([a, b, c])
    <BLANKLINE>
    [[1 2 3 4]
     [4 5 6 8]
     [8 9 1 2]]
    <NDArray 3x4 @cpu_shared(0)>
    >>> # Stack multiple numpy.ndarrays
    >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = np.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    >>> bf.Stack()([a, b])
    <BLANKLINE>
    [[[1 2 3 4]
      [5 6 7 8]]
    <BLANKLINE>
     [[5 6 7 8]
      [1 2 3 4]]]
    <NDArray 2x2x4 @cpu_shared(0)>
    >>> # Stack multiple NDArrays
    >>> a = mx.nd.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = mx.nd.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    >>> bf.Stack()([a, b])
    <BLANKLINE>
    [[[1. 2. 3. 4.]
      [5. 6. 7. 8.]]
    <BLANKLINE>
     [[5. 6. 7. 8.]
      [1. 2. 3. 4.]]]
    <NDArray 2x2x4 @cpu_shared(0)>
    """
    def __init__(self, dtype=None):
        self._dtype = dtype

    def __call__(self, data):
        """Batchify the input data

        Parameters
        ----------
        data : list
            The input data samples

        Returns
        -------
        batch_data : NDArray
        """
        return _stack_arrs(data, False, self._dtype)


class Pad:
    """Return a callable that pads and stacks data.

    Parameters
    ----------
    val : float or int, default 0
        The padding value.
    axis : int, default 0
        The axis to pad the arrays. The arrays will be padded to the largest dimension at
        `axis`. For example, assume the input arrays have shape
        (10, 8, 5), (6, 8, 5), (3, 8, 5) and the `axis` is 0. Each input will be padded into
        (10, 8, 5) and then stacked to form the final output, which has shape（3, 10, 8, 5).
    dtype : str or numpy.dtype, default None
        The value type of the output. If it is set to None, the input data type is used.
    round_to : int, default None
        If specified, the padded dimension will be rounded to be multiple of this argument.

    Examples
    --------
    >>> import gluonnlp.batchify as bf
    >>> # Inputs are multiple lists
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6]
    >>> c = [8, 2]
    >>> bf.Pad()([a, b, c])
    <BLANKLINE>
    [[1. 2. 3. 4.]
     [4. 5. 6. 0.]
     [8. 2. 0. 0.]]
    <NDArray 3x4 @cpu_shared(0)>
    >>> # Inputs are multiple numpy ndarrays
    >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = np.array([[5, 8], [1, 2]])
    >>> bf.Pad(axis=1, val=-1)([a, b])
    <BLANKLINE>
    [[[ 1  2  3  4]
      [ 5  6  7  8]]
    <BLANKLINE>
     [[ 5  8 -1 -1]
      [ 1  2 -1 -1]]]
    <NDArray 2x2x4 @cpu_shared(0)>
    """
    def __init__(self, val=0, axis=0, dtype=None, round_to=None):
        self._axis = axis
        assert isinstance(axis, int), 'axis must be an integer! ' \
                                      'Received axis=%s, type=%s.' % (str(axis),
                                                                      str(type(axis)))
        self._val = val
        self._dtype = dtype
        self._warned = False
        self._round_to = round_to

    def __call__(self, data):
        """Batchify the input data.

        The input can be list of numpy.ndarray, list of numbers or list of
        mxnet.nd.NDArray. Inputting mxnet.nd.NDArray is discouraged as each
        array need to be converted to numpy for efficient padding.

        The arrays will be padded to the largest dimension at `axis` and then
        stacked to form the final output. In addition, the function will output
        the original dimensions at the `axis` if ret_length is turned on.

        Parameters
        ----------
        data : List[np.ndarray] or List[List[dtype]] or List[mx.nd.NDArray]
            List of samples to pad and stack.

        Returns
        -------
        batch_data: NDArray
            Data in the minibatch. Shape is (N, ...)

        """

        if isinstance(data[0], mx.nd.NDArray) and not self._warned:
            self._warned = True
            #TODO(sxjscience) Investigate the warning
            warnings.warn(
                'Using Pad with NDArrays is discouraged for speed reasons. '
                'Instead you should pad your data while it is still a list '
                'and before converting to an NDArray. '
                'Alternatively you can consider inputting a numpy.ndarray.')
        if isinstance(data[0], (mx.nd.NDArray, np.ndarray, list)):
            padded_arr = _pad_arrs_to_max_length(data, self._axis, self._val, False, self._dtype,
                                                 round_to=self._round_to)
            return padded_arr
        else:
            raise NotImplementedError


class Tuple:
    """Wrap multiple batchify functions together. The input functions will be applied
    to the corresponding input fields.

    Each data sample should be a list or tuple containing multiple attributes. The `i`th batchify
    function stored in `Tuple` will be applied on the `i`th attribute. For example, each
    data sample is (nd_data, label). You can wrap two batchify functions using
    `Tuple(DataBatchify, LabelBatchify)` to batchify nd_data and label correspondingly.

    Parameters
    ----------
    fn : list or tuple or callable
        The batchify functions to wrap.
    *args : tuple of callable
        The additional batchify functions to wrap.

    Examples
    --------
    >>> import gluonnlp.data.batchify as bf
    >>> a = ([1, 2, 3, 4], 0)
    >>> b = ([5, 7], 1)
    >>> c = ([1, 2, 3, 4, 5, 6, 7], 0)
    >>> f1, f2 = bf.Tuple(bf.Pad(), bf.Stack())([a, b])
    >>> f1
    <BLANKLINE>
    [[1. 2. 3. 4.]
     [5. 7. 0. 0.]]
    <NDArray 2x4 @cpu_shared(0)>
    >>> f2
    <BLANKLINE>
    [0 1]
    <NDArray 2 @cpu_shared(0)>

    """
    def __init__(self, fn, *args):
        if isinstance(fn, (list, tuple)):
            assert len(args) == 0, 'Input pattern not understood. The input of Tuple can be ' \
                                   'Tuple(A, B, C) or Tuple([A, B, C]) or Tuple((A, B, C)). ' \
                                   'Received fn=%s, args=%s' % (str(fn), str(args))
            self._fn = fn
        else:
            self._fn = (fn, ) + args
        for i, ele_fn in enumerate(self._fn):
            assert hasattr(ele_fn, '__call__'), 'Batchify functions must be callable! ' \
                                                'type(fn[%d]) = %s' % (i, str(type(ele_fn)))

    def __call__(self, data):
        """Batchify the input data.

        Parameters
        ----------
        data : list
            The samples to batchfy. Each sample should contain N attributes.

        Returns
        -------
        ret : tuple
            A tuple of length N. Contains the batchified result of each attribute in the input.
        """
        assert len(data[0]) == len(self._fn),\
            'The number of attributes in each data sample should contains' \
            ' {} elements'.format(len(self._fn))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            ret.append(ele_fn([ele[i] for ele in data]))
        return tuple(ret)

class List:
    """Simply forward the list of input data.

    This is particularly useful when the Dataset contains textual data
    and in conjonction with the `Tuple` batchify function.

    Examples
    --------
    >>> import gluonnlp.data.batchify as bf
    >>> a = ([1, 2, 3, 4], "I am using MXNet")
    >>> b = ([5, 7, 2, 5], "Gluon rocks!")
    >>> c = ([1, 2, 3, 4], "Batchification!")
    >>> _, l = bf.Tuple(bf.Stack(), bf.List())([a, b, c])
    >>> l
    ['I am using MXNet', 'Gluon rocks!', 'Batchification!']
    """
    def __call__(self, data: t_List) -> t_List:
        """
        Parameters
        ----------
        data
            The list of samples

        Returns
        -------
        ret
            The input list
        """
        return list(data)


class Dict:
    """Wrap multiple batchify functions together and apply it to merge inputs from a dict.

    The generated batch samples are stored as a dict with the same keywords.

    Each data sample should be a dict and the fn corresponds to `key` will be applied on the
    input with the keyword `key`.
    For example, each data sample is {'data': nd_data, 'label': nd_label}.
    You can merge the data and labels using
    `Dict({'data': DataBatchify, 'label': LabelBatchify})` to batchify the nd_data and nd_label.

    Parameters
    ----------
    fn_dict
        A dictionary that contains the key-->batchify function mapping.

    Examples
    --------
    >>> from gluonnlp.data.batchify import Dict, Pad, Stack
    >>> a = {'data': [1, 2, 3, 4], 'label': 0}
    >>> b = {'data': [5, 7], 'label': 1}
    >>> c = {'data': [1, 2, 3, 4, 5, 6, 7], 'label': 0}
    >>> batchify_fn = Dict({'data': Pad(), 'label': Stack()})
    >>> sample = batchify_fn([a, b, c])
    >>> sample['data']
    <BLANKLINE>
    [[1. 2. 3. 4. 0. 0. 0.]
     [5. 7. 0. 0. 0. 0. 0.]
     [1. 2. 3. 4. 5. 6. 7.]]
    <NDArray 3x7 @cpu_shared(0)>
    >>> sample['label']
    <BLANKLINE>
    [0 1 0]
    <NDArray 3 @cpu_shared(0)>
    """
    def __init__(self, fn_dict: t_Dict[AnyStr, t_Callable]):
        self._fn_dict = fn_dict
        if not isinstance(fn_dict, dict):
            raise ValueError('Input must be a dictionary! type of input = {}'
                             .format(type(fn_dict)))
        for fn in fn_dict.values():
            if not hasattr(fn, '__call__'):
                raise ValueError('Elements of the dictionary must be callable!')
        self._fn_dict = fn_dict

    def __call__(self, data: t_List[t_Dict]) -> t_Dict:
        """

        Parameters
        ----------
        data
            The samples to batchify. Each sample should be a dictionary

        Returns
        -------
        ret
            The resulting dictionary that stores the merged samples.
        """
        ret = dict()
        for k, ele_fn in self._fn_dict.items():
            ret[k] = ele_fn([ele[k] for ele in data])
        return ret


class NamedTuple:
    """Wrap multiple batchify functions together and apply it to merge inputs from a namedtuple.

    The generated batch samples are stored as a namedtuple with the same structure.

    Each data sample should be a namedtuple. The `i`th batchify
    function stored in `NamedTuple` will be applied on the `i`th attribute of the namedtuple data.
    For example, each data sample is Sample(data=nd_data, label=nd_label).
    You can wrap two batchify functions using
    `NamedTuple(Sample, {'data': DataBatchify, 'label': LabelBatchify})` to
    batchify nd_data and nd_label correspondingly. The result will be stored as a Sample object
    and you can access the data and label via `sample.data` and `sample.label`, correspondingly.

    Parameters
    ----------
    container
        The object that constructs the namedtuple.
    fn_info
        The information of the inner batchify functions.

    Examples
    --------
    >>> from gluonnlp.data.batchify import NamedTuple, Pad, Stack
    >>> from collections import namedtuple
    >>> SampleData = namedtuple('SampleData', ['data', 'label'])
    >>> a = SampleData([1, 2, 3, 4], 0)
    >>> b = SampleData([5, 7], 1)
    >>> c = SampleData([1, 2, 3, 4, 5, 6, 7], 0)
    >>> batchify_fn = NamedTuple(SampleData, {'data': Pad(), 'label': Stack()})
    >>> sample = batchify_fn([a, b, c])
    >>> sample
    SampleData(data=
    [[1. 2. 3. 4. 0. 0. 0.]
     [5. 7. 0. 0. 0. 0. 0.]
     [1. 2. 3. 4. 5. 6. 7.]]
    <NDArray 3x7 @cpu_shared(0)>, label=
    [0 1 0]
    <NDArray 3 @cpu_shared(0)>)
    >>> sample.data
    <BLANKLINE>
    [[1. 2. 3. 4. 0. 0. 0.]
     [5. 7. 0. 0. 0. 0. 0.]
     [1. 2. 3. 4. 5. 6. 7.]]
    <NDArray 3x7 @cpu_shared(0)>
    >>> # Let's consider to use a list
    >>> batchify_fn = NamedTuple(SampleData, [Pad(), Stack()])
    >>> batchify_fn([a, b, c])
    SampleData(data=
    [[1. 2. 3. 4. 0. 0. 0.]
     [5. 7. 0. 0. 0. 0. 0.]
     [1. 2. 3. 4. 5. 6. 7.]]
    <NDArray 3x7 @cpu_shared(0)>, label=
    [0 1 0]
    <NDArray 3 @cpu_shared(0)>)
    """
    def __init__(self,
                 container: t_NamedTuple,
                 fn_info: t_Union[t_List[t_Callable],
                                  t_Tuple[t_Callable],
                                  t_Dict[AnyStr, t_Callable]]):
        self._container = container
        if isinstance(fn_info, (list, tuple)):
            if len(container._fields) != len(fn_info):
                raise ValueError('Attributes mismatch! Required fields={}, fn_info={}'
                                 .format(container._fields, fn_info))
        elif isinstance(fn_info, dict):
            for name in container._fields:
                if name not in fn_info:
                    raise ValueError('Attribute {} has not been assigned a callable. '
                                     'Required fields={}, Found fields={}'
                                     .format(name, container._fields, fn_info.keys()))
            if len(container._fields) != len(fn_info):
                raise ValueError('Attributes mimatch! Required fields={}, Found fields={}'
                                 .format(container._fields, fn_info.keys()))
            fn_info = [fn_info[name] for name in container._fields]
        for fn in fn_info:
            if not hasattr(fn, '__call__'):
                raise ValueError('All batchify functions must be callable.')
        self._fn_l = fn_info

    def __call__(self, data: t_List[t_NamedTuple]) -> t_NamedTuple:
        """Batchify the input data.

        Parameters
        ----------
        data
            The samples to batchfy. Each sample should be a namedtuple.

        Returns
        -------
        ret
            A namedtuple of length N. Contains the batchified result of each attribute in the input.
        """
        if not isinstance(data[0], self._container):
            raise ValueError('The samples should have the same type as the stored namedtuple.'
                             ' data[0]={}, container={}'.format(data[0], self._container))
        ret = []
        for i, ele_fn in enumerate(self._fn_l):
            ret.append(ele_fn([ele[i] for ele in data]))
        return self._container(*ret)
