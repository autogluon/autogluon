import math
import warnings

import numpy as np
import torch


def _pad_arrs_to_max_length(arrs, pad_axis, pad_val, round_to=None, max_length=None):
    """
    Inner Implementation of the Pad collate.

    Parameters
    ----------
        arrs (list)
        pad_axis (int)
        pad_val (number)
        round_to (int, optional). (default: ``None``)
        max_length (int, optional). (default: ``None``)

    Returns
    -------
        ret : torch.Tensor
        original_length : torch.Tensor
    """
    if not isinstance(arrs[0], torch.Tensor):
        arrs = [torch.as_tensor(ele) for ele in arrs]

    original_length = [ele.size(pad_axis) for ele in arrs]
    max_arr_len = max(original_length)

    if round_to is not None:
        max_arr_len = round_to * math.ceil(max_arr_len / round_to)
    elif max_length is not None:
        if max_length < max_arr_len:
            raise ValueError(
                f"If max_length is specified, max_length={max_length} must be larger "
                f"than the maximum length {max_arr_len} of the given arrays at axis={pad_axis}"
            )
        max_arr_len = max_length

    size = arrs[0].size()
    prev_trailing_dims = size[:pad_axis]
    after_trailing_dims = size[pad_axis + 1 :]

    out_dims = (len(arrs),) + prev_trailing_dims + (max_arr_len,) + after_trailing_dims
    out_tensor = arrs[0].new_full(out_dims, pad_val)
    for i, tensor in enumerate(arrs):
        length = tensor.size(pad_axis)
        out_tensor[i].narrow(pad_axis, 0, length)[:] = tensor

    original_length = torch.as_tensor(original_length)

    return out_tensor, original_length


def _stack_arrs(arrs):
    if isinstance(arrs[0], torch.Tensor):
        return torch.stack(arrs, 0)
    else:
        return _stack_arrs([torch.as_tensor(x) for x in arrs])


class StackCollator:
    """
    Stack the input data samples to construct the batch.
    The N input samples must have the same shape/length and will be stacked to construct a batch.
    """

    def __call__(self, data):
        """
        Collate the input data.

        Parameters
        ----------
            data (list): The input data samples.

        Returns
        -------
            batch_data (torch.Tensor)
        """
        return _stack_arrs(data)


class PadCollator:
    """
    Returns a callable that pads and stacks data.

    Parameters
    ----------
        axis (int, optional): The axis to pad the arrays.
            The arrays will be padded to the largest dimension at :attr:`axis`.
            For example, assume the input arrays have shape (10, 8, 5), (6, 8, 5), (3, 8, 5)
            and the `axis` is 0.
            Each input will be padded into (10, 8, 5) and then stacked to form the final output,
            which has shape（3, 10, 8, 5). (default ``0``)
        pad_val (float or int, optional): The padding value. (default ``0``)
        round_to (int, optional):
            If specified, the padded dimension will be rounded to be multiple of this argument.
            Mutually exclusive with :attr:`max_length`. (default ``None``)
        max_length (int, optional):
            If specified, the padded dimension will have length :attr:`max_length`,
            and it must be larger than the maximum length in the arrays at :attr:`axis`.
            Mutually exclusive with :attr:`round_to`.  (default ``None``)
        ret_length (bool, optional): Whether to return the valid length in the output.
            (default ``False``)
    """

    def __init__(self, axis=0, pad_val=0, round_to=None, max_length=None, ret_length=False):
        self._axis = axis
        if not isinstance(axis, int):
            raise ValueError(f"axis must be an integer! Received axis={axis}, type={type(axis)}.")

        if round_to is not None and max_length is not None:
            raise ValueError(f"Only either round_to={round_to} or max_length={max_length} can be specified.")

        self._pad_val = 0 if pad_val is None else pad_val
        self._round_to = round_to
        self._max_length = max_length
        self._ret_length = ret_length

        if pad_val is None:
            warnings.warn(
                "Padding value is not given and will be set automatically to 0 "
                "in data.Pad(). "
                "Please check whether this is intended "
                "(e.g. value of padding index in the tokenizer)."
            )

    def __call__(self, data):
        """
        Collate the input data.

        The arrays will be padded to the largest dimension at `axis` and then
        stacked to form the final output. In addition, the function will output
        the original dimensions at the `axis` if ret_length is turned on.

        Parameters
        ----------
            data : List[np.ndarray] or List[List[dtype]] or List[torch.Tensor]
                List of samples to pad and stack.

        Returns
        -------
            batch_data (torch.Tensor): Data in the minibatch. Shape is (N, ...)
            valid_length (NDArray, optional):
                The sequences' original lengths at the padded axis. Shape is (N,). This will only be
                returned if `ret_length` is True.

        """
        if isinstance(data[0], (torch.Tensor, np.ndarray, list, tuple)):
            padded_arr, original_length = _pad_arrs_to_max_length(
                data,
                pad_axis=self._axis,
                pad_val=self._pad_val,
                round_to=self._round_to,
                max_length=self._max_length,
            )
            if self._ret_length:
                return padded_arr, original_length
            else:
                return padded_arr
        else:
            raise NotImplementedError


class TupleCollator:
    """
    Wrap multiple data collator functions together. The input functions will be applied
    to the corresponding input fields.

    Each data sample should be a list or tuple containing multiple attributes. The `i`th collate
    function stored in `Tuple` will be applied on the `i`th attribute. For example, each
    data sample is (nd_data, label). You can wrap two collate functions using
    `Tuple(DataCollate, LabelCollate)` to collate nd_data and label correspondingly.

    Parameters
    ----------
        fn (list or tuple or callable): The collate functions to wrap.
        *args (tuple of callable, optional): The additional collate functions to wrap.

    """

    def __init__(self, fn, *args):
        if isinstance(fn, (list, tuple)):
            if len(args) != 0:
                raise ValueError(
                    "Input pattern not understood. "
                    "The input of Tuple can be Tuple(A, B, C) "
                    "or Tuple([A, B, C]) or Tuple((A, B, C)). "
                    f"Received fn={str(fn)}, args={str(args)}"
                )
            self._fn = fn
        else:
            self._fn = (fn,) + args
        for i, ele_fn in enumerate(self._fn):
            if not hasattr(ele_fn, "__call__"):
                raise ValueError(f"Collate functions must be callable! type(fn[{i}])={type(ele_fn)}")

    def __call__(self, data):
        """
        Collate the input data.

        Parameters
        ----------
            data (list): The samples to collate. Each sample should contain N attributes.

        Returns
        -------
            ret (tuple):
                A tuple of length N. Contains the collated result of each attribute in the input.
        """
        if len(data[0]) != len(self._fn):
            raise ValueError(f"The number of attributes in each data sample should contains {len(self._fn)} elements")
        ret = []
        for i, ele_fn in enumerate(self._fn):
            ret.append(ele_fn([ele[i] for ele in data]))
        return tuple(ret)


class ListCollator:
    """
    Simply forward the list of input data.

    This is particularly useful when the Dataset contains textual data
    and in conjunction with the `Tuple` collate function.

    """

    def __call__(self, data):
        """
        Parameters
        ----------
            data (list): The list of samples

        Returns
        -------
            ret (list): The input list
        """
        return list(data)


class DictCollator:
    """
    Wrap multiple collate functions together and apply it to merge inputs from a dict.

    The generated batch samples are stored as a dict with the same keywords.

    Each data sample should be a dict and the fn corresponds to `key` will be applied on the
    input with the keyword `key`.
    For example, each data sample is {'data': nd_data, 'label': nd_label}.
    You can merge the data and labels using
    `Dict({'data': DataCollate, 'label': LabelCollate})` to collate the nd_data and nd_label.

    Parameters
    ----------
        fn_dict (dict): A dictionary that contains the key-->collate function mapping.

    """

    def __init__(self, fn_dict):
        self._fn_dict = fn_dict
        if not isinstance(fn_dict, dict):
            raise ValueError(f"Input must be a dictionary! type of input={type(fn_dict)}")
        for fn in fn_dict.values():
            if not hasattr(fn, "__call__"):
                raise ValueError("Elements of the dictionary must be callable!")
        self._fn_dict = fn_dict

    def __call__(self, data):
        """

        Parameters
        ----------
            data (dict): The samples to collate. Each sample should be a dictionary

        Returns
        -------
            ret (dict): The resulting dictionary that stores the merged samples.
        """
        ret = dict()
        for k, ele_fn in self._fn_dict.items():
            ret[k] = ele_fn([ele[k] for ele in data])
        return ret
