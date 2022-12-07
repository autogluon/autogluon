import functools
from collections import OrderedDict
from typing import Callable, Type, Union

import numpy as np
import torch


def assert_tensor_type(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f"{args[0].__class__.__name__} has no attribute " f"{func.__name__} for type {args[0].datatype}"
            )
        return func(*args, **kwargs)

    return wrapper


try:
    from mmcv.parallel import DataContainer, collate
except ImportError:

    class DataContainer:
        """A container for any type of objects.
        Typically tensors will be stacked in the collate function and sliced along
        some dimension in the scatter function. This behavior has some limitations.
        1. All tensors have to be the same size.
        2. Types are limited (numpy array or Tensor).
        We design `DataContainer` and `MMDataParallel` to overcome these
        limitations. The behavior can be either of the following.
        - copy to GPU, pad all tensors to the same size and stack them
        - copy to GPU without stacking
        - leave the objects as is and pass it to the model
        - pad_dims specifies the number of last few dimensions to do padding
        """

        def __init__(
            self,
            data: Union[torch.Tensor, np.ndarray],
            stack: bool = False,
            padding_value: int = 0,
            cpu_only: bool = False,
            pad_dims: int = 2,
        ):
            self._data = data
            self._cpu_only = cpu_only
            self._stack = stack
            self._padding_value = padding_value
            assert pad_dims in [None, 1, 2, 3]
            self._pad_dims = pad_dims

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}({repr(self.data)})"

        def __len__(self) -> int:
            return len(self._data)

        @property
        def data(self) -> Union[torch.Tensor, np.ndarray]:
            return self._data

        @property
        def datatype(self) -> Union[Type, str]:
            if isinstance(self.data, torch.Tensor):
                return self.data.type()
            else:
                return type(self.data)

        @property
        def cpu_only(self) -> bool:
            return self._cpu_only

        @property
        def stack(self) -> bool:
            return self._stack

        @property
        def padding_value(self) -> int:
            return self._padding_value

        @property
        def pad_dims(self) -> int:
            return self._pad_dims

        @assert_tensor_type
        def size(self, *args, **kwargs) -> torch.Size:
            return self.data.size(*args, **kwargs)

        @assert_tensor_type
        def dim(self) -> int:
            return self.data.dim()


def unpack_datacontainers(datacontainers):
    """
    Recursively unpack all `mmcv.parallel.DataContainer` objects from a dictionary.
    """
    if isinstance(datacontainers, DataContainer):
        return unpack_datacontainers(datacontainers.data)

    elif isinstance(datacontainers, dict):
        for k, v in datacontainers.items():
            datacontainers[k] = unpack_datacontainers(v)
        return datacontainers

    elif isinstance(datacontainers, list):
        for idx in range(len(datacontainers)):
            datacontainers[idx] = unpack_datacontainers(datacontainers[idx])
        return datacontainers

    else:
        return datacontainers


def send_datacontainers_to_device(data, device, dont_send=[]):
    """
    Recieves dictionary, send `mmcv.parallel.DataContainer` items that has
    `cpu_only` set to False to the device. Excludes items listed in `dont_send`.
    """
    for k, v in data.items():
        if isinstance(v, DataContainer) and not v.cpu_only:
            if k not in dont_send:
                datacontainer_to_cuda(v, device)


def datacontainer_to_cuda(container, device: Union[str, torch.device]):
    """
    Send :type:`~mmcv.parallel.DataContainer` to device. There are 3 cases.
    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    Parameters
    ----------
    container: DataContainer
        data container object. `container.data` should be either a single torch.Tensor or a
        list / dictionary of tensors.
    device: Union[str, torch.device]
        device to send the data to.
    """

    assert not container.cpu_only, f"{container} is not meant to be moved to {device}"
    if container.stack:
        assert isinstance(container.data, torch.Tensor), f"Expected `torch.Tensor` but got {type(container.data)}"
        container._data = container.data.to(device)
    else:
        if isinstance(container.data, torch.Tensor):
            container._data = container.data.to(device)
        else:
            if isinstance(container.data, list):
                it = range(len(container.data))
            elif isinstance(container.data, dict) or isinstance(container.data, OrderedDict):
                it = container.data.keys()
            else:
                raise TypeError(f"Unidentified iterator type: {type(container.data)}")

            for idx in it:
                assert isinstance(
                    container.data[idx], torch.Tensor
                ), f"Expected `torch.Tensor` but {container.data[idx]} has \
                    type: {type(container.data[idx])}"
                container._data[idx] = container.data[idx].to(device)


class CollateMMDet:
    def __init__(self, samples_per_gpu):
        self.samples_per_gpu = samples_per_gpu

    def __call__(self, x):
        from . import unpack_datacontainers

        ret = collate(x, samples_per_gpu=self.samples_per_gpu)
        ret = unpack_datacontainers(ret)
        if isinstance(ret["img_metas"][0][0], list):
            img_metas = ret["img_metas"][0]
            imgs = [ret["img"][0][0].float()]
            return dict(imgs=imgs, img_metas=img_metas)
        else:
            img_metas = ret["img_metas"][0]
            img = ret["img"][0].float()
            batch_size = img.shape[0]
            gt_bboxes = []
            gt_labels = []
            for i in range(batch_size):
                gt_bboxes.append(ret["gt_bboxes"][0][i].float())
                gt_labels.append(ret["gt_labels"][0][i].long())

            return dict(img=img, img_metas=img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels)


class CollateMMOcr:
    def __init__(self, samples_per_gpu):
        self.samples_per_gpu = samples_per_gpu

    def __call__(self, x):
        return collate(x, samples_per_gpu=self.samples_per_gpu)
