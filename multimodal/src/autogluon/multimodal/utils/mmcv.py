import functools
from typing import Callable

import torch

try:
    from mmengine.dataset import pseudo_collate as collate
except ImportError as e:
    collate = None


def assert_tensor_type(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f"{args[0].__class__.__name__} has no attribute " f"{func.__name__} for type {args[0].datatype}"
            )
        return func(*args, **kwargs)

    return wrapper


class CollateMMDet:
    def __init__(self, samples_per_gpu):
        self.samples_per_gpu = samples_per_gpu

    def __call__(self, x):
        return collate(x)


class CollateMMOcr:
    def __init__(self, samples_per_gpu):
        self.samples_per_gpu = samples_per_gpu

    def __call__(self, x):
        return collate(x, samples_per_gpu=self.samples_per_gpu)
