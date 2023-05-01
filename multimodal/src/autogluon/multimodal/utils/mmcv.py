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
    import mmengine

    # from mmengine.dataset import default_collate as collate
    from mmengine.dataset import pseudo_collate as collate
except ImportError as e:
    mmengine = None


class CollateMMDet:
    def __init__(self, samples_per_gpu):
        self.samples_per_gpu = samples_per_gpu

    def __call__(self, x):
        ret = collate(x)

        return ret

        if isinstance(ret["inputs"], list):
            img_metas = ret["img_metas"][0]
            imgs = [ret["inputs"][0][0].float()]
            return dict(imgs=imgs, img_metas=img_metas)
        else:
            raise NotImplementedError
            img_metas = ret["img_metas"][0]
            img = ret["img"][0].float()
            batch_size = img.shape[0]
            gt_bboxes = []
            gt_labels = []
            for i in range(batch_size):
                gt_bboxes.append(ret["gt_bboxes"][0][i].float())
                gt_labels.append(ret["gt_labels"][0][i].long())

            return dict(img=img, img_metas=img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels)

        return ret


class CollateMMOcr:
    def __init__(self, samples_per_gpu):
        self.samples_per_gpu = samples_per_gpu

    def __call__(self, x):
        return collate(x, samples_per_gpu=self.samples_per_gpu)
