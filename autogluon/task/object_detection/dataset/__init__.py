from ....core import *
from .base import *
from .voc import *
from .coco import *


def get_dataset(dataset_name, *args, **kwargs):
    if dataset_name=="voc":
        print(">>> create dataset: voc")
        return VOC(*args, **kwargs)
    elif dataset_name=="tiny_voc":
        print(">>> create dataset: tiny voc")
        return TinyVOC(*args, **kwargs)
    elif dataset_name=="coco":
        print(">>> create dataset: coco")
        return COCOC(*args, **kwargs)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))

