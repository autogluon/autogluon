from ....core import *
from .base import *
from .voc import *
from .coco import *

import logging
logger = logging.getLogger(__name__)

def get_dataset(dataset_name, *args, **kwargs):
    if dataset_name=="voc":
        logger.info(">>> create dataset: voc")
        return VOC(*args, **kwargs)
    elif dataset_name=="tiny_motorbike":
        logger.info(">>> create dataset: tiny motorbike")
        return TinyVOC(*args, **kwargs)
    elif dataset_name=="coco":
        logger.info(">>> create dataset: coco")
        return COCOC(*args, **kwargs)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))

