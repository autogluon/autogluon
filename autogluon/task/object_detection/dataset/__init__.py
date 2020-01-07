from ....core import *
from .base import *
from .voc import *
from .coco import *

import logging
logger = logging.getLogger(__name__)

def get_dataset(root, index_file_name='trainval', name=None, classes=None, \
                format='voc', Train=True, **kwargs):
    """ Get dataset to use for object detection, which must be in either VOC or COCO format.
    """
    if format=='voc':
        logger.info(">>> create dataset(VOC format) ")

        splits = [('', index_file_name)]
        return CustomVOCDetection(root, splits, name, classes, **kwargs)

    elif format=='coco':
        logger.info(">>> create dataset(COCO format)")
        return COCO(*args, **kwargs)
    else:
        raise NotImplementedError('Other data formats are not implemented.')

