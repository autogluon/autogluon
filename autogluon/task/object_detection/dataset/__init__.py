from ....core import *
from .base import *
from .voc import *
from .coco import *

import logging
logger = logging.getLogger(__name__)

def get_dataset(root='~/.mxnet/datasets/voc', index_file_name='trainval', name=None, \
                classes=None, format='voc', Train=True, **kwargs):
    """
    Parameters
    ----------
    root : str
        root path to dataset folder
    index_file_name : str
        the name of the .txt file which constains images for training or testing. 
        this is only for custom dataset.
    name: str
        name for built-in dataset, ('voc', 'voc2007' or 'voc2012')
        when use built-in dataset, the index_file_name should be None.
    classes: tuple of strs
        users can specify classes for custom dataset 
        ex. classes = ('bike', 'bird', 'cat', ...)
    format: str
        data format. ('voc', 'coco')
    Train : bool
        specify Train/Test mode. It is only valid when name is not None.  
    """
    if format=='voc':
        logger.info(">>> create dataset(VOC format) ")

        # built-in dataset
        if name:
            if Train:
                if name=='voc':
                    splits = [('VOC2007', 'trainval'), ('VOC2012', 'trainval')]
                elif name=='voc2007':
                    splits = [('VOC2007', 'trainval')]
            else:
                splits= [('VOC2007', 'test')] 
        else:  # custom dataset
            splits = [('', index_file_name)]
        return CustomVOCDetection(root, splits, name, classes, **kwargs)

    elif format=='coco':
        logger.info(">>> create dataset(COOC format)")
        return COCO(*args, **kwargs)
    else:
        raise NotImplementedError('Other data formats are not implemented.')

