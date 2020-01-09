from ....core import *
from .base import *
from .voc import *
from .coco import *

import logging
logger = logging.getLogger(__name__)

def get_dataset(root, index_file_name='trainval', name=None, classes=None, \
                format='voc', Train=True, **kwargs):
    """ Load dataset to use for object detection, which must be in either VOC or COCO format.
        
        Parameters
        ----------
        root : str
            Path to folder storing the dataset.
        index_file_name : str
            Name of file containing the training/validation indices of each text example.
        name : str
            Name of the dataset.
        classes : tuple of classes, default = None
            We reuse the neural network weights if the corresponding class appears in the pretrained model. 
            Otherwise, we randomly initialize the neural network weights for new classes.
        format : str
            Format of the object detection dataset, either: 'voc' or 'coco'.
            For details, see: `autogluon/task/object_detection/dataset/voc.py`, `autogluon/task/object_detection/dataset/coco.py`
        Train : bool, default = True
            Whether to use this dataset for training object detection models.
        kwargs : keyword arguments
            Passed to either: :meth:`autogluon.task.object_detection.dataset.CustomVOCDetection` or :meth:`autogluon.task.object_detection.dataset.COCO`.
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

