from .base import *
from .voc import *
from .coco import *

import logging
logger = logging.getLogger(__name__)

def get_dataset(root='~/.mxnet/datasets/voc', index_file_name='trainval', name=None, \
                classes=None, format='voc', Train=True, **kwargs):
    """ Load dataset to use for object detection, which must be in either VOC or COCO format.
        
    Parameters
    ----------
    root : str
        Path to folder storing the dataset.
    index_file_name : str
        Name of file containing the training/validation indices of each text example. The name of the .txt file which constains images for training or testing. 
        this is only for custom dataset.
    name: str
        name for built-in dataset, ('voc', 'voc2007' or 'voc2012')
        when use built-in dataset, the index_file_name should be None.
    classes : tuple of classes, default = None
        users can specify classes for custom dataset ex. classes = ('bike', 'bird', 'cat', ...)
        We reuse the neural network weights if the corresponding class appears in the pretrained model. 
        Otherwise, we randomly initialize the neural network weights for new classes.
    format : str
        Format of the object detection dataset, either: 'voc' or 'coco'.
        For details, see: `autogluon/task/object_detection/dataset/voc.py`, `autogluon/task/object_detection/dataset/coco.py`
    Train : bool, default = True
        pecify Train/Test mode. It is only valid when name is not None.
    kwargs : keyword arguments
        Passed to either: :meth:`autogluon.task.object_detection.dataset.CustomVOCDetection` or :meth:`autogluon.task.object_detection.dataset.COCO`.
    
    Returns
    -------
    Dataset object that can be passed to `task.fit()`, which is actually an :class:`autogluon.space.AutoGluonObject`. 
    To interact with such an object yourself, you must first call `Dataset.init()` to instantiate the object in Python.
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
        logger.info(">>> create dataset(COCO format)")
        return COCO(*args, **kwargs)
    else:
        raise NotImplementedError('Other data formats are not implemented.')

