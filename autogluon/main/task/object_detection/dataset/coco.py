from .base import DatasetBase
from autogluon.core import *

from gluoncv import data as gdata

@obj()
class COCO(DatasetBase):
    """Built-in class to work with the well-known COCO dataset for object detection. 
    
    Returns
    -------
    Dataset object that can be passed to `task.fit()`, which is actually an :class:`autogluon.space.AutoGluonObject`. 
    To interact with such an object yourself, you must first call `Dataset.init()` to instantiate the object in Python.
    """
    def __init__(self):
        super(COCO. self).__init__()
        self.train_dataset = gdata.COCODetection(splits='instances_train2017')
        self.val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        self.val_metric = COCODetectionMetric(
                                self.val_dataset, args.save_prefix + '_eval', cleanup=True,
                                data_shape=(args.data_shape, args.data_shape))
        
        #TODO: whether to use the code below
        """
        # coco validation is slow, consider increase the validation interval
        if args.val_interval == 1:
            args.val_interval = 10
        """
    
    def get_train_val_metric(self):
        return (self.train_dataset, self.val_dataset, self.val_metric)
    
    def get_dataset_name(self):
        return 'coco'






