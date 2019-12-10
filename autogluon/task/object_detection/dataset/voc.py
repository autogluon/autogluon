"""Pascal VOC object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import warnings
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import mxnet as mx
from gluoncv.data.base import VisionDataset

from ....core import *
from .base import DatasetBase
import autogluon as ag

from gluoncv import data as gdata
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric


class TinyVOCDetection(gdata.VOCDetection):
    """Tiny VOC detection Dataset.
    It only contains one category, motorbike
    Parameters
    ----------
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of tuples, default ((2007, 'trainval'), (2012, 'trainval'))
        List of combinations of (year, name)
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.
        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default None
        In default, the 20 classes are mapped into indices from 0 to 19. We can
        customize it by providing a str to int dict specifying how to map class
        names to indices. Use by advanced users only, when you want to swap the orders
        of class labels.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extremely large.
    """
    CLASSES = ('motorbike',)

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'voc'),
                 splits=((2007, 'trainval'), (2012, 'trainval')),
                 transform=None, index_map=None, preload_label=True):
        super(TinyVOCDetection, self).__init__(root=root,
                                               splits=splits,
                                               transform=transform,
                                               index_map=index_map,
                                               preload_label=preload_label),


@obj()
class VOC(DatasetBase):
    def __init__(self):
        super().__init__()
        self.train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        self.val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        self.val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=self.val_dataset.classes)

    def get_train_val_metric(self):
        return (self.train_dataset, self.val_dataset, self.val_metric)

    def get_dataset_name(self):
        return 'voc'

@obj()
class TinyVOC(DatasetBase):
    def __init__(self, root, *args, **kwargs):
        super().__init__()
        self.train_dataset = TinyVOCDetection(
            root = root,
            splits=[(2007, 'tiny_motorbike_train')])
        self.val_dataset = TinyVOCDetection(
            root = root,
            splits=[(2007, 'tiny_motorbike_val')])
        self.val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=self.val_dataset.classes)
    
        self.test_dataset = TinyVOCDetection(
            root = root,
            splits=[(2007, 'tiny_motorbike_test')])


    def get_train_val_metric(self):
        return (self.train_dataset, self.val_dataset, self.val_metric)

    def get_test_metric(self):
        return (self.test_dataset, self.val_metric)

    def get_dataset_name(self):
        return 'tiny_motorbike'

    def get_classes(self):
        return self.train_dataset.classes 

