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

import glob
from xml.etree import ElementTree as ET


class CustomVOCDetectionBase(gdata.VOCDetection):
    """custom Dataset which follows VOC protocol.
    
    Parameters
    ----------
    class: tuple of classes, default None
        reuse the weights if the corresponding class appears in the pretrained model, 
        otherwise, randomly initialize the weights for new categories.
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

    def __init__(self, classes=None, root=os.path.join('~', '.mxnet', 'datasets', 'voc'),
                 splits=((2007, 'trainval'), (2012, 'trainval')),
                 transform=None, index_map=None, preload_label=True):

        # update classes 
        if classes:
            self._set_class(classes)  
        super(CustomVOCDetectionBase, self).__init__(root=root,
                                               splits=splits,
                                               transform=transform,
                                               index_map=index_map,
                                               preload_label=preload_label),
    
    @classmethod
    def _set_class(cls, classes):
        cls.CLASSES = classes
    
    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for subfolder, name in splits:
            root = os.path.join(self._root, subfolder) if subfolder else self._root
            lf = os.path.join(root, 'ImageSets', 'Main', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]
        return ids


@obj()
class CustomVOCDetection():
    def __init__(self, root, splits, name, classes, **kwargs):
        super().__init__()
        self.root = root
        
        if not classes:
            classes = self.generate_gt() if not name else None
        
        self.dataset = CustomVOCDetectionBase(classes=classes,
                                              root=root,
                                              splits=splits)

        self.metric = VOC07MApMetric(iou_thresh=0.5, class_names=self.dataset.classes)

    def get_dataset_and_metric(self):
        return (self.dataset, self.metric)

    def get_classes(self):
        return self.dataset.classes 

    def generate_gt(self):
        classes = []

        all_xml = glob.glob( os.path.join(self.root, 'Annotations', '*.xml') )
        for each_xml_file in all_xml:
            tree = ET.parse(each_xml_file)
            root = tree.getroot()
            for child in root:
                if child.tag=='object':
                    for item in child:
                        if item.tag=='name':
                            object_name = item.text
                            if object_name not in classes:
                                classes.append(object_name)

        classes = sorted(classes)

        return classes


