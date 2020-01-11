"""Pascal VOC object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import warnings
import numpy as np
import glob
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

class CustomVOCDetectionBase(gdata.VOCDetection):
    """Base class for custom Dataset which follows protocol/formatting of the well-known VOC object detection dataset.
    
    Parameters
    ----------
    class: tuple of classes, default = None
        We reuse the neural network weights if the corresponding class appears in the pretrained model. 
        Otherwise, we randomly initialize the neural network weights for new classes.
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of tuples, default ((2007, 'trainval'), (2012, 'trainval'))
        List of combinations of (year, name)
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    transform : callable, default = None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.
        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default = None
        By default, the 20 classes are mapped into indices from 0 to 19. We can
        customize it by providing a str to int dict specifying how to map class
        names to indices. This is only for advanced users, when you want to swap the orders
        of class labels.
    preload_label : bool, default = True
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
                                               preload_label=False),
        self._items_new = [self._items[each_id] for each_id in range(len(self._items)) if self._check_valid(each_id) ]
        self._items = self._items_new
        self._label_cache = self._preload_labels() if preload_label else None
    
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

    def _check_valid(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            try:
                difficult = int(obj.find('difficult').text)
            except ValueError:
                difficult = 0
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)

            if not ((0 <= xmin < width) and (0 <= ymin < height) \
                    and (xmin < xmax <= width) and (ymin < ymax <= height)):
                return False

        return True


@obj()
class CustomVOCDetection():
    """Custom Dataset which follows protocol/formatting of the well-known VOC object detection dataset.
    
    Parameters
    ----------
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of tuples        
        List of combinations of (year, name) to indicate how to split data into training, validation, and test sets.
        For the original VOC dataset, the year candidates can be: 2007, 2012.
        For the original VOC dataset, the name candidates can be: 'train', 'val', 'trainval', 'test'.
        For the original VOC dataset, one might use for example: ((2007, 'trainval'), (2012, 'trainval'))
    classes: tuple of classes
        We reuse the neural network weights if the corresponding class appears in the pretrained model. 
        Otherwise, we randomly initialize the neural network weights for new classes.
    
    Returns
    -------
    Dataset object that can be passed to `task.fit()`, which is actually an :class:`autogluon.space.AutoGluonObject`. 
    To interact with such an object yourself, you must first call `Dataset.init()` to instantiate the object in Python.
    """
    def __init__(self, root, splits, name, classes, **kwargs):
        super().__init__()
        self.root = root
        
        # search classes from gt files for custom dataset
        if not (classes or name):
            classes = self.generate_gt() 
        
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


