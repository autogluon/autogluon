import numpy as np

from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform, SSDDefaultValTransform

from ... import dataset
from ...utils.data_analyzer import DataAnalyzer

__all__ = ['Dataset']

_dataset = {'voc': gdata.VOCDetection}


_transform_fns = {'SSDDefaultTrainTransform': SSDDefaultTrainTransform,
                  'SSDDefaultValTransform': SSDDefaultValTransform}


def get_transform_fn(name, *args):
    """Returns a transform function by name

    Parameters
    ----------
    name : str
        Name of the transform_fn.

    Returns
    -------
    Transform
        The transform function.
    """
    if name not in _transform_fns and name.lower() not in _transform_fns:
        err_str = '"%s" is not among the following transform function list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_transform_fns.keys())))
        raise ValueError(err_str)
    transform_fn = _transform_fns[name](*args)
    return transform_fn

def get_dataset(name, **kwargs):
    """Returns a dataset by name

    Parameters
    ----------
    name : str
        Name of the dataset.

    Returns
    -------
    Dataset
        The dataset.
    """
    name = name.lower()
    if name not in _dataset:
        err_str = '"%s" is not among the following dataset list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_dataset.keys())))
        raise ValueError(err_str)
    dataset = _dataset[name](**kwargs)
    return dataset


def batchify_fn():
    return Tuple(Stack(), Stack(), Stack())


def batchify_val_fn():
    return Tuple(Stack(), Pad(pad_val=-1))

class Dataset(dataset.Dataset):
    def __init__(self, name=None, train_path=None, val_path=None, batch_size=32, num_workers=4,
                 transform_train_fn='SSDDefaultTrainTransform',
                 transform_val_fn='SSDDefaultValTransform',
                 transform_train_list=None, transform_val_list=None,
                 batchify_train_fn=batchify_fn(), batchify_val_fn=batchify_val_fn(),
                 **kwargs):
        super(Dataset, self).__init__(name, train_path, val_path, batch_size, num_workers,
                                      transform_train_fn, transform_val_fn,
                                      transform_train_list, transform_val_list,
                                      batchify_train_fn, batchify_val_fn, **kwargs)
        self._read_dataset(**kwargs)
        # TODO (cgraywang): add search space, handle batch_size, num_workers
        self._add_search_space()

    def _read_dataset(self, **kwargs):
        # TODO (cgraywang): put transform in the search space
        try:
            # Only support VOC in the defined dataset list
            self.train = get_dataset(self.name, splits=[(2007, 'trainval'), (2012, 'trainval')])
            self.val = get_dataset(self.name, splits=[(2007, 'test')])
            self.num_classes = len(self.train.classes)
            # TODO (cgraywang): add DataAnalyzer
            # self.num_classes = DataAnalyzer.stat_dataset(self.train)[0]
            # DataAnalyzer.check_dataset(self.train, self.val)
        except ValueError:
            raise NotImplementedError

    def _add_search_space(self):
        pass

    def _get_search_space_strs(self):
        pass

    def __repr__(self):
        try:
            train_stats = DataAnalyzer.stat_dataset(self.train)
            val_stats = DataAnalyzer.stat_dataset(self.val)
            repr_str = "AutoGluon Dataset: " \
                       "\n ======== " \
                       "\n name = %s" \
                       "\n ======== " \
                       "\n Train data statistic " \
                       "\n number of classes = %d" \
                       "\n number of samples = %d" \
                       "\n mean (label) = %.2f" \
                       "\n std (label) = %.2f" \
                       "\n var (label) = %.2f" \
                       "\n ======== " \
                       "\n Val data statistic " \
                       "\n number of classes = %d" \
                       "\n number of samples = %d" \
                       "\n mean (label) = %.2f" \
                       "\n std (label) = %.2f" \
                       "\n var (label) = %.2f" % (self.name,
                                                train_stats[0], train_stats[1],
                                                train_stats[2], train_stats[3],
                                                train_stats[4],
                                                val_stats[0], val_stats[1],
                                                val_stats[2], val_stats[3],
                                                val_stats[4])
        except AttributeError:
            #TODO: add more info for folder dataset
            repr_str = "AutoGluon Dataset: " \
                       "\n ======== " \
                       "\n name = %s" \
                       "\n ======== " % self.name
        return repr_str
