import numpy as np

from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform, SSDDefaultValTransform

from ... import dataset
from ...utils.data_analyzer import DataAnalyzer

__all__ = ['get_dataset', 'batchify_fn', 'batchify_val_fn']

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

