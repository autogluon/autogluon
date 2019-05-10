import warnings
import logging

import numpy as np
from scipy.stats import ks_2samp
from mxnet import gluon

from .visualizer import Visualizer

__all__ = ['SanityCheck']


class SanityCheck(object):
    def __init__(self):
        pass

    @staticmethod
    def check_dataset_label_num(a, b):
        if isinstance(a, gluon.data.dataset.Dataset) and isinstance(b, gluon.data.dataset.Dataset):
            a_label_num = len(np.unique(a._label))
            b_label_num = len(np.unique(b._label))
            if a_label_num != b_label_num:
                warnings.warn('Warning: '
                              'number of class of labels '
                              'in the compared datasets are different.')
        else:
            pass

    @staticmethod
    def check_dataset_label_KStest(a, b):
        if isinstance(a, gluon.data.dataset.Dataset) and isinstance(b, gluon.data.dataset.Dataset):
            ks_res = ks_2samp(a._label, b._label)
            if ks_res.pvalue < 0.05:
                warnings.warn('Warning: '
                              'data label distribution are different at p-val 0.05 level.')
        else:
            pass

    @staticmethod
    def check_dataset_label_histogram(a, b):
        if isinstance(a, gluon.data.dataset.Dataset) and isinstance(b, gluon.data.dataset.Dataset):
            min_len = min(len(a._label), len(b._label))
            #TODO(cgraywang): sample a min_len?
            a_area, _ = np.histogram(a._label[:min_len], bins=len(np.unique(a._label)))
            b_area, _ = np.histogram(b._label[:min_len], bins=len(np.unique(b._label)))
            if (a_area <= b_area).all() or (a_area >= b_area).all():
                warnings.warn('Warning: '
                              'data label histogram seems not in a good shape')
                logging.info('data label histogram is save at ./histogram.png')
                Visualizer.visualize_dataset_label_histogram(a, b)
        else:
            pass

    @staticmethod
    def check_dataset(a, b):
        SanityCheck.check_dataset_label_num(a, b)
        SanityCheck.check_dataset_label_KStest(a, b)
        SanityCheck.check_dataset_label_histogram(a, b)
