import warnings
import numpy as np
from scipy.stats import ks_2samp
from mxnet import gluon

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
    def check_dataset(a, b):
        SanityCheck.check_dataset_label_num(a, b)
        SanityCheck.check_dataset_label_KStest(a, b)
