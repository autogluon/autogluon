from typing import AnyStr

import ConfigSpace as CS
from mxnet import gluon

from ..space import *

__all__ = ['Dataset']


class Dataset(object):
    def __init__(self, name, train_path=None, val_path=None):
        # TODO (cgraywang): add search space, handle batch_size, num_workers
        self.name: AnyStr = name
        self.train_path: AnyStr = train_path
        self.val_path: AnyStr = val_path
        self.search_space: CS.ConfigurationSpace = None
        self.train_dataset: gluon.data.Dataset = None
        self.val_dataset: gluon.data.Dataset = None
        self.train_data_loader: gluon.data.DataLoader = None
        self.val_data_loader: gluon.data.DataLoader = None

    def _read_dataset(self):
        pass

    def _set_search_space(self, cs: CS.ConfigurationSpace):
        self.search_space = cs

    def add_search_space(self):
        # TODO Think of other hyperparams for data
        cs = CS.ConfigurationSpace()
        data_hyperparams = Exponential(name='batch_size', base=2, lower_exponent=3,
                                       upper_exponent=3).get_hyper_param()
        cs.add_hyperparameter(data_hyperparams)
        self._set_search_space(cs)

    def get_search_space(self):
        return self.search_space

    @staticmethod
    def get_file_extension(path: AnyStr) -> AnyStr:
        """
        Utility method for getting file extension from file path.
        :param path:
        :return:
        """
        from os.path import splitext
        if len(path.split('.')) > 2:
            return '.'.join(path.split('.')[-2:])
        else:
            return splitext(path)[1]
