from typing import AnyStr

import ConfigSpace as CS
from mxnet import gluon

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

    def _set_search_space(self, cs):
        self.search_space = cs

    def add_search_space(self):
        pass

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
