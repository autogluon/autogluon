import numpy as np

from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

from ... import dataset
from ...utils.data_analyzer import DataAnalyzer

__all__ = ['Dataset', 'transform_fn']

_dataset = {'mnist': gluon.data.vision.MNIST,
            'fashionmnist': gluon.data.vision.FashionMNIST,
            'cifar10': gluon.data.vision.CIFAR10,
            'cifar100': gluon.data.vision.CIFAR100}


def get_dataset(name, **kwargs):
    """Returns a dataset by name

    Parameters
    ----------
    name : str
        Name of the model.

    Returns
    -------
    Dataset
        The dataset.
    """
    name = name.lower()
    if name not in _dataset:
        err_str = '"%s" is not among the following loss list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_dataset.keys())))
        raise ValueError(err_str)
    dataset = _dataset[name](*kwargs)
    return dataset


def transform_fn(data, label):
    return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(
        np.float32)


class Dataset(dataset.Dataset):
    def __init__(self, name=None, train_path=None, val_path=None, batch_size=64, num_workers=4,
                 transform_train_fn=None, transform_val_fn=None,
                 transform_train_list=[
                     gcv_transforms.RandomCrop(32, pad=4),
                     transforms.RandomFlipLeftRight(),
                     transforms.ToTensor(),
                     transforms.Normalize([0.4914, 0.4822, 0.4465],
                                          [0.2023, 0.1994, 0.2010])
                 ],
                 transform_val_list=[
                     transforms.ToTensor(),
                     transforms.Normalize([0.4914, 0.4822, 0.4465],
                                          [0.2023, 0.1994, 0.2010])
                 ],
                 **kwargs):
        super(Dataset, self).__init__(name, train_path, val_path, batch_size, num_workers,
                                      transform_train_fn, transform_val_fn,
                                      transform_train_list, transform_val_list, **kwargs)
        self._read_dataset(**kwargs)
        # TODO (cgraywang): add search space, handle batch_size, num_workers
        self._add_search_space()

    def _read_dataset(self, **kwargs):
        # TODO (cgraywang): put transform in the search space
        try:
            self.train = get_dataset(self.name, train=True)
            self.val = get_dataset(self.name, train=False)
            self.num_classes = DataAnalyzer.stat_dataset(self.train)[0]
            DataAnalyzer.check_dataset(self.train, self.val)
        except ValueError:
            self.train = gluon.data.vision.ImageFolderDataset(self.train_path)
            self.val = gluon.data.vision.ImageFolderDataset(self.val_path)
            self.num_classes = kwargs['num_classes']

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
