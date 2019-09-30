import numpy as np

from mxnet import gluon, nd

__all__ = ['get_dataset', 'transform_fn']

_dataset = {'mnist': gluon.data.vision.MNIST,
            'fashionmnist': gluon.data.vision.FashionMNIST,
            'cifar10': gluon.data.vision.CIFAR10,
            'cifar100': gluon.data.vision.CIFAR100}


def get_dataset(name, **kwargs):
    """Returns a dataset by name

    Args:
        name : str
            Name of the model.
    """
    name = name.lower()
    if name not in _dataset:
        err_str = '"%s" is not among the following dataset list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_dataset.keys())))
        raise ValueError(err_str)
    dataset = _dataset[name](*kwargs)
    return dataset


def transform_fn(data, label):
    return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(
        np.float32)
