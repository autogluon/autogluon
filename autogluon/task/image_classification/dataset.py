import math
import numpy as np
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
import gluoncv.data.transforms as gcv_transforms
from ...core import *
from ..base import BaseDataset

__all__ = ['get_built_in_dataset', 'ImageClassificationDataset']

built_in_datasets = [
    'cifar10',
    'cifar100',
]

@autogluon_object()
class ImageClassificationDataset(object):
    """The image classification dataset.
    Args:
        name: the dataset name.
        train_path: the training data location
        val_path: the validation data location.
        batch_size: the batch size.
        num_workers: the number of workers used in DataLoader.
        transform_train_fn: the transformation function for training data.
        transform_val_fn: the transformation function for validation data.
        transform_train_list: the compose list of Transformations for training data.
        transform_val_list: the compose list of Transformations for validation data.
        batchify_train_fn: the batchify function defined for training data.
        batchify_val_fn: the batchify function defined for validation data.
    """
    def __init__(self, name=None, train_path=None, val_path=None,
                 input_size=224, crop_ratio=0.875, jitter_param=0.4,
                 **kwargs):
        self.name = name
        self.train_path = train_path
        self.val_path = val_path
        resize = int(math.ceil(input_size / crop_ratio))
        self.transform_train = transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomResizedCrop(input_size),
                transforms.RandomFlipLeftRight(),
                transforms.RandomColorJitter(brightness=jitter_param,
                                             contrast=jitter_param,
                                             saturation=jitter_param),
                transforms.RandomLighting(0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.transform_val = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self._read_dataset(**kwargs)

    def _read_dataset(self, **kwargs):
        import time
        if self.name in built_in_datasets:
            self.train = get_built_in_dataset(self.name, train=True)._lazy_init()
            self.val = None
            self.test = get_built_in_dataset(self.name, train=False)._lazy_init()
            self.num_classes = len(np.unique(self.train._label))
        else:
            if self.train_path is not None:
                dataset_cls = gluon.data.vision.ImageFolderDataset if '.rec' not in self.train_path \
                        else gluon.data.vision.ImageRecordDataset
                train_set = dataset_cls(self.train_path)
                self.synsets = train_set.synsets
                self.train = train_set.transform_first(self.transform_train)
                self.val = dataset_cls(self.val_path).transform_first(self.transform_val) if self.val_path else None
                if 'test_path' in kwargs:
                    self.test = dataset_cls(kwargs['test_path']).transform_first(self.transform_val)
                self.num_classes = len(np.unique([e[1] for e in self.train]))
            elif 'test_path' in kwargs:
                dataset_cls = gluon.data.vision.ImageFolderDataset if '.rec' not in kwargs['test_path'] \
                        else gluon.data.vision.ImageRecordDataset
                self.test = dataset_cls(kwargs['test_path']).transform_first(self.transform_val)
            else:
                raise NotImplementedError

@autogluon_function()
def get_built_in_dataset(name, train=True):
    if name == 'cifar10':
        transform_split = transforms.Compose([
            gcv_transforms.RandomCrop(32, pad=4),
            transforms.RandomFlipLeftRight(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]) if train else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        return gluon.data.vision.CIFAR10(train=train).transform_first(transform_split)
    elif name == 'cifar100':
        transform_split = transforms.Compose([
            gcv_transforms.RandomCrop(32, pad=4),
            transforms.RandomFlipLeftRight(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]) if train else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        return gluon.data.vision.CIFAR100(train=train).transform_first(transform_split)
    else:
        raise NotImplemented
