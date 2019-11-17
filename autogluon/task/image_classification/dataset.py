import os
import sys
import math
import numpy as np
from PIL import Image

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.data import Dataset as MXDataset
from mxnet.gluon.data.vision import ImageRecordDataset, transforms

from ...core import *
from ..base import BaseDataset
from ...utils import get_data_rec

__all__ = ['get_dataset', 'ImageFolderDataset', 'RecordDataset']

built_in_datasets = [
    'mnist',
    'cifar',
    'cifar10',
    'cifar100',
    'imagenet',
]

@func()
def get_dataset(path=None, train=True, name=None,
               input_size=224, crop_ratio=0.875, jitter_param=0.4,
               *args, **kwargs):
    """A convenient function for image classification dataset, supported datasets given by
    built-in datasets ('mnist', 'cifar10', 'cifar100', 'imagenet'),
    :class:`ImageFolderDataset` and :class:`RecordioDataset`.

    Parameters
    ----------
        name : str, optional
            The name for built-in dataset, overrite other options.
            The options are ('mnist', 'cifar', 'cifar10', 'cifar100', 'imagenet')
        train : bool, default True
            Train or validation mode
        path : str
            The training data location. If using :class:`ImageFolderDataset`,
            image folder`path/to/the/folder` should be provided.
            If using :class:`RecordDataset`, the `path/to/*.rec` should be provided.
        input_size : int
            The input image size.
        crop_ratio : float
            Center crop ratio for evaluation only
    """
    resize = int(math.ceil(input_size / crop_ratio))
    if name in built_in_datasets:
        return get_built_in_dataset(name, train=train, input_size=input_size, *args, **kwargs)

    transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomFlipLeftRight(),
            transforms.RandomColorJitter(brightness=jitter_param,
                                         contrast=jitter_param,
                                         saturation=jitter_param),
            transforms.RandomLighting(0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) if train else transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset_cls = ImageFolderDataset if '.rec' not in path \
            else RecordDataset
    dataset = dataset_cls(path, transform=transform, *args, **kwargs)
    return dataset.init()

@obj()
class RecordDataset(ImageRecordDataset):
    """A dataset wrapping over a RecordIO file containing images.

    Each sample is an image and its corresponding label.

    Parameters
    ----------
    filename : str
        Path to rec file.
    gray_scale : False
        If True, always convert images to greyscale. \
        If False, always convert images to colored (RGB).
    transform : function, default None
        A user defined callback that transforms each sample. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)

    """
    def __init__(self, filename, gray_scale=False, transform=None):
        flag = 0 if gray_scale else 1
        super().__init__(filename, flag=flag, transform=transform)

    @property
    def num_classes(self):
        raise NotImplemented

    @property
    def classes(self):
        raise NotImplemented

@obj()
class ImageFolderDataset(MXDataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Parameters
    ----------
    root : string
        Root directory path.
    transform : callable, optional
        A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
    is_valid_file : callable, optional
        A function that takes path of an Image file
        and check if the file is a valid file (used to check of corrupt files)

    Attributes
    ----------
    classes : list
        List of the class names.
    class_to_idx : dict
        Dict with items (class_name, class_index).
    imgs : list
        List of (image path, class_index) tuples
    """
    _repr_indent = 4
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    def __init__(self, root, extensions=None, transform=None, is_valid_file=None):
        root = os.path.expanduser(root)
        self.root = root
        extensions = extensions if extensions else self.IMG_EXTENSIONS

        self._transform = transform
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples

    @staticmethod
    def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
        images = []
        dir = os.path.expanduser(dir)
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return x.lower().endswith(extensions)
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.abspath(os.path.join(root, fname))
                    if is_valid_file(path):
                        item = (path, class_to_idx[target])
                        images.append(item)

        return images

    @staticmethod
    def loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Parameters
        ----------
        dir : string
            Root directory path.

        Returns
        -------
        tuple: classes, class_to_idx
            where classes are relative to (dir), and class_to_idx is a dictionary.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @property
    def num_classes(self):
        return len(self.classes)

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int
            Index

        Returns
        ----------
        tuple : sample, target
            where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self._sample_transform(sample)
        if self._transform is not None:
            sample = self._transform(sample)

        return sample, target

    def _sample_transform(self, img):
        return nd.array(np.array(img), mx.cpu(0))

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)


def get_built_in_dataset(name, train=True, input_size=224, batch_size=256, num_workers=32,
                         shuffle=True, **kwargs):
    """Built-in image classification dataset.
    """
    print('get_built_in_dataset', name)
    if name in ['cifar10', 'cifar']:
        import gluoncv.data.transforms as gcv_transforms
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
        import gluoncv.data.transforms as gcv_transforms
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
    elif name == 'mnist':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
        return gluon.data.vision.MNIST(train=train, transform=transform)
    elif name == 'imagenet':
        # Please setup the ImageNet dataset following the tutorial from GluonCV
        if train:
            rec_file = '/media/ramdisk/rec/train.rec'
            rec_file_idx = '/media/ramdisk/rec/train.idx'
        else:
            rec_file = '/media/ramdisk/rec/val.rec'
            rec_file_idx = '/media/ramdisk/rec/val.idx'
        data_loader = get_data_rec(input_size, 0.875, rec_file, rec_file_idx,
                                   batch_size, num_workers, train, shuffle=shuffle,
                                   **kwargs)
        return data_loader
    else:
        raise NotImplemented
