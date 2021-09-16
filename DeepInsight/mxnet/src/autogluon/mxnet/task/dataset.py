import logging
import math
import os
import platform
import sys
import warnings

import numpy as np
from PIL import Image
from mxnet import gluon, nd
from mxnet import recordio
from mxnet.gluon.data import RecordFileDataset
from mxnet.gluon.data.vision import ImageFolderDataset as MXImageFolderDataset
from mxnet.gluon.data.vision import ImageRecordDataset, transforms

from autogluon.core import *
from ..utils import get_data_rec
from ..utils.pil_transforms import *

_is_osx = platform.system() == "Darwin"

__all__ = [
    'get_dataset',
    'get_built_in_dataset',
    'ImageFolderDataset',
    'RecordDataset',
    'NativeImageFolderDataset']

logger = logging.getLogger(__name__)

built_in_datasets = [
    'mnist',
    'cifar',
    'cifar10',
    'cifar100',
    'imagenet',
    'fashionmnist',
]


class _TransformFirstClosure(object):
    """Use callable object instead of nested function, it can be pickled."""

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, x, *args):
        if args:
            return (self._fn(x),) + args
        return self._fn(x)


def generate_transform(train, resize, _is_osx, input_size, jitter_param):
    if _is_osx:
        # using PIL to load image (slow)
        if train:
            transform = Compose(
                [
                    RandomResizedCrop(input_size),
                    RandomHorizontalFlip(),
                    ColorJitter(0.4, 0.4, 0.4),
                    ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
        else:
            transform = Compose(
                [
                    Resize(resize),
                    CenterCrop(input_size),
                    ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
    else:
        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomFlipLeftRight(),
                    transforms.RandomColorJitter(
                        brightness=jitter_param,
                        contrast=jitter_param,
                        saturation=jitter_param
                    ),
                    transforms.RandomLighting(0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
    return transform


@func()
def get_dataset(path=None, train=True, name=None,
                input_size=224, crop_ratio=0.875, jitter_param=0.4, scale_ratio_choice=[],
                *args, **kwargs):
    """ Method to produce image classification dataset for AutoGluon, can either be a
    :class:`ImageFolderDataset`, :class:`RecordDataset`, or a
    popular dataset already built into AutoGluon ('mnist', 'cifar10', 'cifar100', 'imagenet').

    Parameters
    ----------
    name : str, optional
        Which built-in dataset to use, will override all other options if specified.
        The options are ('mnist', 'cifar', 'cifar10', 'cifar100', 'imagenet')
    train : bool, default = True
        Whether this dataset should be used for training or validation.
    path : str
        The training data location. If using :class:`ImageFolderDataset`,
        image folder`path/to/the/folder` should be provided.
        If using :class:`RecordDataset`, the `path/to/*.rec` should be provided.
    input_size : int
        The input image size.
    crop_ratio : float
        Center crop ratio (for evaluation only)
    scale_ratio_choice: list
        List of crop_ratio, only in the test dataset, the set of scaling ratios obtained is scaled to the original image, and then cut a fixed size (input_size) and get a set of predictions for averaging.

    Returns
    -------
    Dataset object that can be passed to `task.fit()`, which is actually an :class:`autogluon.space.AutoGluonObject`.
    To interact with such an object yourself, you must first call `Dataset.init()` to instantiate the object in Python.
    """

    resize = int(math.ceil(input_size / crop_ratio))
    transform = generate_transform(train, resize, _is_osx, input_size, jitter_param)

    if isinstance(name, str) and name.lower() in built_in_datasets:
        return get_built_in_dataset(name, train=train, input_size=input_size, *args, **kwargs)

    if '.rec' in path:
        dataset = RecordDataset(
            path,
            *args,
            transform=_TransformFirstClosure(transform),
            **kwargs
        )
    elif _is_osx:
        dataset = ImageFolderDataset(path, transform=transform, *args, **kwargs)
    elif not train:
        if not scale_ratio_choice:
            dataset = TestImageFolderDataset(
                path,
                *args,
                transform=_TransformFirstClosure(transform),
                **kwargs
            )
        else:
            dataset = []
            for i in scale_ratio_choice:
                resize = int(math.ceil(input_size / i))
                dataset_item = TestImageFolderDataset(
                    path,
                    *args,
                    transform=_TransformFirstClosure(
                        generate_transform(train, resize, _is_osx, input_size, jitter_param)
                    ),
                    **kwargs
                )
                dataset.append(dataset_item.init())

    elif 'label_file' in kwargs:
        dataset = IndexImageDataset(
            path,
            transform=_TransformFirstClosure(transform),
            *args,
            **kwargs
        )
    else:
        dataset = NativeImageFolderDataset(
            path,
            *args,
            transform=_TransformFirstClosure(transform),
            **kwargs
        )

    if not scale_ratio_choice:
        dataset = dataset.init()
    return dataset


@obj()
class IndexImageDataset(MXImageFolderDataset):
    """A image classification dataset with a CVS label file
       Each sample is an image and its corresponding label.

    Parameters
    ----------
    root : str
        Path to the image folder.
    indexfile : str
        Local path to the csv index file. The CSV should have two collums
        1. image name (e.g. xxxx or xxxx.jpg)
        2. label name or index (e.g. aaa or 1)
    gray_scale : False
        If True, always convert images to greyscale.
        If False, always convert images to colored (RGB).
    transform : function, default None
        A user defined callback that transforms each sample.
    """

    def __init__(self, root, label_file, gray_scale=False, transform=None,
                 extension='.jpg'):
        self._root = os.path.expanduser(root)
        self.items, self.synsets = self.read_csv(label_file, root, extension)
        self._flag = 0 if gray_scale else 1
        self._transform = transform

    @staticmethod
    def read_csv(filename, root, extension):
        """The CSV should have two collums
        1. image name (e.g. xxxx or xxxx.jpg)
        2. label name or index (e.g. aaa or 1)
        """

        def label_to_index(label_list, name):
            return label_list.index(name)

        import csv
        label_dict = {}
        with open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                assert len(row) == 2
                label_dict[row[0]] = row[1]

        if 'id' in label_dict:
            label_dict.pop('id')

        labels = list(set(label_dict.values()))
        samples = [
            (os.path.join(root, f"{k}{extension}"), label_to_index(labels, v))
            for k, v in label_dict.items()
        ]
        return samples, labels

    @property
    def num_classes(self):
        return len(self.synsets)

    @property
    def classes(self):
        return self.synsets

    @property
    def num_classes(self):
        return len(self.synsets)

    @property
    def classes(self):
        return self.synsets


@obj()
class RecordDataset:
    """A dataset wrapping over a RecordIO file containing images.
       Each sample is an image and its corresponding label.

    Parameters
    ----------
    filename : str
        Local path to the .rec file.
    gray_scale : False
        If True, always convert images to greyscale.
        If False, always convert images to colored (RGB).
    transform : function, default None
        A user defined callback that transforms each sample.
    classes : iterable of str, default is None
        User provided class names. If `None` is provide, will use
        a list of increasing natural number ['0', '1', ..., 'N'] by default.
    """

    def __init__(self, filename, gray_scale=False, transform=None, classes=None):
        flag = 0 if gray_scale else 1
        # retrieve number of classes without decoding images
        td = RecordFileDataset(filename)
        s = set([recordio.unpack(td.__getitem__(i))[0].label[0] for i in range(len(td))])
        self._num_classes = len(s)
        if not classes:
            self._classes = [str(i) for i in range(self._num_classes)]
        else:
            if len(self._num_classes) != len(classes):
                warnings.warn('Provided class names do not match data, expected "num_class" is {} '
                              'vs. provided: {}'.format(self._num_classes, len(classes)))
                self._classes = list(classes) + \
                    [str(i) for i in range(len(classes), self._num_classes)]
        self._dataset = ImageRecordDataset(filename, flag=flag)
        if transform:
            self._dataset = self._dataset.transform_first(transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def classes(self):
        return self._classes

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


@obj()
class NativeImageFolderDataset(MXImageFolderDataset):
    def __init__(self, root, gray_scale=False, transform=None):
        flag = 0 if gray_scale else 1
        super().__init__(root, flag=flag, transform=transform)

    @property
    def num_classes(self):
        return len(self.synsets)

    @property
    def classes(self):
        return self.synsets


@obj()
class TestImageFolderDataset(MXImageFolderDataset):
    def __init__(self, root, gray_scale=False, transform=None):
        flag = 0 if gray_scale else 1
        super().__init__(root, flag=flag, transform=transform)

    def _list_images(self, root):
        self.synsets = []
        self.items = []
        path = os.path.expanduser(root)
        if not os.path.isdir(path):
            raise ValueError('Ignoring %s, which is not a directory.' % path, stacklevel=3)
        for filename in sorted(os.listdir(path)):
            filename = os.path.join(path, filename)
            if os.path.isfile(filename):  # add
                label = len(self.synsets)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn(
                        f'Ignoring {filename} of type {ext}.'
                        f' Only support {", ".join(self._exts)}'
                    )
                    continue
                self.items.append((filename, label))
            else:
                folder = filename
                if not os.path.isdir(folder):
                    raise ValueError(f'Ignoring {path}, which is not a directory.', stacklevel=3)
                label = len(self.synsets)
                for sub_filename in sorted(os.listdir(folder)):
                    sub_filename = os.path.join(folder, sub_filename)
                    ext = os.path.splitext(sub_filename)[1]
                    if ext.lower() not in self._exts:
                        warnings.warn(
                            f'Ignoring {sub_filename} of type {ext}.'
                            f' Only support {", ".join(self._exts)}'
                        )
                        continue
                    self.items.append((sub_filename, label))
                self.synsets.append(label)

    @property
    def num_classes(self):
        return len(self.synsets)

    @property
    def classes(self):
        return self.synsets


@obj()
class ImageFolderDataset(object):
    """A generic data loader where the images are arranged in this way on your local filesystem: ::

        root/dog/a.png
        root/dog/b.png
        root/dog/c.png

        root/cat/x.png
        root/cat/y.png
        root/cat/z.png

    Here, folder-names `dog` and `cat` are the class labels and the images with file-names 'a', `b`, `c` belong to the `dog` class while the others are `cat` images.

    Parameters
    ----------
    root : string
        Root directory path to the folder containing all of the data.
    transform : callable (optional)
        A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
    is_valid_file : callable (optional)
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
            raise RuntimeError(
                f"Found 0 files in subfolders of:  {self.root} "
                f"\nSupported extensions are:  {','.join(extensions)}"
            )

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
                if not x.lower().endswith(extensions):
                    return False
                valid = True
                try:
                    with open(x, 'rb') as f:
                        Image.open(f)
                except OSError:
                    valid = False
                return valid

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
        if not class_to_idx:
            for root, _, fnames in sorted(os.walk(dir)):
                for fname in sorted(fnames):
                    path = os.path.abspath(os.path.join(root, fname))
                    if is_valid_file(path):
                        item = (path, 0)
                        images.append(item)
        return images

    @staticmethod
    def loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    @staticmethod
    def _find_classes(dir):
        """Finds the class folders in a dataset.

        Parameters
        ----------
        dir : string
            Root directory path.

        Returns
        -------
        tuple: (classes, class_to_idx)
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
        tuple : (sample, target)
            where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self._transform is not None:
            sample = self._transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)


def get_built_in_dataset(name, train=True, input_size=224, batch_size=256, num_workers=32,
                         shuffle=True, **kwargs):
    """Returns built-in popular image classification dataset based on provided string name ('cifar10', 'cifar100','mnist','imagenet').
    """
    logger.info(f'get_built_in_dataset {name}')
    name = name.lower()
    if name in ('cifar10', 'cifar'):
        import gluoncv.data.transforms as gcv_transforms
        if train:
            transform_split = transforms.Compose(
                [
                    gcv_transforms.RandomCrop(32, pad=4),
                    transforms.RandomFlipLeftRight(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ]
            )
        else:
            transform_split = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ]
            )
        return gluon.data.vision.CIFAR10(train=train).transform_first(transform_split)
    elif name == 'cifar100':
        import gluoncv.data.transforms as gcv_transforms
        if train:
            transform_split = transforms.Compose(
                [
                    gcv_transforms.RandomCrop(32, pad=4),
                    transforms.RandomFlipLeftRight(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ]
            )
        else:
            transform_split = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ]
            )
        return gluon.data.vision.CIFAR100(train=train).transform_first(transform_split)
    elif name == 'mnist':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)

        return gluon.data.vision.MNIST(train=train, transform=transform)
    elif name == 'fashionmnist':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)

        return gluon.data.vision.FashionMNIST(train=train, transform=transform)
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
        raise NotImplementedError
