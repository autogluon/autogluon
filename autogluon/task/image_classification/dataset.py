from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

from ... import dataset
from ...utils.data_analyzer import DataAnalyzer

__all__ = ['Dataset']


class Dataset(dataset.Dataset):
    def __init__(self, train_path=None, val_path=None):
        # TODO (cgraywang): add search space, handle batch_size, num_workers

        self.train_path = train_path
        self.val_path = val_path
        self.train_data = None
        self.val_data = None
        self._read_dataset()
        self.search_space = None
        self.add_search_space()

    def _read_dataset(self):
        transform_train = transforms.Compose([
            gcv_transforms.RandomCrop(32, pad=4),
            transforms.RandomFlipLeftRight(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2023, 0.1994, 0.2010])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2023, 0.1994, 0.2010])
        ])

        if 'CIFAR10' in self.train_path or 'CIFAR10' in self.val_path:
            train_dataset = gluon.data.vision.CIFAR10(train=True)
            test_dataset = gluon.data.vision.CIFAR10(train=False)
            train_data = gluon.data.DataLoader(
                train_dataset.transform_first(transform_train),
                batch_size=64,
                shuffle=True,
                last_batch="discard",
                num_workers=4)

            test_data = gluon.data.DataLoader(
                test_dataset.transform_first(transform_test),
                batch_size=64,
                shuffle=False,
                num_workers=4)
            DataAnalyzer.check_dataset(train_dataset, test_dataset)
        else:
            train_data = None
            test_data = None
            raise NotImplementedError
        self.train_data = train_data
        self.val_data = test_data
