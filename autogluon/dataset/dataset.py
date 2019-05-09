from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

__all__ = ['Dataset']

class Dataset(object):
    def __init__(self, train_path=None, val_path=None):
        # TODO: add search space, handle batch_size, num_workers
        self.train_path = train_path
        self.val_path = val_path
        self.search_space = None
        self.add_search_space()

    def __new__(cls):
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

        if 'CIFAR10' in cls.train_path or 'CIFAR10' in cls.val_path:
            train_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
                batch_size=64,
                shuffle=True,
                last_batch="discard",
                num_workers=4)

            test_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
                batch_size=64,
                shuffle=False,
                num_workers=4)
        else:
            train_data = None
            test_data = None
            raise NotImplementedError
        return train_data, test_data

    def _set_search_space(self, cs):
        self.search_space = cs

    def add_search_space(self):
        pass

    def get_search_space(self):
        return self.search_space