import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class ImageNetDataset(datasets.ImageFolder):
    BASE_DIR = 'imagenet'
    def __init__(self, root=os.path.expanduser('~/.autogluon/datasets/'), transform=None,
                 target_transform=None, train=True, **kwargs):
        split='train' if train == True else 'val'
        root = os.path.join(root, self.BASE_DIR, split)
        super(ImageNetDataset, self).__init__(
            root, transform, target_transform)
