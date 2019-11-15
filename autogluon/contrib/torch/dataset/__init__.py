import torch
from torchvision.transforms import *
from torchvision.datasets import *

from .imagenet import ImageNetDataset

datasets = {
    'mnist': MNIST,
    'fashionmnist': FashionMNIST,
    'cifar10': CIFAR10,
    'cifar': CIFAR10,
    'imagenet': ImageNetDataset,
}

def get_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)

def get_transform(dataset, large_test_crop=False):
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    if dataset == 'imagenet':
        transform_train = Compose([
            Resize(256),
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4),
            ToTensor(),
            Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
            normalize,
        ])
        if large_test_crop:
            transform_val = Compose([
                Resize(366),
                CenterCrop(320),
                ToTensor(),
                normalize,
            ])
        else:
            transform_val = Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                normalize,
            ])
    elif dataset == 'cifar' or dataset == 'cifar10' or dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                    (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset == 'mnist':
        transform_train = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
           ])
        transform_val = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
           ])
        
    return transform_train, transform_val

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
