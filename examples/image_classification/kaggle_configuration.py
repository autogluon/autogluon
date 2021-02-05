import os
import autogluon.core as ag
from mxnet import optimizer as optim

def download_shopee(dataset, data_path):
    if not os.path.exists(os.path.join(data_path, dataset + '.zip')):
        filename = ag.download('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip',
                               path='data/shopee-iet.zip')
        ag.mkdir(filename[:-4])
        ag.unzip(filename, root=filename[:-4])
    else:
        print(dataset + '.zip already exists.\n')

def config_choice(data_path, dataset):
    global kaggle_choice
    dataset_path = os.path.join(data_path, dataset, 'images')
    if dataset == 'dogs-vs-cats-redux-kernels-edition':
        net_cat = ag.space.Categorical('resnet34_v1b') #resnet34_v1
        @ag.obj(
            learning_rate=ag.space.Real(0.3, 0.5),
            momentum=ag.space.Real(0.86, 0.99),
            wd=ag.space.Real(1e-5, 1e-3, log=True)
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()

        kaggle_choice = {'classes': 2,
                         'net': net_cat,
                         'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 320,#512
                         'epochs': 180,
                         'ngpus_per_trial': 4,
                         'lr_mode': 'step',
                         'lr_decay': 0.1,
                         'lr_decay_period': 0,
                         'lr_decay_epoch': '40,80',
                         'warmup_lr': 0.0,
                         'warmup_epochs': 5,
                         'last_gamma': True,
                         'use_pretrained': True,
                         'use_se': False,
                         'mixup': False,
                         'mixup_alpha': 0.2,
                         'mixup_off_epoch': 0,
                         'label_smoothing': True,
                         'no_wd': True,
                         'teacher_name': None,
                         'temperature': 20.0,
                         'hard_weight': 0.5,
                         'batch_norm': False,
                         'use_gn': False,
                         'num_trials': 16}
    elif dataset == 'aerial-cactus-identification':
        net_aeri = ag.space.Categorical('resnet34_v1b')
        @ag.obj(
            learning_rate=ag.space.Real(0.3, 0.5),
            momentum=ag.space.Real(0.88, 0.95),
            wd=ag.space.Real(1e-5, 1e-3, log=True)
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        kaggle_choice = {'classes': 2, 'net': net_aeri, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 320,#256
                         'epochs': 180,
                         'ngpus_per_trial': 4,
                         'lr_mode': 'step',
                         'lr_decay': 0.1,
                         'lr_decay_period': 0,
                         'lr_decay_epoch': '60,120',
                         'warmup_lr': 0.0,
                         'warmup_epochs': 5,
                         'last_gamma': True,
                         'use_pretrained': True,
                         'use_se': False,
                         'mixup': False,
                         'mixup_alpha': 0.2,
                         'mixup_off_epoch': 0,
                         'label_smoothing': True,
                         'no_wd': True,
                         'teacher_name': None,
                         'temperature': 20.0,
                         'hard_weight': 0.5,
                         'batch_norm': False,
                         'use_gn': False,
                         'num_trials': 30}
    elif dataset == 'plant-seedlings-classification':
        net_plant = ag.space.Categorical('resnet50_v1')
        @ag.obj(
            learning_rate=ag.space.Real(0.3, 0.5),
            momentum=ag.space.Real(0.85, 0.95),
            wd=ag.space.Real(1e-6, 1e-4, log=True)
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        kaggle_choice = {'classes': 12, 'net': net_plant, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 128,
                         'epochs': 120,
                         'ngpus_per_trial': 2,
                         'lr_mode': 'cosine',
                         'lr_decay': 0.1,
                         'lr_decay_period': 0,
                         'lr_decay_epoch': '40,80',
                         'warmup_lr': 0.0,
                         'warmup_epochs': 5,
                         'last_gamma': True,
                         'use_pretrained': True,
                         'use_se': False,
                         'mixup': False,
                         'mixup_alpha': 0.2,
                         'mixup_off_epoch': 0,
                         'label_smoothing': True,
                         'no_wd': True,
                         'teacher_name': None,
                         'temperature': 20.0,
                         'hard_weight': 0.5,
                         'batch_norm': False,
                         'use_gn': False,
                         'num_trials': 30}
    elif dataset == 'fisheries_Monitoring':
        net_fish = ag.space.Categorical('resnet50_v1')
        @ag.obj(
            learning_rate=ag.space.Real(0.3, 0.5),
            momentum=ag.space.Real(0.85, 0.95),
            wd=ag.space.Real(1e-6, 1e-4, log=True)
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        kaggle_choice = {'classes': 8, 'net': net_fish, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 128,
                         'epochs': 120,
                         'ngpus_per_trial': 2,
                         'lr_mode': 'cosine',
                         'lr_decay': 0.1,
                         'lr_decay_period': 0,
                         'lr_decay_epoch': '40,80',
                         'warmup_lr': 0.0,
                         'warmup_epochs': 5,
                         'last_gamma': True,
                         'use_pretrained': True,
                         'use_se': False,
                         'mixup': False,
                         'mixup_alpha': 0.2,
                         'mixup_off_epoch': 0,
                         'label_smoothing': True,
                         'no_wd': True,
                         'teacher_name': None,
                         'temperature': 20.0,
                         'hard_weight': 0.5,
                         'batch_norm': False,
                         'use_gn': False,
                         'num_trials': 30}
    elif dataset == 'dog-breed-identification':
        net_dog = ag.space.Categorical('resnext101_64x4d')
        @ag.obj(
            learning_rate=ag.space.Real(0.3, 0.5),
            momentum=ag.space.Real(0.85, 0.95),
            wd=ag.space.Real(1e-6, 1e-4, log=True)
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        kaggle_choice = {'classes': 120, 'net': net_dog, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 48,
                         'epochs': 180,
                         'ngpus_per_trial': 4,
                         'lr_mode': 'cosine',
                         'lr_decay': 0.1,
                         'lr_decay_period': 0,
                         'lr_decay_epoch': '60,120',
                         'warmup_lr': 0.0,
                         'warmup_epochs': 5,
                         'last_gamma': True,
                         'use_pretrained': True,
                         'use_se': False,
                         'mixup': False,
                         'mixup_alpha': 0.2,
                         'mixup_off_epoch': 0,
                         'label_smoothing': True,
                         'no_wd': True,
                         'teacher_name': None,
                         'temperature': 20.0,
                         'hard_weight': 0.5,
                         'batch_norm': False,
                         'use_gn': False,
                         'num_trials': 30}
    elif dataset == 'shopee-iet-machine-learning-competition':
        net_shopee = ag.space.Categorical('resnet152_v1d')
        @ag.obj(
            learning_rate=ag.space.Real(1e-2, 1e-1, log=True),
            momentum=ag.space.Real(0.85, 0.95),
            wd=ag.space.Real(1e-6, 1e-4, log=True)
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        kaggle_choice = {'classes': 18, 'net': net_shopee, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 48,
                         'epochs': 180,
                         'ngpus_per_trial': 4,
                         'lr_mode': 'cosine',
                         'lr_decay': 0.1,
                         'lr_decay_period': 0,
                         'lr_decay_epoch': '60,120',
                         'warmup_lr': 0.0,
                         'warmup_epochs': 5,
                         'last_gamma': True,
                         'use_pretrained': True,
                         'use_se': False,
                         'mixup': False,
                         'mixup_alpha': 0.2,
                         'mixup_off_epoch': 0,
                         'label_smoothing': True,
                         'no_wd': True,
                         'teacher_name': None,
                         'temperature': 20.0,
                         'hard_weight': 0.5,
                         'batch_norm': False,
                         'use_gn': False,
                         'num_trials': 30}
    elif dataset == 'shopee-iet':
        download_shopee(dataset, data_path)
        dataset_path = os.path.join(data_path, dataset, 'data')
        net_shopee = ag.space.Categorical('resnet18_v1')
        @ag.obj(
            learning_rate=ag.space.Real(1e-4, 1e-2, log=True),
            momentum=ag.space.Real(0.85, 0.95),
            wd=ag.space.Real(1e-6, 1e-2, log=True),
            multi_precision=True
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        kaggle_choice = {'classes': 4, 'net': net_shopee, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 2,
                         'epochs': 1,
                         'ngpus_per_trial': 1,
                         'lr_mode': 'cosine',
                         'lr_decay': 0.1,
                         'lr_decay_period': 0,
                         'lr_decay_epoch': '40,80',
                         'warmup_lr': 0.0,
                         'warmup_epochs': 5,
                         'last_gamma': True,
                         'use_pretrained': True,
                         'use_se': False,
                         'mixup': False,
                         'mixup_alpha': 0.2,
                         'mixup_off_epoch': 0,
                         'label_smoothing': True,
                         'no_wd': True,
                         'teacher_name': None,
                         'temperature': 20.0,
                         'hard_weight': 0.5,
                         'batch_norm': False,
                         'use_gn': False,
                         'num_trials': 1}
    return kaggle_choice
