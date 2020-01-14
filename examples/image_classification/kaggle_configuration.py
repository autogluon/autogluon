import os
import autogluon as ag
from mxnet import optimizer as optim

def download_shopee(dataset, data_path):
    if not os.path.exists(os.path.join(data_path, dataset + '.zip')):
        filename = ag.download('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip',
                               path='data/')
        ag.mkdir(filename[:-4])
        ag.unzip(filename, root=filename[:-4])
    else:
        print(dataset + '.zip already exists.\n')

def config_choice(dataset, data_path):
    global kaggle_choice
    dataset_path = os.path.join(data_path, dataset, 'images')
    if dataset == 'dogs-vs-cats-redux-kernels-edition':
        net_cat = ag.space.Categorical('resnet34_v1b', 'resnet34_v1', 'resnet34_v2')
        @ag.obj(
            learning_rate=ag.space.Real(1e-4, 1e-2, log=True),
            momentum=ag.space.Real(0.86, 0.99),
            wd=ag.space.Real(1e-6, 1e-3, log=True),
            multi_precision=False
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()

        lr_config = ag.space.Dict(
                    lr_mode='cosine',
                    lr_decay=0.1,
                    lr_decay_period=0,
                    lr_decay_epoch='40,80',
                    warmup_lr=0.0,
                    warmup_epochs=5)
        tricks = ag.space.Dict(
                    last_gamma=True,
                    use_pretrained=True,
                    use_se=False,
                    mixup=False,
                    mixup_alpha=0.2,
                    mixup_off_epoch=0,
                    label_smoothing=True,
                    no_wd=True,
                    teacher_name=None,
                    temperature=20.0,
                    hard_weight=0.5,
                    batch_norm=False,
                    use_gn=False)
        kaggle_choice = {'classes': 2, 'net': net_cat, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 384,#512
                         'epochs': 180,
                         'ngpus_per_trial': 8,
                         'lr_config': lr_config,
                         'tricks': tricks,
                         'num_trials': 30}
    elif dataset == 'aerial-cactus-identification':
        net_aeri = ag.space.Categorical('resnet34_v1b')
        @ag.obj(
            learning_rate=ag.space.Real(1e-4, 1e-2, log=True),
            momentum=ag.space.Real(0.88, 0.95),
            wd=ag.space.Real(1e-6, 1e-4, log=True),
            multi_precision=False
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        lr_config = ag.space.Dict(
                    lr_mode='cosine',
                    lr_decay=0.1,
                    lr_decay_period=0,
                    lr_decay_epoch='40,80',
                    warmup_lr=0.0,
                    warmup_epochs=5)
        tricks = ag.space.Dict(
                    last_gamma=True,
                    use_pretrained=True,
                    use_se=False,
                    mixup=False,
                    mixup_alpha=0.2,
                    mixup_off_epoch=0,
                    label_smoothing=True,
                    no_wd=True,
                    teacher_name=None,
                    temperature=20.0,
                    hard_weight=0.5,
                    batch_norm=False,
                    use_gn=False)
        kaggle_choice = {'classes': 2, 'net': net_aeri, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 256,#384
                         'epochs': 180,
                         'ngpus_per_trial': 8,
                         'lr_config': lr_config,
                         'tricks': tricks,
                         'num_trials': 30}
    elif dataset == 'plant-seedlings-classification':
        net_plant = ag.space.Categorical('resnet50_v1', 'resnet50_v1b', 'resnet50_v1c',
                                        'resnet50_v1d', 'resnet50_v1s')

        @ag.obj(
            learning_rate=ag.space.Real(1e-4, 1e-3, log=True),
            momentum=ag.space.Real(0.93, 0.95),
            wd=ag.space.Real(1e-6, 1e-4, log=True),
            multi_precision=False
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        lr_config = ag.space.Dict(
                    lr_mode='cosine',
                    lr_decay=0.1,
                    lr_decay_period=0,
                    lr_decay_epoch='40,80',
                    warmup_lr=0.0,
                    warmup_epochs=5)
        tricks = ag.space.Dict(
                    last_gamma=True,
                    use_pretrained=True,
                    use_se=False,
                    mixup=False,
                    mixup_alpha=0.2,
                    mixup_off_epoch=0,
                    label_smoothing=True,
                    no_wd=True,
                    teacher_name=None,
                    temperature=20.0,
                    hard_weight=0.5,
                    batch_norm=False,
                    use_gn=False)
        kaggle_choice = {'classes': 12, 'net': net_plant, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 16,
                         'epochs': 96,
                         'ngpus_per_trial': 8,
                         'lr_config': lr_config,
                         'tricks': tricks,
                         'num_trials': 30}
    elif dataset == 'fisheries_Monitoring':
        net_fish = ag.space.Categorical('resnet50_v1')
        @ag.obj(
            learning_rate=ag.space.Real(1e-3, 1e-2, log=True),
            momentum=ag.space.Real(0.85, 0.90),
            wd=ag.space.Real(1e-6, 1e-4, log=True),
            multi_precision=False
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()

        lr_config = ag.space.Dict(
                    lr_mode='cosine',
                    lr_decay=0.1,
                    lr_decay_period=0,
                    lr_decay_epoch='40,80',
                    warmup_lr=0.0,
                    warmup_epochs=5)
        tricks = ag.space.Dict(
                    last_gamma=True,
                    use_pretrained=True,
                    use_se=False,
                    mixup=False,
                    mixup_alpha=0.2,
                    mixup_off_epoch=0,
                    label_smoothing=True,
                    no_wd=True,
                    teacher_name=None,
                    temperature=20.0,
                    hard_weight=0.5,
                    batch_norm=False,
                    use_gn=False)
        kaggle_choice = {'classes': 8, 'net': net_fish, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 96,
                         'epochs': 120,
                         'ngpus_per_trial': 8,
                         'lr_config': lr_config,
                         'tricks': tricks,
                         'num_trials': 30}
    elif dataset == 'dog-breed-identification':
        net_dog = ag.space.Categorical('resnet101_v1', 'resnet101_v2', 'resnext101_64x4d', 'resnet101_v1b_gn',
                                       'resnet101_v1b', 'resnet101_v1c', 'resnet101_v1d', 'resnet101_v1e',
                                       'resnet101_v1s', 'resnext101b_64x4d')

        @ag.obj(
            learning_rate=ag.space.Real(1e-4, 1e-3, log=True),
            momentum=ag.space.Real(0.90, 0.95),
            wd=ag.space.Real(1e-6, 1e-4, log=True),
            multi_precision=True  # True fix
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        lr_config = ag.space.Dict(
                    lr_mode='cosine',
                    lr_decay=0.1,
                    lr_decay_period=0,
                    lr_decay_epoch='40,80',
                    warmup_lr=0.0,
                    warmup_epochs=5)
        tricks = ag.space.Dict(
                    last_gamma=True,
                    use_pretrained=True,
                    use_se=False,
                    mixup=False,
                    mixup_alpha=0.2,
                    mixup_off_epoch=0,
                    label_smoothing=True,
                    no_wd=True,
                    teacher_name=None,
                    temperature=20.0,
                    hard_weight=0.5,
                    batch_norm=False,
                    use_gn=False)
        kaggle_choice = {'classes': 120, 'net': net_dog, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 48,
                         'epochs': 180,
                         'ngpus_per_trial': 8,
                         'lr_config': lr_config,
                         'tricks': tricks,
                         'num_trials': 30}
    elif dataset == 'shopee-iet-machine-learning-competition':
        net_shopee = ag.space.Categorical('resnet152_v1','resnet152_v2', 'resnet152_v1b', 'resnet152_v1d','resnet152_v1s')
        @ag.obj(
            learning_rate=ag.space.Real(1e-4, 1e-2, log=True),
            momentum=ag.space.Real(0.90, 1.0),
            wd=ag.space.Real(1e-4, 1e-2, log=True),
            multi_precision=False
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()
        lr_config = ag.space.Dict(
                    lr_mode='cosine',
                    lr_decay=0.1,
                    lr_decay_period=0,
                    lr_decay_epoch='40,80',
                    warmup_lr=0.0,
                    warmup_epochs=5)

        tricks = ag.space.Dict(
                    last_gamma=True,
                    use_pretrained=True,
                    use_se=False,
                    mixup=False,
                    mixup_alpha=0.2,
                    mixup_off_epoch=0,
                    label_smoothing=True,
                    no_wd=True,
                    teacher_name=None,
                    temperature=20.0,
                    hard_weight=0.5,
                    batch_norm=False,
                    use_gn=False)

        kaggle_choice = {'classes': 18, 'net': net_shopee, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 48,
                         'epochs': 180,
                         'ngpus_per_trial': 8,
                         'lr_config': lr_config,
                         'tricks': tricks,
                         'num_trials': 30}
    elif dataset == 'shopee-iet':
        download_shopee(dataset, data_path)
        dataset_path = os.path.join(data_path, dataset, 'data', 'train')
        net_shopee = ag.space.Categorical('resnet18_v1')
        @ag.obj(
            learning_rate=ag.space.Real(1e-4, 1e-2, log=True),
            momentum=ag.space.Real(0.85, 0.95),
            wd=ag.space.Real(1e-6, 1e-2, log=True),
<<<<<<< HEAD
            multi_precision=False
=======
            multi_precision=True
>>>>>>> upstream/master
        )
        class NAG(optim.NAG):
            pass
        optimizer = NAG()

        lr_config = ag.space.Dict(
                    lr_mode='cosine',
                    lr_decay=0.1,
                    lr_decay_period=0,
                    lr_decay_epoch='40,80',
                    warmup_lr=0.0,
                    warmup_epochs=5)

        tricks = ag.space.Dict(
                    last_gamma=True,
                    use_pretrained=True,
                    use_se=False,
                    mixup=False,
                    mixup_alpha=0.2,
                    mixup_off_epoch=0,
                    label_smoothing=True,
                    no_wd=True,
                    teacher_name=None,
                    temperature=20.0,
                    hard_weight=0.5,
                    batch_norm=False,
                    use_gn=False)

        kaggle_choice = {'classes': 4, 'net': net_shopee, 'optimizer': optimizer,
                         'dataset': dataset_path,
                         'batch_size': 2,
                         'epochs': 1,
                         'ngpus_per_trial': 1,
                         'lr_config': lr_config,
                         'tricks': tricks,
                         'num_trials': 1}
    return kaggle_choice

