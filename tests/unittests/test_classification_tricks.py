import os

import pytest

import autogluon as ag
from autogluon import ImageClassification as task
from mxnet import optimizer as optim

tricks_combination = [
    ag.space.Dict(
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
        use_gn=False),
    ag.space.Dict(
        last_gamma=False,
        use_pretrained=False,
        use_se=False,
        mixup=True,
        mixup_alpha=0.2,
        mixup_off_epoch=0,
        label_smoothing=False,
        no_wd=False,
        teacher_name='resnet50_v1',
        temperature=20.0,
        hard_weight=0.5,
        batch_norm=True,
        use_gn=False)]


def download_shopee(data_dir, dataset):
    if not os.path.exists(os.path.join(data_dir, dataset + '.zip')):
        filename = ag.download('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip', path=data_dir)
        ag.mkdir(filename[:-4])
        ag.unzip(filename, root=filename[:-4])
    else:
        print(dataset + '.zip already exists.\n')


def config_choice(dataset, data_path, tricks):
    dataset_path = os.path.join(data_path, dataset, 'data', 'train')
    net_shopee = ag.space.Categorical('resnet50_v1')
    @ag.obj(
        learning_rate=ag.space.Real(1e-4, 1e-2, log=True),
        momentum=ag.space.Real(0.85, 0.95),
        wd=ag.space.Real(1e-6, 1e-2, log=True),
        multi_precision=False
    )
    class NAG(optim.NAG):
        pass
    optimizer = NAG()

    lr_config = ag.space.Dict(
        lr_mode=ag.space.Categorical('step', 'cosine'),
        lr_decay=ag.space.Real(0.1, 0.2),
        lr_decay_period=ag.space.Int(0,1),
        lr_decay_epoch=ag.space.Categorical('40,80'),
        warmup_lr=ag.space.Real(0.0,0.5),
        warmup_epochs=ag.space.Int(0,5)
    )

    kaggle_choice = {'classes': 4, 'net': net_shopee, 'optimizer': optimizer,
                     'dataset': dataset_path,
                     'batch_size': 4,
                     'epochs': 1,
                     'ngpus_per_trial': 1,
                     'lr_config': lr_config,
                     'tricks': tricks,
                     'num_trials': 1}
    return kaggle_choice


@pytest.mark.slow
@pytest.mark.parametrize("combination", tricks_combination)
def test_tricks(combination):
    dataset = 'shopee-iet'
    data_dir = './'
    download_shopee(data_dir, dataset)

    target = config_choice(dataset, data_dir, combination)
    classifier = task.fit(dataset = task.Dataset(target['dataset']),
                          net = target['net'],
                          optimizer = target['optimizer'],
                          epochs = target['epochs'],
                          ngpus_per_trial = target['ngpus_per_trial'],
                          num_trials = target['num_trials'],
                          batch_size = target['batch_size'],
                          verbose = True,
                          search_strategy='random',
                          tricks = target['tricks'],
                          lr_config = target['lr_config'],
                          plot_results = True)

    test_dataset = task.Dataset(target['dataset'].replace('train', 'test/BabyPants'), train=False,
                                scale_ratio_choice=[0.7, 0.8, 0.875])
    inds, probs, probs_all = classifier.predict(test_dataset, set_prob_thresh=0.001)
    print(inds[0],probs[0],probs_all[0])

    print('Top-1 val acc: %.3f' % classifier.results['best_reward'])
    # summary = classifier.fit_summary(output_directory=dataset, verbosity=3)
    # print(summary)




