from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

from ... import dataset

from ...utils.data_analyzer import DataAnalyzer

from .model_zoo import *
from .pipeline import *

import numpy as np
import ConfigSpace as CS
import argparse

import autogluon as ag

from . import pipeline
from . import get_model
from ...network import Nets
from ...optim import Optimizers, get_optim
from ...loss import Losses
from ...metric import Metrics
from ...searcher import *

__all__ = ['fit', 'Dataset'] + model_zoo.__all__ + pipeline.__all__


def fit(data,
        nets=Nets([get_model('resnet18_v1'),
                   get_model('resnet34_v1'),
                   get_model('resnet50_v1'),
                   get_model('resnet101_v1')]),
        optimizers=Optimizers([get_optim('sgd'),
                               get_optim('adam')]),
        metrics=Metrics([]),
        losses=Losses([]),
        searcher=None,
        trial_scheduler=None,
        resume=False,
        savedir='./outputdir/',
        visualizer='tensorboard',
        stop_criterion={'time_limits': 1 * 60 * 60,
                        'max_metric': 0.80,
                        'max_trial_count': 10},
        resources_per_trial={'max_num_gpus': 1,
                             'max_num_cpus': 4,
                             'max_training_epochs': 1},
        backend='autogluon',
        **kwargs):
    r"""
    Fit networks on dataset

    Parameters
    ----------
    data: Input data. It could be:
        autogluon.Datasets
        task.Datasets
    nets: autogluon.Nets
    optimizers: autogluon.Optimizers
    metrics: autogluon.Metrics
    losses: autogluon.Losses
    stop_criterion (dict): The stopping criteria. The keys may be any field in
        the return result of 'train()', whichever is reached first.
        Defaults to empty dict.
    resources_per_trial (dict): Machine resources to allocate per trial,
        e.g. ``{"cpu": 64, "gpu": 8}``. Note that GPUs will not be
        assigned unless you specify them here. Defaults to 1 CPU and 0
        GPUs in ``Trainable.default_resource_request()``.
    savedir (str): Local dir to save training results to.
    checkpoint_freq (int): How many training iterations between
        checkpoints. A value of 0 (default) disables checkpointing.
    checkpoint_at_end (bool): Whether to checkpoint at the end of the
        experiment regardless of the checkpoint_freq. Default is False.
    searcher (SearchAlgorithm): Search Algorithm. Defaults to
        BasicVariantGenerator.
    trial_scheduler (TrialScheduler): Scheduler for executing
        the experiment. Choose among FIFO (default), MedianStopping,
        AsyncHyperBand, and HyperBand.
    resume (bool|"prompt"): If checkpoint exists, the experiment will
        resume from there. If resume is "prompt", Tune will prompt if
        checkpoint detected.
    **kwargs: Used for backwards compatibility.

    Returns
    ----------
    List of Trial objects.
    """
    cs = CS.ConfigurationSpace()
    args = argparse.Namespace()
    args_dict = vars(args)
    assert data is not None
    assert nets is not None
    if data.search_space is not None:
        cs.add_configuration_space(data.search_space.name, data.search_space)
        for hparam in data.search_space.get_hyperparameters():
            if 'None:' in hparam.name:
                args_dict[hparam.name.split('None:')[1]] = None
            else:
                args_dict[hparam.name] = None
    if nets.search_space is not None:
        cs.add_configuration_space(nets.search_space.name, nets.search_space)
        for hparam in nets.search_space.get_hyperparameters():
            if 'None:' in hparam.name:
                args_dict[hparam.name.split('None:')[1]] = None
            else:
                args_dict[hparam.name] = None
    if optimizers.search_space is not None:
        cs.add_configuration_space(optimizers.search_space.name, optimizers.search_space)
        for hparam in optimizers.search_space.get_hyperparameters():
            if 'None:' in hparam.name:
                args_dict[hparam.name.split('None:')[1]] = None
            else:
                args_dict[hparam.name] = None
    if metrics.search_space is not None:
        cs.add_configuration_space(metrics.search_space.name, metrics.search_space)
        for hparam in metrics.search_space.get_hyperparameters():
            if 'None:' in hparam.name:
                args_dict[hparam.name.split('None:')[1]] = None
            else:
                args_dict[hparam.name] = None
    else:
        # TODO: default value leaves for args for now for simplicity
        args_dict['metric'] = 'accuracy'
    if losses.search_space is not None:
        cs.add_configuration_space('fit_space', losses.search_space)
    else:
        # args.loss = 'softmaxcrossentropy'
        args_dict['loss'] = 'softmaxcrossentropy'

    if backend == 'ray':
        from ray import tune
        if searcher is None:
            # ray.tune.automl.search_policy.RandomSearch
            searcher = tune.search_policy.RandomSearch(cs,
                                                       stop_criterion['max_metric'],
                                                       stop_criterion['max_trial_count'])
        if trial_scheduler is None:
            trial_scheduler = tune.schedulers.FIFOScheduler()

        tune.register_trainable(
            "TRAIN_FN", lambda config, reporter: pipeline.train_image_classification(
                args, config, reporter))
        trials = tune.run(
            "TRAIN_FN",
            name=args.expname,
            verbose=2,
            scheduler=trial_scheduler,
            search_alg=searcher,
            **{
                "stop": {
                    "mean_accuracy": stop_criterion['max_metric'],
                    "training_iteration": resources_per_trial['max_training_epochs']
                },
                "resources_per_trial": {
                    "cpu": int(resources_per_trial['max_num_cpus']),
                    "gpu": int(resources_per_trial['max_num_gpus'])
                },
                "num_samples": resources_per_trial['max_trial_count'],
                "config": {
                    "lr": tune.sample_from(lambda spec: np.power(
                        10.0, np.random.uniform(-4, -1))),
                    "momentum": tune.sample_from(lambda spec: np.random.uniform(
                        0.85, 0.95)),
                }
            })
        best_result = max([trial.best_result for trial in trials])
    else:
        args.train_data = data.train_data
        args.val_data = data.val_data
        if searcher is None:
            searcher = ag.searcher.RandomSampling(cs)
        if trial_scheduler == 'hyperband':
            trial_scheduler = ag.scheduler.Hyperband_Scheduler(pipeline.train_image_classification,
                                                               args,
                                                               {'num_cpus': int(
                                                                   resources_per_trial[
                                                                       'max_num_cpus']),
                                                                   'num_gpus': int(
                                                                       resources_per_trial[
                                                                           'max_num_gpus'])},
                                                               searcher,
                                                               time_attr='epoch',
                                                               reward_attr='accuracy',
                                                               max_t=resources_per_trial[
                                                                   'max_training_epochs'],
                                                               grace_period=1)
        else:
            trial_scheduler = ag.scheduler.FIFO_Scheduler(pipeline.train_image_classification, args,
                                                          {'num_cpus': int(
                                                              resources_per_trial['max_num_cpus']),
                                                              'num_gpus': int(
                                                                  resources_per_trial[
                                                                      'max_num_gpus'])},
                                                          searcher, )

        trial_scheduler.run(num_trials=stop_criterion['max_trial_count'])
        # TODO: add results for ag
        trials = None
        best_result = None
    return trials, best_result, cs


class Dataset(dataset.Dataset):
    def __init__(self, train_path=None, val_path=None):
        # TODO (cgraywang): add search space, handle batch_size, num_workers

        self.train_path = train_path
        self.val_path = val_path
        self._read_dataset()
        self.search_space = None
        self.add_search_space()
        self.train_data = None
        self.val_data = None

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
