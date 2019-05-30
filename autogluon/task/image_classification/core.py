import logging
import os
import time

import numpy as np
import ConfigSpace as CS
import argparse
from ConfigSpace import InCondition
from mxboard import SummaryWriter

import autogluon as ag

from .pipeline import *
from .model_zoo import *
from ...network import Nets
from ...optim import Optimizers, get_optim

__all__ = ['fit']

logger = logging.getLogger(__name__)


class Results(object):
    def __init__(self, model, accuracy, config, time):
        self.model = model
        self.val_accuracy = accuracy
        self.config = config
        self.time = time


def fit(data,
        nets=Nets([
            get_model('cifar_resnet20_v1'),
            get_model('cifar_resnet56_v1'),
            get_model('cifar_resnet110_v1')]),
        optimizers=Optimizers(
            [get_optim('sgd'),
             get_optim('adam')]),
        metrics=None,
        losses=None,
        searcher=None,
        trial_scheduler=None,
        resume=False,
        savedir='checkpoint/exp1.ag',
        visualizer='tensorboard',
        stop_criterion={
            'time_limits': 1*60*60,
            'max_metric': 0.80,
            'max_trial_count': 2
        },
        resources_per_trial={
            'max_num_gpus': 0,
            'max_num_cpus': 4,
            'max_training_epochs': 10
        },
        backend='default',
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
        e.g. ``{"max_num_cpus": 64, "max_num_gpus": 8}``. Note that GPUs will not be
        assigned unless you specify them here.
    savedir (str): Local dir to save training results to.
    searcher: Search Algorithm.
    trial_scheduler: Scheduler for executing
        the experiment. Choose among FIFO (default) and HyperBand.
    resume (bool): If checkpoint exists, the experiment will
        resume from there.
    backend: support autogluon default backend, ray. (Will support SageMaker)
    **kwargs: Used for backwards compatibility.

    Returns
    ----------
    results:
        model: the parameters associated with the best model. (TODO:)
        val_accuracy: validation set accuracy
        config: best configuration
        time: total time cost
    """
    logger.info('Start fitting')
    start_fit_time = time.time()

    def _construct_search_space(objs, obj_names):
        def _set_range(obj, name):
            if obj.search_space is not None:
                cs.add_configuration_space(prefix='',
                                           delimiter='',
                                           configuration_space=obj.search_space)

        def _assert_fit_error(obj, name):
            assert obj is not None, '%s cannot be None' % name

        cs = CS.ConfigurationSpace()
        for _, (obj, obj_name) in enumerate(zip(objs, obj_names)):
            if obj is not None:
                _set_range(obj, obj_name)
            else:
                if obj_name == 'data':
                    _assert_fit_error(obj, obj_name)

        def _init_args(cs):
            args = argparse.Namespace()
            args_dict = vars(args)
            args_dict['epochs'] = resources_per_trial['max_training_epochs']
            args_dict['num_gpus'] = resources_per_trial['max_num_gpus']
            args_dict['data'] = data.name
            for hparam in cs.get_hyperparameters():
                args_dict[hparam.name] = hparam.default_value
            return args
        args = _init_args(cs)
        return cs, args
    logger.info('Start constructing search space')
    search_objs = [data, nets, optimizers, losses, metrics]
    # TODO (cgraywang) : replace with autogluon*.name
    search_obj_names = ['data', 'net', 'optimizer', 'loss', 'metric']
    cs, args = _construct_search_space(search_objs, search_obj_names)
    if visualizer is not None:
        logger.info('Start initializing visualizer')
        logdir = os.path.join(os.path.splitext(savedir)[0], 'logs')
        sw = SummaryWriter(logdir=logdir, flush_secs=3)
        vars(args).update({'viz': sw})
        logger.info('Finished.')
    logger.info('Finished.')

    def _run_ray_backend(searcher, trial_scheduler):
        logger.info('Start using ray as backend')
        from ray import tune
        if searcher is None:
            searcher = tune.search_policy.RandomSearch(cs,
                                                       stop_criterion['max_metric'],
                                                       stop_criterion['max_trial_count'])
        if trial_scheduler is None:
            trial_scheduler = tune.schedulers.FIFOScheduler()

        tune.register_trainable(
            "TRAIN_FN", lambda config, reporter: train_image_classification(
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

        # TODO (cgraywang)
        best_config = None
        results = Results(None, best_result, best_config, time.time()-start_fit_time)
        logger.info('Finished.')
        return results

    def _run_backend(searcher, trial_scheduler):
        logger.info('Start using default backend.')

        if searcher is None or searcher == 'random':
            searcher = ag.searcher.RandomSampling(cs)
        if trial_scheduler == 'hyperband':
            trial_scheduler = ag.scheduler.Hyperband_Scheduler(
                train_image_classification,
                args,
                {
                    'num_cpus':
                        int(
                            resources_per_trial[
                                'max_num_cpus']),
                    'num_gpus':
                        int(
                            resources_per_trial[
                                'max_num_gpus'])},
                searcher,
                checkpoint=savedir,
                resume=resume,
                time_attr='epoch',
                reward_attr='accuracy',
                max_t=resources_per_trial[
                    'max_training_epochs'],
                grace_period=1)
        else:
            trial_scheduler = ag.scheduler.FIFO_Scheduler(
                train_image_classification,
                args,
                {
                    'num_cpus':
                        int(
                            resources_per_trial['max_num_cpus']),
                    'num_gpus':
                        int(
                            resources_per_trial[
                                'max_num_gpus'])},
                searcher,
                checkpoint=savedir,
                resume=resume)
        trial_scheduler.run(num_trials=stop_criterion['max_trial_count'])
        if hasattr(args, 'viz'):
            args.viz.export_scalars('{}.json'.format(os.path.splitext(savedir)[0]))
            args.viz.close()
        # TODO (cgraywang)
        trials = None
        best_result = trial_scheduler.get_best_reward()
        best_config = trial_scheduler.get_best_config()
        results = Results(trials, best_result, best_config, time.time() - start_fit_time)
        logger.info('Finished.')
        return results

    if backend == 'ray':
        results = _run_ray_backend(searcher, trial_scheduler)
    else:
        results = _run_backend(searcher, trial_scheduler)
    logger.info('Finished.')
    return results
