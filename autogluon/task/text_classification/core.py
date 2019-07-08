import argparse
import logging
import os
import time
from typing import Any, AnyStr

import ConfigSpace as CS
import numpy as np
from ray import tune
from ray.tune.automl.search_policy import AutoMLSearcher
from ray.tune.schedulers import TrialScheduler

import autogluon as ag
from autogluon import Linear
from autogluon.utils.mxboard_handler import MXBoardHandler
from .dataset import *
from .model_zoo import *
from .pipeline import *
from ...network import Nets
from ...optim import Optimizers, get_optim

__all__ = ['fit']

logger = logging.getLogger(__name__)
# TODO Add More default networks here. Possibly Bert ?
default_nets = Nets([
    get_model('standard_lstm_lm_200', **{'classification_layers': Linear('dense', lower=0, upper=2)}),
    get_model('standard_lstm_lm_650', **{'classification_layers': Linear('dense', lower=0, upper=2)}),
    get_model('standard_lstm_lm_1500', **{'classification_layers': Linear('dense', lower=0, upper=2)}),
    get_model('awd_lstm_lm_600', **{'classification_layers': Linear('dense', lower=0, upper=2)}),
    get_model('awd_lstm_lm_1150', **{'classification_layers': Linear('dense', lower=0, upper=2)}),
    get_model('bert_12_768_12', **{'classification_layers': Linear('dense', lower=0, upper=2)}),
    get_model('bert_24_1024_16', **{'classification_layers': Linear('dense', lower=0, upper=2)})
])

default_optimizers = Optimizers([
    get_optim('adam'),
    get_optim('sgd'),
    get_optim('ftml'),
    get_optim('bertadam')
])

default_stop_criterion = {
    'time_limits': 1 * 60 * 60,
    'max_metric': 0.80,  # TODO Should be place a bound on metric?
    'max_trial_count': 1
}

default_resources_per_trial = {
    'max_num_gpus': 4,
    'max_num_cpus': 4,
    'max_training_epochs': 5
}


class Results(object):
    """
    Python class to hold the results for the trials
    """

    def __init__(self, model: Any, metric: Any, config: Any, time: int):
        self.model = model
        self.metric = metric
        self.config = config
        self.time = time


def fit(data: Dataset,
        nets: Nets = default_nets,
        optimizers: Optimizers = default_optimizers,
        metrics=None,
        losses=None,
        searcher=None,
        trial_scheduler=None,
        resume=False,
        savedir='checkpoint/exp1_tc.ag',
        visualizer='tensorboard',
        stop_criterion=default_stop_criterion,
        resources_per_trial=default_resources_per_trial,
        backend='default',
        **kwargs
        ) -> Results:
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
    model: the parameters associated with the best model. (TODO: use trial to infer for now)
    best_result: accuracy
    best_config: best configuration
    """
    logger.debug('Starting fit method call')
    start_fit_time = time.time()

    def _construct_search_space(search_space_dict: dict) -> (CS.ConfigurationSpace, argparse.Namespace):
        def _init_args() -> argparse.Namespace:
            """
            This method adds some defaults to the args.
            TODO : Remove hardcoding in this method
            :return:
            """
            args = argparse.Namespace()
            args_dict = vars(args)
            args_dict['epochs'] = resources_per_trial['max_training_epochs']
            args_dict['data_name'] = data.name
            args_dict['train_path'] = data.train_path
            args_dict['val_path'] = data.val_path
            args_dict['num_gpus'] = resources_per_trial['max_num_gpus']
            args_dict['pretrained'] = True
            args_dict['data_format'] = data.data_format
            args_dict['dataset'] = data
            args_dict['task_id'] = 0
            return args

        args = _init_args()

        def _set_range(obj, name):
            if obj is not None and obj.search_space is not None:
                # TODO Should this prefix be blank or something else ?
                cs.add_configuration_space(prefix='text_classification',
                                           configuration_space=obj.search_space,
                                           delimiter=':')

        cs = CS.ConfigurationSpace(name='text_classification')
        for obj_name, obj in search_space_dict.items():
            _set_range(obj, obj_name)

        return cs, args

    logger.debug('Start constructing search space')
    search_space_dict = {'data': data,
                         'net': nets,
                         'optimizer': optimizers,
                         'loss': losses,
                         'metric': metrics}

    cs, args = _construct_search_space(search_space_dict)
    logger.debug('Finished constucting search space')
    mxboard_handler = init_visualizer(visualizer, os.path.splitext(savedir)[0])

    if mxboard_handler is not None:
        vars(args).update({'viz': mxboard_handler})

    def _run_ray_backend(searcher: AutoMLSearcher, trial_scheduler: TrialScheduler) -> Results:
        logger.debug('Start using Ray as a backend')
        if searcher is None:
            searcher = tune.search_policy.RandomSearch(cs, stop_criterion['max_metric'],
                                                       stop_criterion['max_trial_count'])

        if trial_scheduler is None:
            trial_scheduler = tune.schedulers.FIFOScheduler()

        tune.register_trainable("TRAIN_FN", lambda config, reporter: train_text_classification(args, config, reporter))

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
                    # TODO : Why is LR being passed in here, when it should come from optimizers
                }
            })

        best_result = max([trial.best_result for trial in trials])
        best_config = None
        results = Results(None, best_result, best_config, time.time() - start_fit_time)
        logger.debug('Finished.')
        return results

    def _run_backend(searcher: AutoMLSearcher, trial_scheduler: AnyStr) -> Results:
        logger.debug('Start using default backend')
        if searcher is None:
            searcher = ag.searcher.RandomSampling(cs)

        if trial_scheduler is None:
            trial_scheduler = 'default'

        if trial_scheduler == 'hyperband':
            trial_scheduler = ag.scheduler.Hyperband_Scheduler(
                train_text_classification,
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
                train_text_classification,
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
        trial_scheduler.get_training_curves('{}.png'.format(os.path.splitext(savedir)[0]))

        trials = None
        best_result = trial_scheduler.get_best_reward()
        best_config = trial_scheduler.get_best_config()
        results = Results(trials, best_result, best_config, time.time() - start_fit_time)

        logger.debug('Finished.')
        return results

    if backend == 'ray':
        results = _run_ray_backend(searcher, trial_scheduler)

    else:
        results = _run_backend(searcher, trial_scheduler)

    logger.debug('Finished experiments!')
    return results


def init_visualizer(visualizer: AnyStr = None, checkpoint_dir: AnyStr = None) -> MXBoardHandler:
    if visualizer is not None:
        logger.info('Initializing visualizer %s' % visualizer)
        log_dir = os.path.join(checkpoint_dir, 'logs')
        # TODO Add events to log and more from the config
        mxboard_handler = MXBoardHandler(-1, log_dir=log_dir, checkpoint_dir=checkpoint_dir)

        return mxboard_handler

    return None
