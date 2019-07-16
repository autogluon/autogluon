import logging
import os
import time
import shutil

import numpy as np
import ConfigSpace as CS
import argparse
from ConfigSpace import InCondition

import autogluon as ag

from .pipeline import *
from .model_zoo import *
from .losses import *
from .metrics import *
from ...network import Nets
from ...optim import Optimizers, get_optim
from ...loss import Losses
from ...metric import Metrics

__all__ = ['fit']

logger = logging.getLogger(__name__)

#TODO: abstract Results
class Results(object):
    def __init__(self, model, metric, config, time):
        self.model = model
        self.val_metric = metric
        self.config = config
        self.time = time


#TODO: abstract fit
def fit(data,
        nets=Nets([
            get_model('ssd_512_resnet50_v1_coco'),
            get_model('faster_rcnn_fpn_resnet101_v1d_coco'),
            get_model('yolo3_mobilenet1.0_coco')]),
        optimizers=Optimizers([
            get_optim('sgd'),
            get_optim('adam')]),
        metrics=Metrics([
            get_metric('VOC07MApMetric')]),
        losses=Losses([
            get_loss('SSDMultiBoxLoss')]),
        searcher=None,
        trial_scheduler=None,
        resume=False,
        savedir='checkpoint/exp1.ag',
        visualizer='tensorboard',
        stop_criterion={
            'time_limits': 1 * 60 * 60,
            'max_metric': 1,
            'max_trial_count': 2
        },
        resources_per_trial={
            'max_num_gpus': 0,
            'max_num_cpus': 4,
            'max_training_epochs': 3
        },
        backend='default',
        demo=False,
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
        metric: validation set metric
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
            if 'lr_factor' in kwargs and 'lr_step' in kwargs:
                args_dict['lr_factor'] = kwargs['lr_factor']
                args_dict['lr_step'] = kwargs['lr_step']
            if 'val_interval' in kwargs:
                args_dict['val_interval'] = kwargs['val_interval']
            args_dict['data'] = data
            args_dict['backend'] = backend
            args_dict['demo'] = demo
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
    logger.info('Finished.')

    def _reset_checkpoint(dir, resume):
        dir = os.path.splitext(dir)[0]
        if not resume and os.path.exists(dir):
            shutil.rmtree(dir)
            os.makedirs(dir)
    _reset_checkpoint(savedir, resume)

    def _run_ray_backend(searcher, trial_scheduler):
        logger.info('Start using ray as backend')
        try:
            from ray import tune
        except ImportError:
            raise ImportError(
                "Unable to import dependency ray. "
                "A quick tip is to install via `pip install ray`. ")
        if searcher is None or searcher == 'random':
            assert isinstance(cs, dict)
            searcher = tune.search_policy.RandomSearch(cs,
                                                       stop_criterion['max_metric'],
                                                       stop_criterion['max_trial_count'])
        if trial_scheduler == 'hyperband':
            trial_scheduler = tune.schedulers.AsyncHyperBandScheduler(
                time_attr="training_iteration",
                reward_attr="meanap",
                max_t=resources_per_trial['max_training_epochs'],
                grace_period=resources_per_trial['max_training_epochs'] // 4)
        else:
            trial_scheduler = tune.schedulers.FIFOScheduler()
        tune.register_trainable(
            "TRAIN_FN", lambda reporter: train_object_detection(
                args, reporter))
        trials = tune.run(
            "TRAIN_FN",
            name=savedir,
            scheduler=trial_scheduler,
            search_alg=searcher,
            **{
                "stop": {
                    "meanap": stop_criterion['max_metric'],
                    "training_iteration": resources_per_trial['max_training_epochs']
                },
                "resources_per_trial": {
                    "cpu": int(resources_per_trial['max_num_cpus']),
                    "gpu": int(resources_per_trial['max_num_gpus'])
                },
                "num_samples": stop_criterion['max_trial_count']
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
                train_object_detection,
                args,
                {'num_cpus': int(resources_per_trial['max_num_cpus']),
                 'num_gpus': int(resources_per_trial['max_num_gpus'])},
                searcher,
                checkpoint=savedir,
                resume=resume,
                time_attr='epoch',
                reward_attr='map',
                max_t=resources_per_trial[
                    'max_training_epochs'],
                grace_period=resources_per_trial[
                    'max_training_epochs']//4,
                visualizer=visualizer)
            # TODO (cgraywang): use empiral val now
        else:
            trial_scheduler = ag.scheduler.FIFO_Scheduler(
                train_object_detection,
                args,
                {'num_cpus': int(resources_per_trial['max_num_cpus']),
                 'num_gpus': int(resources_per_trial['max_num_gpus'])},
                searcher,
                checkpoint=savedir,
                resume=resume,
                reward_attr='map',
                visualizer=visualizer)
        trial_scheduler.run(num_trials=stop_criterion['max_trial_count'])
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
