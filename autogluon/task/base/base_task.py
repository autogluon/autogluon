import os
import shutil
import logging
import argparse
import time
import ConfigSpace as CS

from collections import namedtuple
from abc import ABC

import autogluon as ag
from ...optim import *#Optimizers, get_optim
from ... import dataset

__all__ = ['BaseTask', 'Results']

logger = logging.getLogger(__name__)

Results = namedtuple('Results', 'model metric config time metadata')

class BaseTask(ABC):
    def __init__(self):
        super(BaseTask, self).__init__()

    @staticmethod
    def _reset_checkpoint(metadata):
        dir = os.path.splitext(metadata['savedir'])[0]
        if not metadata['resume'] and os.path.exists(dir):
            shutil.rmtree(dir)
            os.makedirs(dir)

    @staticmethod
    def _run_backend(cs, args, metadata):
        if metadata['searcher'] is None or metadata['searcher'] == 'random':
            searcher = ag.searcher.RandomSampling(cs)
        else:
            raise NotImplementedError
        if metadata['trial_scheduler'] == 'hyperband':
            trial_scheduler = ag.scheduler.Hyperband_Scheduler(
                metadata['kwargs']['train_func'],
                args,
                {'num_cpus': int(metadata['resources_per_trial']['max_num_cpus']),
                 'num_gpus': int(metadata['resources_per_trial']['max_num_gpus'])},
                searcher,
                checkpoint=metadata['savedir'],
                resume=metadata['resume'],
                time_attr='epoch',
                reward_attr=metadata['kwargs']['reward_attr'],
                max_t=metadata['resources_per_trial'][
                    'max_training_epochs'],
                grace_period=metadata['resources_per_trial'][
                                 'max_training_epochs'] // 4,
                visualizer=metadata['visualizer'])
            # TODO (cgraywang): use empiral val now
        else:
            trial_scheduler = ag.scheduler.FIFO_Scheduler(
                metadata['kwargs']['train_func'],
                args,
                {'num_cpus': int(metadata['resources_per_trial']['max_num_cpus']),
                 'num_gpus': int(metadata['resources_per_trial']['max_num_gpus'])},
                searcher,
                checkpoint=metadata['savedir'],
                resume=metadata['resume'],
                time_attr='epoch',
                reward_attr=metadata['kwargs']['reward_attr'],
                visualizer=metadata['visualizer'])
        trial_scheduler.run(num_trials=metadata['stop_criterion']['max_trial_count'])
        # TODO (cgraywang)
        final_model = None
        final_metric = trial_scheduler.get_best_reward()
        final_config = trial_scheduler.get_best_config()
        return final_model, final_metric, final_config

    @staticmethod
    def fit(data,
            *args,
            #nets=None,
            #optimizers=Optimizers([
            #    ]),
            #metrics=None,
            #losses=None,
            #searcher=None,
            #trial_scheduler=None,
            #resume=False,
            #savedir='checkpoint/exp1.ag',
            #visualizer='tensorboard',
            #stop_criterion={
            #    'time_limits': 1 * 60 * 60,
            #    'max_metric': 1.0,
            #    'max_trial_count': 2
            #},
            #resources_per_trial={
            #    'max_num_gpus': 0,
            #    'max_num_cpus': 4,
            #    'max_training_epochs': 3
            #},
            #backend='default',
            **kwargs):
        r"""
        Fit networks on dataset

        Parameters
        ----------
        data: Input data. It could be:
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
        backend: support autogluon default backend
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

        logger.info('Start constructing search space')
        metadata = locals()
        cs, args = BaseTask._construct_search_space(metadata)
        logger.info('Finished.')

        logger.info('Start running trials')
        BaseTask._reset_checkpoint(metadata)
        final_model, final_metric, final_config = BaseTask._run_backend(cs, args, metadata)
        logger.info('Finished.')

        logger.info('Finished.')
        return Results(final_model, final_metric, final_config, time.time() - start_fit_time,
                       metadata)
