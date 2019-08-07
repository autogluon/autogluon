import os
import shutil
import logging
import argparse
import time
import ConfigSpace as CS

from abc import ABC

import autogluon as ag
from ...optim import Optimizers, get_optim
from ... import dataset

__all__ = ['BaseTask']

logger = logging.getLogger(__name__)


class Results(object):
    def __init__(self, model, metric, config, time, metadata):
        self._model = model
        self._metric = metric
        self._config = config
        self._time = time
        self._metadata = metadata

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        self._model = val

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, val):
        self._metric = val

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val):
        self._config = val

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, val):
        self._time = val

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, val):
        self._metadata = val


class BaseTask(ABC):
    class Dataset(dataset.Dataset):
        def __init__(self, name=None, train_path=None, val_path=None, batch_size=None,
                     num_workers=None,
                     transform_train_fn=None, transform_val_fn=None,
                     transform_train_list=None, transform_val_list=None,
                     batchify_train_fn=None, batchify_val_fn=None, **kwargs):
            super(BaseTask.Dataset, self).__init__(name, train_path, val_path, batch_size,
                                                   num_workers,
                                                   transform_train_fn, transform_val_fn,
                                                   transform_train_list, transform_val_list,
                                                   batchify_train_fn, batchify_val_fn, **kwargs)

        def _read_dataset(self, **kwargs):
            pass

        def _add_search_space(self):
            pass

        def _get_search_space_strs(self):
            pass

    def __init__(self):
        super(BaseTask, self).__init__()

    @staticmethod
    def _set_range(obj, cs):
        if obj.search_space is not None:
            cs.add_configuration_space(prefix='',
                                       delimiter='',
                                       configuration_space=obj.search_space)

    @staticmethod
    def _assert_fit_error(obj, name):
        assert obj is not None, '%s cannot be None' % name

    @staticmethod
    def _init_args(cs, metadata):
        args = argparse.Namespace()
        vars(args).update({'data': metadata['data']})
        vars(args).update({'backend': metadata['backend']})
        vars(args).update({'epochs': metadata['resources_per_trial']['max_training_epochs']})
        vars(args).update({'num_gpus': metadata['resources_per_trial']['max_num_gpus']})
        for k, v in metadata['kwargs'].items():
            vars(args).update({k: v})
        for hparam in cs.get_hyperparameters():
            vars(args).update({hparam.name: hparam.default_value})
        return args

    @staticmethod
    def _reset_checkpoint(metadata):
        dir = os.path.splitext(metadata['savedir'])[0]
        if not metadata['resume'] and os.path.exists(dir):
            shutil.rmtree(dir)
            os.makedirs(dir)

    @staticmethod
    def _construct_search_space(metadata):
        cs = CS.ConfigurationSpace()
        for name, obj in metadata.items():
            if hasattr(obj, 'search_space'):
                BaseTask._set_range(obj, cs)
            elif name == 'kwargs':
                for k, v in obj.items():
                    if hasattr(v, 'search_space'):
                        BaseTask._set_range(v, cs)
            elif obj is None and name == 'data':
                BaseTask._assert_fit_error(obj, name)
        args = BaseTask._init_args(cs, metadata)
        return cs, args

    @staticmethod
    def _run_backend(cs, args, metadata):
        if metadata['searcher'] is None or metadata['searcher'] == 'random':
            searcher = ag.searcher.RandomSampling(cs)
        elif metadata['searcher'] == 'bayesopt':
            searcher = ag.searcher.SKoptSearcher(cs)
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
            nets=None,
            optimizers=Optimizers([
                get_optim('sgd'),
                get_optim('adam')]),
            metrics=None,
            losses=None,
            searcher=None,
            trial_scheduler=None,
            resume=False,
            savedir='checkpoint/exp1.ag',
            visualizer='tensorboard',
            stop_criterion={
                'time_limits': 1 * 60 * 60,
                'max_metric': 1.0,
                'max_trial_count': 2
            },
            resources_per_trial={
                'max_num_gpus': 0,
                'max_num_cpus': 4,
                'max_training_epochs': 3
            },
            backend='default',
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
        searcher (str): Search Algorithm to employ, should be one of: 
            'random' (random search), 'bayesopt' (Bayesian optimization using skopt).
            If = None, defaults to 'random'.
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
