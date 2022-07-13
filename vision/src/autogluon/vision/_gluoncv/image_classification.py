"""Auto pipeline for image classification task"""
# pylint: disable=bad-whitespace,missing-class-docstring,bare-except
import time
import os
import math
import copy
import logging
import pprint
import json
import pickle
from typing import Union, Tuple
import uuid
import shutil

import numpy as np
import pandas as pd
import autogluon.core as ag
from autocfg import dataclass
from autogluon.core.constants import MULTICLASS
from autogluon.core.utils import get_cpu_count, get_gpu_count_all
from autogluon.core.task.base import BaseTask
from autogluon.core.searcher import DummySearcher

from gluoncv.auto.estimators.base_estimator import BaseEstimator
from gluoncv.auto.estimators import ImageClassificationEstimator, TorchImageClassificationEstimator
from gluoncv.auto.data.dataset import ImageClassificationDataset
from gluoncv.auto.estimators.conf import _BEST_CHECKPOINT_FILE

from .utils import config_to_nested


__all__ = ['ImageClassification']


logger = logging.getLogger(__name__)


try:
    import timm
except ImportError:
    timm = None
try:
    import torch
except ImportError:
    torch = None
try:
    import mxnet as mx
    from gluoncv.model_zoo import get_model_list
except ImportError:
    mx = None

@dataclass
class LiteConfig:
    model : Union[type(None), str, ag.Space] = ag.Categorical('resnet18', 'mobilenetv3_small_100', 'visformer_tiny')
    lr : Union[ag.Space, float] = 1e-2
    num_trials : int = 1
    epochs : Union[ag.Space, int] = 5
    batch_size : Union[ag.Space, int] = 8
    nthreads_per_trial : int = 32
    ngpus_per_trial : int = 0
    time_limits : int = 7 * 24 * 60 * 60  # 7 days
    search_strategy : str = 'random'
    dist_ip_addrs : Union[type(None), list, Tuple] = None

@dataclass
class DefaultConfig:
    model : Union[type(None), str, ag.Space] = ag.Categorical('resnet50', 'efficientnet_b0', 'visformer_small')
    lr : Union[ag.Space, float] = ag.Categorical(1e-2, 5e-2)
    num_trials : int = 3
    epochs : Union[ag.Space, int] = 15
    batch_size : Union[ag.Space, int] = 16
    nthreads_per_trial : int = 128
    ngpus_per_trial : int = 8
    time_limits : int = 7 * 24 * 60 * 60  # 7 days
    search_strategy : str = 'random'
    dist_ip_addrs : Union[type(None), list, Tuple] = None


def _train_image_classification(args,
                                train_data,
                                val_data,
                                problem_type,
                                wall_clock_tick,
                                log_dir,
                                reporter=None):
    """
    Parameters
    ----------
    args: <class 'autogluon.utils.edict.EasyDict'>
    """
    tic = time.time()
    args = args.copy()
    try:
        task_id = int(args['task_id'])
    except:
        task_id = 0
    final_fit = args.pop('final_fit', False)
    # exponential batch size for Int() space batch sizes
    exp_batch_size = args.pop('exp_batch_size', False)
    if exp_batch_size and 'batch_size' in args:
        args['batch_size'] = 2 ** args['batch_size']

    # mxnet and torch dispatcher
    dispatcher = None
    torch_model_list = None
    mxnet_model_list = None
    custom_net = None
    if args.get('custom_net', None):
        custom_net = args.get('custom_net')
        if torch and timm:
            if isinstance(custom_net, torch.nn.Module):
                dispatcher = 'torch'
        if mx:
            if isinstance(custom_net, mx.gluon.Block):
                dispatcher = 'mxnet'
    else:
        if torch and timm:
            torch_model_list = timm.list_models()
        if mx:
            mxnet_model_list = list(get_model_list())
        model = args.get('model', None)
        if model:
            # timm model has higher priority
            if torch_model_list and model in torch_model_list:
                dispatcher = 'torch'
            elif mxnet_model_list and model in mxnet_model_list:
                dispatcher = 'mxnet'
            else:
                if not torch_model_list:
                    raise ValueError('Model not found in gluoncv model zoo. Install torch and timm if it supports the model.')
                elif not mxnet_model_list:
                    raise ValueError('Model not found in timm model zoo. Install mxnet if it supports the model.')
                else:
                    raise ValueError('Model not supported because it does not exist in both timm and gluoncv model zoo.')
    assert dispatcher in ('torch', 'mxnet'), 'custom net needs to be of type either torch.nn.Module or mx.gluon.Block'
    if dispatcher == 'mxnet':
        logger.log(30, '=============================================================================\n'
                       'WARNING: Using MXNet models in ImagePredictor is deprecated as of v0.4.0 and may contain various bugs and issues!\n'
                       'In v0.6.0, ImagePredictor will no longer support training MXNet models. Please consider switching to specifying Torch models instead.\n'
                       'Users should ensure they update their code that depends on ImagePredictor when upgrading to future AutoGluon releases.\n'
                       'For more information, refer to this GitHub issue: https://github.com/awslabs/autogluon/issues/1560\n'
                       '=============================================================================\n')

    args['estimator'] = TorchImageClassificationEstimator if dispatcher=='torch' else ImageClassificationEstimator
    # convert user defined config to nested form
    args = config_to_nested(args)

    if wall_clock_tick < tic and not final_fit:
        return {'traceback': 'timeout', 'args': str(args),
                'time': 0, 'train_acc': -1, 'valid_acc': -1}

    try:
        valid_summary_file = 'fit_summary_img_cls.ag'
        estimator_cls = args.pop('estimator', None)
        assert estimator_cls in (ImageClassificationEstimator, TorchImageClassificationEstimator)
        if final_fit:
            # load from previous dumps
            estimator = None
            if os.path.isdir(log_dir):
                is_valid_dir_fn = lambda d : d.startswith('.trial_') and os.path.isdir(os.path.join(log_dir, d))
                trial_dirs = [d for d in os.listdir(log_dir) if is_valid_dir_fn(d)]
                best_checkpoint = ''
                best_acc = -1
                result = {}
                for dd in trial_dirs:
                    try:
                        with open(os.path.join(log_dir, dd, valid_summary_file), 'r') as f:
                            result = json.load(f)
                            acc = result.get('valid_acc', -1)
                            if acc > best_acc and os.path.isfile(os.path.join(log_dir, dd, _BEST_CHECKPOINT_FILE)):
                                best_checkpoint = os.path.join(log_dir, dd, _BEST_CHECKPOINT_FILE)
                                best_acc = acc
                    except:
                        pass
                if best_checkpoint:
                    estimator = estimator_cls.load(best_checkpoint)
            if estimator is None:
                if wall_clock_tick < tic:
                    result.update({'traceback': 'timeout'})
                else:
                    # unknown error yet, try reproduce it
                    final_fit = False
        if not final_fit:
            # create independent log_dir for each trial
            trial_log_dir = os.path.join(log_dir, '.trial_{}'.format(task_id))
            args['log_dir'] = trial_log_dir
            custom_optimizer = args.pop('custom_optimizer', None)
            estimator = estimator_cls(args, problem_type=problem_type, reporter=reporter,
                                      net=custom_net, optimizer=custom_optimizer)
            # training
            result = estimator.fit(train_data=train_data, val_data=val_data, time_limit=wall_clock_tick-tic)
            with open(os.path.join(trial_log_dir, valid_summary_file), 'w') as f:
                json.dump(result, f)
    except:
        import traceback
        return {'traceback': traceback.format_exc(), 'args': str(args),
                'time': time.time() - tic, 'train_acc': -1, 'valid_acc': -1}

    if estimator:
        result.update({'model_checkpoint': estimator})
        result.update({'estimator': estimator_cls})
    return result


class ImageClassification(BaseTask):
    """Whole Image Classification general task.
    Parameters
    ----------
    config : dict
        The configurations, can be nested dict.
    logger : logging.Logger
        The desired logger object, use `None` for module specific logger with default setting.
    net : mx.gluon.Block
        The custom network. If defined, the model name in config will be ignored so your
        custom network will be used for training rather than pulling it from model zoo.
    """
    Dataset = ImageClassificationDataset

    def __init__(self, config=None, logger=None, problem_type=None):
        super(ImageClassification, self).__init__()
        if problem_type is None:
            problem_type = MULTICLASS
        self._problem_type = problem_type
        self._fit_summary = {}
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._fit_summary = {}
        self._results = {}


        # cpu and gpu setting
        cpu_count = get_cpu_count()
        gpu_count = get_gpu_count_all()

        # default settings
        if not config:
            if gpu_count < 1:
                self._logger.info('No GPU detected/allowed, using most conservative search space.')
                config = LiteConfig()
            else:
                config = DefaultConfig()
            config = config.asdict()
        else:
            if not config.get('dist_ip_addrs', None):
                ngpus_per_trial = config.get('ngpus_per_trial', gpu_count)
                ngpus_per_trial = min(ngpus_per_trial, gpu_count)
                if ngpus_per_trial < 1:
                    self._logger.info('No GPU detected/allowed, using most conservative search space.')
                    default_config = LiteConfig()
                else:
                    default_config = DefaultConfig()
                config = default_config.merge(config, allow_new_key=True).asdict()

        # adjust cpu/gpu resources
        if not config.get('dist_ip_addrs', None):
            nthreads_per_trial = config.get('nthreads_per_trial', cpu_count)
            nthreads_per_trial = min(nthreads_per_trial, cpu_count)
            ngpus_per_trial = config.get('ngpus_per_trial', gpu_count)
            if ngpus_per_trial > gpu_count:
                ngpus_per_trial = gpu_count
                self._logger.warning(
                    "The number of requested GPUs is greater than the number of available GPUs."
                    "Reduce the number to %d", ngpus_per_trial)
        else:
            raise ValueError('Please specify `nthreads_per_trial` and `ngpus_per_trial` '
                             'given that dist workers are available')


        # additional configs
        config['num_workers'] = nthreads_per_trial
        config['gpus'] = [int(i) for i in range(ngpus_per_trial)]
        config['seed'] = config.get('seed', np.random.randint(32,767))
        config['final_fit'] = False
        self._cleanup_disk = config.get('cleanup_disk', True)
        self._config = config

        # scheduler options
        self.scheduler = config.pop('scheduler', 'local')
        self.search_strategy = config.pop('search_strategy', 'random')
        self.search_options = config.pop('search_options', {})
        self.scheduler_options = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'checkpoint': config.get('checkpoint', 'checkpoint/exp1.ag'),
            'num_trials': config.get('num_trials', 2),
            'time_out': config.get('time_limits', 60 * 60),
            'resume': (len(config.get('resume', '')) > 0),
            'visualizer': config.get('visualizer', 'none'),
            'time_attr': 'epoch',
            'reward_attr': 'acc_reward',
            'dist_ip_addrs': config.get('dist_ip_addrs', None),
            'searcher': self.search_strategy,
            'search_options': self.search_options,
            'max_reward': config.get('max_reward', None)}

    def fit(self, train_data, val_data=None, train_size=0.9, random_state=None, time_limit=None):
        """Fit auto estimator given the input data.
        Parameters
        ----------
        train_data : pd.DataFrame or iterator
            Training data.
        val_data : pd.DataFrame or iterator, optional
            Validation data, optional. If `train_data` is DataFrame, `val_data` will be split from
            `train_data` given `train_size`.
        train_size : float
            The portion of train data split from original `train_data` if `val_data` is not provided.
        random_state : int
            Random state for splitting, for `np.random.seed`.
        time_limit : int, default is None
            The wall clock time limit(second) for fit process, if `None`, time limit is not enforced.
            If `fit` takes longer than `time_limit`, the process will terminate early and return the
            model prematurally.
            Due to callbacks and additional validation functions, the `time_limit` may not be very precise
            (few minutes allowance), but you can use it to safe-guard a very long training session.
            If `time_limits` key set in __init__ with config, the `time_limit` value will overwrite configuration
            if not `None`.
        Returns
        -------
        Estimator
            The estimator obtained by training on the specified dataset.
        """
        config = self._config.copy()
        if time_limit is None:
            if config.get('time_limits', None):
                time_limit = config['time_limits']
            else:
                time_limit = math.inf
        elif not isinstance(time_limit, int):
            raise TypeError(f'Invalid type `time_limit={time_limit}`, int or None expected')
        self.scheduler_options['time_out'] = time_limit
        wall_clock_tick = time.time() + time_limit
        # split train/val before HPO to make fair comparisons
        if not isinstance(train_data, pd.DataFrame):
            assert val_data is not None, \
                "Please provide `val_data` as we do not know how to split `train_data` of type: \
                {}".format(type(train_data))

        if val_data is None:
            assert 0 <= train_size <= 1.0
            if random_state:
                np.random.seed(random_state)
            split_mask = np.random.rand(len(train_data)) < train_size
            train = train_data[split_mask]
            val = train_data[~split_mask]
            self._logger.info('Randomly split train_data into train[%d]/validation[%d] splits.',
                              len(train), len(val))
            train_data, val_data = train, val

        estimator = config.get('estimator', None)
        if estimator:
            if isinstance(estimator, ag.Space):
                estimator = estimator.data
            elif isinstance(estimator, str):
                estimator = [estimator]
            for i, e in enumerate(estimator):
                if e != 'img_cls':
                    estimator.pop(e)
            if not estimator:
                raise ValueError('Unable to determine the estimator for fit function.')

        # register args
        log_dir = os.path.join(config.pop('log_dir', os.getcwd()), str(uuid.uuid4())[:8])
        self.scheduler_options['checkpoint'] = os.path.join(log_dir, 'exp1.ag')

        train_fn_kwargs = dict(
            train_data=train_data,
            val_data=val_data,
            wall_clock_tick=wall_clock_tick,
            log_dir=log_dir,
            problem_type=self._problem_type,
        )

        start_time = time.time()
        self._fit_summary = {}
        self._results = {}

        num_trials = config.pop('num_trials', 1)
        hpo_enabled = num_trials > 1
        if not hpo_enabled:
            default_config = DummySearcher(search_space=config).get_config()
            self._logger.info("Starting fit without HPO")
            cur_config = {**config}
            cur_config.update(default_config)
            results = _train_image_classification({**cur_config}, **train_fn_kwargs)
            best_config = cur_config
            self._fit_summary.update({'train_acc': results.get('train_acc', -1),
                                      'valid_acc': results.get('valid_acc', -1),
                                      'total_time': results.get('time', time.time() - start_time),
                                      'best_config': best_config})
            self._results = self._fit_summary
        else:
            self._logger.info("Starting HPO experiments")
            results = self.run_fit(_train_image_classification, config, self.scheduler,
                                   self.scheduler_options, plot_results=False,
                                   train_fn_kwargs=train_fn_kwargs
                                   )
            if isinstance(results, dict):
                ks = ('best_reward', 'best_config', 'total_time', 'config_history', 'reward_attr')
                self._results.update({k: v for k, v in results.items() if k in ks})
        end_time = time.time()
        self._logger.info("Finished, total runtime is %.2f s", end_time - start_time)
        if hpo_enabled:
            best_config = {**config}
            best_config.update(results['best_config'])
            best_config.update({'estimator': results['estimator']})
            # convert best config to nested form
            best_config = config_to_nested(best_config)
            best_config.pop('train_data', None)
            best_config.pop('val_data', None)
            self._fit_summary.update({'train_acc': results.get('train_acc', -1),
                                      'valid_acc': results.get('valid_acc', results.get('best_reward', -1)),
                                      'total_time': results.get('total_time', time.time() - start_time),
                                      'best_config': best_config})
        self._logger.info(pprint.pformat(self._fit_summary, indent=2))

        if self._cleanup_disk:
            shutil.rmtree(log_dir, ignore_errors=True)
        model_checkpoint = results.get('model_checkpoint', None)
        if model_checkpoint is None:
            if results.get('traceback', '') == 'timeout':
                raise TimeoutError(f'Unable to fit a usable model given `time_limit={time_limit}`')
            raise RuntimeError(f'Unexpected error happened during fit: {pprint.pformat(results, indent=2)}')
        if isinstance(results['model_checkpoint'], bytes):
            estimator = pickle.loads(results['model_checkpoint'])
        else:
            estimator = results['model_checkpoint']
        return estimator

    def fit_summary(self):
        return copy.copy(self._fit_summary)

    def fit_history(self):
        return copy.copy(self._results)

    @classmethod
    def load(cls, filename):
        obj = BaseEstimator.load(filename)
        # make sure not accidentally loading e.g. classification model
        assert isinstance(obj, ImageClassificationEstimator)
        return obj
