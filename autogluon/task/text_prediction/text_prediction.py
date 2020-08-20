import logging
import copy
import warnings
from packaging import version

import numpy as np
import pandas as pd
import mxnet
# Ensure the MXNet version
if version.parse(mxnet.__version__) < version.parse('1.7.0')\
        or version.parse(mxnet.__version__) >= version.parse('2.0.0'):
    raise ImportError('You will need to ensure that you have mxnet>=1.7.0, <2.0.0.')
from mxnet.util import use_np
from autogluon_contrib_nlp.utils.registry import Registry
from autogluon_contrib_nlp.utils.misc import logging_config

from . import constants as _C
from .dataset import random_split_train_val, TabularDataset, infer_problem_type
from .models.basic_v1 import BertForTextPredictionBasic
from .. import tabular_prediction
from ..base import BaseTask
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ...core import space
from ...utils import in_ipynb
from ...utils.tabular.utils.loaders import load_pd
from ...utils.tabular.ml.utils import default_holdout_frac
from ...utils.miscs import verbosity2loglevel

__all__ = ['TextPrediction', 'ag_text_prediction_params']

logger = logging.getLogger()  # return root logger

ag_text_prediction_params = Registry('ag_text_prediction_params')


@ag_text_prediction_params.register()
def default() -> dict:
    """The default hyper-parameters

    It will have a version key and a list of candidate models.
    Each model has its own search space inside.
    """
    ret = {
        'version': 1,
        'models': {
            'BertForTextPredictionBasic': {
                'search_space': {
                    'model.backbone.name': 'google_electra_small',
                    'optimization.batch_size': 32,
                    'optimization.num_train_epochs': 4,
                    'optimization.lr': space.Real(1E-5, 1E-4)
                }
            },
        },
        'hpo_params': {
            'scheduler': 'fifo',           # Can be 'fifo', 'hyperband'
            'search_strategy': 'random',   # Can be 'random', 'skopt', or 'bayesopt'
            'search_options': None,        # The search option
            'time_limits': None,           # The total time limit
            'num_trials': 4,               # The number of trials
            'reduction_factor': 4,         # The reduction factor
            'grace_period': 10,            # The grace period
            'max_t': 50,                   # The max_t in the hyperband
            'time_attr': 'report_idx'      # The time attribute used in hyperband searcher.
                                           # We report the validation accuracy 10 times each epoch.
        }
    }
    return ret


def merge_params(base_params, partial_params=None):
    """Merge a partial change to the base configuration.

    Parameters
    ----------
    base_params
        The base parameters
    partial_params
        The partial parameters

    Returns
    -------
    final_params
        The final parameters
    """
    if partial_params is None:
        return base_params
    else:
        if not isinstance(partial_params, dict):
            return partial_params
        final_params = copy.deepcopy(base_params)
        for key in partial_params:
            if key in base_params:
                final_params[key] = merge_params(base_params[key], partial_params[key])
            else:
                final_params[key] = partial_params[key]
        return final_params


def get_recommended_resource(nthreads_per_trial=None,
                             ngpus_per_trial=None):
    """Get the recommended resource.

    Parameters
    ----------
    nthreads_per_trial
        The number of threads per trial
    ngpus_per_trial
        The number of GPUs per trial

    Returns
    -------
    resource
        The resource
    """
    if nthreads_per_trial is None and ngpus_per_trial is None:
        nthreads_per_trial = get_cpu_count()
        ngpus_per_trial = get_gpu_count()
    elif nthreads_per_trial is not None and ngpus_per_trial is None:
        ngpus_per_trial = get_gpu_count()
    elif nthreads_per_trial is None and ngpus_per_trial is not None:
        if ngpus_per_trial != 0:
            num_parallel_jobs = get_gpu_count() // ngpus_per_trial
            nthreads_per_trial = max(get_cpu_count() // num_parallel_jobs, 1)
        else:
            nthreads_per_trial = min(get_cpu_count(), 4)
    nthreads_per_trial = min(nthreads_per_trial, get_cpu_count())
    ngpus_per_trial = min(ngpus_per_trial, get_gpu_count())
    assert nthreads_per_trial > 0 and ngpus_per_trial >= 0,\
        'Invalid number of threads and number of GPUs.'
    return {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial}


def infer_eval_stop_log_metrics(problem_type,
                                label_shape,
                                eval_metric=None,
                                stopping_metric=None):
    """Infer the evaluate, stopping and logging metrics

    Parameters
    ----------
    problem_type
        Type of the problem
    label_shape
        Shape of the label
    eval_metric
        The eval metric provided by the user
    stopping_metric
        The stopping metric provided by the user

    Returns
    -------
    eval_metric
        The updated evaluation metric
    stopping_metric
        The updated stopping metric
    log_metrics
        The updated logging metric
    """
    if eval_metric is not None and stopping_metric is None:
        stopping_metric = eval_metric
        if isinstance(eval_metric, list):
            stopping_metric = eval_metric[0]
    if problem_type == _C.CLASSIFICATION:
        if stopping_metric is None:
            stopping_metric = 'acc'
        if eval_metric is None:
            eval_metric = 'acc'
        if label_shape == 2:
            log_metrics = ['f1', 'mcc', 'auc', 'acc', 'nll']
        else:
            log_metrics = ['acc', 'nll']
    elif problem_type == _C.REGRESSION:
        if stopping_metric is None:
            stopping_metric = 'mse'
        if eval_metric is None:
            eval_metric = 'mse'
        log_metrics = ['mse', 'rmse', 'mae']
    else:
        raise NotImplementedError('The problem type is not supported yet!')
    for other_log_metric in [stopping_metric, eval_metric]:
        if isinstance(other_log_metric, str) and other_log_metric not in log_metrics:
            log_metrics.append(other_log_metric)
        else:
            if isinstance(other_log_metric, list):
                for ele in other_log_metric:
                    if ele not in log_metrics:
                        log_metrics.append(ele)
    return eval_metric, stopping_metric, log_metrics


@use_np
class TextPrediction(BaseTask):
    """AutoGluon Task for predicting labels based on text data."""
    Dataset = tabular_prediction.TabularDataset

    @classmethod
    def fit(cls, train_data,
            label,
            tuning_data=None,
            time_limits=None,
            output_directory='./ag_text',
            feature_columns=None,
            holdout_frac=None,
            eval_metric=None,
            stopping_metric=None,
            nthreads_per_trial=None,
            ngpus_per_trial=None,
            dist_ip_addrs=None,
            scheduler=None,
            num_trials=None,
            search_strategy=None,
            search_options=None,
            hyperparameters=None,
            plot_results=None,
            seed=None,
            verbosity=2):
        """

        Parameters
        ----------
        train_data
            Training dataset
        label
            Name of the label column. It can be a stringBy default, we will search for a column named
        tuning_data
            The tuning dataset. We will tune the model
        time_limits
            The time limits. By default, there won't be any time limit and we will try to
            find the best model.
        output_directory
            The output directory
        feature_columns
            The feature columns
        holdout_frac
            Ratio of the training data that will be held out as the tuning data.
            By default, we will choose the appropriate holdout_frac based on the number of
            training samples.
        eval_metric
            The evaluation metric, i.e., how you will finally evaluate the model.
        stopping_metric
            The intrinsic metric used for early stopping.
            By default, we will select the best metric that
        nthreads_per_trial
            The number of threads per trial. By default, we will use all available CPUs.
        ngpus_per_trial
            The number of GPUs to use for the fit job. By default, we decide the usage
            based on the total number of GPUs available.
        dist_ip_addrs
            The distributed IP address
        scheduler
            The scheduler of HPO
        num_trials
            The number of trials in the HPO search
        search_strategy
            The search strategy
        search_options
            The search options
        hyperparameters
            The hyper-parameters of the search-space.
        plot_results
            Whether to plot the fitting results
        seed
            The seed of the random state
        verbosity
            Verbosity levels range from 0 to 4 and control how much information is printed
            during fit().
            Higher levels correspond to more detailed print statements
            (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed
            via `logger.setLevel(L)`,
            where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print
            statements, opposite of verbosity levels)

        Returns
        -------
        model
            A model object
        """
        assert dist_ip_addrs is None, 'Training on remote machine is currently not supported.'
        if verbosity < 0:
            verbosity = 0
        elif verbosity > 4:
            verbosity = 4
        console_log = verbosity >= 2
        logging_config(folder=output_directory, name='ag_text_prediction',
                       logger=logger, level=verbosity2loglevel(verbosity),
                       console=console_log)
        # Parse the hyper-parameters
        if hyperparameters is None:
            hyperparameters = ag_text_prediction_params.create('default')
        elif isinstance(hyperparameters, str):
            hyperparameters = ag_text_prediction_params.create(hyperparameters)
        else:
            base_params = ag_text_prediction_params.create('default')
            hyperparameters = merge_params(base_params, hyperparameters)
        np.random.seed(seed)
        if not isinstance(train_data, pd.DataFrame):
            train_data = load_pd.load(train_data)
        # Inference the label
        if not isinstance(label, list):
            label = [label]
        label_columns = []
        for ele in label:
            if isinstance(ele, int):
                label_columns.append(train_data.columns[ele])
            else:
                label_columns.append(ele)
        if feature_columns is None:
            all_columns = list(train_data.columns)
            feature_columns = [ele for ele in all_columns if ele not in label_columns]
        else:
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]
            for col in feature_columns:
                assert col not in label_columns, 'Feature columns and label columns cannot overlap.'
            all_columns = feature_columns + label_columns
            all_columns = [ele for ele in train_data.columns if ele in all_columns]
        if tuning_data is None:
            if holdout_frac is None:
                holdout_frac = default_holdout_frac(len(train_data), True)
            train_data, tuning_data = random_split_train_val(train_data,
                                                             valid_ratio=holdout_frac)
        else:
            if not isinstance(tuning_data, pd.DataFrame):
                tuning_data = load_pd.load(tuning_data)
        train_data = TabularDataset(train_data,
                                    columns=all_columns,
                                    label_columns=label_columns)
        tuning_data = TabularDataset(tuning_data, column_properties=train_data.column_properties)

        logger.info('Train Dataset:')
        logger.info(train_data)
        logger.info('Tuning Dataset:')
        logger.info(tuning_data)
        logger.debug('Hyperparameters:')
        logger.debug(hyperparameters)
        column_properties = train_data.column_properties

        problem_types = []
        label_shapes = []
        for label_col_name in label_columns:
            problem_type, label_shape = infer_problem_type(column_properties=column_properties,
                                                           label_col_name=label_col_name)
            problem_types.append(problem_type)
            label_shapes.append(label_shape)
        logging.info('Label columns={}, Problem types={}, Label shapes={}'.format(
            label_columns, problem_types, label_shapes))
        eval_metric, stopping_metric, log_metrics =\
            infer_eval_stop_log_metrics(problem_types[0],
                                        label_shapes[0],
                                        eval_metric=eval_metric,
                                        stopping_metric=stopping_metric)
        logging.info('Eval Metric={}, Stop Metric={}, Log Metrics={}'.format(eval_metric,
                                                                             stopping_metric,
                                                                             log_metrics))
        model_candidates = []
        for model_type, kwargs in hyperparameters['models'].items():
            search_space = kwargs['search_space']
            if model_type == 'BertForTextPredictionBasic':
                model = BertForTextPredictionBasic(column_properties=column_properties,
                                                   label_columns=label_columns,
                                                   feature_columns=feature_columns,
                                                   label_shapes=label_shapes,
                                                   problem_types=problem_types,
                                                   stopping_metric=stopping_metric,
                                                   log_metrics=log_metrics,
                                                   base_config=None,
                                                   search_space=search_space,
                                                   output_directory=output_directory,
                                                   logger=logger)
                model_candidates.append(model)
            else:
                raise ValueError('model_type = "{}" is not supported. You can try to use '
                                 'model_type = "BertForTextPredictionBasic"'.format(model_type))
        assert len(model_candidates) == 1, 'Only one model is supported currently'
        recommended_resource = get_recommended_resource(nthreads_per_trial=nthreads_per_trial,
                                                        ngpus_per_trial=ngpus_per_trial)
        if scheduler is None:
            scheduler = hyperparameters['hpo_params']['scheduler']
        if search_strategy is None:
            search_strategy = hyperparameters['hpo_params']['search_strategy']
        if time_limits is None:
            time_limits = hyperparameters['hpo_params']['time_limits']
        else:
            if isinstance(time_limits, str):
                if time_limits.endswith('min'):
                    time_limits = int(float(time_limits[:-3]) * 60)
                elif time_limits.endswith('hour'):
                    time_limits = int(float(time_limits[:-4]) * 60 * 60)
                else:
                    raise ValueError('The given time_limits="{}" cannot be parsed!'
                                     .format(time_limits))
        if num_trials is None:
            num_trials = hyperparameters['hpo_params']['num_trials']

        # Setting the HPO-specific parameters.
        reduction_factor = hyperparameters['hpo_params']['reduction_factor']
        grace_period = hyperparameters['hpo_params']['grace_period']
        max_t = hyperparameters['hpo_params']['max_t']
        if recommended_resource['num_gpus'] == 0:
            warnings.warn('Recommend to use GPU to run the TextPrediction task!')
        model = model_candidates[0]
        if plot_results is None:
            if in_ipynb():
                plot_results = True
            else:
                plot_results = False
        model.train(train_data=train_data,
                    tuning_data=tuning_data,
                    resource=recommended_resource,
                    time_limits=time_limits,
                    scheduler=scheduler,
                    searcher=search_strategy,
                    num_trials=num_trials,
                    reduction_factor=reduction_factor,
                    grace_period=grace_period,
                    max_t=max_t,
                    plot_results=plot_results,
                    console_log=verbosity > 2,
                    ignore_warning=verbosity <= 2)
        return model

    @staticmethod
    def load(dir_path):
        """Load model from the directory

        Parameters
        ----------
        dir_path


        Returns
        -------
        model
            The loaded model
        """
        return BertForTextPredictionBasic.load(dir_path)
