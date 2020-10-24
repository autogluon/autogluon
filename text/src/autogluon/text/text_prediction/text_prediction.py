import logging
import copy
import warnings
from packaging import version

import numpy as np
import pandas as pd
import mxnet
from mxnet.util import use_np
from autogluon_contrib_nlp.utils.registry import Registry
from autogluon_contrib_nlp.utils.misc import logging_config

from . import constants as _C
from .dataset import random_split_train_val, TabularDataset, infer_problem_type,\
    get_column_properties
from .models.basic_v1 import BertForTextPredictionBasic
from autogluon.tabular.task import tabular_prediction
from autogluon.core.task.base import BaseTask
from autogluon.core.scheduler import get_cpu_count, get_gpu_count
from autogluon.core import space
from autogluon.core.utils import in_ipynb
from autogluon.core.utils.loaders import load_pd
from autogluon.core.utils.utils import default_holdout_frac
from autogluon.core.utils.miscs import verbosity2loglevel

__all__ = ['TextPrediction', 'ag_text_prediction_params']

logger = logging.getLogger()  # return root logger

ag_text_prediction_params = Registry('ag_text_prediction_params')


@ag_text_prediction_params.register()
def default() -> dict:
    """The default hyperparameters.

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
            'search_strategy': 'random',   # Can be 'random', 'bayesopt', 'skopt',
                                           # 'hyperband', 'bayesopt_hyperband'
            'search_options': None,        # Extra kwargs passed to searcher
            'scheduler_options': None,     # Extra kwargs passed to scheduler
            'time_limits': None,           # The total time limit
            'num_trials': 4,               # The number of trials
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
    elif base_params is None:
        return partial_params
    else:
        if not isinstance(partial_params, dict):
            return partial_params
        assert isinstance(base_params, dict)
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
    """Decide default evaluation, stopping, and logging metrics (based on type of prediction problem).

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
    """AutoGluon Task for classification/regression with text data."""
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
            num_trials=None,
            search_strategy=None,
            search_options=None,
            scheduler_options=None,
            hyperparameters=None,
            plot_results=None,
            seed=None,
            verbosity=2):
        """Fit models to make predictions based on text inputs.

        Parameters
        ----------
        train_data : :class:`autogluon.task.tabular_prediction.TabularDataset` or `pandas.DataFrame`
            Training dataset where rows = individual training examples, columns = features.
        label : str
            Name of the label column. It can be a stringBy default, we will search for a column named
        tuning_data : :class:`autogluon.task.tabular_prediction.TabularDataset` or `pandas.DataFrame`, default = None
            Another dataset containing validation data reserved for hyperparameter tuning (in same format as training data).
            If `tuning_data = None`, `fit()` will automatically hold out random examples from `train_data` for validation.
        time_limits : int or str, default = None
            Approximately how long `fit()` should run for (wallclock time in seconds if int).
            String values may instead be used to specify time in different units such as: '1min' or '1hour'.
            Longer `time_limits` will usually improve predictive accuracy.
            If not specified, `fit()` will run until all models to try by default have completed training.
        output_directory : str, default = './ag_text'
            Path to directory where models and intermediate outputs should be saved.
        feature_columns : List[str], default = None
            Which columns of table to consider as predictive features (other columns will be ignored, except for label-column).
            If None (by default), all columns of table are considered predictive features.
        holdout_frac : float, default = None
            Fraction of train_data to holdout as tuning data for optimizing hyperparameters (ignored unless `tuning_data = None`).
            If None, default value is selected based on the number of training examples.
        eval_metric : str, default = None
            The evaluation metric that will be used to evaluate the model's predictive performance.
            If None, an appropriate default metric will be selected (accuracy for classification, mean-squared-error for regression).
            Options for classification include: 'acc' (accuracy), 'nll' (negative log-likelihood).
            Additional options for binary classification include: 'f1' (F1 score), 'mcc' (Matthews coefficient), 'auc' (area under ROC curve).
            Options for regression include: 'mse' (mean squared error), 'rmse' (root mean squared error), 'mae' (mean absolute error).
        stopping_metric, default = None
            Metric which iteratively-trained models use to early stop to avoid overfitting.
            Defaults to `eval_metric` value (if None).
            Options are identical to options for `eval_metric`.
        nthreads_per_trial, default = None
            The number of threads per individual model training run. By default, all available CPUs are used.
        ngpus_per_trial, default = None
            The number of GPUs to use per individual model training run. If unspecified, a default value is chosen based on total number of GPUs available.
        dist_ip_addrs, default = None
            List of IP addresses corresponding to remote workers, in order to leverage distributed computation.
        num_trials : , default = None
            The number of trials in the HPO search
        search_strategy : str, default = None
            Which hyperparameter search algorithm to use. Options include:
            'random' (random search), 'bayesopt' (Gaussian process Bayesian optimization),
            'skopt' (SKopt Bayesian optimization), 'grid' (grid search),
            'hyperband' (Hyperband scheduling with random search), 'bayesopt-hyperband'
            (Hyperband scheduling with GP-BO search).
            If unspecified, the default is 'random'.
        search_options : dict, default = None
            Options passed to searcher.
        scheduler_options : dict, default = None
            Additional kwargs passed to scheduler __init__.
        hyperparameters : dict, default = None
            Determines the hyperparameters used by the models. Each hyperparameter may be either fixed value or search space of many values.
            For example of default hyperparameters, see: `autogluon.task.text_prediction.text_prediction.default()`
        plot_results : bool, default = None
            Whether or not to plot intermediate training results during `fit()`.
        seed : int, default = None
            Seed value for random state used inside `fit()`. 
        verbosity : int, default = 2
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
            A `BertForTextPredictionBasic` object that can be used for making predictions on new data.
        """
        assert dist_ip_addrs is None, 'Training on remote machine is currently not supported.'
        # Version check of MXNet
        if version.parse(mxnet.__version__) < version.parse('1.7.0') \
                or version.parse(mxnet.__version__) >= version.parse('2.0.0'):
            raise ImportError('You will need to ensure that you have mxnet>=1.7.0, <2.0.0. '
                              'For more information about how to install mxnet, you can refer to '
                              'https://sxjscience.github.io/KDD2020/ .')

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
                assert col in train_data.columns,\
                    'Feature columns must be in the pandas dataframe! Received col = "{}", ' \
                    'all columns = "{}"'.format(col, train_data.columns)
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
        train_data = train_data[all_columns]
        tuning_data = tuning_data[all_columns]
        column_properties = get_column_properties(
            pd.concat([train_data, tuning_data]),
            metadata=None,
            label_columns=label_columns,
            provided_column_properties=None,
            categorical_default_handle_missing_value=True)
        train_data = TabularDataset(train_data,
                                    column_properties=column_properties,
                                    label_columns=label_columns)
        tuning_data = TabularDataset(tuning_data,
                                     column_properties=train_data.column_properties,
                                     label_columns=label_columns)

        logger.info('Train Dataset:')
        logger.info(train_data)
        logger.info('Tuning Dataset:')
        logger.info(tuning_data)
        logger.debug('Hyperparameters:')
        logger.debug(hyperparameters)
        has_text_column = False
        for k, v in column_properties.items():
            if v.type == _C.TEXT:
                has_text_column = True
                break
        if not has_text_column:
            raise NotImplementedError('No Text Column is found! This is currently not supported by '
                                      'the TextPrediction task. You may try to use '
                                      'TabularPrediction.fit().\n' \
                                      'The inferred column properties of the training data is {}'
                                      .format(train_data))
        problem_types = []
        label_shapes = []
        for label_col_name in label_columns:
            problem_type, label_shape = infer_problem_type(column_properties=column_properties,
                                                           label_col_name=label_col_name)
            problem_types.append(problem_type)
            label_shapes.append(label_shape)
        logging.info('Label columns={}, Feature columns={}, Problem types={}, Label shapes={}'
            .format(label_columns, feature_columns,
                    problem_types, label_shapes))
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
        if scheduler_options is None:
            scheduler_options = hyperparameters['hpo_params']['scheduler_options']
            if scheduler_options is None:
                scheduler_options = dict()
        if search_strategy.endswith('hyperband'):
            # Specific defaults for hyperband scheduling
            scheduler_options['reduction_factor'] = scheduler_options.get(
                'reduction_factor', 4)
            scheduler_options['grace_period'] = scheduler_options.get(
                'grace_period', 10)
            scheduler_options['max_t'] = scheduler_options.get(
                'max_t', 50)

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
                    search_strategy=search_strategy,
                    search_options=search_options,
                    scheduler_options=scheduler_options,
                    num_trials=num_trials,
                    plot_results=plot_results,
                    console_log=verbosity > 2,
                    ignore_warning=verbosity <= 2)
        return model

    @staticmethod
    def load(dir_path):
        """Load a model object previously produced by `fit()` from disk and return this object.
           It is highly recommended the model be loaded with the exact AutoGluon version it was previously fit with.

        Parameters
        ----------
        dir_path : str
            Path to directory where this model was previously saved (i.e. `output_directory` specified in previous call to `fit`).


        Returns
        -------
        model
            A `BertForTextPredictionBasic` object that can be used for making predictions on new data.
        """
        return BertForTextPredictionBasic.load(dir_path)
