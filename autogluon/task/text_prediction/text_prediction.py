import pandas as pd
import logging
import numpy as np
from . import constants as _C
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ...core import space
from ...contrib.nlp.utils.registry import Registry
from ..base import BaseTask
from ...utils.tabular.utils.loaders import load_pd
from .dataset import random_split_train_val, TabularDataset, infer_problem_type
from .models.basic_v1 import BertForTextPredictionBasic

__all__ = ['TextPrediction']

logger = logging.getLogger()  # return root logger

ag_text_params = Registry('ag_text_params')


@ag_text_params.register()
def default():
    """The default hyper-parameters

    It will have a version key and a list of candidate models.
    We will store the name and search space of each model.
    """
    ret = {
        'version': 1,
        'models':
            [
                {
                    'name': 'BertForTextPredictionBasic',
                    'search_space': {
                        'model.backbone.name': 'google_electra_base',
                        'optimization.batch_size': space.Categorical(32, 64),
                        'optimization.num_train_epochs': space.Categorical(3, 10),
                        'optimization.lr': space.Real(1E-5, 1E-4)
                    }
                }
            ]
    }
    return ret


def infer_eval_stop_log_metrics(problem_type, label_shape,
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
    stopping_metric
    log_metrics
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
        raise NotImplementedError
    for other_log_metric in [stopping_metric, eval_metric]:
        if isinstance(other_log_metric, str) and other_log_metric not in log_metrics:
            log_metrics.append(other_log_metric)
        else:
            if isinstance(other_log_metric, list):
                for ele in other_log_metric:
                    if ele not in log_metrics:
                        log_metrics.append(ele)
    return eval_metric, stopping_metric, log_metrics


class TextPrediction(BaseTask):
    Dataset = pd.DataFrame

    @staticmethod
    def fit(train_data,
            label=None,
            tuning_data=None,
            time_limits=None,
            output_directory='./ag_text',
            feature_columns=None,
            holdout_frac=0.15,
            eval_metric=None,
            stopping_metric=None,
            nthreads_per_trial=None,
            ngpus_per_trial=None,
            search_strategy='random',
            search_options=None,
            hyperparameters=None,
            seed=None):
        """

        Parameters
        ----------
        train_data
            Training dataset
        label
            Name of the label column. By default, we will search for a column named "
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
            Ratio of the training data that will be held out as the tuning data / or dev data.
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
        search_strategy
            The search strategy
        search_options
            The options for running the hyper-parameter search
        hyperparameters
            The hyper-parameters of the fit function.
            Including the configuration of the search space.
            There are two options:
            1) You are given a predefined search space
        seed
            The seed of the random state

        Returns
        -------
        model
            A model object
        """
        np.random.seed(seed)
        train_data = load_pd.load(train_data)
        if label is None:
            # Perform basic label inference
            if 'label' in train_data.columns:
                label = 'label'
            elif 'score' in train_data.columns:
                label = 'score'
            else:
                label = train_data.columns[-1]
        if not isinstance(label, list):
            label = [label]
        if feature_columns is None:
            all_columns = train_data.columns
            feature_columns = [ele for ele in all_columns if ele is not label]
        else:
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]
            all_columns = feature_columns + label
            all_columns = [ele for ele in train_data.columns if ele in all_columns]
        if tuning_data is None:
            train_data, tuning_data = random_split_train_val(train_data,
                                                             valid_ratio=holdout_frac)
        else:
            tuning_data = load_pd.load(tuning_data)
        if nthreads_per_trial is None:
            nthreads_per_trial = get_cpu_count()
        else:
            nthreads_per_trial = min(get_cpu_count(), nthreads_per_trial)
        if ngpus_per_trial is None:
            ngpus_per_trial = get_gpu_count()
        else:
            ngpus_per_trial = min(get_gpu_count(), ngpus_per_trial)
        if hyperparameters is None:
            hyperparameters = 'default'
        if isinstance(hyperparameters, str):
            hyperparameters = ag_text_params.create(hyperparameters)
        train_data = TabularDataset(train_data, columns=all_columns, label_columns=label)
        tuning_data = TabularDataset(tuning_data, column_properties=train_data.column_properties)
        logger.info('Train Dataset:')
        logger.info(train_data)
        logger.info('Tuning Dataset:')
        logger.info(tuning_data)
        column_properties = train_data.column_properties

        problem_types = []
        label_shapes = []
        for label_col_name in label:
            problem_type, label_shape = infer_problem_type(column_properties=column_properties,
                                                           label_col_name=label_col_name)
            problem_types.append(problem_type)
            label_shapes.append(label_shape)
        logging.info('Label={}, Problem={}, Label shape={}'.format(label,
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
        for model_name, model_search_space in hyperparameters.items():
            if model_name == 'BertForTextPredictionBasic':
                model = BertForTextPredictionBasic(column_properties=column_properties,
                                                   label_columns=label,
                                                   feature_columns=feature_columns,
                                                   label_shapes=label_shapes,
                                                   problem_types=problem_types,
                                                   eval_metric=eval_metric,
                                                   stopping_metric=stopping_metric,
                                                   log_metrics=log_metrics,
                                                   base_config=None,
                                                   search_space=model_search_space,
                                                   output_directory=output_directory,
                                                   logger=logger)
                model_candidates.append(model)
            else:
                raise NotImplementedError
        assert len(model_candidates) == 1, 'Only one model is supported currently'
        resources = {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial}
        model = model_candidates[0].train(train_data=train_data,
                                          tuning_data=tuning_data,
                                          label_columns=label,
                                          feature_columns=feature_columns,
                                          resources=resources,
                                          time_limits=time_limits)
        return model

    @staticmethod
    def load(dir_path):
        """

        Parameters
        ----------
        dir_path

        Returns
        -------
        model
            The loaded model
        """
        return BertForTextPredictionBasic.load(dir_path)
