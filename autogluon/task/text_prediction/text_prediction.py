import pandas as pd
import logging
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ... import core
from ...contrib.nlp.utils.registry import Registry
from ..base import BaseTask
from ...utils.tabular.utils.loaders import load_pd
from .dataset import random_split_train_val, TabularDataset
from .estimators.basic_v1 import BertForTextPredictionBasic

__all__ = ['TextPrediction']

logger = logging.getLogger()  # return root logger

ag_text_params = Registry('ag_text_params')


@ag_text_params.register()
def default():
    """Default configuration"""
    ret = {
        'BertForTextPredictionBasic': {
            'model.backbone.name': 'google_uncased_mobilebert',
            'optimization.num_train_epochs': core.space.Choice([3, 10]),
            'optimization.lr': core.space.Real(1E-5, 1E-4)
        }
    }
    return ret


class TextPrediction(BaseTask):
    Dataset = pd.DataFrame

    @staticmethod
    def fit(train_data,
            label=None,
            tuning_data=None,
            time_limits=5 * 60 * 60,
            output_directory='./ag_text',
            feature_columns=None,
            holdout_frac=0.15,
            eval_metric=None,
            stopping_metric=None,
            nthreads_per_trial=None,
            ngpus_per_trial=None,
            search_strategy='random',
            search_options=None,
            hyperparameters=None):
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
            The time limits.
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

        Returns
        -------
        estimator
            An estimator object
        """
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
            all_columns = feature_columns + [label]
        train_data = TabularDataset(train_data,
                                    columns=all_columns,
                                    label_columns=label)
        column_properties = train_data.column_properties
        if tuning_data is None:
            train_data, tuning_data = random_split_train_val(train_data.table,
                                                             valid_ratio=holdout_frac)
            train_data = TabularDataset(train_data,
                                        columns=all_columns,
                                        column_properties=column_properties)
        else:
            tuning_data = load_pd.load(tuning_data)
        tuning_data = TabularDataset(tuning_data,
                                     columns=all_columns,
                                     column_properties=column_properties)
        if nthreads_per_trial is None:
            nthreads_per_trial = min(get_cpu_count(), 4)
        if ngpus_per_trial is None:
            ngpus_per_trial = min(get_gpu_count(), 2)
        if hyperparameters is None:
            hyperparameters = 'default'
        if isinstance(hyperparameters, str):
            hyperparameters = ag_text_params.create(hyperparameters)
        model_candidates = []
        for model_name, model_search_space in hyperparameters.items():
            if model_name == 'BertForTextPredictionBasic':
                estimator = BertForTextPredictionBasic(model_search_space)
                model_candidates.append(estimator)
            else:
                raise NotImplementedError
        args_decorator = core.args(hyperparameters)
        cfg = BertForTextPredictionBasic.get_cfg()
        cfg.defrost()
        if exp_dir is not None:
            cfg.misc.exp_dir = exp_dir
        if log_metrics is not None:
            cfg.learning.log_metrics = log_metrics
        if stop_metric is not None:
            cfg.learning.stop_metric = stop_metric
        cfg.freeze()
        estimator = BertForTextPredictionBasic(cfg)
        estimator.fit(train_data=train_data, valid_data=valid_data,
                      feature_columns=feature_columns,
                      label=label)
        return estimator

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
        BertForTextPredictionBasic.load(dir_path)
