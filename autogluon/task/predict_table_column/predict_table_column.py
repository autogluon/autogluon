import logging, multiprocessing
import numpy as np
import pandas as pd
import mxnet as mx


# TODO: Perhaps change this to import from autogluon module rather than entirely separate tabular module. Need to replace all imports with the proper autogluon module once tabular has been fully integrated as a submodule of autogluon 
from tabular.ml.learner.default_learner import DefaultLearner as Learner
from tabular.ml.trainer.auto_trainer import AutoTrainer
from tabular.feature_generators.auto_ml_feature_generator import AutoMLFeatureGenerator

from .dataset import TabularDataset

from ...core import *
from ...searcher import *
from ...scheduler import *
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ..base import BaseTask
from ..base.base_task import schedulers


__all__ = ['PredictTableColumn']

logger = logging.getLogger(__name__)


class PredictTableColumn(BaseTask):
    """AutoGluon task for predicting a column of tabular dataset (classification & regression)
    
    """
    Dataset = TabularDataset
    
    @staticmethod
    def load(output_directory):
        """ output_directory (str): path to directory where models are stored """
        return Learner.load(output_directory)
    
    # TODO: need flag use_trees, use_nets to control whether NN / lightGBM are used at all.
    @staticmethod
    def fit(train_data, label, tuning_data=None, output_directory='', problem_type=None, objective_func=None, 
            submission_columns=[], feature_generator=None, trainer_type=AutoTrainer, threshold=10,
            hyperparameter_tune=True, feature_prune=False,
            nn_options = {'num_epochs': 300}, gbm_options = {'num_boost_round': 10000},
            time_limits=None, num_trials=None, dist_ip_addrs=[], visualizer='none',
            nthreads_per_trial=None, ngpus_per_trial=None,
            search_strategy='random', search_options={}):
        """
        train_data: Dataset object, which is highly similar to pandas DataFrame.
        label (str): name of column that contains the target variable to predict
        tuning_data: Another Dataset object containing validation data reserved for hyperparameter tuning (in same format as training data).
            Note: final model returned may be fit on this tuning_data as well as train_data! Do not provide your test data here.
        output_directory (str): Path to directory where models and intermediate outputs should be saved.
            Note: To call fit() twice and save all results of each fit, you must specify different locations for output_directory.
                  Otherwise files from first fit() will be overwritten by second fit().
        problem_type (str): Type of prediction problem, ie. is this a binary/multiclass classification or regression problem (options: 'binary', 'multiclass', 'regression').
            If problem_type = None, the prediction problem type will be automatically inferred based on target LABEL column in dataset.
        objective_func (function): Metric by which performance will be evaluated on test data (for examples see: sklearn.metrics).
            If = None, objective_func is automatically chosen based on problem_type.
        submission_columns (list): banned subset of column names that model may not use as predictive features (eg. contains label).
            DataFrame of just these columns may be submitted in a ML competition.
        feature_generator: which feature engineering protocol to follow. Default is AutoMLFeatureGenerator.
        trainer_type: which trainer class to use
        threshold: TODO: describe
        hyperparameter_tune (bool): whether to tune hyperparameters or just use fixed hyperparameter values for each model
        feature_prune (bool): whether to perform feature selection
        nn_options (dict): Keyword arguments specifying hyperparameters of neural network models.
            Each hyperparameter can be fixed value or search space. For full list of options, see: TODO.
            Hyperparameters not specified will be set to default values (or default search spaces if hyperparameter_tune = True).
            If nn_options = None, then fit() will not train any neural network models.
        gbm_options (dict): Keyword arguments specifying hyperparameters of gradient boosting models.
            Each hyperparameter can be fixed value or search space. For full list of options, see: TODO.
            Hyperparameters not specified will be set to default values (or default search spaces if hyperparameter_tune = True).
            If gbm_options = None, then fit() will not train any gradient boosting models.
        search_strategy (str): which hyperparameter search algorithm to use
        search_options (dict): auxiliary keyword arguments for the searcher that performs hyperparameter optimization
        time_limits (int): Approximately how long this call to fit() should run for (wallclock time in seconds).
        num_trials (int): Maximal number of different hyperparameter settings of each model type to evaluate.
            If both time_limits and num_trials are specified, time_limits takes precedent.
        dist_ip_addrs: List of IP addresses corresponding to remote workers.
        visualizer (str): Method to visualize training progress during fit().
        nthreads_per_trial (int): how many CPUs to use in each trial (ie. training run of a single model)
        ngpus_per_trial (int): how many GPUs to use in each trial (ie. training run on a single model).
        """
        if len(set(train_data.columns)) < len(train_data.columns):
            raise ValueError("Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")
        if tuning_data is not None and np.any(train_data.columns != tuning_data.columns):
            raise ValueError("Column names must match between training and tuning data")

        # Create feature generator, schedulers, searchers for each model:
        if feature_generator is None:
            feature_generator = AutoMLFeatureGenerator()
        if nthreads_per_trial is None:
            nthreads_per_trial = multiprocessing.cpu_count()  # Use all of processing power / trial by default. To use just half: # int(np.floor(multiprocessing.cpu_count()/2))
        if ngpus_per_trial is None:
            if mx.test_utils.list_gpus():
                ngpus_per_trial = 1  # Single GPU / trial
            else:
                ngpus_per_trial = 0
        if ngpus_per_trial > 1:
            ngpus_per_trial = 1
            print("predict_table_column currently does not use more than 1 GPU per training run. ngpus_per_trial has been set = 1")
        # Adjust default time limits / trials:
        if num_trials is None:
            if time_limits is None:
                time_limits = 10 * 60  # run for 10min by default
            if time_limits <= 20:  # threshold = 20 sec, ie. too little time to run >1 trial.
                num_trials = 1
            else:
                num_trials = 1000000  # run as many trials as you can within the given time_limits
        elif time_limits is None:
            time_limits = 100000000  # user only specified num_trials, so run all of them regardless of time-limits
        time_limits *= 0.9  # reduce slightly to account for extra time overhead
        if (nn_options is not None) and (gbm_options is not None):
            time_limits /= 2  # each model type gets half the available time

        # All models use same scheduler (TODO: grant each model their own scheduler to run simultaneously):
        scheduler_options = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'num_trials': num_trials,
            'time_out': time_limits,
            'visualizer': visualizer,
            'time_attr': 'epoch',  # For lightGBM, one boosting round = one epoch
            'reward_attr': 'validation_performance',
            'dist_ip_addrs': dist_ip_addrs,
            'searcher': search_strategy,
            'search_options': search_options,
        }
        if isinstance(search_strategy, str):
            scheduler = schedulers[search_strategy.lower()]
        else:
            assert callable(search_strategy)
            scheduler = search_strategy
            scheduler_options['searcher'] = 'random'
        scheduler_options = (scheduler, scheduler_options)  # wrap into tuple
        predictor = Learner(path_context=output_directory, label=label, problem_type=problem_type, objective_func=objective_func, 
                          submission_columns=submission_columns, feature_generator=feature_generator, trainer_type=trainer_type, threshold=threshold)
        predictor.fit(X=train_data, X_test=tuning_data, scheduler_options=scheduler_options, 
                      hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, 
                      nn_options=nn_options, gbm_options=gbm_options)
        return predictor

