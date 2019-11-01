import logging, multiprocessing
import numpy as np
import pandas as pd
import mxnet as mx

from .dataset import TabularDataset


# TODO: Perhaps change this to import from autogluon module rather than entirely separate tabular module. Need to replace all imports with the proper autogluon module once tabular has been fully integrated as a submodule of autogluon 
from tabular.ml.learner.default_learner import DefaultLearner as Learner
from tabular.feature_generators.auto_ml_feature_generator import AutoMLFeatureGenerator

from ..base import *
from ..base import schedulers # dick of possible schedulers, maps string -> scheduler function
from ...scheduler import *


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
            submission_columns=[], feature_generator=None, threshold=100,
            hyperparameter_tune=True, feature_prune=False, 
            nn_options = {}, 
            num_cpus=None, num_gpus=None, time_limits=None, num_trials=2, dist_ip_addrs=[], visualizer='none',
            search_strategy='random', search_options={}):
        """
        train_data: Dataset object, which is similar to pandas DataFrame.
        label (str): name of column that contains the target variable to predict
        tuning_data: Another Dataset object containing validation data reserved for hyperparameter tuning (in same format as training data). Note: final model returned will be fit on this data as well as train_data!
        output_directory (str): path to directory where models should be saved
        problem_type (str): is this classification or regression problem (TODO: options). If = None, will be inferred based on target LABEL column in dataset.
        objective_func (function): metric by which performance will be evaluated on test data (for examples see: sklearn.metrics). If = None, automatically chosen based on problem_type
        submission_columns (list): banned subset of column names that model may not use as predictive features (eg. contains label). DataFrame of just these columns may be submitted in ML competition.
        feature_generator: which feature engineering protocol to follow. Default is AutoMLFeatureGenerator.
        threshold: TODO: describe
        stratified_split: whether to stratify training/validation data split based on labels (ignored when val_data is provided). TODO: not implemented at the moment! 
        model_list: List of models to try, will replace get_preset_models() in auto_trainer.py.  TODO: not implemented
        hyperparameter_tune (bool): whether to tune hyperparameters or just use default values for each model
        feature_prune (bool): whether to perform feature selection
        search_strategy (str): which hyperparameter search algorithm to use
        search_options (dict): kwargs for searcher used for hyperparameter optimization
        """
        if len(set(train_data.columns)) < len(train_data.columns):
            raise ValueError("Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")
        if tuning_data is not None and np.any(train_data.columns != tuning_data.columns):
            raise ValueError("Column names must match between training and tuning data")
        # Create feature generator, schedulers, searchers for each model:
        if feature_generator is None:
            feature_generator = AutoMLFeatureGenerator()
        if num_cpus is None:
            num_cpus = int(np.floor(multiprocessing.cpu_count()/2)) # At most half of processing power / trial
        if num_gpus is None:
            if mx.test_utils.list_gpus():
                num_gpus = 1 # single GPU / trial
            else:
                num_gpus = 0
        
        # All models use same scheduler (TODO: grant each model their own scheduler to run simultaneously):
        scheduler_options = {
            'resource': {'num_cpus': num_cpus, 'num_gpus': num_gpus},
            'num_trials': num_trials,
            'time_out': time_limits,
            'visualizer': visualizer,
            'time_attr': 'epoch', # For lightGBM, one boosting round = one epoch
            'reward_attr': 'validation_performance',
            'dist_ip_addrs': dist_ip_addrs,
            'searcher': search_strategy,
            'search_options': search_options,
        }
        if isinstance(self.search_strategy, str):
            scheduler = schedulers[search_strategy.lower()]
        else:
            assert callable(self.search_strategy)
            scheduler = self.search_strategy
            scheduler_options['searcher'] = 'random'
        scheduler_options = (scheduler, scheduler_options) # wrap into tuple
        
        predictor = Learner(path_context=savedir, label=label, problem_type=problem_type, objective_func=objective_func, 
                          submission_columns=submission_columns, feature_generator=feature_generator, threshold=threshold)
        predictor.fit(X=train_data, X_test=tuning_data, scheduler_options=scheduler_options, 
                      hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, 
                      nn_options=nn_options)
        return predictor

