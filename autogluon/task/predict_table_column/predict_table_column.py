import logging
import numpy as np
import pandas as pd
from mxnet import gluon, nd

# TODO: Perhaps change this to import from autogluon module rather than entirely separate tabular module. Need to replace all imports with the proper autogluon module once tabular has been fully integrated as a submodule of autogluon 
from tabular.utils.loaders import load_pd
from tabular.ml.learner.default_learner import DefaultLearner as Learner
from tabular.feature_generators.auto_ml_feature_generator import AutoMLFeatureGenerator

from ..base import *


__all__ = ['PredictTableColumn']

logger = logging.getLogger(__name__)


class PredictTableColumn(BaseTask):
    
    @staticmethod
    def load_data(file_path):
        """ Missing values must be represented as NA fields.
        """
        df = load_pd.load(file_path)
        return df
    
    @staticmethod
    def load(file_path):
        return Learner.load(file_path)
    
    @staticmethod
    def fit(train_data, label, val_data=None, savedir='', problem_type=None, objective_func=None, 
            submission_columns=[], feature_generator=None, threshold=100, 
            hyperparameter_tune=False, feature_prune=False, searcher=None, scheduler=None):
        """
        train_data: DataFrame of training data containing both features and label as columns
        label (str): name of column that contains the target variable to predict
        val_data: DataFrame containing validation data in same format as training data
        savedir (str): path to directory where models should be saved
        problem_type (str): is this classification or regression problem (TODO: options). If = None, will be inferred based on target LABEL column in dataset.
        objective_func (function): metric by which performance will be evaluated on test data (for examples see: sklearn.metrics). If = None, automatically chosen based on problem_type
        submission_columns (list): banned subset of column names that model may not use as predictive features (eg. contains label). DataFrame of just these columns may be submitted in ML competition.
        feature_generator: which feature engineering protocol to follow. Default is AutoMLFeatureGenerator.
        threshold: TODO: describe
        stratified_split: whether to stratify training/validation data split based on labels (ignored when val_data is provided). TODO: not implemented at the moment! 
        model_list: List of models to try, will replace get_preset_models() in auto_trainer.py.  TODO: not implemented
        hyperparameter_tune (bool): whether to tune lightGBM hyperparameters
        feature_prune (bool): whether to perform feature selection
        searcher: autogluon.searcher object
        scheduler: autogluon.scheduler object
        """
        if len(set(train_data.columns)) < len(train_data.columns):
            raise ValueError("Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")
        if val_data and np.any(train_data.columns != val_data.columns):
            raise ValueError("Column names must match between training and validation data")
        if feature_generator is None:
            feature_generator = AutoMLFeatureGenerator()
        learner = Learner(path_context=savedir, label=label, problem_type=problem_type, objective_func=objective_func, 
                          submission_columns=submission_columns, feature_generator=feature_generator, threshold=threshold)
        learner.fit(X=train_data, X_test=val_data, searcher=searcher, scheduler=scheduler)
        return learner










