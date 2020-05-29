import gc
import logging
import os
import random
import re
import time
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from . import lgb_utils
from .callbacks import early_stopping_custom
from .hyperparameters.lgb_trial import lgb_trial
from .hyperparameters.parameters import get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace
from .lgb_utils import construct_dataset
from ..abstract.abstract_model import AbstractModel, fixedvals_from_searchspaces
from ...constants import BINARY, MULTICLASS, REGRESSION
from ....utils.savers import save_pkl
from .....try_import import try_import_lightgbm
from ......core import Int, Space

warnings.filterwarnings("ignore", category=UserWarning, message="Starting from version")  # lightGBM brew libomp warning
logger = logging.getLogger(__name__)


# TODO: Save dataset to binary and reload for HPO. This will avoid the memory spike overhead when training each model and instead it will only occur once upon saving the dataset.
class LGBModel(AbstractModel):
    def __init__(self, path: str, name: str, problem_type: str, objective_func, stopping_metric=None, num_classes=None, hyperparameters=None, features=None, debug=0, **kwargs):
        super().__init__(path=path, name=name, problem_type=problem_type, objective_func=objective_func, stopping_metric=stopping_metric, num_classes=num_classes, hyperparameters=hyperparameters, features=features, debug=debug, **kwargs)

        self.eval_metric_name = self.stopping_metric.name
        self.is_higher_better = True
        self._internal_feature_map = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type, num_classes=self.num_classes)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return get_default_searchspace(problem_type=self.problem_type, num_classes=self.num_classes)

    def get_eval_metric(self):
        return lgb_utils.func_generator(metric=self.stopping_metric, is_higher_better=True, needs_pred_proba=not self.stopping_metric_needs_y_pred, problem_type=self.problem_type)

    def fit(self, X_train=None, Y_train=None, X_test=None, Y_test=None, dataset_train=None, dataset_val=None, time_limit=None, **kwargs):
        start_time = time.time()
        params = self.params.copy()

        # TODO: kwargs can have num_cpu, num_gpu. Currently these are ignored.
        verbosity = kwargs.get('verbosity', 2)
        params = fixedvals_from_searchspaces(params)

        if verbosity <= 1:
            verbose_eval = False
        elif verbosity == 2:
            verbose_eval = 1000
        elif verbosity == 3:
            verbose_eval = 50
        else:
            verbose_eval = 1

        eval_metric = self.get_eval_metric()
        dataset_train, dataset_val = self.generate_datasets(X_train=X_train, Y_train=Y_train, params=params, X_test=X_test, Y_test=Y_test, dataset_train=dataset_train, dataset_val=dataset_val)
        gc.collect()

        num_boost_round = params.pop('num_boost_round', 1000)
        logger.log(15, f'Training Gradient Boosting Model for {num_boost_round} rounds...')
        logger.log(15, "with the following hyperparameter settings:")
        logger.log(15, params)

        num_rows_train = len(dataset_train.data)
        if 'min_data_in_leaf' in params:
            if params['min_data_in_leaf'] > num_rows_train:  # TODO: may not be necessary
                params['min_data_in_leaf'] = max(1, int(num_rows_train / 5.0))

        # TODO: Better solution: Track trend to early stop when score is far worse than best score, or score is trending worse over time
        if (dataset_val is not None) and (dataset_train is not None):
            modifier = 1 if num_rows_train <= 10000 else 10000 / num_rows_train
            early_stopping_rounds = max(round(modifier * 150), 10)
        else:
            early_stopping_rounds = 150

        callbacks = []
        valid_names = ['train_set']
        valid_sets = [dataset_train]
        if dataset_val is not None:
            reporter = kwargs.get('reporter', None)
            train_loss_name = self._get_train_loss_name() if reporter is not None else None
            callbacks += [
                # Note: Don't use self.params_aux['max_memory_usage_ratio'] here as LightGBM handles memory per iteration optimally.  # TODO: Consider using when ratio < 1.
                early_stopping_custom(early_stopping_rounds, metrics_to_use=[('valid_set', self.eval_metric_name)], max_diff=None, start_time=start_time, time_limit=time_limit,
                                      ignore_dart_warning=True, verbose=False, manual_stop_file=False, reporter=reporter, train_loss_name=train_loss_name),
            ]
            valid_names = ['valid_set'] + valid_names
            valid_sets = [dataset_val] + valid_sets

        seed_val = params.pop('seed_value', 0)
        train_params = {
            'params': params,
            'train_set': dataset_train,
            'num_boost_round': num_boost_round,
            'valid_sets': valid_sets,
            'valid_names': valid_names,
            'callbacks': callbacks,
            'verbose_eval': verbose_eval,
        }
        if not isinstance(eval_metric, str):
            train_params['feval'] = eval_metric
        if seed_val is not None:
            train_params['params']['seed'] = seed_val
            random.seed(seed_val)
            np.random.seed(seed_val)

        # Train LightGBM model:
        try_import_lightgbm()
        import lightgbm as lgb
        self.model = lgb.train(**train_params)
        self.params_trained['num_boost_round'] = self.model.best_iteration

    def predict_proba(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X)
        if self.problem_type == REGRESSION:
            return self.model.predict(X)

        y_pred_proba = self.model.predict(X)
        if self.problem_type == BINARY:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif self.problem_type == MULTICLASS:
            return y_pred_proba
        else:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 2:  # Should this ever happen?
                return y_pred_proba
            else:  # Should this ever happen?
                return y_pred_proba[:, 1]

    def preprocess(self, X, is_train=False):
        X = super().preprocess(X=X)

        if is_train:
            for column in X.columns:
                new_column = re.sub(r'[",:{}[\]]', '', column)
                if new_column != column:
                    self._internal_feature_map = {feature: i for i, feature in enumerate(list(X.columns))}
                    break

        if self._internal_feature_map:
            new_columns = [self._internal_feature_map[column] for column in list(X.columns)]
            X_new = X.copy(deep=False)
            X_new.columns = new_columns
            return X_new
        else:
            return X

    def generate_datasets(self, X_train: DataFrame, Y_train: Series, params, X_test=None, Y_test=None, dataset_train=None, dataset_val=None, save=False):
        lgb_dataset_params_keys = ['objective', 'two_round', 'num_threads', 'num_classes', 'verbose']  # Keys that are specific to lightGBM Dataset object construction.
        data_params = {key: params[key] for key in lgb_dataset_params_keys if key in params}.copy()

        W_train = None  # TODO: Add weight support
        W_test = None  # TODO: Add weight support
        if X_train is not None:
            X_train = self.preprocess(X_train, is_train=True)
        if X_test is not None:
            X_test = self.preprocess(X_test)
        # TODO: Try creating multiple Datasets for subsets of features, then combining with Dataset.add_features_from(), this might avoid memory spike
        if not dataset_train:
            # X_train, W_train = self.convert_to_weight(X=X_train)
            dataset_train = construct_dataset(x=X_train, y=Y_train, location=f'{self.path}datasets{os.path.sep}train', params=data_params, save=save, weight=W_train)
            # dataset_train = construct_dataset_lowest_memory(X=X_train, y=Y_train, location=self.path + 'datasets/train', params=data_params)
        if (not dataset_val) and (X_test is not None) and (Y_test is not None):
            # X_test, W_test = self.convert_to_weight(X=X_test)
            dataset_val = construct_dataset(x=X_test, y=Y_test, location=f'{self.path}datasets{os.path.sep}val', reference=dataset_train, params=data_params, save=save, weight=W_test)
            # dataset_val = construct_dataset_lowest_memory(X=X_test, y=Y_test, location=self.path + 'datasets/val', reference=dataset_train, params=data_params)
        return dataset_train, dataset_val

    def debug_features_to_use(self, X_test_in):
        feature_splits = self.model.feature_importance()
        total_splits = feature_splits.sum()
        feature_names = list(X_test_in.columns.values)
        feature_count = len(feature_names)
        feature_importances = pd.DataFrame(data=feature_names, columns=['feature'])
        feature_importances['splits'] = feature_splits
        feature_importances_unused = feature_importances[feature_importances['splits'] == 0]
        feature_importances_used = feature_importances[feature_importances['splits'] >= (total_splits / feature_count)]
        logger.debug(feature_importances_unused)
        logger.debug(feature_importances_used)
        logger.debug(f'feature_importances_unused: {len(feature_importances_unused)}')
        logger.debug(f'feature_importances_used: {len(feature_importances_used)}')
        features_to_use = list(feature_importances_used['feature'].values)
        logger.debug(str(features_to_use))
        return features_to_use

    # FIXME: Requires major refactor + refactor lgb_trial.py
    #  model names are not aligned with what is communicated to trainer!
    # FIXME: Likely tabular_nn_trial.py and abstract trial also need to be refactored heavily + hyperparameter functions
    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options, **kwargs):
        time_start = time.time()
        logger.log(15, "Beginning hyperparameter tuning for Gradient Boosting Model...")
        self._set_default_searchspace()
        params_copy = self.params.copy()
        if isinstance(params_copy['min_data_in_leaf'], Int):
            upper_minleaf = params_copy['min_data_in_leaf'].upper
            if upper_minleaf > X_train.shape[0]:  # TODO: this min_data_in_leaf adjustment based on sample size may not be necessary
                upper_minleaf = max(1, int(X_train.shape[0] / 5.0))
                lower_minleaf = params_copy['min_data_in_leaf'].lower
                if lower_minleaf > upper_minleaf:
                    lower_minleaf = max(1, int(upper_minleaf / 3.0))
                params_copy['min_data_in_leaf'] = Int(lower=lower_minleaf, upper=upper_minleaf)

        directory = self.path  # also create model directory if it doesn't exist
        # TODO: This will break on S3! Use tabular/utils/savers for datasets, add new function
        os.makedirs(directory, exist_ok=True)
        scheduler_func, scheduler_options = scheduler_options  # Unpack tuple
        if scheduler_func is None or scheduler_options is None:
            raise ValueError("scheduler_func and scheduler_options cannot be None for hyperparameter tuning")
        num_threads = scheduler_options['resource'].get('num_cpus', -1)
        params_copy['num_threads'] = num_threads
        # num_gpus = scheduler_options['resource']['num_gpus'] # TODO: unused

        dataset_train, dataset_val = self.generate_datasets(X_train=X_train, Y_train=Y_train, params=params_copy, X_test=X_test, Y_test=Y_test)
        dataset_train_filename = "dataset_train.bin"
        train_file = self.path + dataset_train_filename
        if os.path.exists(train_file):  # clean up old files first
            os.remove(train_file)
        dataset_train.save_binary(train_file)
        dataset_val_filename = "dataset_val.bin"  # names without directory info
        val_file = self.path + dataset_val_filename
        if os.path.exists(val_file):  # clean up old files first
            os.remove(val_file)
        dataset_val.save_binary(val_file)
        dataset_val_pkl_filename = 'dataset_val.pkl'
        val_pkl_path = directory + dataset_val_pkl_filename
        save_pkl.save(path=val_pkl_path, object=(X_test, Y_test))

        if not np.any([isinstance(params_copy[hyperparam], Space) for hyperparam in params_copy]):
            logger.warning("Attempting to do hyperparameter optimization without any search space (all hyperparameters are already fixed values)")
        else:
            logger.log(15, "Hyperparameter search space for Gradient Boosting Model: ")
            for hyperparam in params_copy:
                if isinstance(params_copy[hyperparam], Space):
                    logger.log(15, f'{hyperparam}:   {params_copy[hyperparam]}')

        util_args = dict(
            dataset_train_filename=dataset_train_filename,
            dataset_val_filename=dataset_val_filename,
            dataset_val_pkl_filename=dataset_val_pkl_filename,
            directory=directory,
            model=self,
            time_start=time_start,
            time_limit=scheduler_options['time_out']
        )
        lgb_trial.register_args(util_args=util_args, **params_copy)
        scheduler = scheduler_func(lgb_trial, **scheduler_options)
        if ('dist_ip_addrs' in scheduler_options) and (len(scheduler_options['dist_ip_addrs']) > 0):
            # This is multi-machine setting, so need to copy dataset to workers:
            logger.log(15, "Uploading data to remote workers...")
            scheduler.upload_files([train_file, val_file, val_pkl_path])  # TODO: currently does not work.
            directory = self.path  # TODO: need to change to path to working directory used on every remote machine
            lgb_trial.update(directory=directory)
            logger.log(15, "uploaded")

        scheduler.run()
        scheduler.join_jobs()

        return self._get_hpo_results(scheduler=scheduler, scheduler_options=scheduler_options, time_start=time_start)

    # TODO: Consider adding _internal_feature_map functionality to abstract_model
    def compute_feature_importance(self, **kwargs):
        permutation_importance = super().compute_feature_importance(**kwargs)
        if self._internal_feature_map is not None:
            inverse_internal_feature_map = {i: feature for feature, i in self._internal_feature_map.items()}
            permutation_importance = {inverse_internal_feature_map[i]: importance for i, importance in permutation_importance.items()}
        return permutation_importance

    def _get_train_loss_name(self):
        if self.problem_type == BINARY:
            train_loss_name = 'binary_logloss'
        elif self.problem_type == MULTICLASS:
            train_loss_name = 'multi_logloss'
        elif self.problem_type == REGRESSION:
            train_loss_name = 'l2'
        else:
            raise ValueError(f"unknown problem_type for LGBModel: {self.problem_type}")
        return train_loss_name

    def get_model_feature_importance(self, use_original_feature_names=False):
        feature_names = self.model.feature_name()
        importances = self.model.feature_importance()
        importance_dict = {feature_name: importance for (feature_name, importance) in zip(feature_names, importances)}
        if use_original_feature_names and (self._internal_feature_map is not None):
            inverse_internal_feature_map = {i: feature for feature, i in self._internal_feature_map.items()}
            importance_dict = {inverse_internal_feature_map[i]: importance for i, importance in importance_dict.items()}
        return importance_dict
