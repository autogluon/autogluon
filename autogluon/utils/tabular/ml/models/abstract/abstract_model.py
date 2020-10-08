import copy
import gc
import logging
import math
import os
import pickle
import sys
import time
from typing import Union

import numpy as np
import pandas as pd
import psutil

from .model_trial import model_trial
from ...constants import AG_ARGS_FIT, BINARY, REGRESSION, REFIT_FULL_SUFFIX, OBJECTIVES_TO_NORMALIZE
from ...tuning.feature_pruner import FeaturePruner
from ...utils import get_pred_from_proba, generate_train_test_split, shuffle_df_rows, normalize_pred_probas, infer_eval_metric
from .... import metrics
from ....features.feature_metadata import FeatureMetadata
from ....utils.exceptions import TimeLimitExceeded, NoValidFeatures
from ....utils.loaders import load_pkl
from ....utils.savers import save_pkl, save_json
from ......core import Space, Categorical, List, NestedSpace
from ......scheduler.fifo import FIFOScheduler
from ......task.base import BasePredictor

logger = logging.getLogger(__name__)


class AbstractModel:
    model_file_name = 'model.pkl'
    model_info_name = 'info.pkl'
    model_info_json_name = 'info.json'

    def __init__(self, path: str, name: str, problem_type: str, eval_metric: Union[str, metrics.Scorer] = None, num_classes=None, stopping_metric=None, model=None, hyperparameters=None, features=None, feature_metadata: FeatureMetadata = None, debug=0, **kwargs):
        """ Creates a new model.
            Args:
                path (str): directory where to store all outputs.
                name (str): name of subdirectory inside path where model will be saved.
                problem_type (str): type of problem this model will handle. Valid options: ['binary', 'multiclass', 'regression'].
                eval_metric (str or autogluon.utils.tabular.metrics.Scorer): objective function the model intends to optimize. If None, will be inferred based on problem_type.
                hyperparameters (dict): various hyperparameters that will be used by model (can be search spaces instead of fixed values).
                feature_metadata (autogluon.utils.tabular.features.feature_metadata.FeatureMetadata): contains feature type information that can be used to identify special features such as text ngrams and datetime as well as which features are numerical vs categorical
        """
        self.name = name
        self.path_root = path
        self.path_suffix = self.name + os.path.sep  # TODO: Make into function to avoid having to reassign on load?
        self.path = self.create_contexts(self.path_root + self.path_suffix)  # TODO: Make this path a function for consistency.
        self.num_classes = num_classes
        self.model = model
        self.problem_type = problem_type
        if eval_metric is not None:
            self.eval_metric = metrics.get_metric(eval_metric, self.problem_type, 'eval_metric')  # Note: we require higher values = better performance
        else:
            self.eval_metric = infer_eval_metric(problem_type=self.problem_type)
            logger.log(20, f"Model {self.name}'s eval_metric inferred to be '{self.eval_metric.name}' because problem_type='{self.problem_type}' and eval_metric was not specified during init.")

        if stopping_metric is None:
            self.stopping_metric = self.eval_metric
        else:
            self.stopping_metric = stopping_metric

        if self.eval_metric.name in OBJECTIVES_TO_NORMALIZE:
            self.normalize_pred_probas = True
            logger.debug(self.name +" predicted probabilities will be transformed to never =0 since eval_metric=" + self.eval_metric.name)
        else:
            self.normalize_pred_probas = False

        if isinstance(self.eval_metric, metrics._ProbaScorer):
            self.metric_needs_y_pred = False
        elif isinstance(self.eval_metric, metrics._ThresholdScorer):
            self.metric_needs_y_pred = False
        else:
            self.metric_needs_y_pred = True

        if isinstance(self.stopping_metric, metrics._ProbaScorer):
            self.stopping_metric_needs_y_pred = False
        elif isinstance(self.stopping_metric, metrics._ThresholdScorer):
            self.stopping_metric_needs_y_pred = False
        else:
            self.stopping_metric_needs_y_pred = True

        self.feature_metadata = feature_metadata  # TODO: Should this be passed to a model on creation? Should it live in a Dataset object and passed during fit? Currently it is being updated prior to fit by trainer
        self.features = features
        self.debug = debug

        self.fit_time = None  # Time taken to fit in seconds (Training data)
        self.predict_time = None  # Time taken to predict in seconds (Validation data)
        self.val_score = None  # Score with eval_metric (Validation data)

        self.params = {}
        self.params_aux = {}

        self._set_default_auxiliary_params()
        if hyperparameters is not None:
            hyperparameters = hyperparameters.copy()
            if AG_ARGS_FIT in hyperparameters:
                ag_args_fit = hyperparameters.pop(AG_ARGS_FIT)
                self.params_aux.update(ag_args_fit)
        self._set_default_params()
        self.nondefault_params = []
        if hyperparameters is not None:
            self.params.update(hyperparameters)
            self.nondefault_params = list(hyperparameters.keys())[:]  # These are hyperparameters that user has specified.
        self.params_trained = dict()

    # Checks if model is capable of inference on new data (if normal model) or has produced out-of-fold predictions (if bagged model)
    def is_valid(self) -> bool:
        return self.is_fit()

    # Checks if model is capable of inference on new data
    def can_infer(self) -> bool:
        return self.is_valid()

    # Checks if a model has been fit
    def is_fit(self) -> bool:
        return self.model is not None

    def _set_default_params(self):
        pass

    def _set_default_auxiliary_params(self):
        # TODO: Consider adding to get_info() output
        default_auxiliary_params = dict(
            max_memory_usage_ratio=1.0,  # Ratio of memory usage allowed by the model. Values > 1.0 have an increased risk of causing OOM errors.
            # TODO: Add more params
            # max_memory_usage=None,
            # max_disk_usage=None,
            max_time_limit_ratio=1.0,  # ratio of given time_limit to use during fit(). If time_limit == 10 and max_time_limit_ratio=0.3, time_limit would be changed to 3.
            max_time_limit=None,  # max time_limit value during fit(). If the provided time_limit is greater than this value, it will be replaced by max_time_limit. Occurs after max_time_limit_ratio is applied.
            min_time_limit=0,  # min time_limit value during fit(). If the provided time_limit is less than this value, it will be replaced by min_time_limit. Occurs after max_time_limit is applied.
            # num_cpu=None,
            # num_gpu=None,
            # ignore_hpo=False,
            # max_early_stopping_rounds=None,
            # use_orig_features=True,  # TODO: Only for stackers
            # TODO: add option for only top-k ngrams
            ignored_type_group_special=[],  # List, drops any features in `self.feature_metadata.type_group_map_special[type]` for type in `ignored_type_group_special`. | Currently undocumented in task.
            ignored_type_group_raw=[],  # List, drops any features in `self.feature_metadata.type_group_map_raw[type]` for type in `ignored_type_group_raw`. | Currently undocumented in task.
        )
        for key, value in default_auxiliary_params.items():
            self._set_default_param_value(key, value, params=self.params_aux)

    def _set_default_param_value(self, param_name, param_value, params=None):
        if params is None:
            params = self.params
        if param_name not in params:
            params[param_name] = param_value

    def _get_default_searchspace(self) -> dict:
        """
        Get the default hyperparameter searchspace of the model.
        See `autogluon.core.space` for available space classes.

        Returns
        -------
        dict of hyperparameter search spaces.
        """
        return {}

    def _set_default_searchspace(self):
        """ Sets up default search space for HPO. Each hyperparameter which user did not specify is converted from
            default fixed value to default search space.
        """
        def_search_space = self._get_default_searchspace().copy()
        # Note: when subclassing AbstractModel, you must define or import get_default_searchspace() from the appropriate location.
        for key in self.nondefault_params:  # delete all user-specified hyperparams from the default search space
            def_search_space.pop(key, None)
        if self.params is not None:
            self.params.update(def_search_space)

    def set_contexts(self, path_context):
        self.path = self.create_contexts(path_context)
        self.path_suffix = self.name + os.path.sep
        # TODO: This should be added in future once naming conventions have been standardized for WeightedEnsembleModel
        # if self.path_suffix not in self.path:
        #     raise ValueError('Expected path_suffix not in given path! Values: (%s, %s)' % (self.path_suffix, self.path))
        self.path_root = self.path.rsplit(self.path_suffix, 1)[0]

    @staticmethod
    def create_contexts(path_context):
        path = path_context
        return path

    def rename(self, name: str):
        """Renames the model and updates self.path to reflect the updated name."""
        self.path = self.path[:-len(self.name) - 1] + name + os.path.sep
        self.name = name

    # Extensions of preprocess must act identical in bagged situations, otherwise test-time predictions will be incorrect
    # This means preprocess cannot be used for normalization
    # TODO: Add preprocess_stateful() to enable stateful preprocessing for models such as KNN
    def preprocess(self, X):
        if self.features is not None:
            # TODO: In online-inference this becomes expensive, add option to remove it (only safe in controlled environment where it is already known features are present
            if list(X.columns) != self.features:
                return X[self.features]
        else:
            self.features = list(X.columns)  # TODO: add fit and transform versions of preprocess instead of doing this
            ignored_type_group_raw = self.params_aux.get('ignored_type_group_raw', [])
            ignored_type_group_special = self.params_aux.get('ignored_type_group_special', [])
            valid_features = self.feature_metadata.get_features(invalid_raw_types=ignored_type_group_raw, invalid_special_types=ignored_type_group_special)
            self.features = [feature for feature in self.features if feature in valid_features]
            if not self.features:
                raise NoValidFeatures
            if list(X.columns) != self.features:
                X = X[self.features]
        return X

    def _preprocess_fit_args(self, **kwargs):
        time_limit = kwargs.get('time_limit', None)
        max_time_limit_ratio = self.params_aux.get('max_time_limit_ratio', 1)
        if time_limit is not None:
            time_limit *= max_time_limit_ratio
        max_time_limit = self.params_aux.get('max_time_limit', None)
        if max_time_limit is not None:
            if time_limit is None:
                time_limit = max_time_limit
            else:
                time_limit = min(time_limit, max_time_limit)
        min_time_limit = self.params_aux.get('min_time_limit', 0)
        if min_time_limit is None:
            time_limit = min_time_limit
        elif time_limit is not None:
            time_limit = max(time_limit, min_time_limit)
        kwargs['time_limit'] = time_limit
        return kwargs

    def fit(self, **kwargs):
        kwargs = self._preprocess_fit_args(**kwargs)
        if 'time_limit' not in kwargs or kwargs['time_limit'] is None or kwargs['time_limit'] > 0:
            self._fit(**kwargs)
        else:
            logger.warning(f'\tWarning: Model has no time left to train, skipping model... (Time Left = {round(kwargs["time_limit"], 1)}s)')
            raise TimeLimitExceeded

    def _fit(self, X_train, y_train, **kwargs):
        # kwargs may contain: num_cpus, num_gpus
        X_train = self.preprocess(X_train)
        self.model = self.model.fit(X_train, y_train)

    def predict(self, X, preprocess=True):
        y_pred_proba = self.predict_proba(X, preprocess=preprocess)
        y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
        return y_pred

    def predict_proba(self, X, preprocess=True, normalize=None):
        if normalize is None:
            normalize = self.normalize_pred_probas
        y_pred_proba = self._predict_proba(X=X, preprocess=preprocess)
        if normalize:
            y_pred_proba = normalize_pred_probas(y_pred_proba, self.problem_type)
        y_pred_proba = y_pred_proba.astype(np.float32)
        return y_pred_proba

    def _predict_proba(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X)

        if self.problem_type == REGRESSION:
            return self.model.predict(X)

        y_pred_proba = self.model.predict_proba(X)
        if self.problem_type == BINARY:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif y_pred_proba.shape[1] > 2:
            return y_pred_proba
        else:
            return y_pred_proba[:, 1]

    def score(self, X, y, eval_metric=None, metric_needs_y_pred=None, preprocess=True):
        if eval_metric is None:
            eval_metric = self.eval_metric
        if metric_needs_y_pred is None:
            metric_needs_y_pred = self.metric_needs_y_pred
        if metric_needs_y_pred:
            y_pred = self.predict(X=X, preprocess=preprocess)
            return eval_metric(y, y_pred)
        else:
            y_pred_proba = self.predict_proba(X=X, preprocess=preprocess)
            return eval_metric(y, y_pred_proba)

    def score_with_y_pred_proba(self, y, y_pred_proba, eval_metric=None, metric_needs_y_pred=None):
        if eval_metric is None:
            eval_metric = self.eval_metric
        if metric_needs_y_pred is None:
            metric_needs_y_pred = self.metric_needs_y_pred
        if metric_needs_y_pred:
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
            return eval_metric(y, y_pred)
        else:
            return eval_metric(y, y_pred_proba)

    def save(self, path: str = None, verbose=True) -> str:
        """
        Saves the model to disk.

        Parameters
        ----------
        path : str, default None
            Path to the saved model, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            If None, self.path is used.
            The final model file is typically saved to path + self.model_file_name.
        verbose : bool, default True
            Whether to log the location of the saved file.

        Returns
        -------
        path : str
            Path to the saved model, minus the file name.
            Use this value to load the model from disk via cls.load(path), cls being the class of the model object, such as model = RFModel.load(path)
        """
        if path is None:
            path = self.path
        file_path = path + self.model_file_name
        save_pkl.save(path=file_path, object=self, verbose=verbose)
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        """
        Loads the model from disk to memory.

        Parameters
        ----------
        path : str
            Path to the saved model, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            The model file is typically located in path + cls.model_file_name.
        reset_paths : bool, default True
            Whether to reset the self.path value of the loaded model to be equal to path.
            It is highly recommended to keep this value as True unless accessing the original self.path value is important.
            If False, the actual valid path and self.path may differ, leading to strange behaviour and potential exceptions if the model needs to load any other files at a later time.
        verbose : bool, default True
            Whether to log the location of the loaded file.

        Returns
        -------
        model : cls
            Loaded model object.
        """
        file_path = path + cls.model_file_name
        model = load_pkl.load(path=file_path, verbose=verbose)
        if reset_paths:
            model.set_contexts(path)
        return model

    # TODO: Consider disabling feature pruning when num_features is high (>1000 for example), or using a faster feature importance calculation method
    def compute_feature_importance(self, X, y, features_to_use=None, preprocess=True, subsample_size=10000, silent=False, **kwargs) -> pd.Series:
        if (subsample_size is not None) and (len(X) > subsample_size):
            # Reset index to avoid error if duplicated indices.
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

            X = X.sample(subsample_size, random_state=0)
            y = y.loc[X.index]
        else:
            X = X.copy()
            y = y.copy()

        if preprocess:
            X = self.preprocess(X)

        if not features_to_use:
            features = list(X.columns.values)
        else:
            features = list(features_to_use)

        feature_importance_quick_dict = self.get_model_feature_importance()
        # TODO: Also consider banning features with close to 0 importance
        # TODO: Consider adding 'golden' features if the importance is high enough to avoid unnecessary computation when doing feature selection
        banned_features = [feature for feature, importance in feature_importance_quick_dict.items() if importance == 0 and feature in features]
        features = [feature for feature in features if feature not in banned_features]

        permutation_importance_dict = self.compute_permutation_importance(X=X, y=y, features=features, preprocess=False, silent=silent)

        feature_importances = pd.Series(permutation_importance_dict)
        results_banned = pd.Series(data=[0 for _ in range(len(banned_features))], index=banned_features)
        feature_importances = pd.concat([feature_importances, results_banned])
        feature_importances = feature_importances.sort_values(ascending=False)

        return feature_importances

    # TODO: Consider repeating with different random seeds and averaging to increase confidence
    # TODO: Optimize this
    # Compute feature importance via permutation importance
    # Note: Expensive to compute
    #  Time to compute is O(predict_time*num_features)
    def compute_permutation_importance(self, X, y, features: list, preprocess=True, silent=False) -> dict:
        time_start = time.time()

        feature_count = len(features)
        if not silent:
            logger.log(20, f'Computing permutation importance for {feature_count} features on {self.name} ...')
        if preprocess:
            X = self.preprocess(X)

        time_start_score = time.time()
        model_score_base = self.score(X=X, y=y, preprocess=False)
        time_score = time.time() - time_start_score

        if not silent:
            time_estimated = (feature_count + 1) * time_score + time_start_score - time_start
            logger.log(20, f'\t{round(time_estimated, 2)}s\t= Expected runtime')

        X_shuffled = shuffle_df_rows(X=X, seed=0)
        row_count = X.shape[0]

        # calculating maximum number of features, which is safe to process parallel
        X_memory_ratio_max = 0.2
        compute_count_max = 200

        X_size_bytes = sys.getsizeof(pickle.dumps(X, protocol=4))
        available_mem = psutil.virtual_memory().available
        X_memory_ratio = X_size_bytes / available_mem

        compute_count_safe = math.floor(X_memory_ratio_max / X_memory_ratio)
        compute_count = max(1, min(compute_count_max, compute_count_safe))
        compute_count = min(compute_count, feature_count)

        # creating copy of original data N=compute_count times for parallel processing
        X_raw = pd.concat([X.copy() for _ in range(compute_count)], ignore_index=True, sort=False).reset_index(drop=True)

        #  TODO: Make this faster by multi-threading?
        permutation_importance_dict = {}
        for i in range(0, feature_count, compute_count):
            parallel_computed_features = features[i:i + compute_count]

            # if final iteration, leaving only necessary part of X_raw
            num_features_processing = len(parallel_computed_features)
            final_iteration = i + num_features_processing == feature_count
            if (num_features_processing < compute_count) and final_iteration:
                X_raw = X_raw.loc[:row_count * num_features_processing - 1]

            row_index = 0
            for feature in parallel_computed_features:
                row_index_end = row_index + row_count
                X_raw.loc[row_index:row_index_end - 1, feature] = X_shuffled[feature].values
                row_index = row_index_end

            if self.metric_needs_y_pred:
                y_pred = self.predict(X_raw, preprocess=False)
            else:
                y_pred = self.predict_proba(X_raw, preprocess=False)

            row_index = 0
            for feature in parallel_computed_features:
                # calculating importance score for given feature
                row_index_end = row_index + row_count
                y_pred_cur = y_pred[row_index:row_index_end]
                score = self.eval_metric(y, y_pred_cur)
                permutation_importance_dict[feature] = model_score_base - score

                if not final_iteration:
                    # resetting to original values for processed feature
                    X_raw.loc[row_index:row_index_end - 1, feature] = X[feature].values

                row_index = row_index_end

        if not silent:
            logger.log(20, f'\t{round(time.time() - time_start, 2)}s\t= Actual runtime')

        return permutation_importance_dict

    # Custom feature importance values for a model (such as those calculated from training)
    def get_model_feature_importance(self) -> dict:
        return dict()

    # Hyperparameters of trained model
    def get_trained_params(self) -> dict:
        trained_params = self.params.copy()
        trained_params.update(self.params_trained)
        return trained_params

    # After calling this function, returned model should be able to be fit as if it was new, as well as deep-copied.
    def convert_to_template(self):
        model = self.model
        self.model = None
        template = copy.deepcopy(self)
        template.reset_metrics()
        self.model = model
        return template

    # After calling this function, model should be able to be fit without test data using the iterations trained by the original model
    def convert_to_refitfull_template(self):
        params_trained = self.params_trained.copy()
        template = self.convert_to_template()
        template.params.update(params_trained)
        template.name = template.name + REFIT_FULL_SUFFIX
        template.set_contexts(self.path_root + template.name + os.path.sep)
        return template

    def hyperparameter_tune(self, X_train, y_train, X_val, y_val, scheduler_options, **kwargs):
        # verbosity = kwargs.get('verbosity', 2)
        time_start = time.time()
        logger.log(15, "Starting generic AbstractModel hyperparameter tuning for %s model..." % self.name)
        self._set_default_searchspace()
        params_copy = self.params.copy()
        directory = self.path  # also create model directory if it doesn't exist
        # TODO: This will break on S3. Use tabular/utils/savers for datasets, add new function
        scheduler_func, scheduler_options = scheduler_options  # Unpack tuple
        if scheduler_func is None or scheduler_options is None:
            raise ValueError("scheduler_func and scheduler_options cannot be None for hyperparameter tuning")
        params_copy['num_threads'] = scheduler_options['resource'].get('num_cpus', None)
        params_copy['num_gpus'] = scheduler_options['resource'].get('num_gpus', None)
        dataset_train_filename = 'dataset_train.p'
        train_path = directory + dataset_train_filename
        save_pkl.save(path=train_path, object=(X_train, y_train))

        dataset_val_filename = 'dataset_val.p'
        val_path = directory + dataset_val_filename
        save_pkl.save(path=val_path, object=(X_val, y_val))

        if not any(isinstance(params_copy[hyperparam], Space) for hyperparam in params_copy):
            logger.warning("Attempting to do hyperparameter optimization without any search space (all hyperparameters are already fixed values)")
        else:
            logger.log(15, "Hyperparameter search space for %s model: " % self.name)
            for hyperparam in params_copy:
                if isinstance(params_copy[hyperparam], Space):
                    logger.log(15, f"{hyperparam}:   {params_copy[hyperparam]}")

        util_args = dict(
            dataset_train_filename=dataset_train_filename,
            dataset_val_filename=dataset_val_filename,
            directory=directory,
            model=self,
            time_start=time_start,
            time_limit=scheduler_options['time_out'],
        )

        model_trial.register_args(util_args=util_args, **params_copy)
        scheduler: FIFOScheduler = scheduler_func(model_trial, **scheduler_options)
        if ('dist_ip_addrs' in scheduler_options) and (len(scheduler_options['dist_ip_addrs']) > 0):
            # This is multi-machine setting, so need to copy dataset to workers:
            logger.log(15, "Uploading data to remote workers...")
            scheduler.upload_files([train_path, val_path])  # TODO: currently does not work.
            directory = self.path  # TODO: need to change to path to working directory used on every remote machine
            model_trial.update(directory=directory)
            logger.log(15, "uploaded")

        scheduler.run()
        scheduler.join_jobs()

        return self._get_hpo_results(scheduler=scheduler, scheduler_options=scheduler_options, time_start=time_start)

    def _get_hpo_results(self, scheduler, scheduler_options, time_start):
        # Store results / models from this HPO run:
        best_hp = scheduler.get_best_config()  # best_hp only contains searchable stuff
        hpo_results = {
            'best_reward': scheduler.get_best_reward(),
            'best_config': best_hp,
            'total_time': time.time() - time_start,
            'metadata': scheduler.metadata,
            'training_history': scheduler.training_history,
            'config_history': scheduler.config_history,
            'reward_attr': scheduler._reward_attr,
            'args': model_trial.args
        }

        hpo_results = BasePredictor._format_results(hpo_results)  # results summarizing HPO for this model
        if ('dist_ip_addrs' in scheduler_options) and (len(scheduler_options['dist_ip_addrs']) > 0):
            raise NotImplementedError("need to fetch model files from remote Workers")
            # TODO: need to handle locations carefully: fetch these files and put them into self.path directory:
            # 1) hpo_results['trial_info'][trial]['metadata']['trial_model_file']

        hpo_models = {}  # stores all the model names and file paths to model objects created during this HPO run.
        hpo_model_performances = {}
        for trial in sorted(hpo_results['trial_info'].keys()):
            # TODO: ignore models which were killed early by scheduler (eg. in Hyperband). How to ID these?
            file_id = "trial_" + str(trial)  # unique identifier to files from this trial
            trial_model_name = self.name + os.path.sep + file_id
            trial_model_path = self.path_root + trial_model_name + os.path.sep
            hpo_models[trial_model_name] = trial_model_path
            hpo_model_performances[trial_model_name] = hpo_results['trial_info'][trial][scheduler._reward_attr]

        logger.log(15, "Time for %s model HPO: %s" % (self.name, str(hpo_results['total_time'])))
        logger.log(15, "Best hyperparameter configuration for %s model: " % self.name)
        logger.log(15, str(best_hp))
        return hpo_models, hpo_model_performances, hpo_results

    def feature_prune(self, X_train, X_holdout, y_train, y_holdout):
        feature_pruner = FeaturePruner(model_base=self)
        X_train, X_val, y_train, y_val = generate_train_test_split(X_train, y_train, problem_type=self.problem_type, test_size=0.2)
        feature_pruner.tune(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_holdout=X_holdout, y_holdout=y_holdout)
        features_to_keep = feature_pruner.features_in_iter[feature_pruner.best_iteration]
        logger.debug(str(features_to_keep))
        self.features = features_to_keep

    # Resets metrics for the model
    def reset_metrics(self):
        self.fit_time = None
        self.predict_time = None
        self.val_score = None
        self.params_trained = dict()

    # TODO: Experimental, currently unused
    #  Has not been tested on Windows
    #  Does not work if model is located in S3
    #  Does not work if called before model was saved to disk (Will output 0)
    def get_disk_size(self) -> int:
        # Taken from https://stackoverflow.com/a/1392549
        from pathlib import Path
        model_path = Path(self.path)
        model_disk_size = sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file())
        return model_disk_size

    # TODO: This results in a doubling of memory usage of the model to calculate its size.
    #  If the model takes ~40%+ of memory, this may result in an OOM error.
    #  This is generally not an issue because the model already needed to do this when being saved to disk, so the error would have been triggered earlier.
    #  Consider using Pympler package for memory efficiency: https://pympler.readthedocs.io/en/latest/asizeof.html#asizeof
    def get_memory_size(self) -> int:
        gc.collect()  # Try to avoid OOM error
        return sys.getsizeof(pickle.dumps(self, protocol=4))

    # Removes non-essential objects from the model to reduce memory and disk footprint.
    # If `remove_fit=True`, enables the removal of variables which are required for fitting the model. If the model is already fully trained, then it is safe to remove these.
    # If `remove_info=True`, enables the removal of variables which are used during model.get_info(). The values will be None when calling model.get_info().
    # If `requires_save=True`, enables the removal of variables which are part of the model.pkl object, requiring an overwrite of the model to disk if it was previously persisted.
    def reduce_memory_size(self, remove_fit=True, remove_info=False, requires_save=True, **kwargs):
        pass

    # Deletes the model from disk.
    # WARNING: This will DELETE ALL FILES in the self.path directory, regardless if they were created by AutoGluon or not.
    #  DO NOT STORE FILES INSIDE OF THE MODEL DIRECTORY THAT ARE UNRELATED TO AUTOGLUON.
    def delete_from_disk(self):
        logger.log(30, f'Deleting model {self.name}. All files under {self.path} will be removed.')
        from pathlib import Path
        import shutil
        model_path = Path(self.path)
        # TODO: Report errors?
        shutil.rmtree(path=model_path, ignore_errors=True)

    def get_info(self) -> dict:
        info = dict(
            name=self.name,
            model_type=type(self).__name__,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric.name,
            stopping_metric=self.stopping_metric.name,
            fit_time=self.fit_time,
            predict_time=self.predict_time,
            val_score=self.val_score,
            hyperparameters=self.params,
            hyperparameters_fit=self.params_trained,  # TODO: Explain in docs that this is for hyperparameters that differ in final model from original hyperparameters, such as epochs (from early stopping)
            hyperparameters_nondefault=self.nondefault_params,
            AG_args_fit=self.params_aux,
            num_features=len(self.features) if self.features else None,
            features=self.features,
            # disk_size=self.get_disk_size(),
            memory_size=self.get_memory_size(),  # Memory usage of model in bytes
        )
        return info

    @classmethod
    def load_info(cls, path, load_model_if_required=True) -> dict:
        load_path = path + cls.model_info_name
        try:
            return load_pkl.load(path=load_path)
        except:
            if load_model_if_required:
                model = cls.load(path=path, reset_paths=True)
                return model.get_info()
            else:
                raise

    def save_info(self) -> dict:
        info = self.get_info()

        save_pkl.save(path=self.path + self.model_info_name, object=info)
        json_path = self.path + self.model_info_json_name
        save_json.save(path=json_path, obj=info)
        return info
