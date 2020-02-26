import copy
import logging
import os
import time

import numpy as np
import pandas as pd

from .model_trial import model_trial
from ...constants import BINARY, REGRESSION
from ...tuning.feature_pruner import FeaturePruner
from ...utils import get_pred_from_proba, generate_train_test_split
from .... import metrics
from ....utils.loaders import load_pkl
from ....utils.savers import save_pkl
from ......core import Space, Categorical, List, NestedSpace
from ......task.base import BasePredictor
from ......scheduler.scheduler import TaskScheduler

logger = logging.getLogger(__name__)


# Methods useful for all models:
def fixedvals_from_searchspaces(params):
    """ Converts any search space hyperparams in params dict into fixed default values. """
    if any(isinstance(params[hyperparam], Space) for hyperparam in params):
        logger.warning("Attempting to fit model without HPO, but search space is provided. fit() will only consider default hyperparameter values from search space.")
        bad_keys = [hyperparam for hyperparam in params if isinstance(params[hyperparam], Space)][:]  # delete all keys which are of type autogluon Space
        params = params.copy()
        for hyperparam in bad_keys:
            params[hyperparam] = hp_default_value(params[hyperparam])
        return params
    else:
        return params


def hp_default_value(hp_value):
    """ Extracts default fixed value from hyperparameter search space hp_value to use a fixed value instead of a search space.
    """
    if not isinstance(hp_value, Space):
        return hp_value
    if isinstance(hp_value, Categorical):
        return hp_value[0]
    elif isinstance(hp_value, List):
        return [z[0] for z in hp_value]
    elif isinstance(hp_value, NestedSpace):
        raise ValueError("Cannot extract default value from NestedSpace. Please specify fixed value instead of: %s" % str(hp_value))
    else:
        return hp_value.get_hp('dummy_name').default_value


class AbstractModel:
    model_file_name = 'model.pkl'

    def __init__(self, path: str, name: str, problem_type: str, objective_func, stopping_metric=None, model=None, hyperparameters=None, features=None, feature_types_metadata=None, debug=0):
        """ Creates a new model. 
            Args:
                path (str): directory where to store all outputs
                name (str): name of subdirectory inside path where model will be saved
                hyperparameters (dict): various hyperparameters that will be used by model (can be search spaces instead of fixed values)
        """
        self.name = name
        self.path_root = path
        self.path_suffix = self.name + os.path.sep  # TODO: Make into function to avoid having to reassign on load?
        self.path = self.create_contexts(self.path_root + self.path_suffix)  # TODO: Make this path a function for consistency.
        self.model = model
        self.problem_type = problem_type
        self.objective_func = objective_func  # Note: we require higher values = better performance

        if stopping_metric is None:
            self.stopping_metric = self.objective_func
        else:
            self.stopping_metric = stopping_metric

        if isinstance(self.objective_func, metrics._ProbaScorer):
            self.metric_needs_y_pred = False
        elif isinstance(self.objective_func, metrics._ThresholdScorer):
            self.metric_needs_y_pred = False
        else:
            self.metric_needs_y_pred = True

        if isinstance(self.stopping_metric, metrics._ProbaScorer):
            self.stopping_metric_needs_y_pred = False
        elif isinstance(self.stopping_metric, metrics._ThresholdScorer):
            self.stopping_metric_needs_y_pred = False
        else:
            self.stopping_metric_needs_y_pred = True

        self.feature_types_metadata = feature_types_metadata  # TODO: Should this be passed to a model on creation? Should it live in a Dataset object and passed during fit? Currently it is being updated prior to fit by trainer
        self.features = features
        self.debug = debug

        self.fit_time = None  # Time taken to fit in seconds (Training data)
        self.predict_time = None  # Time taken to predict in seconds (Validation data)
        self.val_score = None  # Score with eval_metric (Validation data)

        self.params = {}
        self._set_default_params()
        self.nondefault_params = []
        if hyperparameters is not None:
            self.params.update(hyperparameters.copy())
            self.nondefault_params = list(hyperparameters.keys())[:]  # These are hyperparameters that user has specified.
        self.params_trained = dict()

    # Checks if model is ready to make predictions for final result
    def is_valid(self):
        return self.is_fit()

    # Checks if a model has been fit
    def is_fit(self):
        return self.model is not None

    def _set_default_params(self):
        pass

    def _set_default_param_value(self, param_name, param_value):
        if param_name not in self.params:
            self.params[param_name] = param_value

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

    def rename(self, name):
        self.path = self.path[:-len(self.name) - 1] + name + os.path.sep
        self.name = name

    # Extensions of preprocess must act identical in bagged situations, otherwise test-time predictions will be incorrect
    # This means preprocess cannot be used for normalization
    # TODO: Add preprocess_stateful() to enable stateful preprocessing for models such as KNN
    def preprocess(self, X):
        if self.features is not None:
            return X[self.features]
        return X

    def fit(self, X_train, Y_train, **kwargs):
        # kwargs may contain: num_cpus, num_gpus
        X_train = self.preprocess(X_train)
        self.model = self.model.fit(X_train, Y_train)

    def predict(self, X, preprocess=True):
        y_pred_proba = self.predict_proba(X, preprocess=preprocess)
        y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
        return y_pred

    def predict_proba(self, X, preprocess=True):
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

    def score(self, X, y, eval_metric=None, metric_needs_y_pred=None):
        if eval_metric is None:
            eval_metric = self.objective_func
        if metric_needs_y_pred is None:
            metric_needs_y_pred = self.metric_needs_y_pred
        if metric_needs_y_pred:
            y_pred = self.predict(X=X)
            return eval_metric(y, y_pred)
        else:
            y_pred_proba = self.predict_proba(X=X)
            return eval_metric(y, y_pred_proba)

    def score_with_y_pred_proba(self, y, y_pred_proba, eval_metric=None, metric_needs_y_pred=None):
        if eval_metric is None:
            eval_metric = self.objective_func
        if metric_needs_y_pred is None:
            metric_needs_y_pred = self.metric_needs_y_pred
        if metric_needs_y_pred:
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
            return eval_metric(y, y_pred)
        else:
            return eval_metric(y, y_pred_proba)

    def save(self, file_prefix="", directory=None, return_filename=False, verbose=True):
        if directory is None:
            directory = self.path
        file_name = directory + file_prefix + self.model_file_name
        save_pkl.save(path=file_name, object=self, verbose=verbose)
        if return_filename:
            return file_name

    @classmethod
    def load(cls, path, file_prefix="", reset_paths=False, verbose=True):
        load_path = path + file_prefix + cls.model_file_name
        if not reset_paths:
            return load_pkl.load(path=load_path, verbose=verbose)
        else:
            obj = load_pkl.load(path=load_path, verbose=verbose)
            obj.set_contexts(path)
            return obj

    def compute_feature_importance(self, X, y, features_to_use=None):
        sample_size = 10000
        if len(X) > sample_size:
            X = X.sample(sample_size, random_state=0)
            y = y.loc[X.index]
        else:
            X = X.copy()
            y = y.copy()

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        X = self.preprocess(X)

        if not features_to_use:
            features = X.columns.values
        else:
            features = features_to_use
        feature_count = len(features)

        model_score_base = self.score(X=X, y=y)

        model_score_diff = []

        row_count = X.shape[0]
        rand_shuffle = np.random.randint(0, row_count, size=row_count)

        X_test_shuffled = X.iloc[rand_shuffle].reset_index(drop=True)
        compute_count = 200
        indices = [x for x in range(0, feature_count, compute_count)]

        # TODO: Make this faster by multi-threading?
        for i, indice in enumerate(indices):
            if indice + compute_count > feature_count:
                compute_count = feature_count - indice

            logger.debug(indice)
            x = [X.copy() for _ in range(compute_count)]  # TODO Make this much faster, only make this and concat it once. Then just update values and reset the values edited each iteration
            for j, val in enumerate(x):
                feature = features[indice + j]
                val[feature] = X_test_shuffled[feature]
            X_test_raw = pd.concat(x, ignore_index=True)

            if self.metric_needs_y_pred:
                Y_pred = self.predict(X_test_raw, preprocess=False)
            else:
                Y_pred = self.predict_proba(X_test_raw, preprocess=False)

            row_index = 0
            for j in range(compute_count):
                row_index_end = row_index + row_count
                Y_pred_cur = Y_pred[row_index:row_index_end]
                row_index = row_index_end
                score = self.objective_func(y, Y_pred_cur)
                model_score_diff.append(model_score_base - score)

        results = pd.Series(data=model_score_diff, index=features)
        results = results.sort_values(ascending=False)

        return results

    def _get_default_searchspace(self, problem_type):
        return NotImplementedError

    def _set_default_searchspace(self):
        """ Sets up default search space for HPO. Each hyperparameter which user did not specify is converted from
            default fixed value to default spearch space.
        """
        def_search_space = self._get_default_searchspace(problem_type=self.problem_type).copy()
        # Note: when subclassing AbstractModel, you must define or import get_default_searchspace() from the appropriate location.
        for key in self.nondefault_params:  # delete all user-specified hyperparams from the default search space
            def_search_space.pop(key, None)
        if self.params is not None:
            self.params.update(def_search_space)

    # Hyperparameters of trained model
    def get_trained_params(self):
        trained_params = self.params.copy()
        trained_params.update(self.params_trained)
        return trained_params

    # TODO: currently inplace and destructive. This won't work well for in-memory models.
    #  Problem with not inplace -> 2x memory usage, try to avoid somehow
    # After calling this function, model should be able to be fit as if it was new, as well as deep-copied.
    def convert_to_template(self):
        model = self.model
        self.model = None
        template = copy.deepcopy(self)
        template.params_trained = dict()
        self.model = model
        return template

    # After calling this function, model should be able to be fit without test data using the iterations trained by the original model
    def convert_to_compressed_template(self):
        params_trained = self.params_trained.copy()
        template = self.convert_to_template()
        template.params.update(params_trained)
        template.name = template.name + '_compressed'
        template.path = template.create_contexts(self.path + template.name + os.path.sep)
        return template

    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options, **kwargs):
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
        save_pkl.save(path=train_path, object=(X_train, Y_train))

        dataset_val_filename = 'dataset_val.p'
        val_path = directory + dataset_val_filename
        save_pkl.save(path=val_path, object=(X_test, Y_test))

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
        scheduler: TaskScheduler = scheduler_func(model_trial, **scheduler_options)
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

    def feature_prune(self, X_train, X_holdout, Y_train, Y_holdout):
        feature_pruner = FeaturePruner(model_base=self)
        X_train, X_test, y_train, y_test = generate_train_test_split(X_train, Y_train, problem_type=self.problem_type, test_size=0.2)
        feature_pruner.tune(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_holdout=X_holdout, y_holdout=Y_holdout)
        features_to_keep = feature_pruner.features_in_iter[feature_pruner.best_iteration]
        logger.debug(str(features_to_keep))
        self.features = features_to_keep
