import logging, time
import pickle
import psutil
import sys
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor

from ..sklearn.sklearn_model import SKLearnModel
from ...constants import MULTICLASS, REGRESSION
from ....utils.exceptions import NotEnoughMemoryError, TimeLimitExceeded

logger = logging.getLogger(__name__)


class RFModel(SKLearnModel):
    def __init__(self, path: str, name: str, problem_type: str, objective_func, num_classes=None, hyperparameters=None, features=None, feature_types_metadata=None, debug=0):
        self.num_classes = num_classes
        super().__init__(path=path, name=name, problem_type=problem_type, objective_func=objective_func, hyperparameters=hyperparameters, features=features, feature_types_metadata=feature_types_metadata, debug=debug)
        if self.params['model_type'] == 'rf':
            if self.problem_type == REGRESSION:
                self._model_type = RandomForestRegressor
            else:
                self._model_type = RandomForestClassifier
        elif self.params['model_type'] == 'xt':
            if self.problem_type == REGRESSION:
                self._model_type = ExtraTreesRegressor
            else:
                self._model_type = ExtraTreesClassifier
        else:
            raise ValueError('model_type arg must be one of [\'rf\', \'xt\'], but value %s was given.' % self.params['model_type'])

    def preprocess(self, X):
        X = super().preprocess(X)
        X = X.fillna(0)
        return X

    def _set_default_params(self):
        default_params = {
            'model_type': 'rf',
            'n_estimators': 300,
            'n_jobs': -1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # TODO: Add in documentation that Categorical default is the first index
    # TODO: enable HPO for RF models
    def _get_default_searchspace(self, problem_type):
        spaces = {
            # 'n_estimators': Int(lower=10, upper=1000, default=300),
            # 'max_features': Categorical(['auto', 0.5, 0.25]),
            # 'criterion': Categorical(['gini', 'entropy']),
        }

        return spaces

    def fit(self, X_train, Y_train, X_test=None, Y_test=None, time_limit=None, **kwargs):
        time_start = time.time()
        hyperparams = self.params.copy()
        hyperparams.pop('model_type')
        n_estimators_final = hyperparams['n_estimators']
        n_estimators_test = 8
        # minimum_n_estimators = min(50, n_estimators_final)  # TODO: Add in for early stopping RF models based on memory/time constraints

        X_train = self.preprocess(X_train)
        n_estimator_increments = [n_estimators_final]

        # Very rough guess to size of a single tree before training
        if self.problem_type == MULTICLASS:
            if self.num_classes is None:
                num_trees_per_estimator = 10  # Guess since it wasn't passed in, could also check y_train for a better value
            else:
                num_trees_per_estimator = self.num_classes
        else:
            num_trees_per_estimator = 1
        bytes_per_estimator = num_trees_per_estimator * len(X_train) / 60000 * 1e6  # Underestimates by 3x on ExtraTrees
        available_mem = psutil.virtual_memory().available
        expected_memory_usage = bytes_per_estimator * n_estimators_final / available_mem
        # expected_min_memory_usage = bytes_per_estimator * minimum_n_estimators / available_mem
        if expected_memory_usage > 0.8:  # if estimated size is greater than 80% memory
            logger.warning('\tWarning: Model is expected to require %s percent of available memory (Estimated before training)...' % (expected_memory_usage * 100))
            raise NotEnoughMemoryError

        if n_estimators_final > n_estimators_test * 2:
            if self.problem_type == MULTICLASS:
                n_estimator_increments = [n_estimators_test, n_estimators_final]
                hyperparams['warm_start'] = True
            else:
                if expected_memory_usage > 0.05:  # Somewhat arbitrary, consider finding a better value, should it scale by cores?
                    # Causes ~10% training slowdown, so try to avoid if memory is not an issue
                    n_estimator_increments = [n_estimators_test, n_estimators_final]
                    hyperparams['warm_start'] = True

        hyperparams['n_estimators'] = n_estimator_increments[0]
        self.model = self._model_type(**hyperparams)

        time_train_start = time.time()
        for i, n_estimators in enumerate(n_estimator_increments):
            if i != 0:
                self.model.n_estimators = n_estimators
            self.model = self.model.fit(X_train, Y_train)
            if (i == 0) and (len(n_estimator_increments) > 1):
                time_elapsed = time.time() - time_train_start
                time_expected = time_train_start - time_start + (time_elapsed * n_estimators_final / n_estimators)
                if (time_limit is not None) and (time_expected > time_limit):
                    logger.warning('\tWarning: Model is expected to require %ss to train, which exceeds the maximum time limit of %ss, skipping model...' % (round(time_expected, 2), round(time_limit, 2)))
                    raise TimeLimitExceeded
                model_size_bytes = sys.getsizeof(pickle.dumps(self.model))
                expected_final_model_size_bytes = model_size_bytes * (n_estimators_final / self.model.n_estimators)
                available_mem = psutil.virtual_memory().available
                model_memory_ratio = expected_final_model_size_bytes / available_mem
                if model_memory_ratio > 0.20:
                    logger.warning('\tWarning: Model is expected to require %s percent of available memory...' % (model_memory_ratio*100))
                if model_memory_ratio > 0.30:
                    raise NotEnoughMemoryError  # don't train full model to avoid OOM error

    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options=None, **kwargs):

        self.fit(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, **kwargs)
        hpo_model_performances = {self.name: self.score(X_test, Y_test)}
        hpo_results = {}
        self.save()
        hpo_models = {self.name: self.path}

        return hpo_models, hpo_model_performances, hpo_results
