import logging
import pickle
import psutil
import sys
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ...constants import REGRESSION
from ..sklearn.sklearn_model import SKLearnModel
from ....utils.exceptions import NotEnoughMemoryError

logger = logging.getLogger(__name__)


# TODO: Normalize data!
class KNNModel(SKLearnModel):
    def __init__(self, path: str, name: str, problem_type: str, objective_func, hyperparameters=None, features=None, feature_types_metadata=None, debug=0):
        super().__init__(path=path, name=name, problem_type=problem_type, objective_func=objective_func, hyperparameters=hyperparameters, features=features, feature_types_metadata=feature_types_metadata, debug=debug)
        if self.problem_type == REGRESSION:
            self._model_type = KNeighborsRegressor
        else:
            self._model_type = KNeighborsClassifier

    def preprocess(self, X):
        cat_columns = X.select_dtypes(['category']).columns
        X = X.drop(cat_columns, axis=1).fillna(0)  # TODO: Test if crash when all columns are categorical
        return X

    def _set_default_params(self):
        default_params = {
            'weights': 'uniform',
            'n_jobs': -1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # TODO: Enable HPO for KNN
    def _get_default_searchspace(self, problem_type):
        spaces = {}
        return spaces

    def fit(self, X_train, Y_train, X_test=None, Y_test=None, **kwargs):
        X_train = self.preprocess(X_train)

        model_size_bytes = sys.getsizeof(pickle.dumps(X_train))
        expected_final_model_size_bytes = model_size_bytes * 2.1  # Roughly what can be expected of the final KNN model in memory size
        if expected_final_model_size_bytes > 10000000:  # Only worth checking if expected model size is >10MB
            available_mem = psutil.virtual_memory().available
            model_memory_ratio = expected_final_model_size_bytes / available_mem
            if model_memory_ratio > 0.35:
                logger.warning('\tWarning: Model is expected to require %s percent of available memory...' % (model_memory_ratio * 100))
            if model_memory_ratio > 0.45:
                raise NotEnoughMemoryError  # don't train full model to avoid OOM error

        model = self._model_type(**self.params)
        self.model = model.fit(X_train, Y_train)

    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options=None, **kwargs):
        # verbosity = kwargs.get('verbosity', 2)
        self.fit(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, **kwargs)
        hpo_model_performances = {self.name: self.score(X_test, Y_test)}
        hpo_results = {}
        self.save()
        hpo_models = {self.name: self.path}

        return hpo_models, hpo_model_performances, hpo_results
