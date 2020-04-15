import logging
import pickle
import sys
import time

import psutil
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ..abstract import model_trial
from ..abstract.abstract_model import SKLearnModel
from ...constants import REGRESSION
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
        X = X.drop(cat_columns, axis=1)  # TODO: Test if crash when all columns are categorical
        X = super().preprocess(X).fillna(0)
        return X

    def _set_default_params(self):
        default_params = {
            'weights': 'uniform',
            'n_jobs': -1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # TODO: Enable HPO for KNN
    def _get_default_searchspace(self):
        spaces = {}
        return spaces

    def fit(self, X_train, Y_train, **kwargs):
        X_train = self.preprocess(X_train)

        model_size_bytes = sys.getsizeof(pickle.dumps(X_train, protocol=4))
        expected_final_model_size_bytes = model_size_bytes * 2.1  # Roughly what can be expected of the final KNN model in memory size
        if expected_final_model_size_bytes > 10000000:  # Only worth checking if expected model size is >10MB
            available_mem = psutil.virtual_memory().available
            model_memory_ratio = expected_final_model_size_bytes / available_mem
            if model_memory_ratio > 0.35:
                logger.warning(f'\tWarning: Model is expected to require {model_memory_ratio * 100} percent of available memory...')
            if model_memory_ratio > 0.45:
                raise NotEnoughMemoryError  # don't train full model to avoid OOM error

        model = self._model_type(**self.params)
        self.model = model.fit(X_train, Y_train)

    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options=None, **kwargs):
        fit_model_args = dict(X_train=X_train, Y_train=Y_train, **kwargs)
        predict_proba_args = dict(X=X_test)
        model_trial.fit_and_save_model(model=self, params=dict(), fit_args=fit_model_args, predict_proba_args=predict_proba_args, y_test=Y_test, time_start=time.time(), time_limit=None)
        hpo_results = {'total_time': self.fit_time}
        hpo_model_performances = {self.name: self.val_score}
        hpo_models = {self.name: self.path}
        return hpo_models, hpo_model_performances, hpo_results
