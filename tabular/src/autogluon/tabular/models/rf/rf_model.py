import logging
import math
import pickle
import sys
import time

import numpy as np
import psutil

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS, QUANTILE
from autogluon.core.utils.exceptions import NotEnoughMemoryError, TimeLimitExceeded
from autogluon.core.features.types import R_OBJECT
from autogluon.core.utils.utils import normalize_pred_probas

from autogluon.core.models.abstract.model_trial import skip_hpo
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

logger = logging.getLogger(__name__)


class RFModel(AbstractModel):
    """
    Random Forest model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._daal = False  # Whether daal4py backend is being used

    def _get_model_type(self):
        if self.problem_type == QUANTILE:
            from .rf_quantile import RandomForestQuantileRegressor
            return RandomForestQuantileRegressor
        if self.params_aux.get('use_daal', False):
            # Disabled by default because it appears to degrade performance
            try:
                # TODO: Use sklearnex instead once a suitable toggle option is provided that won't impact future models
                # FIXME: DAAL OOB score is broken, returns biased predictions. Without this optimization, can't compute Efficient OOF.
                from daal4py.sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                logger.log(15, '\tUsing daal4py RF backend...')
                self._daal = True
            except:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                self._daal = False
        else:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            self._daal = False
        if self.problem_type in [REGRESSION, SOFTCLASS]:
            return RandomForestRegressor
        else:
            return RandomForestClassifier

    # TODO: X.fillna -inf? Add extra is_missing column?
    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X

    def _set_default_params(self):
        default_params = {
            # TODO: 600 is much better, but increases info leakage in stacking -> therefore 300 is ~equal in stack ensemble final quality.
            #  Consider adding targeted noise to OOF to avoid info leakage, or increase `min_samples_leaf`.
            'n_estimators': 300,
            'n_jobs': -1,
            'random_state': 0,
            'bootstrap': True,  # Required for OOB estimates, setting to False will raise exception if bagging.
            # TODO: min_samples_leaf=5 is too large on most problems, however on some datasets it helps a lot (airlines likes >40 min_samples_leaf, adult likes 2 much better than 1)
            #  This value would need to be tuned per dataset, likely very worthwhile.
            #  Higher values = less OOF info leak, default = 1, which maximizes info leak.
            # 'min_samples_leaf': 5,  # Significantly reduces info leakage to stacker models. Never use the default/1 when using as base model.
            # 'oob_score': True,  # Disabled by default as it is better to do it post-fit via custom logic.
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # TODO: Add in documentation that Categorical default is the first index
    # TODO: enable HPO for RF models
    def _get_default_searchspace(self):
        spaces = {
            # 'n_estimators': Int(lower=10, upper=1000, default=300),
            # 'max_features': Categorical(['auto', 0.5, 0.25]),
            # 'criterion': Categorical(['gini', 'entropy']),
        }
        return spaces

    def _fit(self,
             X,
             y,
             time_limit=None,
             sample_weight=None,
             **kwargs):
        time_start = time.time()

        model_cls = self._get_model_type()

        max_memory_usage_ratio = self.params_aux['max_memory_usage_ratio']
        params = self._get_model_params()
        n_estimators_final = params['n_estimators']

        n_estimators_minimum = min(40, n_estimators_final)
        n_estimators_test = min(4, max(1, math.floor(n_estimators_minimum/5)))

        X = self.preprocess(X)
        n_estimator_increments = [n_estimators_final]

        # Very rough guess to size of a single tree before training
        if self.problem_type in [MULTICLASS, SOFTCLASS]:
            if self.num_classes is None:
                num_trees_per_estimator = 10  # Guess since it wasn't passed in, could also check y for a better value
            else:
                num_trees_per_estimator = self.num_classes
        else:
            num_trees_per_estimator = 1
        bytes_per_estimator = num_trees_per_estimator * len(X) / 60000 * 1e6  # Underestimates by 3x on ExtraTrees
        available_mem = psutil.virtual_memory().available
        expected_memory_usage = bytes_per_estimator * n_estimators_final / available_mem
        expected_min_memory_usage = bytes_per_estimator * n_estimators_minimum / available_mem
        if expected_min_memory_usage > (0.5 * max_memory_usage_ratio):  # if minimum estimated size is greater than 50% memory
            logger.warning(f'\tWarning: Model is expected to require {round(expected_min_memory_usage * 100, 2)}% of available memory (Estimated before training)...')
            raise NotEnoughMemoryError

        if n_estimators_final > n_estimators_test * 2:
            if self.problem_type == MULTICLASS:
                n_estimator_increments = [n_estimators_test, n_estimators_final]
                params['warm_start'] = True
            else:
                if expected_memory_usage > (0.05 * max_memory_usage_ratio):  # Somewhat arbitrary, consider finding a better value, should it scale by cores?
                    # Causes ~10% training slowdown, so try to avoid if memory is not an issue
                    n_estimator_increments = [n_estimators_test, n_estimators_final]
                    params['warm_start'] = True

        params['n_estimators'] = n_estimator_increments[0]
        if self._daal:
            if params.get('warm_start', False):
                params['warm_start'] = False

        model = model_cls(**params)

        time_train_start = time.time()
        for i, n_estimators in enumerate(n_estimator_increments):
            if i != 0:
                if params.get('warm_start', False):
                    model.n_estimators = n_estimators
                else:
                    params['n_estimators'] = n_estimators
                    model = model_cls(**params)
            model = model.fit(X, y, sample_weight=sample_weight)
            if (i == 0) and (len(n_estimator_increments) > 1):
                time_elapsed = time.time() - time_train_start
                model_size_bytes = 0
                for estimator in model.estimators_:  # Uses far less memory than pickling the entire forest at once
                    model_size_bytes += sys.getsizeof(pickle.dumps(estimator))
                expected_final_model_size_bytes = model_size_bytes * (n_estimators_final / model.n_estimators)
                available_mem = psutil.virtual_memory().available
                model_memory_ratio = expected_final_model_size_bytes / available_mem

                ideal_memory_ratio = 0.15 * max_memory_usage_ratio
                n_estimators_ideal = min(n_estimators_final, math.floor(ideal_memory_ratio / model_memory_ratio * n_estimators_final))

                if n_estimators_final > n_estimators_ideal:
                    if n_estimators_ideal < n_estimators_minimum:
                        logger.warning(f'\tWarning: Model is expected to require {round(model_memory_ratio*100, 2)}% of available memory...')
                        raise NotEnoughMemoryError  # don't train full model to avoid OOM error
                    logger.warning(f'\tWarning: Reducing model \'n_estimators\' from {n_estimators_final} -> {n_estimators_ideal} due to low memory. Expected memory usage reduced from {round(model_memory_ratio*100, 2)}% -> {round(ideal_memory_ratio*100, 2)}% of available memory...')

                if time_limit is not None:
                    time_expected = time_train_start - time_start + (time_elapsed * n_estimators_ideal / n_estimators)
                    n_estimators_time = math.floor((time_limit - time_train_start + time_start) * n_estimators / time_elapsed)
                    if n_estimators_time < n_estimators_ideal:
                        if n_estimators_time < n_estimators_minimum:
                            logger.warning(f'\tWarning: Model is expected to require {round(time_expected, 1)}s to train, which exceeds the maximum time limit of {round(time_limit, 1)}s, skipping model...')
                            raise TimeLimitExceeded
                        logger.warning(f'\tWarning: Reducing model \'n_estimators\' from {n_estimators_ideal} -> {n_estimators_time} due to low time. Expected time usage reduced from {round(time_expected, 1)}s -> {round(time_limit, 1)}s...')
                        n_estimators_ideal = n_estimators_time

                for j in range(len(n_estimator_increments)):
                    if n_estimator_increments[j] > n_estimators_ideal:
                        n_estimator_increments[j] = n_estimators_ideal

        self.model = model
        self.params_trained['n_estimators'] = self.model.n_estimators

    # TODO: Remove this after simplifying _predict_proba to reduce code duplication. This is only present for SOFTCLASS support.
    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)

        if self.problem_type == REGRESSION:
            return self.model.predict(X)
        elif self.problem_type == SOFTCLASS:
            return self.model.predict(X)
        elif self.problem_type == QUANTILE:
            return self.model.predict(X, quantile_levels=self.quantile_levels)

        y_pred_proba = self.model.predict_proba(X)
        return self._convert_proba_to_unified_form(y_pred_proba)

    def get_oof_pred_proba(self, X, normalize=None, **kwargs):
        """X should be the same X passed to `.fit`"""
        y_oof_pred_proba = self._get_oof_pred_proba(X=X, **kwargs)
        if normalize is None:
            normalize = self.normalize_pred_probas
        if normalize:
            y_oof_pred_proba = normalize_pred_probas(y_oof_pred_proba, self.problem_type)
        y_oof_pred_proba = y_oof_pred_proba.astype(np.float32)
        return y_oof_pred_proba

    # FIXME: Unknown if this works with quantile regression
    def _get_oof_pred_proba(self, X, y, **kwargs):
        if self._daal:
            raise AssertionError('DAAL forest backend does not support out-of-bag predictions.')
        if not self.model.bootstrap:
            raise ValueError('Forest models must set `bootstrap=True` to compute out-of-fold predictions via out-of-bag predictions.')

        # TODO: This can also be done via setting `oob_score=True` in model params,
        #  but getting the correct `pred_time_val` that way is not easy, since we can't time the internal call.
        if (getattr(self.model, "oob_decision_function_", None) is None and getattr(self.model, "oob_prediction_", None) is None) \
                and callable(getattr(self.model, "_set_oob_score", None)):
            X = self.preprocess(X)

            if getattr(self.model, "n_classes_", None) is not None:
                if self.model.n_outputs_ == 1:
                    self.model.n_classes_ = [self.model.n_classes_]
            from sklearn.tree._tree import DTYPE, DOUBLE
            X, y = self.model._validate_data(X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE)
            if y.ndim == 1:
                # reshape is necessary to preserve the data contiguity against vs
                # [:, np.newaxis] that does not.
                y = np.reshape(y, (-1, 1))
            if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
                y = np.ascontiguousarray(y, dtype=DOUBLE)
            self.model._set_oob_score(X, y)
            if getattr(self.model, "n_classes_", None) is not None:
                if self.model.n_outputs_ == 1:
                    self.model.n_classes_ = self.model.n_classes_[0]

        if getattr(self.model, "oob_decision_function_", None) is not None:
            y_oof_pred_proba = self.model.oob_decision_function_
            self.model.oob_decision_function_ = None  # save memory
        elif getattr(self.model, "oob_prediction_", None) is not None:
            y_oof_pred_proba = self.model.oob_prediction_
            self.model.oob_prediction_ = None  # save memory
        else:
            raise AssertionError(f'Model class {type(self.model)} does not support out-of-fold prediction generation.')

        # TODO: Regression does not return NaN for missing rows, instead it sets them to 0. This makes life hard.
        #  The below code corrects the missing rows to NaN instead of 0.
        # Don't bother if >60 trees, near impossible to have missing
        # If using 68% of data for training, chance of missing for each row is 1 in 11 billion.
        if self.problem_type == REGRESSION and self.model.n_estimators <= 60:
            from sklearn.ensemble._forest import _get_n_samples_bootstrap, _generate_unsampled_indices
            n_samples = len(y)

            n_predictions = np.zeros(n_samples)
            n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, self.model.max_samples)
            for estimator in self.model.estimators_:
                unsampled_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples_bootstrap)
                n_predictions[unsampled_indices] += 1
            missing_row_mask = n_predictions == 0
            y_oof_pred_proba[missing_row_mask] = np.nan

        # fill missing prediction rows with average of non-missing rows
        if np.isnan(np.sum(y_oof_pred_proba)):
            if len(y_oof_pred_proba.shape) == 1:
                col_mean = np.nanmean(y_oof_pred_proba)
                y_oof_pred_proba[np.isnan(y_oof_pred_proba)] = col_mean
            else:
                col_mean = np.nanmean(y_oof_pred_proba, axis=0)
                inds = np.where(np.isnan(y_oof_pred_proba))
                y_oof_pred_proba[inds] = np.take(col_mean, inds[1])

        return self._convert_proba_to_unified_form(y_oof_pred_proba)

    # TODO: Add HPO
    def _hyperparameter_tune(self, **kwargs):
        return skip_hpo(self, **kwargs)

    def get_model_feature_importance(self):
        if self._features is None:
            # TODO: Consider making this raise an exception
            logger.warning('Warning: get_model_feature_importance called when self._features is None!')
            return dict()
        return dict(zip(self._features, self.model.feature_importances_))

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[R_OBJECT],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, problem_type=None, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(problem_type=problem_type, **kwargs)
        if problem_type != QUANTILE:  # use_child_oof not supported in quantile regression
            extra_ag_args_ensemble = {'use_child_oof': True}
            default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _more_tags(self):
        return {'valid_oof': True}
