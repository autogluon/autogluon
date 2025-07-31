from __future__ import annotations

import logging
import math
import pickle
import sys
import time

import numpy as np
import pandas as pd

from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS
from autogluon.core.models import AbstractModel
from autogluon.core.utils.exceptions import NotEnoughMemoryError, TimeLimitExceeded
from autogluon.core.utils.utils import normalize_pred_probas
from autogluon.features.generators import LabelEncoderFeatureGenerator

from .compilers.native import RFNativeCompiler
from .compilers.onnx import RFOnnxCompiler

logger = logging.getLogger(__name__)


class RFModel(AbstractModel):
    """
    Random Forest model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    ag_key = "RF"
    ag_name = "RandomForest"
    ag_priority = 80

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._daal = False  # Whether daal4py backend is being used
        self._num_features_post_process = None

    # noinspection PyUnresolvedReferences
    def _get_model_type(self):
        if self.problem_type == QUANTILE:
            from .rf_quantile import RandomForestQuantileRegressor

            return RandomForestQuantileRegressor
        if self.params_aux.get("use_daal", False):
            # Disabled by default because OOB score does not yet work properly
            try:
                # FIXME: sklearnex OOB score is broken, returns biased predictions. Without this optimization, can't compute Efficient OOF.
                #  Refer to https://github.com/intel/scikit-learn-intelex/issues/933
                #  Current workaround: Forcibly set oob_score=True during fit to compute OOB during train time.
                #  Downsides:
                #    1. Slows down training slightly by forcing computation of OOB even if OOB is not needed (such as in medium_quality)
                #    2. Makes computing the correct pred_time_val difficult, as the time is instead added to the fit_time,
                #       and we would need to waste extra time to compute the proper pred_time_val post-fit.
                #       Therefore with sklearnex enabled, pred_time_val is incorrect.
                from sklearnex.ensemble import RandomForestClassifier, RandomForestRegressor

                logger.log(15, "\tUsing sklearnex RF backend...")
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
            "n_estimators": 300,
            # Cap leaf nodes to 15000 to avoid large datasets using unreasonable amounts of memory/disk for RF/XT.
            #  Ensures that memory and disk usage of RF model with 300 n_estimators is at most ~500 MB for binary/regression, ~200 MB per class for multiclass.
            #  This has no effect on datasets with <=15000 rows, and minimal to no impact on datasets with <50000 rows.
            #  For large datasets, will often make the model worse, but will significantly speed up inference speed and massively reduce memory and disk usage.
            #  For example, when left uncapped, RF can use 5 GB of disk for a regression dataset with 2M rows.
            #  Multiply by the 8 RF/XT models in config for best quality / high quality and this is 40 GB of tree models, which is unreasonable.
            #  This size scales linearly with number of rows.
            "max_leaf_nodes": 15000,
            "n_jobs": -1,
            "bootstrap": True,  # Required for OOB estimates, setting to False will raise exception if bagging.
            # TODO: min_samples_leaf=5 is too large on most problems, however on some datasets it helps a lot (airlines likes >40 min_samples_leaf, adult likes 2 much better than 1)
            #  This value would need to be tuned per dataset, likely very worthwhile.
            #  Higher values = less OOF info leak, default = 1, which maximizes info leak.
            # 'min_samples_leaf': 5,  # Significantly reduces info leakage to stacker models. Never use the default/1 when using as base model.
            # 'oob_score': True,  # Disabled by default as it is better to do it post-fit via custom logic.
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_random_seed_from_hyperparameters(self, hyperparameters: dict | None = None) -> int | None | str:
        return hyperparameters.get("random_state", "N/A")

    # TODO: Add in documentation that Categorical default is the first index
    # TODO: enable HPO for RF models
    def _get_default_searchspace(self):
        spaces = {
            # 'n_estimators': Int(lower=10, upper=1000, default=300),
            # 'max_features': Categorical(['auto', 0.5, 0.25]),
            # 'criterion': Categorical(['gini', 'entropy']),
        }
        return spaces

    def _get_num_trees_per_estimator(self) -> int:
        return self._get_num_trees_per_estimator_static(problem_type=self.problem_type, num_classes=self.num_classes)

    @classmethod
    def _get_num_trees_per_estimator_static(cls, problem_type: str, num_classes: int | None) -> int:
        # Very rough guess to size of a single tree before training
        if problem_type in [MULTICLASS, SOFTCLASS]:
            if num_classes is None:
                num_trees_per_estimator = 10  # Guess since it wasn't passed in, could also check y for a better value
            else:
                num_trees_per_estimator = num_classes
        else:
            num_trees_per_estimator = 1
        return num_trees_per_estimator

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(X=X, problem_type=self.problem_type, num_classes=self.num_classes, hyperparameters=hyperparameters, **kwargs)

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict = None,
        problem_type: str = None,
        num_classes: int = 1,
        **kwargs,
    ) -> int:
        if hyperparameters is None:
            hyperparameters = {}
        n_estimators_final = hyperparameters.get("n_estimators", 300)
        if isinstance(n_estimators_final, int):
            n_estimators_minimum = min(40, n_estimators_final)
        else:  # if search space
            n_estimators_minimum = 40
        num_trees_per_estimator = cls._get_num_trees_per_estimator_static(problem_type=problem_type, num_classes=num_classes)
        bytes_per_estimator = num_trees_per_estimator * len(X) / 60000 * 1e6  # Underestimates by 3x on ExtraTrees
        expected_min_memory_usage = int(bytes_per_estimator * n_estimators_minimum)
        return expected_min_memory_usage

    def _validate_fit_memory_usage(self, mem_error_threshold: float = 0.5, mem_warning_threshold: float = 0.4, mem_size_threshold: int = 1e7, **kwargs):
        return super()._validate_fit_memory_usage(
            mem_error_threshold=mem_error_threshold, mem_warning_threshold=mem_warning_threshold, mem_size_threshold=mem_size_threshold, **kwargs
        )

    def _expected_mem_usage(self, n_estimators_final, bytes_per_estimator):
        available_mem = ResourceManager.get_available_virtual_mem()
        return n_estimators_final * bytes_per_estimator / available_mem

    def _fit(self, X, y, num_cpus=-1, time_limit=None, sample_weight=None, **kwargs):
        time_start = time.time()

        model_cls = self._get_model_type()

        max_memory_usage_ratio = self.params_aux["max_memory_usage_ratio"]
        params = self._get_model_params()
        if "n_jobs" not in params:
            params["n_jobs"] = num_cpus
        n_estimators_final = params["n_estimators"]

        n_estimators_minimum = min(40, n_estimators_final)
        n_estimators_test = min(4, max(1, math.floor(n_estimators_minimum / 5)))

        X = self.preprocess(X)
        n_estimator_increments = [n_estimators_final]

        num_trees_per_estimator = self._get_num_trees_per_estimator()
        bytes_per_estimator = num_trees_per_estimator * len(X) / 60000 * 1e6  # Underestimates by 3x on ExtraTrees
        expected_memory_usage = self._expected_mem_usage(n_estimators_final, bytes_per_estimator)

        if n_estimators_final > n_estimators_test * 2:
            if self.problem_type == MULTICLASS:
                n_estimator_increments = [n_estimators_test, n_estimators_final]
                params["warm_start"] = True
            else:
                if expected_memory_usage > (0.05 * max_memory_usage_ratio):  # Somewhat arbitrary, consider finding a better value, should it scale by cores?
                    # Causes ~10% training slowdown, so try to avoid if memory is not an issue
                    n_estimator_increments = [n_estimators_test, n_estimators_final]
                    params["warm_start"] = True

        params["n_estimators"] = n_estimator_increments[0]
        if self._daal:
            if params.get("warm_start", False):
                params["warm_start"] = False
            # FIXME: This is inefficient but sklearnex doesn't support computing oob_score after training
            params["oob_score"] = True

        model = model_cls(random_state=self.random_seed, **params)

        time_train_start = time.time()
        for i, n_estimators in enumerate(n_estimator_increments):
            if i != 0:
                if params.get("warm_start", False):
                    model.n_estimators = n_estimators
                else:
                    params["n_estimators"] = n_estimators
                    model = model_cls(**params)
            model = model.fit(X, y, sample_weight=sample_weight)
            if (i == 0) and (len(n_estimator_increments) > 1):
                time_elapsed = max(time.time() - time_train_start, 0.001)  # avoid it being too small and being truncated to 0
                model_size_bytes = 0
                for estimator in model.estimators_:  # Uses far less memory than pickling the entire forest at once
                    model_size_bytes += sys.getsizeof(pickle.dumps(estimator))
                expected_final_model_size_bytes = model_size_bytes * (n_estimators_final / model.n_estimators)
                available_mem = ResourceManager.get_available_virtual_mem()
                model_memory_ratio = expected_final_model_size_bytes / available_mem

                ideal_memory_ratio = 0.15 * max_memory_usage_ratio
                n_estimators_ideal = min(n_estimators_final, math.floor(ideal_memory_ratio / model_memory_ratio * n_estimators_final))

                if n_estimators_final > n_estimators_ideal:
                    if n_estimators_ideal < n_estimators_minimum:
                        logger.warning(f"\tWarning: Model is expected to require {round(model_memory_ratio*100, 2)}% of available memory...")
                        raise NotEnoughMemoryError  # don't train full model to avoid OOM error
                    logger.warning(
                        f"\tWarning: Reducing model 'n_estimators' from {n_estimators_final} -> {n_estimators_ideal} due to low memory. Expected memory usage reduced from {round(model_memory_ratio*100, 2)}% -> {round(ideal_memory_ratio*100, 2)}% of available memory..."
                    )

                if time_limit is not None:
                    time_expected = time_train_start - time_start + (time_elapsed * n_estimators_ideal / n_estimators)
                    n_estimators_time = math.floor((time_limit - time_train_start + time_start) * n_estimators / time_elapsed)
                    if n_estimators_time < n_estimators_ideal:
                        if n_estimators_time < n_estimators_minimum:
                            logger.warning(
                                f"\tWarning: Model is expected to require {round(time_expected, 1)}s to train, which exceeds the maximum time limit of {round(time_limit, 1)}s, skipping model..."
                            )
                            raise TimeLimitExceeded
                        logger.warning(
                            f"\tWarning: Reducing model 'n_estimators' from {n_estimators_ideal} -> {n_estimators_time} due to low time. Expected time usage reduced from {round(time_expected, 1)}s -> {round(time_limit, 1)}s..."
                        )
                        n_estimators_ideal = n_estimators_time

                for j in range(len(n_estimator_increments)):
                    if n_estimator_increments[j] > n_estimators_ideal:
                        n_estimator_increments[j] = n_estimators_ideal
        self.model = model
        self.params_trained["n_estimators"] = self.model.n_estimators

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

    def predict_proba_oof(self, X, normalize=None, **kwargs):
        """X should be the same X passed to `.fit`"""
        y_oof_pred_proba = self._predict_proba_oof(X=X, **kwargs)
        if normalize is None:
            normalize = self.normalize_pred_probas
        if normalize:
            y_oof_pred_proba = normalize_pred_probas(y_oof_pred_proba, self.problem_type)
        y_oof_pred_proba = y_oof_pred_proba.astype(np.float32)
        return y_oof_pred_proba

    def _is_sklearn_1(self) -> bool:
        """Returns True if the trained model is from sklearn>=1.0"""
        return callable(getattr(self.model, "_set_oob_score_and_attributes", None))

    def _model_supports_oob_pred_proba(self) -> bool:
        """Returns True if model supports computing out-of-bag prediction probabilities"""
        # TODO: Remove `_set_oob_score` after sklearn version requirement is >=1.0
        return callable(getattr(self.model, "_set_oob_score", None)) or self._is_sklearn_1()

    # FIXME: Unknown if this works with quantile regression
    def _predict_proba_oof(self, X, y, **kwargs):
        if not self.model.bootstrap:
            raise ValueError("Forest models must set `bootstrap=True` to compute out-of-fold predictions via out-of-bag predictions.")

        oob_is_not_set = getattr(self.model, "oob_decision_function_", None) is None and getattr(self.model, "oob_prediction_", None) is None

        if oob_is_not_set and self._daal:
            raise AssertionError("DAAL forest backend does not support out-of-bag predictions.")

        # TODO: This can also be done via setting `oob_score=True` in model params,
        #  but getting the correct `pred_time_val` that way is not easy, since we can't time the internal call.
        if oob_is_not_set and self._model_supports_oob_pred_proba():
            X = self.preprocess(X)

            if getattr(self.model, "n_classes_", None) is not None:
                if self.model.n_outputs_ == 1:
                    self.model.n_classes_ = [self.model.n_classes_]
            from sklearn.tree._tree import DOUBLE, DTYPE
            from sklearn.utils.validation import check_X_y

            X, y = check_X_y(X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE)
            if y.ndim == 1:
                # reshape is necessary to preserve the data contiguity against vs
                # [:, np.newaxis] that does not.
                y = np.reshape(y, (-1, 1))
            if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
                y = np.ascontiguousarray(y, dtype=DOUBLE)
            if self._is_sklearn_1():
                # sklearn >= 1.0
                # TODO: Can instead do `_compute_oob_predictions` but requires post-processing. Skips scoring func.
                self.model._set_oob_score_and_attributes(X, y)
            else:
                # sklearn < 1.0
                # TODO: Remove once sklearn < 1.0 support is dropped
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
            raise AssertionError(f"Model class {type(self.model)} does not support out-of-fold prediction generation.")

        # TODO: Regression does not return NaN for missing rows, instead it sets them to 0. This makes life hard.
        #  The below code corrects the missing rows to NaN instead of 0.
        # Don't bother if >60 trees, near impossible to have missing
        # If using 68% of data for training, chance of missing for each row is 1 in 11 billion.
        if self.problem_type == REGRESSION and self.model.n_estimators <= 60:
            from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap

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

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, problem_type=None, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(problem_type=problem_type, **kwargs)
        if problem_type != QUANTILE:  # use_child_oof not supported in quantile regression
            extra_ag_args_ensemble = {"use_child_oof": True}
            default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression", "quantile", "softclass"]

    @classmethod
    def _class_tags(cls):
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self):
        # `can_refit_full=True` because final n_estimators is communicated at end of `_fit`:
        #  `self.params_trained['n_estimators'] = self.model.n_estimators`
        tags = {"can_refit_full": True}
        if self.problem_type == QUANTILE:
            tags["valid_oof"] = False  # not supported in quantile regression
        else:
            tags["valid_oof"] = True
        return tags

    @classmethod
    def _valid_compilers(cls):
        return [RFNativeCompiler, RFOnnxCompiler]

    @classmethod
    def _default_compiler(cls):
        return RFNativeCompiler
