import logging
import math
import time
from typing import Dict, Union

import numpy as np

from autogluon.common.features.types import R_FLOAT, R_INT, S_BOOL
from autogluon.common.utils.log_utils import fix_sklearnex_logging_if_kaggle
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel
from autogluon.core.utils.exceptions import NotEnoughMemoryError
from autogluon.core.utils.utils import normalize_pred_probas

logger = logging.getLogger(__name__)


# TODO: Normalize data!
class KNNModel(AbstractModel):
    """
    KNearestNeighbors model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._X_unused_index = None  # Keeps track of unused training data indices, necessary for LOO OOF generation

    def _get_model_type(self):
        if self.params_aux.get("use_daal", True):
            try:
                from sklearnex.neighbors import KNeighborsClassifier, KNeighborsRegressor

                fix_sklearnex_logging_if_kaggle()  # Fix logging verbosity if in Kaggle notebook environment

                # sklearnex backend for KNN seems to be 20-40x+ faster than native sklearn with no downsides.
                logger.log(15, "\tUsing sklearnex KNN backend...")
            except:
                from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        else:
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        if self.problem_type == REGRESSION:
            return KNeighborsRegressor
        else:
            return KNeighborsClassifier

    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X

    def _set_default_params(self):
        default_params = {
            "weights": "uniform",
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_INT, R_FLOAT],  # TODO: Eventually use category features
            ignored_type_group_special=[S_BOOL],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args(cls) -> dict:
        default_ag_args = super()._get_default_ag_args()
        extra_ag_args = {
            "valid_stacker": False,
            "problem_types": [BINARY, MULTICLASS, REGRESSION],
        }
        default_ag_args.update(extra_ag_args)
        return default_ag_args

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {"use_child_oof": True}
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    # TODO: Enable HPO for KNN
    def _get_default_searchspace(self):
        spaces = {}
        return spaces

    def _fit(self, X, y, num_cpus=-1, time_limit=None, sample_weight=None, **kwargs):
        time_start = time.time()
        X = self.preprocess(X)
        params = self._get_model_params()
        if "n_jobs" not in params:
            params["n_jobs"] = num_cpus
        if sample_weight is not None:  # TODO: support
            logger.log(15, "sample_weight not yet supported for KNNModel, this model will ignore them in training.")

        num_rows_max = len(X)
        # FIXME: v0.1 Must store final num rows for refit_full or else will use everything! Worst case refit_full could train far longer than the original model.
        if time_limit is None or num_rows_max <= 10000:
            self.model = self._get_model_type()(**params).fit(X, y)
        else:
            self.model = self._fit_with_samples(X=X, y=y, model_params=params, time_limit=time_limit - (time.time() - time_start))

    def _estimate_memory_usage(self, X, **kwargs):
        model_size_bytes = 4 * X.shape[0] * X.shape[1]  # Assuming float32 types
        expected_final_model_size_bytes = model_size_bytes * 3.6  # Roughly what can be expected of the final KNN model in memory size
        return expected_final_model_size_bytes

    def _validate_fit_memory_usage(self, mem_error_threshold: float = 0.2, mem_warning_threshold: float = 0.15, mem_size_threshold: int = 1e7, **kwargs):
        return super()._validate_fit_memory_usage(
            mem_error_threshold=mem_error_threshold, mem_warning_threshold=mem_warning_threshold, mem_size_threshold=mem_size_threshold, **kwargs
        )

    # TODO: Won't work for RAPIDS without modification
    # TODO: Technically isn't OOF, but can be used inplace of OOF. Perhaps rename to something more accurate?
    def predict_proba_oof(self, X, normalize=None, **kwargs):
        """X should be the same X passed to `.fit`"""
        y_oof_pred_proba = self._predict_proba_oof(X=X, **kwargs)
        if normalize is None:
            normalize = self.normalize_pred_probas
        if normalize:
            y_oof_pred_proba = normalize_pred_probas(y_oof_pred_proba, self.problem_type)
        y_oof_pred_proba = y_oof_pred_proba.astype(np.float32)
        return y_oof_pred_proba

    def _predict_proba_oof(self, X, **kwargs):
        from ._knn_loo_variants import KNeighborsClassifierLOOMixin, KNeighborsRegressorLOOMixin

        if self.problem_type in [BINARY, MULTICLASS]:
            y_oof_pred_proba = KNeighborsClassifierLOOMixin.predict_proba_loo(self.model)
        else:
            y_oof_pred_proba = KNeighborsRegressorLOOMixin.predict_loo(self.model)
        y_oof_pred_proba = self._convert_proba_to_unified_form(y_oof_pred_proba)
        if X is not None and self._X_unused_index:
            X_unused = X.iloc[self._X_unused_index]
            y_pred_proba_new = self.predict_proba(X_unused)
            X_unused_index = set(self._X_unused_index)
            num_rows = len(X)
            X_used_index = [i for i in range(num_rows) if i not in X_unused_index]
            oof_pred_shape = y_oof_pred_proba.shape
            if len(oof_pred_shape) == 1:
                y_oof_tmp = np.zeros(num_rows, dtype=np.float32)
                y_oof_tmp[X_used_index] = y_oof_pred_proba
                y_oof_tmp[self._X_unused_index] = y_pred_proba_new
            else:
                y_oof_tmp = np.zeros((num_rows, oof_pred_shape[1]), dtype=np.float32)
                y_oof_tmp[X_used_index, :] = y_oof_pred_proba
                y_oof_tmp[self._X_unused_index, :] = y_pred_proba_new
            y_oof_pred_proba = y_oof_tmp
        return y_oof_pred_proba

    # TODO: Consider making this fully generic and available to all models
    def _fit_with_samples(self, X, y, model_params, time_limit, start_samples=10000, max_samples=None, sample_growth_factor=2, sample_time_growth_factor=8):
        """
        Fit model with samples of the data repeatedly, gradually increasing the amount of data until time_limit is reached or all data is used.

        X and y must already be preprocessed.

        Parameters
        ----------
        X : np.ndarray
            The training data features (preprocessed).
        y : Series
            The training data ground truth labels.
        time_limit : float, default = None
            Time limit in seconds to adhere to when fitting model.
        start_samples : int, default = 10000
            Number of samples to start with. This will be multiplied by sample_growth_factor after each model fit to determine the next number of samples.
            For example, if start_samples=10000, sample_growth_factor=2, then the number of samples per model fit would be [10000, 20000, 40000, 80000, ...]
        max_samples : int, default = None
            The maximum number of samples to use.
            If None or greater than the number of rows in X, then it is set equal to the number of rows in X.
        sample_growth_factor : float, default = 2
            The rate of growth in sample size between each model fit. If 2, then the sample size doubles after each fit.
        sample_time_growth_factor : float, default = 8
            The multiplier to the expected fit time of the next model. If `sample_time_growth_factor=8` and a model took 10 seconds to train, the next model fit will be expected to take 80 seconds.
            If an expected time is greater than the remaining time in `time_limit`, the model will not be trained and the method will return early.
        """
        time_start = time.time()

        num_rows_samples = []
        if max_samples is None:
            num_rows_max = len(X)
        else:
            num_rows_max = min(len(X), max_samples)
        num_rows_cur = start_samples
        while True:
            num_rows_cur = min(num_rows_cur, num_rows_max)
            num_rows_samples.append(num_rows_cur)
            if num_rows_cur == num_rows_max:
                break
            num_rows_cur *= sample_growth_factor
            num_rows_cur = math.ceil(num_rows_cur)
            if num_rows_cur * 1.5 >= num_rows_max:
                num_rows_cur = num_rows_max

        def sample_func(chunk, frac):
            # Guarantee at least 1 sample (otherwise log_loss would crash or model would return different column counts in pred_proba)
            n = max(math.ceil(len(chunk) * frac), 1)
            return chunk.sample(n=n, replace=False, random_state=0)

        if self.problem_type != REGRESSION:
            y_df = y.to_frame(name="label").reset_index(drop=True)
        else:
            y_df = None

        time_start_sample_loop = time.time()
        time_limit_left = time_limit - (time_start_sample_loop - time_start)
        model_type = self._get_model_type()
        idx = None
        for i, samples in enumerate(num_rows_samples):
            if samples != num_rows_max:
                if self.problem_type == REGRESSION:
                    idx = np.random.choice(num_rows_max, size=samples, replace=False)
                else:
                    idx = y_df.groupby("label", group_keys=False).apply(sample_func, frac=samples / num_rows_max).index
                X_samp = X[idx, :]
                y_samp = y.iloc[idx]
            else:
                X_samp = X
                y_samp = y
                idx = None
            self.model = model_type(**model_params).fit(X_samp, y_samp)
            time_limit_left_prior = time_limit_left
            time_fit_end_sample = time.time()
            time_limit_left = time_limit - (time_fit_end_sample - time_start)
            time_fit_sample = time_limit_left_prior - time_limit_left
            time_required_for_next = time_fit_sample * sample_time_growth_factor
            logger.log(15, f"\t{round(time_fit_sample, 2)}s \t= Train Time (Using {samples}/{num_rows_max} rows) ({round(time_limit_left, 2)}s remaining time)")
            if time_required_for_next > time_limit_left and i != len(num_rows_samples) - 1:
                logger.log(
                    20,
                    f"\tNot enough time to train KNN model on all training rows. Fit {samples}/{num_rows_max} rows. (Training KNN model on {num_rows_samples[i+1]} rows is expected to take {round(time_required_for_next, 2)}s)",
                )
                break
        if idx is not None:
            idx = set(idx)
            self._X_unused_index = [i for i in range(num_rows_max) if i not in idx]
        return self.model

    def _get_maximum_resources(self) -> Dict[str, Union[int, float]]:
        # use at most 32 cpus to avoid OpenBLAS error: https://github.com/autogluon/autogluon/issues/1020
        return {"num_cpus": 32}

    def _get_default_resources(self):
        # use at most 32 cpus to avoid OpenBLAS error: https://github.com/autogluon/autogluon/issues/1020
        num_cpus = ResourceManager.get_cpu_count()
        num_gpus = 0
        return num_cpus, num_gpus

    def _more_tags(self):
        return {
            "valid_oof": True,
            "can_refit_full": True,
        }


class FAISSModel(KNNModel):
    def _get_model_type(self):
        from .knn_utils import FAISSNeighborsClassifier, FAISSNeighborsRegressor

        if self.problem_type == REGRESSION:
            return FAISSNeighborsRegressor
        else:
            return FAISSNeighborsClassifier

    def _set_default_params(self):
        default_params = {
            "index_factory_string": "Flat",
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        super()._set_default_params()

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {"use_child_oof": False}
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _more_tags(self):
        return {"valid_oof": False}
