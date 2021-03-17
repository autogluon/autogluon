import logging

import numpy as np
import math
import psutil
import time

from autogluon.core.constants import REGRESSION
from autogluon.core.utils.exceptions import NotEnoughMemoryError
from autogluon.core.features.types import R_CATEGORY, R_OBJECT, S_TEXT_NGRAM, S_TEXT_SPECIAL, S_DATETIME_AS_INT
from autogluon.core.models.abstract.model_trial import skip_hpo
from autogluon.core.models import AbstractModel

logger = logging.getLogger(__name__)


# TODO: Normalize data!
class KNNModel(AbstractModel):
    """
    KNearestNeighbors model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """
    def _get_model_type(self):
        try:
            from ._knn_loo_variants import KNeighborsClassifier, KNeighborsRegressor
        except:
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            logger.warning('WARNING: Leave-one-out variants of KNN failed to import. Falling back to standard KNN implementations.')
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
            'weights': 'uniform',
            'n_jobs': -1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[R_CATEGORY, R_OBJECT],  # TODO: Eventually use category features
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_SPECIAL, S_DATETIME_AS_INT],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble()
        extra_ag_args_ensemble = {'use_child_oof': True}
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    # TODO: Enable HPO for KNN
    def _get_default_searchspace(self):
        spaces = {}
        return spaces

    def _fit(self,
             X,
             y,
             time_limit=None,
             sample_weight=None,
             **kwargs):
        time_start = time.time()
        X = self.preprocess(X)
        self._validate_fit_memory_usage(X=X)  # TODO: Can incorporate this into samples, can fit on portion of data to satisfy memory instead of raising exception immediately
        if sample_weight is not None:  # TODO: support
            logger.log(15, "sample_weight not yet supported for KNNModel, this model will ignore them in training.")

        num_rows_max = len(X)
        # FIXME: v0.1 Must store final num rows for refit_full or else will use everything! Worst case refit_full could train far longer than the original model.
        if time_limit is None or num_rows_max <= 10000:
            self.model = self._get_model_type()(**self.params).fit(X, y)
        else:
            self.model = self._fit_with_samples(X=X, y=y, time_limit=time_limit - (time.time() - time_start))

    def _validate_fit_memory_usage(self, X):
        max_memory_usage_ratio = self.params_aux['max_memory_usage_ratio']
        model_size_bytes = 4 * X.shape[0] * X.shape[1]  # Assuming float32 types
        expected_final_model_size_bytes = model_size_bytes * 3.6  # Roughly what can be expected of the final KNN model in memory size
        if expected_final_model_size_bytes > 10000000:  # Only worth checking if expected model size is >10MB
            available_mem = psutil.virtual_memory().available
            model_memory_ratio = expected_final_model_size_bytes / available_mem
            if model_memory_ratio > (0.15 * max_memory_usage_ratio):
                logger.warning(f'\tWarning: Model is expected to require {round(model_memory_ratio * 100, 2)}% of available memory...')
            if model_memory_ratio > (0.20 * max_memory_usage_ratio):
                raise NotEnoughMemoryError  # don't train full model to avoid OOM error

    # TODO: Won't work for RAPIDS without modification
    # TODO: support normalization?
    # TODO: Technically isn't OOF, but can be used inplace of OOF. Perhaps rename to something more accurate?
    def get_oof_pred_proba(self):
        if callable(getattr(self.model, "predict_proba_loo", None)):
            y_oof_pred_proba = self.model.predict_proba_loo()
        elif callable(getattr(self.model, "predict_loo", None)):
            y_oof_pred_proba = self.model.predict_loo()
        else:
            raise AssertionError(f'Model class {type(self.model)} does not support out-of-fold prediction generation.')
        return self._convert_proba_to_unified_form(y_oof_pred_proba)

    # TODO: Consider making this fully generic and available to all models
    def _fit_with_samples(self, X, y, time_limit):
        """
        Fit model with samples of the data repeatedly, gradually increasing the amount of data until time_limit is reached or all data is used.

        X and y must already be preprocessed
        """
        time_start = time.time()

        sample_growth_factor = 2  # Growth factor of each sample in terms of row count
        sample_time_growth_factor = 8  # Assume next sample will take 8x longer than previous (Somewhat safe but there are datasets where it is even >8x.

        num_rows_samples = []
        num_rows_max = len(X)
        num_rows_cur = 10000
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
            y_df = y.to_frame(name='label').reset_index(drop=True)
        else:
            y_df = None

        time_start_sample_loop = time.time()
        time_limit_left = time_limit - (time_start_sample_loop - time_start)
        model_type = self._get_model_type()
        for i, samples in enumerate(num_rows_samples):
            if samples != num_rows_max:
                if self.problem_type == REGRESSION:
                    idx = np.random.choice(num_rows_max, size=samples, replace=False)
                else:
                    idx = y_df.groupby('label', group_keys=False).apply(sample_func, frac=samples/num_rows_max).index
                X_samp = X[idx, :]
                y_samp = y.iloc[idx]
            else:
                X_samp = X
                y_samp = y
            self.model = model_type(**self.params).fit(X_samp, y_samp)
            time_limit_left_prior = time_limit_left
            time_fit_end_sample = time.time()
            time_limit_left = time_limit - (time_fit_end_sample - time_start)
            time_fit_sample = time_limit_left_prior - time_limit_left
            time_required_for_next = time_fit_sample * sample_time_growth_factor
            logger.log(15, f'\t{round(time_fit_sample, 2)}s \t= Train Time (Using {samples}/{num_rows_max} rows) ({round(time_limit_left, 2)}s remaining time)')
            if time_required_for_next > time_limit_left and i != len(num_rows_samples) - 1:
                logger.log(20, f'\tNot enough time to train KNN model on all training rows. Fit {samples}/{num_rows_max} rows. (Training KNN model on {num_rows_samples[i+1]} rows is expected to take {round(time_required_for_next, 2)}s)')
                break
        return self.model

    # TODO: Add HPO
    def _hyperparameter_tune(self, **kwargs):
        return skip_hpo(self, **kwargs)


class FAISSModel(KNNModel):
    def _get_model_type(self):
        from .knn_utils import FAISSNeighborsClassifier, FAISSNeighborsRegressor
        if self.problem_type == REGRESSION:
            return FAISSNeighborsRegressor
        else:
            return FAISSNeighborsClassifier

    def _set_default_params(self):
        default_params = {
            'index_factory_string': 'Flat',
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        super()._set_default_params()

    @classmethod
    def _get_default_ag_args_ensemble(cls) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble()
        extra_ag_args_ensemble = {'use_child_oof': False}
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble
