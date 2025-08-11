import copy
import os
import time
from typing import Optional

import numpy as np
from pandas import DataFrame, Series

from autogluon.common.utils.try_import import try_import_lightgbm
from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS
from autogluon.core.utils.exceptions import TimeLimitExceeded

# Mapping to specialized LightGBM metrics that are much faster than the standard metric computation
_ag_to_lgbm_metric_dict = {
    BINARY: dict(
        accuracy="binary_error",
        log_loss="binary_logloss",
        roc_auc="auc",
    ),
    MULTICLASS: dict(
        accuracy="multi_error",
        log_loss="multi_logloss",
    ),
    QUANTILE: dict(
        pinball_loss="quantile",
    ),
    REGRESSION: dict(
        mean_absolute_error="l1",
        mean_squared_error="l2",
        root_mean_squared_error="rmse",
    ),
}


def convert_ag_metric_to_lgbm(ag_metric_name, problem_type):
    return _ag_to_lgbm_metric_dict.get(problem_type, dict()).get(ag_metric_name, None)


def func_generator(metric, is_higher_better, needs_pred_proba, problem_type, error=False):
    if error:
        is_higher_better = False

    compute = metric.error if error else metric
    if problem_type in [REGRESSION, QUANTILE]:
        # TODO: Might not work for custom quantile metrics
        def function_template(y_hat, data):
            y_true = data.get_label()
            return metric.name, compute(y_true, y_hat), is_higher_better

    elif needs_pred_proba:
        if problem_type == MULTICLASS:

            def function_template(y_hat, data):
                y_true = data.get_label()
                return metric.name, compute(y_true, y_hat), is_higher_better

        elif problem_type == SOFTCLASS:  # metric must take in soft labels array, like soft_log_loss

            def function_template(y_hat, data):
                y_true = data.softlabels
                y_hat = y_hat.reshape(y_true.shape[1], -1).T
                y_hat = np.exp(y_hat)
                y_hat = np.multiply(y_hat, 1 / np.sum(y_hat, axis=1)[:, np.newaxis])
                return metric.name, compute(y_true, y_hat), is_higher_better

        else:

            def function_template(y_hat, data):
                y_true = data.get_label()
                return metric.name, compute(y_true, y_hat), is_higher_better

    else:
        if problem_type == MULTICLASS:

            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = y_hat.argmax(axis=1)
                return metric.name, compute(y_true, y_hat), is_higher_better

        else:

            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = np.round(y_hat)
                return metric.name, compute(y_true, y_hat), is_higher_better

    # allows lgb library to output autogluon metric name in the evaluation logs
    function_template.__name__ = metric.name

    return function_template


def softclass_lgbobj(preds, train_data):
    """Custom LightGBM loss function for soft (probabilistic, vector-valued) class-labels only,
    which have been appended to lgb.Dataset (train_data) as additional ".softlabels" attribute (2D numpy array).
    """
    softlabels = train_data.softlabels
    num_classes = softlabels.shape[1]
    preds = np.reshape(preds, (len(softlabels), num_classes), order="F")
    preds = np.exp(preds)
    preds = np.multiply(preds, 1 / np.sum(preds, axis=1)[:, np.newaxis])
    grad = preds - softlabels
    hess = 2.0 * preds * (1.0 - preds)
    return grad.flatten("F"), hess.flatten("F")


def construct_dataset(x: DataFrame, y: Series, location=None, reference=None, params=None, save=False, weight=None):
    try_import_lightgbm()
    import lightgbm as lgb

    dataset = lgb.Dataset(data=x, label=y, reference=reference, free_raw_data=True, params=params, weight=weight)

    if save:
        assert location is not None
        saving_path = f"{location}.bin"
        if os.path.exists(saving_path):
            os.remove(saving_path)

        os.makedirs(os.path.dirname(saving_path), exist_ok=True)
        dataset.save_binary(saving_path)
        # dataset_binary = lgb.Dataset(location + '.bin', reference=reference, free_raw_data=False)# .construct()

    return dataset


def train_lgb_model(early_stopping_callback_kwargs=None, **train_params):
    import lightgbm as lgb

    if train_params["params"]["objective"] == "quantile":
        quantile_levels = train_params["params"].pop("quantile_levels")
        booster = QuantileBooster(
            quantile_levels=quantile_levels, early_stopping_callback_kwargs=early_stopping_callback_kwargs
        )
        return booster.fit(**train_params)
    else:
        return lgb.train(**train_params)


class QuantileBooster:
    """Wrapper that trains a separate LGBM Booster for each quantile level."""

    def __init__(self, quantile_levels: list[float], early_stopping_callback_kwargs: Optional[dict] = None):
        if quantile_levels is None:
            raise AssertionError
        if not all(0 < q < 1 for q in quantile_levels):
            raise AssertionError(
                f"quantile_levels must fulfill 0 < q < 1, provided quantile_levels: {quantile_levels}"
            )

        self.quantile_levels = quantile_levels

        self.early_stopping_callback_kwargs = None
        self.time_limit_global = None

        if early_stopping_callback_kwargs is not None:
            self.early_stopping_callback_kwargs = early_stopping_callback_kwargs
            self.time_limit_global = early_stopping_callback_kwargs.pop("time_limit")
        self.model_dict = {}

    def fit(self, **train_params_base):
        import lightgbm as lgb

        from .callbacks import early_stopping_custom

        start_time_global = time.time()

        for q in self.quantile_levels:
            train_params = copy.deepcopy(train_params_base)
            train_params["params"]["alpha"] = q
            if self.early_stopping_callback_kwargs is not None:
                es_kwargs = copy.deepcopy(self.early_stopping_callback_kwargs)
                if self.time_limit_global is not None:
                    es_kwargs["start_time"] = time.time()
                    es_kwargs["time_limit"] = self.time_limit_global / len(self.quantile_levels)
                # Don't add a logging callback to avoid printing logs for each base booster
                train_params["callbacks"] = [early_stopping_custom(**es_kwargs)]
            else:
                train_params["callbacks"] = []

            self.model_dict[q] = lgb.train(**train_params)
            if self.time_limit_global is not None:
                time_left = self.time_limit_global - (time.time() - start_time_global)
                if time_left <= 0 and len(self.model_dict) != len(self.quantile_levels):
                    raise TimeLimitExceeded
        return self

    def predict(self, X, num_threads=0):
        predictions = {}
        for q in self.quantile_levels:
            predictions[q] = self.model_dict[q].predict(X, num_threads=num_threads)
        return DataFrame(predictions)

    @property
    def best_iteration(self):
        return int(np.ceil(np.mean([model.best_iteration for model in self.model_dict.values()])))

    def current_iteration(self):
        return int(np.ceil(np.mean([model.current_iteration() for model in self.model_dict.values()])))
