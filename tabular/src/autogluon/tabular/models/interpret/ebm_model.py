from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer


class EbmCallback:
    """Time limit callback for EBM."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        self.end_time: int | None = None

    def __call__(self, **kwargs):
        if self.end_time is None:
            self.end_time = time.monotonic() + self.seconds
            return False
        return time.monotonic() > self.end_time


class EBMModel(AbstractModel):
    ag_key = "EBM"
    ag_name = "ExplainableBM"

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
        num_cpus: int | str = "auto",
        **kwargs,
    ):
        # Preprocess data.
        X = self.preprocess(X)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        features = self._features
        if features is None:
            features = X.columns

        params = construct_ebm_params(
            self.problem_type,
            self._get_model_params(),
            features,
            self.stopping_metric,
            num_cpus,
            time_limit,
        )

        # Init Class
        model_cls = get_class_from_problem_type(self.problem_type)
        self.model = model_cls(random_state=self.random_seed, **params)

        # Handle validation data format for EBM
        fit_X = X
        fit_y = y
        fit_sample_weight = sample_weight
        bags = None
        if X_val is not None:
            fit_X = pd.concat([X, X_val], ignore_index=True)
            fit_y = pd.concat([y, y_val], ignore_index=True)
            if sample_weight is not None:
                fit_sample_weight = np.hstack([sample_weight, sample_weight_val])
            bags = np.full((len(fit_X), 1), 1, np.int8)
            bags[len(X) :, 0] = -1

        with warnings.catch_warnings():  # try to filter joblib warnings
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*resource_tracker: process died.*",
            )
            self.model.fit(fit_X, fit_y, sample_weight=fit_sample_weight, bags=bags)

    def _get_random_seed_from_hyperparameters(
        self, hyperparameters: dict
    ) -> int | None | str:
        return hyperparameters.get("random_state", "N/A")

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = {
            "valid_raw_types": ["int", "float", "category"],
        }
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        """EBMs do not yet support refit full."""
        return {"can_refit_full": False}

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=self._get_model_params(),
            features=self._features,
            **kwargs,
        )

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        problem_type: str,
        hyperparameters: dict | None = None,
        num_classes: int = 1,
        features=None,
        **kwargs,
    ) -> int:
        """Returns the expected peak memory usage in bytes of the EBM model during fit."""
        # TODO: we can improve the memory estimate slightly by using num_classes

        if features is None:
            features = X.columns

        params = construct_ebm_params(problem_type, hyperparameters, features)

        model_cls = get_class_from_problem_type(problem_type)

        baseline_memory_bytes = 400_000_000  # 400 MB baseline memory

        # assuming we call pd.concat([X, X_val], ignore_index=True) above, then it will be doubled
        return baseline_memory_bytes + model_cls(**params).estimate_mem(
            X, data_multiplier=2.0
        )

    def _validate_fit_memory_usage(self, mem_error_threshold: float = 1, **kwargs):
        # Given the good mem estimates with overhead, we set the threshold to 1.
        return super()._validate_fit_memory_usage(
            mem_error_threshold=mem_error_threshold, **kwargs
        )


def construct_ebm_params(
    problem_type,
    hyperparameters=None,
    features=None,
    stopping_metric=None,
    num_cpus=-1,
    time_limit=None,
):
    if hyperparameters is None:
        hyperparameters = {}

    hyperparameters = hyperparameters.copy()  # we pop values below, so copy.

    # The user can specify nominal and continuous columns.
    continuous_columns = hyperparameters.pop("continuous_columns", [])
    nominal_columns = hyperparameters.pop("nominal_columns", [])

    feature_types = None
    if features is not None:
        feature_types = []
        for c in features:
            if c in continuous_columns:
                f_type = "continuous"
            elif c in nominal_columns:
                f_type = "nominal"
            else:
                f_type = "auto"
            feature_types.append(f_type)

    # Default parameters for EBM
    params = {
        "outer_bags": 1,  # AutoGluon ensemble creates outer bags, no need for this overhead.
        "feature_names": features,
        "feature_types": feature_types,
        "n_jobs": -1 if isinstance(num_cpus, str) else num_cpus,
    }
    if stopping_metric is not None:
        params["objective"] = get_metric_from_ag_metric(
            metric=stopping_metric, problem_type=problem_type
        )
    if time_limit is not None:
        params["callback"] = EbmCallback(time_limit)

    params.update(hyperparameters)
    return params


def get_class_from_problem_type(problem_type: str):
    if problem_type in [BINARY, MULTICLASS]:
        from interpret.glassbox import ExplainableBoostingClassifier

        model_cls = ExplainableBoostingClassifier
    elif problem_type == REGRESSION:
        from interpret.glassbox import ExplainableBoostingRegressor

        model_cls = ExplainableBoostingRegressor
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
    return model_cls


def get_metric_from_ag_metric(*, metric: Scorer, problem_type: str):
    """Map AutoGluon metric to EBM metric for early stopping."""
    if problem_type in [BINARY, MULTICLASS]:
        metric_class = "log_loss"
    elif problem_type == REGRESSION:
        metric_class = "rmse"
    else:
        raise AssertionError(f"EBM does not support {problem_type} problem type.")

    return metric_class
