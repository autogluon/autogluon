from __future__ import annotations

from typing import ClassVar

import pandas as pd

from .tabpfnv2_5_model import TabPFNModel


class TabPFNv26Model(TabPFNModel):
    """TabPFN-2.6 version."""

    ag_key = "TABPFN-2.6"
    ag_name = "TabPFN-2.6"
    license_noncommercial: ClassVar[bool] = True

    fixed_random_state: int = 0
    """We found that the validation score is misleading for TabPFN, when one uses a
    different random state for the refit model than for models fit during CV.
    This is because TabPFN's random state determines the preprocessing of TabPFN
    """

    default_classification_model: str | None = "tabpfn-v2.6-classifier-v2.6_default.ckpt"
    default_regression_model: str | None = "tabpfn-v2.6-regressor-v2.6_default.ckpt"

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Peak CPU RSS: ~1.9 GB process baseline plus ~5 float64 copies of the
        train + prediction-batch data made by TabPFN-2.6's preprocessing.

        Calibrated on synthetic fit+predict measurements (1k-100k rows, 10-2000
        features) plus real TabArena datasets (categorical-heavy data needs a
        higher copy count than numeric-only synthetic data); accurate within
        0.9-1.2x on both.
        """
        n_train, n_features = X.shape
        n_test = cls._n_test_for_memory_estimate(n_train=n_train, hyperparameters=hyperparameters)
        baseline_mem_est = 1.9e9
        preprocessing_mem_est = 5.0 * 8 * (n_train + n_test) * n_features
        return int(baseline_mem_est + preprocessing_mem_est)

    @classmethod
    def _estimate_gpu_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        problem_type: str | None = None,
        **kwargs,
    ) -> int:
        """Peak VRAM (reserved + CUDA context) across fit and prediction.

        TabPFN-2.6 materializes the joint train + prediction-batch sequence, so peak
        VRAM is symmetric in total rows, with a two-segment per-cell (row x feature)
        slope: steep to ~300 features, shallow to the hard cap at ~1000, plus a
        ~25 KB/row per-row cost independent of feature count (dominant on narrow
        data). Peak is set by the most expensive ensemble-member preprocessing
        config, reached by ``n_estimators >= 8`` (coefficients below assume
        default-sized ensembles; n_estimators=1 peaks ~1.8x lower). Regression's
        distributional output costs ~2.5x the classification slopes. Calibrated on
        synthetic measurements (1k-100k train rows, up to 100k prediction rows,
        10-2000 features), cross-checked on real TabArena datasets.
        """
        n_train, n_features = X.shape
        n_test = cls._n_test_for_memory_estimate(n_train=n_train, hyperparameters=hyperparameters)
        total_rows = n_train + n_test
        regression_multiplier = 2.5 if problem_type == "regression" else 1.0
        return int(
            0.76e9  # CUDA context + model weights floor
            + regression_multiplier
            * (
                25e3 * total_rows
                + 3.4e3 * total_rows * min(n_features, 300)
                + 0.65e3 * total_rows * max(0, min(n_features, 1000) - 300)
            )
        )

    @classmethod
    def _class_tags(cls):
        return {**super()._class_tags(), "can_estimate_gpu_memory_usage_static": True}

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 100_000,
                "max_features": 2500,
                "max_classes": 10,
                "model_telemetry": False,
            }
        )
        return default_auxiliary_params

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        """The list of checkpoints to use for hyperparameter tuning."""
        raise NotImplementedError("We did not benchmark more checkpoints or tuning.")
