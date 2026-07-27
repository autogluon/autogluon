from __future__ import annotations

from typing import ClassVar

import pandas as pd

from .tabpfnv2_5_model import TabPFNModel


class TabPFN3Model(TabPFNModel):
    """TabPFN-3 version: https://priorlabs.ai/.

    Technical Report: https://arxiv.org/abs/2605.13986
    Requires ``tabpfn>=8.0`` for the v3 checkpoints.

    .. versionadded:: 1.6.0
    """

    ag_key = "TABPFN-3"
    ag_name = "TabPFN-3"
    license_noncommercial: ClassVar[bool] = True

    fixed_random_state: int = 0
    """We found that the validation score is misleading for TabPFN, when one uses a
    different random state for the refit model than for models fit during CV.
    This is because TabPFN's random state determines the preprocessing of TabPFN.
    """

    default_classification_model: str | None = "tabpfn-v3-classifier-v3_default.ckpt"
    default_regression_model: str | None = "tabpfn-v3-regressor-v3_default.ckpt"

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 500_000,
                "max_features": 2500,
                "max_classes": 160,
                # max_batch_size (prediction chunking) is the model's only bound on
                # test-side VRAM (peak grows linearly in unchunked prediction rows);
                # "auto" resolves at fit time to min(1M, max(100k, n_train)).
                "max_batch_size": "auto",
                "model_telemetry": False,
            }
        )
        return default_auxiliary_params

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        """The list of checkpoints to use for hyperparameter tuning."""
        raise NotImplementedError("We did not benchmark more checkpoints or tuning.")

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Peak CPU RSS: ~2.5 GB process baseline plus ~7.5 float64 copies of the
        train + prediction-batch data made by TabPFN-3's preprocessing.

        Calibrated on synthetic fit+predict measurements (1k-800k rows, 10-2000
        features); accurate within 0.9-1.5x including 500k-row x 500-feature cells.
        """
        n_train, n_features = X.shape
        n_test = cls._n_test_for_memory_estimate(n_train=n_train, hyperparameters=hyperparameters)
        baseline_mem_est = 2.5e9
        preprocessing_mem_est = 7.5 * 8 * (n_train + n_test) * n_features
        return int(baseline_mem_est + preprocessing_mem_est)

    @classmethod
    def _estimate_gpu_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Peak VRAM (reserved + CUDA context) across fit and prediction.

        TabPFN-3's peak is asymmetric in train vs prediction rows: train rows persist
        as attention context (~40 KB/row reserved) while prediction rows are transient
        (~26 KB/row) and bounded by ``ag.max_batch_size`` chunking. Features saturate
        quickly (~1.5 GB above 100). Flat in ``n_estimators``. Calibrated on synthetic
        measurements up to 500k train rows / 800k prediction rows / 2000 features.
        """
        n_train, n_features = X.shape
        n_test = cls._n_test_for_memory_estimate(n_train=n_train, hyperparameters=hyperparameters)
        return int(
            1.1e9  # CUDA context + model weights floor
            + 40e3 * n_train
            + 26e3 * n_test
            + 1.5e9 * (n_features > 100)
        )

    @classmethod
    def _class_tags(cls):
        return {**super()._class_tags(), "can_estimate_gpu_memory_usage_static": True}
