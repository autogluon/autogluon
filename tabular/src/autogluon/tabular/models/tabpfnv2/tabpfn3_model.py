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

    max_batch_size_min: int = 100_000
    """TabPFN-3 reuses the training context (KV cache) across prediction chunks, so
    chunks smaller than the training set multiply predict time while saving little
    memory — unlike TabPFN-2.5/2.6 (the base default), which re-process the joint
    train + batch sequence per chunk and benefit from a low floor."""

    _default_auxiliary_params_extra = {
        "max_rows": 500_000,
        # No feature cap: TabPFN-2.6 and -3 handle very wide data well, and on BeyondArena's
        # widest tasks (up to 22k columns) they are the two strongest methods of 28. Set to None
        # rather than removed, because the base TabPFNModel (2.5) caps at 2000 and subclass
        # `_default_auxiliary_params_extra` entries are merged base-most first, so an absent key
        # would inherit that tighter cap instead of lifting it. Memory remains bounded by
        # `_estimate_memory_usage_static`, which is what skips a fit that genuinely will not fit.
        "max_features": None,
        "max_classes": 160,
        # max_batch_size (prediction chunking) is the model's only bound on
        # test-side VRAM (peak grows linearly in unchunked prediction rows);
        # "auto" resolves at fit time to min(1M, max(100k, n_train)).
        "max_batch_size": "auto",
        "model_telemetry": False,
    }

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
        """Peak CPU RSS: ~2.8 GB process baseline plus ~10 float64 copies of the
        train + prediction-batch data made by TabPFN-3's preprocessing. Features
        count up to the model's internal 500-feature subsampling cap.

        Calibrated on synthetic fit+predict measurements (1k-800k rows, 10-2000
        features) and all 136 real TabArena and BeyondArena tasks (100 to 1M rows):
        1.03-2.5x of measured, no underestimates.
        """
        n_train, n_features = X.shape
        n_test = cls._n_test_for_memory_estimate(n_train=n_train, hyperparameters=hyperparameters)
        baseline_mem_est = 2.8e9
        preprocessing_mem_est = 10 * 8 * (n_train + n_test) * min(n_features, 500)
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

        TabPFN-3's peak is asymmetric in train vs prediction rows: train rows persist
        as attention context (~40 KB/row reserved) while prediction rows are transient
        (~26 KB/row) and bounded by the fit-time prediction batch (see
        :meth:`_n_test_for_memory_estimate`). Features cost ~18 MB each (x1.5 for
        regression), saturating at ~230 where internal subsampling caps the cost.
        Flat in ``n_estimators``.

        Regression's distributional (full-support) output costs more per prediction
        row (~110 KB, saturating to ~40 KB past 100k rows) plus buffers that grow
        with the training size to a ~2.9 GB ceiling. Calibrated on synthetic
        measurements up to 500k train rows / 800k prediction rows / 2000 features
        plus all 136 real TabArena and BeyondArena tasks (100 to 1M rows):
        1.0-4.3x of measured, no underestimates.
        """
        n_train, n_features = X.shape
        n_test = cls._n_test_for_memory_estimate(n_train=n_train, hyperparameters=hyperparameters)
        is_regression = problem_type == "regression"
        if is_regression:
            prediction_mem_est = 110e3 * min(n_test, 100_000) + 40e3 * max(n_test - 100_000, 0)
            output_buffer_mem_est = min(2.9e9, 1e6 * n_train)
        else:
            prediction_mem_est = 26e3 * n_test
            output_buffer_mem_est = 0
        return int(
            1.1e9  # CUDA context + model weights floor
            + 40e3 * n_train
            + prediction_mem_est
            + output_buffer_mem_est
            + 18e6 * min(n_features, 230) * (1.5 if is_regression else 1.0)
        )
