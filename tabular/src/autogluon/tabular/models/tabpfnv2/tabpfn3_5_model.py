from __future__ import annotations

from typing import ClassVar

import pandas as pd

from .tabpfnv2_5_model import TabPFNModel


class TabPFN35Model(TabPFNModel):
    """TabPFN-3.5 version: https://priorlabs.ai/.

    Technical Report: https://arxiv.org/abs/2609.17895
    One multitask checkpoint serves classification and regression; tabpfn infers the model
    version from its file name and downloads it from https://huggingface.co/Prior-Labs/tabpfn_3_5.
    Requires ``tabpfn>=9.0``.

    .. versionadded:: 1.6.4
    """

    ag_key = "TABPFN-3.5"
    ag_name = "TabPFN-3.5"
    license_noncommercial: ClassVar[bool] = True

    fixed_random_state: int = 0
    """We found that the validation score is misleading for TabPFN, when one uses a
    different random state for the refit model than for models fit during CV.
    This is because TabPFN's random state determines the preprocessing of TabPFN.
    """

    default_classification_model: str | None = "tabpfn-v3.5-20260909.safetensors"
    default_regression_model: str | None = "tabpfn-v3.5-20260909.safetensors"

    max_batch_size_min: int = 100_000
    """Every prediction chunk re-runs the forward pass over the whole training
    context, so a chunk costs about as much as a predict on the training set even
    when it holds a few rows; chunks smaller than the training set multiply predict
    time while saving little memory."""

    max_batch_size_slack: int = 100_000
    """A held-out fold of a two-fold bag is at most one row, or a few group or time
    blocks, larger than its training set; without slack that split the fold's
    prediction into a full chunk and a small second chunk that re-ran the whole
    training context. Chunking starts once the prediction set exceeds the training
    set by more than this."""

    _default_auxiliary_params_extra = {
        # The checkpoint's native limits are 1M rows, 20k features and 160 classes.
        "max_rows": 1_000_000,
        # Set to None rather than removed, because the base TabPFNModel (2.5) caps at 2000 and
        # subclass `_default_auxiliary_params_extra` entries are merged base-most first, so an
        # absent key would inherit that tighter cap. Memory remains bounded by
        # `_estimate_memory_usage_static`, which is what skips a fit that genuinely will not fit.
        "max_features": None,
        "max_classes": 160,
        # max_batch_size (prediction chunking) is the model's only bound on
        # test-side VRAM (peak grows linearly in unchunked prediction rows);
        # "auto" resolves at fit time to min(1M, n_train + 100k).
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
        """Peak CPU RSS: a ~3.6 GB process baseline plus ~10 float64 copies of the
        train + prediction-batch data, with features counted up to 2000.

        Calibrated on synthetic fit+predict measurements (2k-300k rows, 20-20k features,
        binary, multiclass and regression): 1.02-1.83x of measured, no underestimates.
        """
        n_train, n_features = X.shape
        n_test = cls._n_test_for_memory_estimate(n_train=n_train, hyperparameters=hyperparameters)
        baseline_mem_est = 3.6e9
        preprocessing_mem_est = 10 * 8 * (n_train + n_test) * min(n_features, 2000)
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

        Train rows persist as attention context (~50 KB/row for classification, ~40 KB/row
        for regression) while prediction rows are transient and bounded by the fit-time
        prediction batch (see :meth:`_n_test_for_memory_estimate`). A prediction row costs
        ~30 KB for classification and ~180 KB for regression, whose distributional output
        dominates at large prediction batches. Features cost ~17 MB each, saturating at ~1000
        where internal subsampling caps the cost.

        Calibrated on synthetic measurements up to 300k train rows / 300k prediction rows /
        20k features, including prediction batches smaller than the training set: 1.06-1.43x
        of measured, no underestimates.
        """
        n_train, n_features = X.shape
        n_test = cls._n_test_for_memory_estimate(n_train=n_train, hyperparameters=hyperparameters)
        if problem_type in ("regression", "quantile"):
            floor_mem_est, train_row_mem_est, test_row_mem_est = 2.0e9, 40e3, 180e3
        else:
            floor_mem_est, train_row_mem_est, test_row_mem_est = 1.8e9, 50e3, 30e3
        return int(
            floor_mem_est  # CUDA context + model weights floor
            + train_row_mem_est * n_train
            + test_row_mem_est * n_test
            + 17e6 * min(n_features, 1000)
        )
