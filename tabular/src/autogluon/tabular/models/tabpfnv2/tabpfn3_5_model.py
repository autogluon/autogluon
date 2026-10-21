from __future__ import annotations

from typing import ClassVar

import pandas as pd

from .tabpfn3_model import (
    estimate_fit_time_v3,
    estimate_gpu_memory_v3_5,
    n_pred_for_gpu_memory_estimate_v3,
    prediction_batch_size_v3,
)
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
    time while saving little memory.
    The GPU memory check covers a prediction batch of this size (or the training size,
    if smaller), the smallest batch "auto" settles for when the free memory is short."""

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
        # "auto" predicts in the largest batch that fits the free GPU memory, at most
        # min(1M, n_train + 100k) rows.
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
    def _estimate_fit_time_static(
        cls,
        *,
        X: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        hyperparameters: dict | None = None,
        num_gpus: int | float = 0,
        **kwargs,
    ) -> float | None:
        """Seconds to fit and predict the validation rows on GPU; see `estimate_fit_time_v3`."""
        hyperparameters = hyperparameters or {}
        n_train, n_features = X.shape
        max_batch_size = hyperparameters.get("ag.max_batch_size", "auto")
        if max_batch_size == "auto":
            max_batch_size = cls._resolve_auto_max_batch_size(n_train=n_train)
        return estimate_fit_time_v3(
            n_train=n_train,
            n_val=0 if X_val is None else len(X_val),
            n_features=n_features,
            n_estimators=hyperparameters.get("n_estimators", "auto"),
            max_batch_size=max_batch_size,
            num_gpus=num_gpus,
            fit_mode=hyperparameters.get("fit_mode", "fit_preprocessors"),
        )

    @classmethod
    def _estimate_gpu_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        problem_type: str | None = None,
        X_val: pd.DataFrame | None = None,
        **kwargs,
    ) -> int:
        """Peak VRAM of the fit and the prediction batch it is checked on; see `estimate_gpu_memory_v3` and
        `n_pred_for_gpu_memory_estimate_v3`."""
        hyperparameters = hyperparameters or {}
        n_train, n_features = X.shape
        n_pred = n_pred_for_gpu_memory_estimate_v3(
            n_train=n_train,
            n_val=None if X_val is None else len(X_val),
            max_batch_size=kwargs.get("max_batch_size", hyperparameters.get("ag.max_batch_size", "auto")),
            batch_min=cls.max_batch_size_min,
        )
        return cls._estimate_gpu_memory_for_batch(
            n_train=n_train,
            n_pred=n_pred,
            n_features=n_features,
            hyperparameters=hyperparameters,
            problem_type=problem_type,
        )

    @classmethod
    def _estimate_gpu_memory_for_batch(
        cls, *, n_train: int, n_pred: int, n_features: int, hyperparameters: dict, problem_type: str | None
    ) -> int:
        """Peak VRAM of a fit on ``n_train`` rows that predicts a batch of ``n_pred`` rows; see
        `estimate_gpu_memory_v3_5`."""
        return estimate_gpu_memory_v3_5(
            n_train=n_train,
            n_pred=n_pred,
            n_features=n_features,
            n_estimators=hyperparameters.get("n_estimators", "auto"),
            problem_type=problem_type,
        )

    def _get_max_batch_size(self) -> int | None:
        return prediction_batch_size_v3(self, super()._get_max_batch_size())
