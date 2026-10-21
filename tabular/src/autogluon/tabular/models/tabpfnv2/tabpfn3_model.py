from __future__ import annotations

import math
from typing import ClassVar

import pandas as pd

from .tabpfnv2_5_model import TabPFNModel


def estimate_fit_time_v3(
    *,
    n_train: int,
    n_val: int,
    n_features: int,
    n_estimators: int | str,
    max_batch_size: int | None,
    num_gpus: int | float,
    fit_mode: str = "fit_preprocessors",
) -> float | None:
    """Seconds a TabPFN-3 / 3.5 estimator takes to fit on ``n_train`` rows and predict ``n_val`` rows on GPU.

    As TabPFN-3.5's checkpoints run it: ``n_estimators="auto"`` is 8, and each estimator sees at most
    768 features plus one column the library adds. A prediction runs the network over the training
    rows and the predicted rows, in which every row attends to the training rows, so each chunk of
    ``max_batch_size`` predicted rows costs a pass over the training rows. Per estimator and row
    passed: ``1.9e-7 s`` per column, ``2.7e-10 s`` per training row and ``3.3e-6 s``, plus ``0.17 s``
    per prediction. Preprocessing costs ``4.8e-8 s`` per training cell and estimator.

    ``fit_mode`` decides where the work falls, so it matters for a fit without validation rows (a
    refit): ``"fit_preprocessors"`` preprocesses at fit, ``"low_memory"`` keeps the data at fit and
    preprocesses at each prediction, and ``"fit_with_cache"`` also runs the training rows' pass at fit,
    once, so predictions pass only the predicted rows. A fit plus a prediction costs the same in the
    first two; ``"fit_with_cache"`` was not measured.

    Calibrated on BeyondArena's hosted TabPFN-3.5 run (2,010 splits up to 1M rows and 22k features,
    8-fold bags on one RTX PRO 6000; each split's train and test time, each size band weighed the
    same): predicted / measured train time has a median of 0.96-1.07 per size band, and 0.99-1.03
    (10th-90th percentile) above 300k rows. Estimators are spread over the GPUs. None on CPU, which
    it is not calibrated for.
    """
    if not num_gpus or isinstance(num_gpus, str):
        return None
    if n_estimators == "auto":
        n_estimators = 8
    columns_per_estimator = min(n_features, 768) + 1
    estimators_per_gpu = math.ceil(n_estimators / max(1, min(int(num_gpus), n_estimators)))
    preprocess_time = 4.8e-8 * n_train * columns_per_estimator * n_estimators
    per_row = 1.9e-7 * columns_per_estimator + 2.7e-10 * n_train + 3.3e-6
    fit_time = 0.07 if fit_mode == "low_memory" else preprocess_time + 0.07
    if fit_mode == "fit_with_cache":
        fit_time += estimators_per_gpu * n_train * per_row
    if n_val == 0:
        return fit_time
    if fit_mode == "fit_with_cache":
        rows_passed = n_val
    else:
        n_chunks = 1 if max_batch_size is None else math.ceil(n_val / max_batch_size)
        rows_passed = n_chunks * n_train + n_val
    predict_time = estimators_per_gpu * rows_passed * per_row + 0.17
    if fit_mode == "low_memory":
        predict_time += preprocess_time
    return fit_time + predict_time


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
    """Every prediction chunk re-runs the forward pass over the whole training
    context, so a chunk costs about as much as a predict on the training set even
    when it holds a few rows; chunks smaller than the training set multiply predict
    time while saving little memory. TabPFN-2.5/2.6 (the base default) re-process
    the joint train + batch sequence per chunk, where a low floor bounds memory."""

    max_batch_size_slack: int = 100_000
    """A held-out fold of a two-fold bag is at most one row, or a few group or time
    blocks, larger than its training set; without slack that split the fold's
    prediction into a full chunk and a small second chunk that re-ran the whole
    training context. Chunking starts once the prediction set exceeds the training
    set by more than this."""

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
