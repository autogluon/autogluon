from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
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
    passed: ``2.1e-7 s`` per column, ``2.75e-10 s`` per training row and ``3.3e-6 s``, plus ``0.165 s``
    per prediction. Fitting preprocesses every feature: ``4.2e-8 s`` per training cell up to 1,000
    features and half that beyond, times ``n_estimators ** 0.53``, plus ``0.07 s``.

    ``fit_mode`` decides where the work falls: ``"low_memory"`` fits in 0.6 of that time and predicts
    in the same time, and ``"fit_with_cache"`` runs the training rows' pass once, at fit, so
    predictions pass only the predicted rows (not measured).

    Calibrated on BeyondArena's hosted TabPFN-3.5 run (2,010 splits up to 1M rows and 22k features,
    8-fold bags; each split's train and test time, each size band weighed the same) and on fits and
    predictions measured up to 1M training rows, 1M predicted rows and 9,284 features, all on one
    RTX PRO 6000: predicted / measured train time has a median of 0.93-1.02 per size band and
    0.98-1.01 (10th-90th percentile) above 300k rows; measured fits 0.84-1.16, predictions 0.95-1.03
    above 400k rows. Estimators are spread over the GPUs. None on CPU, which it is not calibrated for.
    """
    if not num_gpus or isinstance(num_gpus, str):
        return None
    if n_estimators == "auto":
        n_estimators = 8
    columns_per_estimator = min(n_features, 768) + 1
    estimators_per_gpu = math.ceil(n_estimators / max(1, min(int(num_gpus), n_estimators)))
    cells = n_train * (4.2e-8 * min(n_features, 1000) + 2.1e-8 * max(n_features - 1000, 0))
    fit_time = 0.07 + cells * n_estimators**0.53 * (0.6 if fit_mode == "low_memory" else 1.0)
    per_row = 2.1e-7 * columns_per_estimator + 2.75e-10 * n_train + 3.3e-6
    if fit_mode == "fit_with_cache":
        fit_time += estimators_per_gpu * n_train * per_row
    if n_val == 0:
        return fit_time
    if fit_mode == "fit_with_cache":
        rows_passed = n_val
    else:
        n_chunks = 1 if max_batch_size is None else math.ceil(n_val / max_batch_size)
        rows_passed = n_chunks * n_train + n_val
    return fit_time + estimators_per_gpu * rows_passed * per_row + 0.165


def estimate_gpu_memory_v3_5(
    *,
    n_train: int,
    n_pred: int,
    n_features: int,
    n_estimators: int | str,
    problem_type: str | None,
) -> int:
    """Peak VRAM (CUDA context and what the caching allocator needs) of a TabPFN-3.5 estimator that is fit on
    ``n_train`` rows and predicts a batch of ``n_pred`` rows.

    Fitting only places the weights on the device; the memory goes to predicting, one batch at a time against the
    whole training context. Two stages bound the peak. The cell embedding and column aggregation run in chunks of a
    fixed cell count, a working set of ~7 GB, plus a row term that matters for very narrow tables; on a tiny table
    one forward carries as many ensemble members as fit in 32,768 rows. The in-context stage costs ~26.5 KB per
    training row, ~19 KB per predicted row and ~4.5 B per cell, features counted up to 768 (each estimator's
    subsample). Regression's distributional output adds 80 KB per predicted row and, with several estimators, keeps
    ~70 KB per row through every forward. Classification is flat in ``n_estimators``. The caching allocator needs
    1.12-1.26x the allocated peak under a hard memory cap, hence the factor 1.3.

    Calibrated on one RTX PRO 6000 with TabPFN-3.5's checkpoint (tabpfn 9.2): 2k-1M training rows, 2k-1M predicted
    rows, 3-9,284 features, binary, multiclass (10-160 classes) and regression, 1-8 estimators, the
    "fit_preprocessors" and "low_memory" fit modes, and nine BeyondArena tables of ~1M rows: 1.30-1.53x the allocated
    peak from 400k training rows, and above every hard cap a fit failed under.
    """
    features = min(n_features, 768)
    rows = n_train + n_pred
    if n_estimators == "auto":
        n_estimators = 8
    members = 1 if n_estimators == 1 else min(n_estimators, max(1, 32_768 // rows))
    first_stages = (
        0.9e9
        + 4.5e3 * min(members * rows * features, 1_572_864)
        + 3.2e6 * features
        + 24e3 * min(members * rows, 1_572_864 // features)
    )
    in_context = 2.6e9 + 26.5e3 * n_train + 19e3 * n_pred + 4.5 * rows * features
    activations = max(first_stages, in_context)
    if problem_type in ("regression", "quantile"):
        if n_estimators == 1:
            activations = max(activations, 0.9e9 + 80e3 * n_pred)
        else:
            activations = max(activations + 70e3 * n_pred, 0.9e9 + 100e3 * n_pred)
    return int(0.6e9 + 1.3 * activations)


def estimate_gpu_memory_v3(*, n_train: int, n_pred: int, n_features: int, problem_type: str | None) -> int:
    """Peak VRAM (reserved + CUDA context) of a TabPFN-3 estimator on TabPFN-3's own checkpoints that is fit on
    ``n_train`` rows and predicts a batch of ``n_pred`` rows.

    Train rows persist as attention context (~40 KB/row reserved) while prediction rows are transient (~26 KB/row).
    Features cost ~18 MB each (x1.5 for regression), saturating at ~230. Regression's distributional output costs
    more per prediction row (~110 KB, saturating to ~40 KB past 100k rows) plus buffers that grow with the training
    size to a ~2.9 GB ceiling. Calibrated on synthetic measurements up to 500k train rows / 800k prediction rows /
    2000 features plus 136 real TabArena and BeyondArena tasks (100 to 1M rows).
    """
    is_regression = problem_type in ("regression", "quantile")
    if is_regression:
        prediction_mem_est = 110e3 * min(n_pred, 100_000) + 40e3 * max(n_pred - 100_000, 0)
        output_buffer_mem_est = min(2.9e9, 1e6 * n_train)
    else:
        prediction_mem_est = 26e3 * n_pred
        output_buffer_mem_est = 0
    return int(
        1.1e9  # CUDA context + model weights floor
        + 40e3 * n_train
        + prediction_mem_est
        + output_buffer_mem_est
        + 18e6 * min(n_features, 230) * (1.5 if is_regression else 1.0)
    )


def n_pred_for_gpu_memory_estimate_v3(
    *, n_train: int, n_val: int | None, max_batch_size: int | str | None, batch_min: int
) -> int:
    """The prediction batch a fit's GPU memory check covers.

    A fit with validation rows (a bag's fold) predicts them; a fit without (a refit) predicts nothing until
    inference, and is checked on a batch of its training size. ``max_batch_size="auto"`` predicts in the largest
    batch that fits the free GPU memory (`max_batch_size_for_vram_v3`), so the check covers the smallest batch it
    settles for, ``batch_min`` rows.
    """
    n_pred = n_train if n_val is None else n_val
    if max_batch_size == "auto":
        return min(n_pred, batch_min)
    if max_batch_size is None:
        return n_pred
    return min(n_pred, int(max_batch_size))


def max_batch_size_for_vram_v3(*, estimate: Callable[[int], int], available_vram: float, max_batch_size: int) -> int:
    """The largest prediction batch of at most ``max_batch_size`` rows whose ``estimate`` (peak VRAM of a batch of
    that many rows) is within 90 % of ``available_vram``, the margin of AutoGluon's GPU memory check; at least
    1,000 rows."""
    budget = 0.9 * available_vram
    if estimate(max_batch_size) <= budget:
        return max_batch_size
    low, high = min(1_000, max_batch_size), max_batch_size
    if estimate(low) > budget:
        return low
    while high - low > 1_000:
        middle = (low + high) // 2
        low, high = (middle, high) if estimate(middle) <= budget else (low, middle)
    return low


def prediction_batch_size_v3(model: TabPFNModel, max_batch_size: int | None) -> int | None:
    """``model``'s prediction batch: with ``ag.max_batch_size="auto"`` on a GPU, the largest batch of at most
    ``max_batch_size`` rows that fits the free GPU memory now (`max_batch_size_for_vram_v3`), else
    ``max_batch_size``."""
    if model.aux_params.max_batch_size != "auto" or max_batch_size is None or model.get_device() != "cuda":
        return max_batch_size
    from autogluon.common.utils.resource_utils import ResourceManager

    available_vram = ResourceManager.get_available_vram(device=model.model.devices_[0].index or 0)
    if available_vram is None:
        return max_batch_size
    hyperparameters = model._get_model_params()
    n_features = len(model.features)

    def estimate(n_pred: int) -> int:
        return model._estimate_gpu_memory_for_batch(
            n_train=model._n_train_fit,
            n_pred=n_pred,
            n_features=n_features,
            hyperparameters=hyperparameters,
            problem_type=model.problem_type,
        )

    ratio = model.params_aux.get("max_gpu_memory_usage_ratio") or 1.0
    return max_batch_size_for_vram_v3(
        estimate=estimate, available_vram=available_vram * ratio, max_batch_size=max_batch_size
    )


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
    the joint train + batch sequence per chunk, where a low floor bounds memory.
    The GPU memory check covers a prediction batch of this size (or the training size,
    if smaller), the smallest batch "auto" settles for when the free memory is short."""

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
        """Peak VRAM of a fit on ``n_train`` rows that predicts a batch of ``n_pred`` rows, for the checkpoint
        ``hyperparameters`` select: `estimate_gpu_memory_v3_5` for a TabPFN-3.5 checkpoint, and the larger of it and
        `estimate_gpu_memory_v3` for TabPFN-3's own (on those, the first misses wide regression tables and the second
        tiny tables with several estimators)."""
        estimate = estimate_gpu_memory_v3_5(
            n_train=n_train,
            n_pred=n_pred,
            n_features=n_features,
            n_estimators=hyperparameters.get("n_estimators", "auto"),
            problem_type=problem_type,
        )
        if "v3.5" in Path(cls._checkpoint_file(hyperparameters=hyperparameters, problem_type=problem_type)).name:
            return estimate
        return max(
            estimate,
            estimate_gpu_memory_v3(n_train=n_train, n_pred=n_pred, n_features=n_features, problem_type=problem_type),
        )

    @classmethod
    def _checkpoint_file(cls, *, hyperparameters: dict, problem_type: str | None) -> str:
        """The checkpoint ``hyperparameters`` select for ``problem_type``, as `_resolve_model_path` picks it."""
        zip_model_path = hyperparameters.get("zip_model_path") or {}
        if isinstance(zip_model_path, (list, tuple)) and len(zip_model_path) == 2:
            zip_model_path = dict.fromkeys(("binary", "multiclass"), zip_model_path[0]) | dict.fromkeys(
                ("regression", "quantile"), zip_model_path[1]
            )
        is_regression = problem_type in ("regression", "quantile")
        default = cls.default_regression_model if is_regression else cls.default_classification_model
        default = (cls.default_model_map or {}).get(problem_type, default)
        return str(zip_model_path.get(problem_type) or default or "")

    def _get_max_batch_size(self) -> int | None:
        return prediction_batch_size_v3(self, super()._get_max_batch_size())
