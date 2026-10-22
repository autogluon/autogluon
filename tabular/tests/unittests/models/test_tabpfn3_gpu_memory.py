from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from autogluon.tabular.models import TabPFN3Model, TabPFN35Model
from autogluon.tabular.models.tabpfnv2.tabpfn3_model import (
    estimate_gpu_memory_v3_5,
    max_batch_size_for_vram_v3,
    n_pred_for_gpu_memory_estimate_v3,
    prediction_batch_size_v3,
)

GB = 1e9
GiB = 2**30
#: Free VRAM a fresh process sees on a 96 GB RTX PRO 6000.
RTX_PRO_6000_FREE = 94.4 * GiB


class _Shape:
    """Stands in for a frame in the static estimates, which read only its shape and length."""

    def __init__(self, n_rows: int, n_features: int):
        self.shape = (n_rows, n_features)

    def __len__(self) -> int:
        return self.shape[0]


@pytest.mark.parametrize(
    "problem_type, n_train, n_pred, n_features, n_estimators, allocated_gb",
    [
        ("regression", 875_000, 125_000, 230, 1, 28.3),
        ("regression", 875_000, 125_000, 230, 8, 37.1),
        ("binary", 875_000, 125_000, 1000, 1, 30.7),
        ("regression", 1_000_000, 250_000, 230, 1, 33.5),
        ("regression", 1_000_000, 250_000, 1000, 1, 36.3),
        ("regression", 1_000_000, 1_000_000, 230, 1, 80.9),
        ("binary", 1_000_000, 1_000_000, 230, 1, 43.9),
    ],
)
def test_estimate_gpu_memory_v3_covers_measured_peaks(
    problem_type, n_train, n_pred, n_features, n_estimators, allocated_gb
):
    """Allocated peaks measured on one RTX PRO 6000 with TabPFN-3.5's checkpoint: the estimate keeps the 1.3x the
    caching allocator needs under a hard cap, without the 1.6-2.6x it reserves on a free device."""
    estimate = estimate_gpu_memory_v3_5(
        n_train=n_train, n_pred=n_pred, n_features=n_features, n_estimators=n_estimators, problem_type=problem_type
    )
    assert 1.3 <= estimate / (allocated_gb * GB) <= 1.6


TABPFN_3_5 = {
    "zip_model_path": dict.fromkeys(("binary", "multiclass", "regression"), "tabpfn-v3.5-20260909.safetensors")
}


@pytest.mark.parametrize("model_cls, hyperparameters", [(TabPFN3Model, TABPFN_3_5), (TabPFN35Model, {})])
@pytest.mark.parametrize("n_features", [103, 225, 990])
def test_refit_on_a_million_regression_rows_fits_a_96gb_gpu(model_cls, hyperparameters, n_features):
    """A refit on 1M rows of a regression table is checked on a 100k-row prediction batch (allocated 33-36 GB as a
    250k-row batch), within 90 % of the free memory of a 96 GB GPU."""
    estimate = model_cls._estimate_gpu_memory_usage_static(
        X=_Shape(1_000_000, n_features), problem_type="regression", hyperparameters=hyperparameters
    )
    assert estimate < 0.9 * RTX_PRO_6000_FREE


@pytest.mark.parametrize(
    "problem_type, n_train, n_pred, n_features, n_estimators, allocated_gb",
    [
        ("regression", 1_000_000, 250_000, 1000, 1, 47.5),
        ("regression", 1_000_000, 250_000, 230, 1, 22.1),
        ("binary", 1_000_000, 250_000, 230, 1, 22.9),
        ("binary", 2_000, 2_000, 100, 8, 6.7),
        ("regression", 2_000, 2_000, 100, 8, 6.9),
        ("regression", 400_000, 100_000, 100, 8, 17.0),
    ],
)
def test_tabpfn3_checkpoints_take_the_larger_estimate(
    problem_type, n_train, n_pred, n_features, n_estimators, allocated_gb
):
    """TabPFN-3's own checkpoints, measured like TabPFN-3.5's: TabPFN-3.5's estimate misses wide regression tables
    and the previous TabPFN-3 estimate tiny tables with several estimators, so they take the larger of the two."""
    kwargs = dict(n_train=n_train, n_pred=n_pred, n_features=n_features, problem_type=problem_type)
    hyperparameters = {"n_estimators": n_estimators}
    tabpfn3 = TabPFN3Model._estimate_gpu_memory_for_batch(hyperparameters=hyperparameters, **kwargs)
    tabpfn3_5 = TabPFN3Model._estimate_gpu_memory_for_batch(
        hyperparameters={**hyperparameters, **TABPFN_3_5}, **kwargs
    )
    assert tabpfn3 >= 1.3 * allocated_gb * GB
    assert tabpfn3_5 == estimate_gpu_memory_v3_5(n_estimators=n_estimators, **kwargs) <= tabpfn3


def test_n_pred_for_gpu_memory_estimate_v3():
    def n_pred(n_train, n_val, max_batch_size):
        return n_pred_for_gpu_memory_estimate_v3(
            n_train=n_train, n_val=n_val, max_batch_size=max_batch_size, batch_min=100_000
        )

    # a fold predicts its validation rows; "auto" is checked on the smallest batch it settles for
    assert n_pred(875_000, 125_000, "auto") == 100_000
    assert n_pred(875_000, 50_000, "auto") == 50_000
    # a refit predicts nothing until inference, and is checked on a batch of its training size
    assert n_pred(1_000_000, None, "auto") == 100_000
    assert n_pred(500, None, "auto") == 500
    assert n_pred(1_000_000, None, 150_000) == 150_000
    assert n_pred(875_000, 125_000, 150_000) == 125_000
    # no chunking: every row at once
    assert n_pred(1_000_000, None, None) == 1_000_000


def test_max_batch_size_for_vram_v3():
    kwargs = dict(n_train=1_000_000, n_features=230, n_estimators=8, problem_type="regression")

    def estimate(n_pred: int) -> int:
        return estimate_gpu_memory_v3_5(n_pred=n_pred, **kwargs)

    batch = max_batch_size_for_vram_v3(estimate=estimate, available_vram=RTX_PRO_6000_FREE, max_batch_size=1_000_000)
    assert estimate(batch) <= 0.9 * RTX_PRO_6000_FREE < estimate(batch + 1_000)
    assert 300_000 < batch < 600_000
    # a batch that fits is kept; the smallest batch is 1,000 rows
    assert max_batch_size_for_vram_v3(estimate=estimate, available_vram=1e12, max_batch_size=600_000) == 600_000
    assert max_batch_size_for_vram_v3(estimate=estimate, available_vram=1e9, max_batch_size=600_000) == 1_000


def test_gpu_memory_estimate_reads_ag_max_batch_size():
    """`ag.max_batch_size` reaches the estimate, though `_get_model_params()` leaves out the `ag.` parameters."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1_000, 5)))
    y = pd.Series(rng.normal(size=1_000))

    def estimate(hyperparameters: dict) -> int:
        model = TabPFN3Model(path="", name="m", problem_type="regression", hyperparameters=hyperparameters)
        model.initialize(X=X, y=y)
        return model.estimate_gpu_memory_usage(X=X)

    assert estimate({"ag.max_batch_size": 100}) < estimate({})


def _fitted(max_batch_size, device: str = "cuda") -> SimpleNamespace:
    import torch

    return SimpleNamespace(
        aux_params=SimpleNamespace(max_batch_size=max_batch_size),
        get_device=lambda: device,
        model=SimpleNamespace(devices_=[torch.device(device, 0) if device == "cuda" else torch.device(device)]),
        _n_train_fit=1_000_000,
        features=[f"f{i}" for i in range(230)],
        _get_model_params=lambda: {},
        _estimate_gpu_memory_for_batch=TabPFN35Model._estimate_gpu_memory_for_batch,
        problem_type="regression",
        params_aux={"max_gpu_memory_usage_ratio": 1.0},
    )


def test_prediction_batch_size_v3_sizes_auto_to_the_free_gpu_memory():
    vram = "autogluon.common.utils.resource_utils.ResourceManager.get_available_vram"
    with patch(vram, return_value=RTX_PRO_6000_FREE):
        batch = prediction_batch_size_v3(_fitted("auto"), 1_000_000)
        # an explicit batch, and a fit on the CPU, keep the given size
        assert prediction_batch_size_v3(_fitted(150_000), 150_000) == 150_000
        assert prediction_batch_size_v3(_fitted("auto", device="cpu"), 1_000_000) == 1_000_000
    expected = max_batch_size_for_vram_v3(
        estimate=lambda n_pred: estimate_gpu_memory_v3_5(
            n_train=1_000_000, n_pred=n_pred, n_features=230, n_estimators="auto", problem_type="regression"
        ),
        available_vram=RTX_PRO_6000_FREE,
        max_batch_size=1_000_000,
    )
    assert batch == expected < 1_000_000
