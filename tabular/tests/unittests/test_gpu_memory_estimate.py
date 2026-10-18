from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models.dummy.dummy_model import DummyModel
from autogluon.core.utils.exceptions import NotEnoughCudaMemoryError

GB = 1024**3


class GpuDummyModel(DummyModel):
    """DummyModel that declares a fixed GPU memory estimate."""

    gpu_mem_estimate: int = 10 * GB

    def _estimate_gpu_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return self.gpu_mem_estimate


def _init_model(model_cls):
    X = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    y = pd.Series([0, 1] * 5)
    model = model_cls(path="", name=model_cls.__name__, problem_type="binary", eval_metric="accuracy")
    model.initialize(X=X, y=y)
    return model, X, y


def test__estimate_gpu_memory_usage__default_is_none():
    model, X, y = _init_model(DummyModel)
    assert model.estimate_gpu_memory_usage(X=X, y=y) is None
    # no estimate -> check is skipped even with a GPU assigned
    approx, _ = model._validate_fit_gpu_memory_usage(X=X, y=y, num_gpus=1, available_mem=1)
    assert approx is None


def test__validate_fit_gpu_memory_usage__raises_when_estimate_exceeds_vram():
    model, X, y = _init_model(GpuDummyModel)
    assert model.estimate_gpu_memory_usage(X=X, y=y) == 10 * GB
    with pytest.raises(NotEnoughCudaMemoryError):
        model._validate_fit_gpu_memory_usage(X=X, y=y, num_gpus=1, available_mem=8 * GB)


def test__validate_fit_gpu_memory_usage__passes_when_estimate_fits():
    model, X, y = _init_model(GpuDummyModel)
    approx, avail = model._validate_fit_gpu_memory_usage(X=X, y=y, num_gpus=1, available_mem=100 * GB)
    assert approx == 10 * GB
    assert avail == 100 * GB


def test__validate_fit_gpu_memory_usage__skipped_without_gpu():
    model, X, y = _init_model(GpuDummyModel)
    # num_gpus=0 -> not fitting on GPU -> no check, no raise
    model._validate_fit_gpu_memory_usage(X=X, y=y, num_gpus=0, available_mem=1)


def test__validate_fit_gpu_memory_usage__skipped_when_ratio_none():
    model, X, y = _init_model(GpuDummyModel)
    model.params_aux["max_gpu_memory_usage_ratio"] = None
    model._validate_fit_gpu_memory_usage(X=X, y=y, num_gpus=1, available_mem=1)


def test__estimate_gpu_memory_usage_static__not_implemented_by_default():
    assert DummyModel.can_estimate_gpu_memory_usage_static() is False
    X = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(NotImplementedError):
        DummyModel.estimate_gpu_memory_usage_static(X=X, y=pd.Series([0, 1]))


def test__get_available_vram__returns_bytes_or_none():
    vram = ResourceManager.get_available_vram()
    assert vram is None or (isinstance(vram, float) and vram > 0)
