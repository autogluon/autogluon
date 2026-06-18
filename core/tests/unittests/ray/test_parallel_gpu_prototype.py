"""Tests for the EXPERIMENTAL GPU-parallel prototype (``AG_PARALLEL_GPU``).

Covers the flag gate and the per-child GPU sizing used by ``ParallelFitManager.schedule_jobs`` to
cap concurrent fits by available GPUs. The end-to-end parallel-fit-on-GPU path requires Ray, GPU
models, and real GPUs, so it is not exercised here.
"""

import os
from types import SimpleNamespace

import pytest

from autogluon.core.models import AbstractModel
from autogluon.core.ray.distributed_jobs_managers import (
    ParallelFitManager,
    gpu_parallel_fit_enabled,
)


@pytest.fixture
def _restore_flag():
    """Save/restore AG_PARALLEL_GPU so tests don't leak the env var."""
    prev = os.environ.get("AG_PARALLEL_GPU")
    os.environ.pop("AG_PARALLEL_GPU", None)
    yield
    if prev is None:
        os.environ.pop("AG_PARALLEL_GPU", None)
    else:
        os.environ["AG_PARALLEL_GPU"] = prev


def test_flag_disabled_by_default(_restore_flag):
    assert gpu_parallel_fit_enabled() is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [("True", True), ("False", False), ("true", False), ("1", False), ("", False)],
)
def test_flag_only_true_string_enables(_restore_flag, value, expected):
    # Matches the AG_FORCE_PARALLEL convention: strictly the string "True" enables it.
    os.environ["AG_PARALLEL_GPU"] = value
    assert gpu_parallel_fit_enabled() is expected


class _DummyModel(AbstractModel):
    pass


def _model(num_gpus=None) -> _DummyModel:
    # Skip AbstractModel.__init__; _num_gpus_per_child only needs isinstance + _user_params_aux.
    model = object.__new__(_DummyModel)
    model._user_params_aux = {} if num_gpus is None else {"num_gpus": num_gpus}
    return model


def _num_gpus_per_child(total_num_gpus, model):
    # Call the unbound method with a minimal stub for `self`.
    return ParallelFitManager._num_gpus_per_child(SimpleNamespace(total_num_gpus=total_num_gpus), model)


def test_num_gpus_per_child_reads_declared_gpus_when_enabled(_restore_flag):
    os.environ["AG_PARALLEL_GPU"] = "True"
    assert _num_gpus_per_child(total_num_gpus=8, model=_model(num_gpus=2)) == 2


def test_num_gpus_per_child_zero_when_flag_off(_restore_flag):
    os.environ.pop("AG_PARALLEL_GPU", None)
    assert _num_gpus_per_child(total_num_gpus=8, model=_model(num_gpus=2)) == 0


def test_num_gpus_per_child_zero_when_no_gpus_available(_restore_flag):
    os.environ["AG_PARALLEL_GPU"] = "True"
    assert _num_gpus_per_child(total_num_gpus=0, model=_model(num_gpus=2)) == 0


def test_num_gpus_per_child_zero_when_model_declares_none(_restore_flag):
    os.environ["AG_PARALLEL_GPU"] = "True"
    assert _num_gpus_per_child(total_num_gpus=8, model=_model(num_gpus=None)) == 0


def test_num_gpus_per_child_zero_for_non_model(_restore_flag):
    # refit mode passes a model name (str), which must not be treated as GPU-bound.
    os.environ["AG_PARALLEL_GPU"] = "True"
    assert _num_gpus_per_child(total_num_gpus=8, model="SomeModel_name") == 0
