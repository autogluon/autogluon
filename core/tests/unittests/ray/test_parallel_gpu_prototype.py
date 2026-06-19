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
    prepare_model_resources_for_fit,
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


def test_prepare_model_resources_does_not_multiply_gpus_by_num_parallel():
    """Parent (bagged orchestrator / refit_full) GPU count must equal the per-fit count, not the
    CPU-style ``num_gpus * num_parallel`` aggregate.

    Multiplying made the parent fit see more GPUs than Ray actually made visible to it -> a model
    that selects devices by absolute index (e.g. TabPFN-3) crashed with "invalid device ordinal".
    """
    parent = object.__new__(_DummyModel)
    parent._user_params_aux = {}
    child = object.__new__(_DummyModel)
    child._user_params_aux = {}
    parent.model_base = child  # bagged: parent orchestrates a separate child model

    prepare_model_resources_for_fit(
        model=parent,
        total_num_cpus=64,
        total_num_gpus=8,
        num_cpus=4,
        num_gpus=1,
        num_parallel=2,
        num_children=2,
    )

    # CPUs ARE aggregated across the parallel folds the parent orchestrates...
    assert parent._user_params_aux["num_cpus"] == 4 * 2
    # ...but GPUs are NOT: the parent fit uses the per-child count, matching its reservation.
    assert parent._user_params_aux["num_gpus"] == 1
    assert child._user_params_aux["num_gpus"] == 1
