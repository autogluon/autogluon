"""Tests for the EXPERIMENTAL GPU-parallel prototype (``AG_PARALLEL_GPU``).

Covers the flag gate and the per-child GPU sizing used by ``ParallelFitManager.schedule_jobs`` to
cap concurrent fits by available GPUs. The end-to-end parallel-fit-on-GPU path requires Ray, GPU
models, and real GPUs, so it is not exercised here.
"""

import os
from types import SimpleNamespace

import pytest

from autogluon.core.models import AbstractModel, BaggedEnsembleModel
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


# ---------------------------------------------------------------------------
# Strategy-aware GPU reservation in get_resources_for_model_fit
# ---------------------------------------------------------------------------
def _bagged_model(*, num_gpus=1, fold_fitting_strategy=None, refit_folds=False, num_cpus=4):
    """A minimal bagged-model stub for get_resources_for_model_fit (no __init__/ray)."""
    bag = object.__new__(BaggedEnsembleModel)
    bag._user_params = {"refit_folds": refit_folds}
    if fold_fitting_strategy is not None:
        bag._user_params["fold_fitting_strategy"] = fold_fitting_strategy
    bag._user_params_aux = {"num_cpus": num_cpus * 2}  # parent (post prepare *num_parallel)
    child = object.__new__(_DummyModel)
    child._user_params_aux = {"num_cpus": num_cpus, "num_gpus": num_gpus}
    bag.model_base = child
    return bag


def _resources_for_fit(model, *, num_splits=2):
    stub = SimpleNamespace(num_splits=num_splits, max_cpu_resources_per_node=64, total_num_cpus=64)
    return ParallelFitManager.get_resources_for_model_fit(stub, model=model)


def test_sequential_local_reserves_only_model_worker_gpu(_restore_flag):
    """sequential_local (flag on): no nested fold-workers, so total GPUs == the model-worker's
    per-fold GPU (not num_gpus * num_splits)."""
    os.environ["AG_PARALLEL_GPU"] = "True"
    res = _resources_for_fit(_bagged_model(num_gpus=1, fold_fitting_strategy="sequential_local", refit_folds=True))
    assert res.num_gpus_for_model_worker == 1
    assert res.num_gpus_for_fold_worker == 0
    assert res.total_num_gpus == 1  # was 1 + 1*num_splits = 3 before the fix


def test_sequential_local_gives_gpu_even_without_refit(_restore_flag):
    """sequential_local + refit_folds=False (flag on): the in-process model-worker still gets a GPU
    (previously it reserved 0 -> CUDA error for the in-process folds)."""
    os.environ["AG_PARALLEL_GPU"] = "True"
    res = _resources_for_fit(_bagged_model(num_gpus=1, fold_fitting_strategy="sequential_local", refit_folds=False))
    assert res.num_gpus_for_model_worker == 1
    assert res.total_num_gpus == 1


def test_parallel_local_reservation_unchanged(_restore_flag):
    """parallel_local keeps the nested-fold-worker reservation (model-worker GPU only for refit)."""
    os.environ["AG_PARALLEL_GPU"] = "True"
    res_refit = _resources_for_fit(
        _bagged_model(num_gpus=1, fold_fitting_strategy="parallel_local", refit_folds=True), num_splits=2
    )
    assert res_refit.num_gpus_for_model_worker == 1  # refit_full on the worker
    assert res_refit.num_gpus_for_fold_worker == 1
    assert res_refit.total_num_gpus == 1 + 1 * 2  # = 3

    res_norefit = _resources_for_fit(
        _bagged_model(num_gpus=1, fold_fitting_strategy="parallel_local", refit_folds=False), num_splits=2
    )
    assert res_norefit.num_gpus_for_model_worker == 0
    assert res_norefit.total_num_gpus == 0 + 1 * 2  # = 2


def test_sequential_local_unchanged_when_flag_off(_restore_flag):
    """With the flag off, sequential_local is NOT special-cased (historical parallel reservation)."""
    os.environ.pop("AG_PARALLEL_GPU", None)
    res = _resources_for_fit(_bagged_model(num_gpus=1, fold_fitting_strategy="sequential_local", refit_folds=True))
    # Falls through to the parallel branch: 1 (refit) + 1*num_splits.
    assert res.total_num_gpus == 1 + 1 * 2


# ---------------------------------------------------------------------------
# Concurrent-children sizing (CPU/GPU/memory reservation for sequential_local)
# ---------------------------------------------------------------------------
def _concurrent(model, num_children=8):
    # `self` is unused by _num_concurrent_children; a bare stub suffices.
    return ParallelFitManager._num_concurrent_children(SimpleNamespace(), model, num_children)


def _bag_with_strategy(strategy):
    bag = object.__new__(BaggedEnsembleModel)
    bag._user_params = {} if strategy is None else {"fold_fitting_strategy": strategy}
    return bag


def test_concurrent_children_sequential_local_is_one(_restore_flag):
    # sequential_local fits folds one-at-a-time -> concurrent footprint is 1, not num_splits.
    os.environ["AG_PARALLEL_GPU"] = "True"
    assert _concurrent(_bag_with_strategy("sequential_local"), num_children=8) == 1


def test_concurrent_children_parallel_local_unchanged(_restore_flag):
    os.environ["AG_PARALLEL_GPU"] = "True"
    assert _concurrent(_bag_with_strategy("parallel_local"), num_children=8) == 8


def test_concurrent_children_default_strategy_unchanged(_restore_flag):
    os.environ["AG_PARALLEL_GPU"] = "True"
    assert _concurrent(_bag_with_strategy(None), num_children=8) == 8


def test_concurrent_children_sequential_local_noop_when_flag_off(_restore_flag):
    os.environ.pop("AG_PARALLEL_GPU", None)
    assert _concurrent(_bag_with_strategy("sequential_local"), num_children=8) == 8


def test_concurrent_children_non_model_unchanged(_restore_flag):
    # refit mode passes a model name (str), which must not be special-cased.
    os.environ["AG_PARALLEL_GPU"] = "True"
    assert _concurrent("SomeModel_name", num_children=8) == 8


def test_sequential_local_reserves_one_fold_cpu_not_num_splits(_restore_flag):
    """End-to-end: with the scheduler's sequential num_parallel=1, the model reserves ONE fold's
    CPUs (memory pressure is a single resident fold) -- not num_cpus * num_splits."""
    os.environ["AG_PARALLEL_GPU"] = "True"
    bag = object.__new__(BaggedEnsembleModel)
    bag._user_params = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}
    bag._user_params_aux = {}
    child = object.__new__(_DummyModel)
    child._user_params_aux = {}
    bag.model_base = child

    # num_parallel=1 is what schedule_jobs now passes for sequential_local (was num_splits).
    prepare_model_resources_for_fit(
        model=bag,
        total_num_cpus=192,
        total_num_gpus=8,
        num_cpus=7,
        num_gpus=1,
        num_parallel=1,
        num_children=8,
    )
    res = _resources_for_fit(bag, num_splits=8)
    assert res.total_num_cpus == 7  # one fold's worth; was 7 * 8 = 56 before the fix
    assert res.total_num_gpus == 1
