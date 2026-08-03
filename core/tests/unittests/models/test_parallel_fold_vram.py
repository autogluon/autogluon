"""VRAM accounting when budgeting parallel bagging folds."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from autogluon.core.models.ensemble.fold_fitting_strategy import ParallelLocalFoldFittingStrategy
from autogluon.core.utils.exceptions import NotEnoughCudaMemoryError

VRAM_PATH = "autogluon.core.models.ensemble.fold_fitting_strategy.ResourceManager.get_available_vram"


class _Strategy:
    """Exercises the VRAM cap without building a full fold-fitting strategy."""

    _max_folds_in_parallel_with_vram = ParallelLocalFoldFittingStrategy._max_folds_in_parallel_with_vram
    _per_device_available_vram = ParallelLocalFoldFittingStrategy._per_device_available_vram

    def __init__(self, *, num_gpus: int, can_estimate: bool, estimate: int | None):
        self.num_gpus = num_gpus
        self.X = pd.DataFrame({"feature": np.arange(10.0)})

        class _Model:
            def can_estimate_gpu_memory_usage_static(_self) -> bool:
                return can_estimate

            def estimate_gpu_memory_usage(_self, X) -> int | None:
                return estimate

        self._initialized_model_base = _Model()


def test__vram_cap__limits_folds_to_what_the_card_holds():
    """Folds share one device, so the cap is free VRAM divided by the per-fold estimate."""
    with patch(VRAM_PATH, return_value=96e9):
        strategy = _Strategy(num_gpus=1, can_estimate=True, estimate=int(12e9))
        assert strategy._max_folds_in_parallel_with_vram(1.0) == 8
        assert strategy._max_folds_in_parallel_with_vram(0.8) == 6  # ratio applies


def test__vram_cap__rejects_a_model_that_cannot_fit_even_one_fold():
    """Bottoming out at one fold is ambiguous, so the model's VRAM check settles it.

    Without this the fit would be attempted at one fold and die on a CUDA OOM mid-fit; the
    raise is what the trainer converts into a graceful model skip.
    """
    calls = []

    class _Rejecting(_Strategy):
        def __init__(self):
            super().__init__(num_gpus=1, can_estimate=True, estimate=int(200e9))
            strategy = self

            class _Model(type(self._initialized_model_base)):
                def _validate_fit_gpu_memory_usage(_self, **kwargs):
                    calls.append(kwargs)
                    raise NotEnoughCudaMemoryError("estimate exceeds available VRAM")

            self._initialized_model_base = _Model()

    with patch(VRAM_PATH, return_value=96e9), pytest.raises(NotEnoughCudaMemoryError):
        _Rejecting()._max_folds_in_parallel_with_vram(1.0)
    # the check is handed the single-fold estimate and the bag's GPU count, not a fold fraction
    assert calls[0]["approx_mem_size_req"] == int(200e9)
    assert calls[0]["num_gpus"] == 1


def test__vram_cap__does_not_constrain_when_it_cannot_know():
    """Each fall-through leaves the existing RAM-only budget in charge."""
    with patch(VRAM_PATH, return_value=96e9):
        # CPU-only fit
        assert (
            _Strategy(num_gpus=0, can_estimate=True, estimate=int(12e9))._max_folds_in_parallel_with_vram(1.0)
            == math.inf
        )
        # model does not implement a GPU estimate (most do not; it is opt-in)
        assert (
            _Strategy(num_gpus=1, can_estimate=False, estimate=None)._max_folds_in_parallel_with_vram(1.0) == math.inf
        )
        # model implements it but returns nothing usable
        assert _Strategy(num_gpus=1, can_estimate=True, estimate=0)._max_folds_in_parallel_with_vram(1.0) == math.inf
    # free VRAM cannot be read
    with patch(VRAM_PATH, return_value=None):
        assert (
            _Strategy(num_gpus=1, can_estimate=True, estimate=int(12e9))._max_folds_in_parallel_with_vram(1.0)
            == math.inf
        )


class TestGpuAssignmentPacking:
    """Device selection packs GPUs toward 1.0 utilization, then prefers VRAM headroom."""

    def test__mixed_requests_pack_one_device_before_the_next(self):
        from autogluon.core.models.ensemble.fold_fitting_strategy import plan_gpu_assignments

        # 0.5 -> empty device 0; 1.0 needs a whole device -> 1; 0.25 -> tops up device 0
        # (highest utilization that still fits); 0.5 no longer fits device 0 -> fresh device 2.
        plan = plan_gpu_assignments(requests=[0.5, 1, 0.25, 0.5], total_gpus=4)
        assert plan == [[0], [1], [0], [2]]

        utilization = [0.0] * 4
        for request, devices in zip([0.5, 1, 0.25, 0.5], plan):
            for device in devices:
                utilization[device] += 1.0 if request >= 1 else request
        assert utilization == [0.75, 1.0, 0.5, 0.0]

    def test__homogeneous_fractions_fill_devices_to_exactly_one(self):
        from autogluon.core.models.ensemble.fold_fitting_strategy import plan_gpu_assignments

        # eight half-GPU folds on four devices: pairs share a device, no device oversubscribed
        plan = plan_gpu_assignments(requests=[0.5] * 8, total_gpus=4)
        assert plan == [[0], [0], [1], [1], [2], [2], [3], [3]]

    def test__vram_headroom_breaks_utilization_ties_and_gates_candidates(self):
        from autogluon.core.models.ensemble.fold_fitting_strategy import plan_gpu_assignments

        # equal utilization: the device with more free VRAM wins the tie
        plan = plan_gpu_assignments(requests=[0.5], total_gpus=2, per_device_vram=[10e9, 40e9], vram_est_per_task=8e9)
        assert plan == [[1]]
        # a device without VRAM headroom is not a candidate even if its fraction fits
        plan = plan_gpu_assignments(
            requests=[0.25, 0.25], total_gpus=2, per_device_vram=[9e9, 40e9], vram_est_per_task=8e9
        )
        assert plan == [[1], [1]]  # device 0 fits one fold by fraction but not a second by VRAM

    def test__whole_device_requests_take_idle_devices(self):
        from autogluon.core.models.ensemble.fold_fitting_strategy import plan_gpu_assignments

        plan = plan_gpu_assignments(requests=[2, 1, 1], total_gpus=4)
        assert plan == [[0, 1], [2], [3]]

    def test__overflow_falls_back_to_least_utilized_instead_of_failing(self):
        from autogluon.core.models.ensemble.fold_fitting_strategy import plan_gpu_assignments

        # three 0.75 requests on two devices: the third fits nowhere, lands on the emptier one
        plan = plan_gpu_assignments(requests=[0.75, 0.75, 0.75], total_gpus=2)
        assert plan[0] == [0] and plan[1] == [1]
        assert len(plan[2]) == 1  # assigned somewhere rather than raising
