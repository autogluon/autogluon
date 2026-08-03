"""VRAM accounting when budgeting parallel bagging folds."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd

from autogluon.core.models.ensemble.fold_fitting_strategy import ParallelLocalFoldFittingStrategy

VRAM_PATH = "autogluon.core.models.ensemble.fold_fitting_strategy.ResourceManager.get_available_vram"


class _Strategy:
    """Exercises the VRAM cap without building a full fold-fitting strategy."""

    _max_folds_in_parallel_with_vram = ParallelLocalFoldFittingStrategy._max_folds_in_parallel_with_vram

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

        # a model larger than the card still gets one fold, matching the RAM budget's floor
        assert _Strategy(num_gpus=1, can_estimate=True, estimate=int(200e9))._max_folds_in_parallel_with_vram(1.0) == 1


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
