"""Worker recycling of the parallel fold fitting strategy: fresh processes only where a reused one would leak state."""

import time

import numpy as np
import pandas as pd
import pytest

from autogluon.core.models import AbstractModel, BaggedEnsembleModel
from autogluon.core.models.ensemble.fold_fitting_strategy import ParallelLocalFoldFittingStrategy


class _Model(AbstractModel):
    def get_minimum_resources(self, **kwargs):
        return {"num_cpus": 1, "num_gpus": 0}

    def _get_default_resources(self):
        return 1, 0


def _strategy(*, num_gpus: int, num_folds_parallel: int, num_jobs: int = 4) -> ParallelLocalFoldFittingStrategy:
    pytest.importorskip("ray")
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([10.0, 20.0, 30.0, 40.0])
    model_base = _Model(hyperparameters={})
    return ParallelLocalFoldFittingStrategy(
        num_jobs=num_jobs,
        num_folds_parallel=num_folds_parallel,
        model_base=model_base,
        model_base_kwargs={},
        bagged_ensemble_model=BaggedEnsembleModel(model_base, hyperparameters={}),
        X=X,
        y=y,
        X_pseudo=None,
        y_pseudo=None,
        sample_weight=None,
        time_start=time.time(),
        time_limit=None,
        models=[],
        oof_pred_proba=np.array([]),
        oof_pred_model_repeats=np.array([]),
        save_folds=True,
        num_cpus=4,
        num_gpus=num_gpus,
    )


def test__cpu_folds_in_parallel__reuse_workers():
    strategy = _strategy(num_gpus=0, num_folds_parallel=4)
    assert strategy.resources["num_gpus"] == 0 and not strategy._pseudo_sequential
    assert strategy._ray_fit._default_options["max_calls"] == 0


def test__gpu_folds__get_a_fresh_worker_each():
    strategy = _strategy(num_gpus=1, num_folds_parallel=4)
    assert strategy.resources["num_gpus"] > 0
    assert strategy._ray_fit._default_options["max_calls"] == 1


def test__pseudo_sequential_folds__get_a_fresh_worker_each():
    strategy = _strategy(num_gpus=0, num_folds_parallel=1)
    assert strategy._pseudo_sequential
    assert strategy._ray_fit._default_options["max_calls"] == 1
