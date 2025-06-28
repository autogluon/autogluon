import math
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from autogluon.common import space
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel
from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from autogluon.core.models.ensemble.fold_fitting_strategy import FoldFittingStrategy, ParallelLocalFoldFittingStrategy
from autogluon.core.ray.resources_calculator import CpuResourceCalculator
from autogluon.core.searcher import LocalRandomSearcher

NUM_CPU = 8
NUM_GPU = 1


class DummyBigModel(AbstractModel):
    def _estimate_memory_usage(self, **kwargs):
        return 1e9


def _prepare_data():
    # prepare an all numeric data so that we don't need to clean labels and features
    data = [[1, 10], [2, 20], [3, 30]]
    df = pd.DataFrame(data, columns=["Number", "Age"])
    label = "Age"
    X = df.drop(columns=[label])
    y = df[label]
    return X, y


def _construct_dummy_fold_strategy(
    num_jobs, model_base_cls=AbstractModel, time_limit=None, num_folds_parallel=8
):
    dummy_model_base = model_base_cls()
    dummy_bagged_ensemble_model = BaggedEnsembleModel(dummy_model_base)
    train_data, test_data = _prepare_data()
    args = dict(
        model_base=dummy_model_base,
        model_base_kwargs=dict(),
        bagged_ensemble_model=dummy_bagged_ensemble_model,
        X=train_data,
        y=test_data,
        X_pseudo=None,
        y_pseudo=None,
        sample_weight=None,
        time_limit=time_limit,
        time_start=time.time(),
        models=[],
        oof_pred_proba=np.array([]),
        oof_pred_model_repeats=np.array([]),
        save_folds=True,
        num_cpus=NUM_CPU,
        num_gpus=NUM_GPU,
        num_jobs=num_jobs,
        num_folds_parallel=num_folds_parallel,
        time_limit_fold_ratio=1,
        max_memory_usage_ratio=1,
    )
    return ParallelLocalFoldFittingStrategy(**args)


def _test_resource_allocation_and_time_limit(num_jobs, num_folds_parallel, time_limit):
    num_cpus = NUM_CPU
    num_gpus = NUM_GPU
    time_start = time.time()
    fold_fitting_strategy = _construct_dummy_fold_strategy(
        num_jobs=num_jobs, time_limit=time_limit, num_folds_parallel=num_folds_parallel
    )
    for i in range(num_jobs):
        fold_fitting_strategy.schedule_fold_model_fit(dict())
    resources, batches, num_parallel_jobs = (
        fold_fitting_strategy.resources,
        fold_fitting_strategy.batches,
        fold_fitting_strategy.num_parallel_jobs,
    )
    time_elapsed = time.time() - time_start
    time_remaining = time_limit - time_elapsed
    time_limit_fold = fold_fitting_strategy._get_fold_time_limit()
    num_cpus_per_job = resources.get("num_cpus", 0)
    num_gpus_per_job = resources.get("num_gpus", 0)
    assert batches >= 1
    if batches > 1:
        assert num_jobs <= num_parallel_jobs * batches <= (num_jobs + num_parallel_jobs)
    assert num_cpus_per_job * num_parallel_jobs <= num_cpus
    if num_gpus != 0:
        assert num_gpus_per_job * num_parallel_jobs <= num_gpus
    else:
        assert num_gpus_per_job == 0
    assert math.isclose(time_limit_fold, (time_remaining / batches), abs_tol=0.5)


def test_resource_allocation_and_time_limit():
    num_iterations = 100

    search_space = dict(
        num_jobs=space.Int(1, 100),
        num_folds_parallel=space.Int(1, 200),
        time_limit=space.Int(60, 60 * 60 * 24),
    )

    searcher = LocalRandomSearcher(search_space=search_space)

    for i in range(num_iterations):
        config = searcher.get_config()
        _test_resource_allocation_and_time_limit(**config)


@patch(
    "autogluon.common.utils.resource_utils.ResourceManager.get_available_virtual_mem"
)
@patch(
    "autogluon.core.ray.resources_calculator.ResourceCalculatorFactory.get_resource_calculator"
)
def test_dynamic_resource_allocation(resource_cal, mock_get_mem):
    mock_get_mem.return_value = 2.5 * 1e9
    resource_cal.return_value = CpuResourceCalculator()
    fold_fitting_strategy = _construct_dummy_fold_strategy(
        model_base_cls=DummyBigModel, num_jobs=8, num_folds_parallel=8
    )
    assert (
        fold_fitting_strategy.num_parallel_jobs == 2
        and fold_fitting_strategy.batches == 4
    )
    mock_get_mem.return_value = 7.5 * 1e9
    fold_fitting_strategy = _construct_dummy_fold_strategy(
        model_base_cls=DummyBigModel, num_jobs=8, num_folds_parallel=8
    )
    # If memory is not sufficient to train num_folds_parallel, reduce to max power of 2 folds that's smaller than folds_can_be_fit_in_parallel.
    # Here memory can only train 7 folds, therefore we train 4 folds instead in two batches
    assert (
        fold_fitting_strategy.num_parallel_jobs == 4
        and fold_fitting_strategy.batches == 2
    )
    mock_get_mem.return_value = 6 * 1e9
    fold_fitting_strategy = _construct_dummy_fold_strategy(
        model_base_cls=DummyBigModel, num_jobs=10, num_folds_parallel=10
    )
    # Here memory can only train 10 folds, therefore we train 4 folds instead in three batches, the last batch would train 2 folds in parallel
    assert (
        fold_fitting_strategy.num_parallel_jobs == 4
        and fold_fitting_strategy.batches == 3
    )
