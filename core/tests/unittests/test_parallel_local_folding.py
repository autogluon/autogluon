import math
import time

import numpy as np
import pandas as pd

from autogluon.common import space
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel
from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from autogluon.core.models.ensemble.fold_fitting_strategy import ParallelLocalFoldFittingStrategy
from autogluon.core.searcher import LocalRandomSearcher


def _prepare_data():
    # prepare an all numeric data so that we don't need to clean labels and features
    data = [[1, 10], [2, 20], [3, 30]]
    df = pd.DataFrame(data, columns=["Number", "Age"])
    label = "Age"
    X = df.drop(columns=[label])
    y = df[label]
    return X, y


def _construct_dummy_fold_strategy(num_jobs, time_limit=None, num_folds_parallel=8):
    dummy_model_base = AbstractModel()
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
        num_cpus=ResourceManager.get_cpu_count(),
        num_gpus=ResourceManager.get_gpu_count_all(),
        num_jobs=num_jobs,
        num_folds_parallel=num_folds_parallel,
        time_limit_fold_ratio=1,
    )
    return ParallelLocalFoldFittingStrategy(**args)


def _test_resource_allocation_and_time_limit(num_jobs, num_folds_parallel, time_limit):
    num_cpus = ResourceManager.get_cpu_count()
    num_gpus = ResourceManager.get_gpu_count_all()
    time_start = time.time()
    fold_fitting_strategy = _construct_dummy_fold_strategy(num_jobs=num_jobs, time_limit=time_limit, num_folds_parallel=num_folds_parallel)
    for i in range(num_jobs):
        fold_fitting_strategy.schedule_fold_model_fit(dict())
    resources, batches, num_parallel_jobs = fold_fitting_strategy.resources, fold_fitting_strategy.batches, fold_fitting_strategy.num_parallel_jobs
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
