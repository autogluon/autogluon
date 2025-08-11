import time

import numpy as np
import pandas as pd
import pytest

from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from autogluon.core.models.ensemble.fold_fitting_strategy import (
    ParallelLocalFoldFittingStrategy,
    SequentialLocalFoldFittingStrategy,
)
from autogluon.tabular.models import AbstractModel


class DummyBaseModel(AbstractModel):
    def __init__(self, minimum_resources=None, default_resources=None, **kwargs):
        self._minimum_resources = minimum_resources
        self._default_resources = default_resources
        super().__init__(**kwargs)

    def get_minimum_resources(self, **kwargs):
        return self._minimum_resources

    def _get_default_resources(self):
        num_cpus = self._default_resources.get("num_cpus")
        num_gpus = self._default_resources.get("num_gpus")
        return num_cpus, num_gpus


class DummyModel(DummyBaseModel):
    pass


class DummyBaggedModel(BaggedEnsembleModel):
    pass


def _prepare_data():
    # prepare an all numeric data so that we don't need to clean labels and features
    data = [[1, 10], [2, 20], [3, 30]]
    df = pd.DataFrame(data, columns=["Number", "Age"])
    label = "Age"
    X = df.drop(columns=[label])
    y = df[label]
    return X, y


def _construct_dummy_fold_strategy(
    fold_strategy_cls,
    num_jobs,
    num_folds_parallel,
    bagged_resources=None,
    model_base_resources=None,
    model_base_minimum_resources=None,
    model_base_default_resources=None,
):
    if bagged_resources is None:
        bagged_resources = {}
    if model_base_resources is None:
        model_base_resources = {}
    if model_base_minimum_resources is None:
        model_base_minimum_resources = {}
    dummy_model_base = DummyModel(
        minimum_resources=model_base_minimum_resources,
        default_resources=model_base_default_resources,
        hyperparameters={"ag_args_fit": model_base_resources},
    )
    dummy_bagged_ensemble_model = DummyBaggedModel(dummy_model_base, hyperparameters={"ag_args_fit": bagged_resources})
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
        time_start=time.time(),
        time_limit=0,
        models=[],
        oof_pred_proba=np.array([]),
        oof_pred_model_repeats=np.array([]),
        save_folds=True,
        time_limit_fold_ratio=1,
        **bagged_resources,  # These will be processed at abstract model level in a real tabular predictor
    )
    if fold_strategy_cls == ParallelLocalFoldFittingStrategy:
        args["num_jobs"] = num_jobs
        args["num_folds_parallel"] = num_folds_parallel
    return fold_strategy_cls(**args)


@pytest.mark.parametrize("fold_strategy_cls", [ParallelLocalFoldFittingStrategy, SequentialLocalFoldFittingStrategy])
def test_bagging_invalid_resources_per_fold(fold_strategy_cls):
    # resources per fold more than resources to ensemble model
    with pytest.raises(AssertionError) as e:
        _construct_dummy_fold_strategy(
            fold_strategy_cls=fold_strategy_cls,
            num_jobs=8,
            num_folds_parallel=8,
            bagged_resources={"num_cpus": 1, "num_gpus": 1},
            model_base_resources={"num_cpus": 2, "num_gpus": 1},
            model_base_minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
        )
    # resources per fold less than minimum resources
    with pytest.raises(AssertionError) as e:
        _construct_dummy_fold_strategy(
            fold_strategy_cls=fold_strategy_cls,
            num_jobs=8,
            num_folds_parallel=8,
            bagged_resources={"num_cpus": 8, "num_gpus": 1},
            model_base_resources={"num_cpus": 1, "num_gpus": 0.1},
            model_base_minimum_resources={"num_cpus": 1, "num_gpus": 0.5},
        )


def test_parallel_bagging_resources_per_fold():
    fold_fitting_strategy = _construct_dummy_fold_strategy(
        fold_strategy_cls=ParallelLocalFoldFittingStrategy,
        num_jobs=8,
        num_folds_parallel=8,
        bagged_resources={"num_cpus": 8, "num_gpus": 1},
        model_base_resources={"num_cpus": 4, "num_gpus": 1},
        model_base_minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
        model_base_default_resources={"num_cpus": 8, "num_gpus": 1},
    )
    assert fold_fitting_strategy.resources == {"num_cpus": 4, "num_gpus": 1}
    assert fold_fitting_strategy.resources_model == {"num_cpus": 4, "num_gpus": 1}
    assert fold_fitting_strategy.num_parallel_jobs == 1
    assert fold_fitting_strategy.batches == 8

    fold_fitting_strategy = _construct_dummy_fold_strategy(
        fold_strategy_cls=ParallelLocalFoldFittingStrategy,
        num_jobs=8,
        num_folds_parallel=8,
        bagged_resources={"num_cpus": 8, "num_gpus": 1},
        model_base_resources={"num_cpus": 1, "num_gpus": 0.1},
        model_base_minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
        model_base_default_resources={"num_cpus": 8, "num_gpus": 1},
    )
    assert fold_fitting_strategy.resources == {"num_cpus": 1, "num_gpus": 0.1}
    assert fold_fitting_strategy.resources_model == {"num_cpus": 1, "num_gpus": 0.1}
    assert fold_fitting_strategy.num_parallel_jobs == 8
    assert fold_fitting_strategy.batches == 1


def test_parallel_bagging_no_resources_per_fold():
    fold_fitting_strategy = _construct_dummy_fold_strategy(
        fold_strategy_cls=ParallelLocalFoldFittingStrategy,
        num_jobs=8,
        num_folds_parallel=4,
        bagged_resources={"num_cpus": 8, "num_gpus": 1},
        model_base_minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
        model_base_default_resources={"num_cpus": 8, "num_gpus": 1},
    )
    assert fold_fitting_strategy.resources == {"num_cpus": 2, "num_gpus": 0.25}
    assert fold_fitting_strategy.resources_model == {"num_cpus": 2, "num_gpus": 0.25}
    assert fold_fitting_strategy.num_parallel_jobs == 4
    assert fold_fitting_strategy.batches == 2

    fold_fitting_strategy = _construct_dummy_fold_strategy(
        fold_strategy_cls=ParallelLocalFoldFittingStrategy,
        num_jobs=4,
        num_folds_parallel=2,
        bagged_resources={"num_cpus": 8, "num_gpus": 1},
        model_base_minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
        model_base_default_resources={"num_cpus": 2, "num_gpus": 0.2},
    )
    assert fold_fitting_strategy.resources == {"num_cpus": 4, "num_gpus": 0.5}
    assert fold_fitting_strategy.resources_model == {"num_cpus": 2, "num_gpus": 0.2}
    assert fold_fitting_strategy.num_parallel_jobs == 2
    assert fold_fitting_strategy.batches == 2


def test_sequential_bagging_resources_per_fold():
    fold_fitting_strategy = _construct_dummy_fold_strategy(
        fold_strategy_cls=SequentialLocalFoldFittingStrategy,
        num_jobs=8,
        num_folds_parallel=8,
        bagged_resources={"num_cpus": 8, "num_gpus": 1},
        model_base_resources={"num_cpus": 4, "num_gpus": 1},
        model_base_default_resources={"num_cpus": 1, "num_gpus": 0},
        model_base_minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
    )
    assert fold_fitting_strategy.resources == {"num_cpus": 4, "num_gpus": 1}
    assert fold_fitting_strategy.user_resources_per_job == {"num_cpus": 4, "num_gpus": 1}

    fold_fitting_strategy = _construct_dummy_fold_strategy(
        fold_strategy_cls=SequentialLocalFoldFittingStrategy,
        num_jobs=8,
        num_folds_parallel=8,
        bagged_resources={"num_cpus": 8, "num_gpus": 1},
        model_base_resources={"num_cpus": 1, "num_gpus": 0.1},
        model_base_default_resources={"num_cpus": 1, "num_gpus": 0},
        model_base_minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
    )
    assert fold_fitting_strategy.resources == {"num_cpus": 1, "num_gpus": 0.1}
    assert fold_fitting_strategy.user_resources_per_job == {"num_cpus": 1, "num_gpus": 0.1}


def test_sequential_bagging_no_resources_per_fold():
    fold_fitting_strategy = _construct_dummy_fold_strategy(
        fold_strategy_cls=SequentialLocalFoldFittingStrategy,
        num_jobs=8,
        num_folds_parallel=4,
        bagged_resources={"num_cpus": 8, "num_gpus": 1},
        model_base_minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
        model_base_default_resources={"num_cpus": 4, "num_gpus": 0.5},
    )
    assert fold_fitting_strategy.num_cpus == 8
    assert fold_fitting_strategy.num_gpus == 1
    assert fold_fitting_strategy.resources == {"num_cpus": 4, "num_gpus": 0.5}
    assert fold_fitting_strategy.user_resources_per_job == None
