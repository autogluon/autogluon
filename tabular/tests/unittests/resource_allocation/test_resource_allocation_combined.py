import time

import numpy as np
import pandas as pd
import pytest

from autogluon.core.hpo.executors import CustomHpoExecutor, RayHpoExecutor
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


dummy_x = pd.DataFrame([1, 2, 3], columns=["Dummy"])


def _initialize_executor(executor_cls, hyperparameter_tune_kwargs):
    executor = executor_cls()
    executor.initialize(hyperparameter_tune_kwargs)
    return executor


def _prepare_data():
    # prepare an all numeric data so that we don't need to clean labels and features
    data = [[1, 10], [2, 20], [3, 30]]
    df = pd.DataFrame(data, columns=["Number", "Age"])
    label = "Age"
    X = df.drop(columns=[label])
    y = df[label]
    return X, y


def _construct_dummy_fold_strategy(fold_strategy_cls, num_jobs, num_folds_parallel, resource_granted, model_base, bagged_model):
    train_data, test_data = _prepare_data()
    args = dict(
        model_base=model_base,
        model_base_kwargs=dict(),
        bagged_ensemble_model=bagged_model,
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
        **resource_granted,  # These will be processed at abstract model level in a real tabular predictor
    )
    if fold_strategy_cls == ParallelLocalFoldFittingStrategy:
        args["num_jobs"] = num_jobs
        args["num_folds_parallel"] = num_folds_parallel
    return fold_strategy_cls(**args)


def _test_bagging(
    fold_strategy_cls,
    num_jobs,
    num_folds_parallel,
    model_base,
    bagged_model,
    resources,
    expected_answer,
):
    fold_fitting_strategy = _construct_dummy_fold_strategy(
        fold_strategy_cls=fold_strategy_cls,
        num_jobs=num_jobs,
        num_folds_parallel=num_folds_parallel,
        resource_granted=resources,
        model_base=model_base,
        bagged_model=bagged_model,
    )
    expected_resources_per_model = expected_answer["resources_per_model"]
    assert fold_fitting_strategy.resources == expected_resources_per_model
    if fold_strategy_cls == ParallelLocalFoldFittingStrategy:
        expected_model_in_parallel = expected_answer["model_in_parallel"]
        assert fold_fitting_strategy.num_parallel_jobs == expected_model_in_parallel


def _test_functionality(mock_system_resources_ctx_mgr, test_args):
    system_resources = test_args.get("system_resources", {"num_cpus": 16, "num_gpus": 4})
    total_resources = test_args.get("total_resources", {"num_cpus": "auto", "num_gpus": "auto"})
    model_default_resources = test_args.get("model_default_resources", {"num_cpus": 1, "num_gpus": 0})
    model_minimum_resources = test_args.get("model_minimum_resources", {"num_cpus": 1, "num_gpus": 0})
    ag_args_ensemble = test_args.get("ag_args_ensemble", {})
    ag_args_fit = test_args.get("ag_args_fit", {})
    num_bag_folds = test_args.get("num_bag_folds", 0)
    fold_strategy_cls = test_args.get("fold_strategy_cls", ParallelLocalFoldFittingStrategy)
    num_trials = test_args.get("num_trials", 0)
    executor_cls = test_args.get("executor_cls", RayHpoExecutor)
    expected_answer = test_args.get("expected_answer")
    hpo = num_trials > 0
    parallel_hpo = hpo and executor_cls == RayHpoExecutor
    with mock_system_resources_ctx_mgr(num_cpus=system_resources.get("num_cpus"), num_gpus=system_resources.get("num_gpus")):
        model = DummyModel(minimum_resources=model_minimum_resources, default_resources=model_default_resources, hyperparameters={"ag_args_fit": ag_args_fit})
        model.initialize()
        if num_bag_folds > 0:
            model = DummyBaggedModel(model, hyperparameters={"ag_args_fit": ag_args_ensemble})
            model.initialize()
        resources = model._preprocess_fit_resources(total_resources=total_resources, k_fold=num_bag_folds, parallel_hpo=parallel_hpo)
        resources.pop("k_fold")
        if hpo:
            hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": num_trials}
            executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
            executor.register_resources(model, k_fold=num_bag_folds, X=dummy_x, **resources)
            resources_per_trial = executor.hyperparameter_tune_kwargs["resources_per_trial"]
            assert resources_per_trial == expected_answer["resources_per_trial"]
            if num_bag_folds > 0:
                _test_bagging(
                    fold_strategy_cls=fold_strategy_cls,
                    num_jobs=num_bag_folds,
                    num_folds_parallel=num_bag_folds,
                    model_base=model.model_base,
                    bagged_model=model,
                    resources=resources_per_trial,
                    expected_answer=expected_answer,
                )
        else:
            if num_bag_folds > 0:
                _test_bagging(
                    fold_strategy_cls=fold_strategy_cls,
                    num_jobs=num_bag_folds,
                    num_folds_parallel=num_bag_folds,
                    model_base=model.model_base,
                    bagged_model=model,
                    resources=resources,
                    expected_answer=expected_answer,
                )
            else:
                assert resources == expected_answer["resources_per_model"]


tests_dict = {
    "valid_ag_args_fit": ({"ag_args_fit": {"num_cpus": 8, "num_gpus": 2}, "expected_answer": {"resources_per_model": {"num_cpus": 8, "num_gpus": 2}}}),
    "valid_ag_args_fit_without_gpu_default_no_gpu": (
        {
            "ag_args_fit": {"num_cpus": 8},
            "model_default_resources": {"num_cpus": 1, "num_gpus": 0},
            "expected_answer": {"resources_per_model": {"num_cpus": 8, "num_gpus": 0}},
        }
    ),
    "valid_ag_args_fit_without_gpu_default_gpu": (
        {
            "ag_args_fit": {"num_cpus": 8},
            "model_default_resources": {"num_cpus": 1, "num_gpus": 1},
            "expected_answer": {"resources_per_model": {"num_cpus": 8, "num_gpus": 1}},
        }
    ),
    "valid_ag_args_ensemble_and_ag_args_fit": (
        {
            "ag_args_ensemble": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_fit": {"num_cpus": 4, "num_gpus": 1},
            "expected_answer": {"resources_per_model": {"num_cpus": 4, "num_gpus": 1}},
        }
    ),
    "valid_ag_args_ensemble": (
        {
            "ag_args_ensemble": {"num_cpus": 8, "num_gpus": 2},  # should be ignored
            "model_default_resources": {"num_cpus": 2, "num_gpus": 1},
            "expected_answer": {"resources_per_model": {"num_cpus": 2, "num_gpus": 1}},
        }
    ),
    "total_resources_with_valid_ag_args_ensemble_and_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 0.5},
            "expected_answer": {"resources_per_model": {"num_cpus": 2, "num_gpus": 0.5}},
        }
    ),
    "total_resources_with_valid_ag_args_ensemble": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},  # should be ignored
            "model_default_resources": {"num_cpus": 2, "num_gpus": 1},
            "expected_answer": {"resources_per_model": {"num_cpus": 8, "num_gpus": 2}},
        }
    ),
    "total_resources_with_valid_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_fit": {"num_cpus": 4, "num_gpus": 1},
            "expected_answer": {"resources_per_model": {"num_cpus": 4, "num_gpus": 1}},
        }
    ),
    "total_resources": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "expected_answer": {"resources_per_model": {"num_cpus": 8, "num_gpus": 2}},
        }
    ),
    "without_anything": (
        {"model_default_resources": {"num_cpus": 2, "num_gpus": 1}, "expected_answer": {"resources_per_model": {"num_cpus": 2, "num_gpus": 1}}}
    ),
    "bagging_with_total_resources_and_valid_ag_args_ensemble_and_valid_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 1},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 2},
            "num_bag_folds": 8,
            "expected_answer": {"resources_per_model": {"num_cpus": 2, "num_gpus": 1}, "model_in_parallel": 2},
        }
    ),
    "bagging_with_valid_ag_args_ensemble_and_valid_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 1},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 2},
            "num_bag_folds": 8,
            "expected_answer": {"resources_per_model": {"num_cpus": 2, "num_gpus": 1}, "model_in_parallel": 2},
        }
    ),
    "bagging_with_valid_ag_args_ensemble": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_ensemble": {"num_cpus": 8, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 1},
            "num_bag_folds": 8,
            "expected_answer": {"resources_per_model": {"num_cpus": 2, "num_gpus": 1}, "model_in_parallel": 4},
        }
    ),
    "bagging_with_valid_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 1},
            "num_bag_folds": 8,
            "expected_answer": {"resources_per_model": {"num_cpus": 2, "num_gpus": 1}, "model_in_parallel": 4},
        }
    ),
    "bagging_without_anything": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0},
            "model_default_resources": {"num_cpus": 1, "num_gpus": 0},
            "num_bag_folds": 8,
            "expected_answer": {"resources_per_model": {"num_cpus": 2, "num_gpus": 0}, "model_in_parallel": 8},
        }
    ),
    "bagging_without_anything_with_gpu": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "model_default_resources": {"num_cpus": 2, "num_gpus": 1},
            "num_bag_folds": 8,
            "expected_answer": {"resources_per_model": {"num_cpus": 2, "num_gpus": 0.5}, "model_in_parallel": 8},
        }
    ),
    "sequential_bagging_with_total_resources_and_valid_ag_args_ensemble_and_valid_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 0.5},
            "num_bag_folds": 8,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "expected_answer": {
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
            },
        }
    ),
    "sequential_bagging_with_valid_ag_args_ensemble_and_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_ensemble": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 1},
            "num_bag_folds": 8,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "expected_answer": {
                "resources_per_model": {"num_cpus": 2, "num_gpus": 1},
            },
        }
    ),
    "sequential_bagging_with_valid_ag_args_ensemble": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_ensemble": {"num_cpus": 2, "num_gpus": 1},
            "num_bag_folds": 8,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "expected_answer": {
                # "resources_per_model": {"num_cpus": 2, "num_gpus": 1},
                # FIXME: Above is commented out in v1.4 to fix bug in sequential.
                #  But this creates inconsistency with Parallel logic. Resolve in v1.5.
                #  To me, the below seems correct,
                #  as we don't want to use GPUs for a model that doesn't default to using them.
                "resources_per_model": {"num_cpus": 1, "num_gpus": 0},
            },
        }
    ),
    "sequential_bagging_with_valid_ag_args_fit_and_ag_args_ensemble": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 1},
            "num_bag_folds": 8,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "expected_answer": {
                "resources_per_model": {"num_cpus": 2, "num_gpus": 1},
            },
        }
    ),
    "sequential_bagging_without_anything": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "model_default_resources": {"num_cpus": 2, "num_gpus": 1},
            "num_bag_folds": 8,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "expected_answer": {
                "resources_per_model": {"num_cpus": 2, "num_gpus": 1},
            },
        }
    ),
    "hpo_with_total_resources_and_ag_args_ensemble_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 1},
            "num_trials": 2,
            "expected_answer": {"resources_per_trial": {"num_cpus": 2, "num_gpus": 1}},
        }
    ),
    "hpo_with_total_resources_and_ag_args_ensemble": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_ensemble": {"num_cpus": 2, "num_gpus": 1},
            "num_trials": 2,
            "expected_answer": {"resources_per_trial": {"num_cpus": 4, "num_gpus": 1}},  # ag_args_ensemble shouldn't affect hpo without bagging
        }
    ),
    "hpo_with_total_resources_and_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 1},
            "num_trials": 2,
            "expected_answer": {"resources_per_trial": {"num_cpus": 2, "num_gpus": 1}},
        }
    ),
    "hpo_with_ag_args_ensemble": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_ensemble": {"num_cpus": 2, "num_gpus": 1},
            "model_default_resources": {"num_cpus": 2, "num_gpus": 1},
            "num_trials": 2,
            "expected_answer": {"resources_per_trial": {"num_cpus": 8, "num_gpus": 2}},  # ag_args_ensemble shouldn't affect hpo without bagging
        }
    ),
    "hpo_with_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 1},
            "num_trials": 2,
            "expected_answer": {"resources_per_trial": {"num_cpus": 2, "num_gpus": 1}},
        }
    ),
    "hpo_without_anything": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_default_resources": {"num_cpus": 2, "num_gpus": 0},
            "num_trials": 2,
            "expected_answer": {"resources_per_trial": {"num_cpus": 8, "num_gpus": 0}},
        }
    ),
    "hpo_without_anything_with_gpu": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_default_resources": {"num_cpus": 2, "num_gpus": 1},
            "num_trials": 2,
            "expected_answer": {"resources_per_trial": {"num_cpus": 8, "num_gpus": 2}},
        }
    ),
    "custom_hpo_with_total_resources_and_ag_args_ensemble_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 1},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "expected_answer": {"resources_per_trial": {"num_cpus": 2, "num_gpus": 1}},
        }
    ),
    "custom_hpo_with_total_resources_and_ag_args_ensemble": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_ensemble": {"num_cpus": 2, "num_gpus": 1},
            "model_default_resources": {"num_cpus": 2, "num_gpus": 1},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "expected_answer": {"resources_per_trial": {"num_cpus": 8, "num_gpus": 2}},  # ag_args_ensemble shouldn't affect hpo without bagging
        }
    ),
    "custom_hpo_with_total_resources_and_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 1},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "expected_answer": {"resources_per_trial": {"num_cpus": 2, "num_gpus": 1}},
        }
    ),
    "custom_hpo_with_ag_args_ensemble": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_ensemble": {"num_cpus": 2, "num_gpus": 1},
            "model_default_resources": {"num_cpus": 1, "num_gpus": 1},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "expected_answer": {"resources_per_trial": {"num_cpus": 1, "num_gpus": 1}},  # ag_args_ensemble shouldn't affect hpo without bagging
        }
    ),
    "custom_hpo_with_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 1},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "expected_answer": {"resources_per_trial": {"num_cpus": 2, "num_gpus": 1}},
        }
    ),
    "custom_hpo_without_anything": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_default_resources": {"num_cpus": 1, "num_gpus": 0},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "expected_answer": {"resources_per_trial": {"num_cpus": 1, "num_gpus": 0}},
        }
    ),
    "custom_hpo_without_anything_with_gpu": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_default_resources": {"num_cpus": 1, "num_gpus": 1},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "expected_answer": {"resources_per_trial": {"num_cpus": 1, "num_gpus": 1}},
        }
    ),
    "hpo_and_bagging_with_total_resources": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 1},
            "num_trials": 2,
            "num_bag_folds": 2,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 8, "num_gpus": 2},
                "resources_per_model": {"num_cpus": 4, "num_gpus": 1},
                "model_in_parallel": 2,  # This is models running in parallel in a bagged model
            },
        }
    ),
    "hpo_and_bagging_with_total_resources_and_ag_args_ensemble_and_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 0.5},
            "num_trials": 2,
            "num_bag_folds": 4,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 4, "num_gpus": 1},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
                "model_in_parallel": 2,  # This is models running in parallel in a bagged model
            },
        }
    ),
    "hpo_and_bagging_with_total_resources_and_ag_args_ensemble": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 1, "num_gpus": 0.5},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "num_trials": 2,
            "num_bag_folds": 4,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 4, "num_gpus": 1},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
                "model_in_parallel": 2,  # This is models running in parallel in a bagged model
            },
        }
    ),
    "hpo_and_bagging_with_total_resources_and_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 1, "num_gpus": 0.5},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 0.5},
            "num_trials": 2,
            "num_bag_folds": 4,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 8, "num_gpus": 2},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
                "model_in_parallel": 4,  # This is models running in parallel in a bagged model
            },
        }
    ),
    "hpo_and_bagging_with_ag_args_ensemble_and_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 0.5},
            "num_trials": 2,
            "num_bag_folds": 4,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 4, "num_gpus": 1},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
                "model_in_parallel": 2,  # This is models running in parallel in a bagged model
            },
        }
    ),
    "hpo_and_bagging_with_ag_args_ensemble": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 0.5},
            "num_trials": 2,
            "num_bag_folds": 4,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 4, "num_gpus": 1},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
                "model_in_parallel": 2,  # This is models running in parallel in a bagged model
            },
        }
    ),
    "hpo_and_bagging_with_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 0.5},
            "num_trials": 2,
            "num_bag_folds": 4,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 8, "num_gpus": 2},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
                "model_in_parallel": 4,  # This is models running in parallel in a bagged model
            },
        }
    ),
    "hpo_and_bagging_without_anything": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_default_resources": {"num_cpus": 1, "num_gpus": 0},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0},
            "num_trials": 2,
            "num_bag_folds": 4,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 8, "num_gpus": 0},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0},
                "model_in_parallel": 4,  # This is models running in parallel in a bagged model
            },
        }
    ),
    "hpo_and_bagging_without_anything_with_gpu": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_default_resources": {"num_cpus": 1, "num_gpus": 1},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 1},
            "num_trials": 2,
            "num_bag_folds": 4,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 16, "num_gpus": 4},
                "resources_per_model": {"num_cpus": 4, "num_gpus": 1},
                "model_in_parallel": 4,  # This is models running in parallel in a bagged model
            },
        }
    ),
    "hpo_and_sequential_bagging_with_total_resources_and_ag_args_ensemble_and_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 0.5},
            "num_trials": 2,
            "num_bag_folds": 4,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 4, "num_gpus": 1},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
            },
        }
    ),
    "hpo_and_sequential_bagging_with_total_resources_and_ag_args_ensemble": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "num_trials": 2,
            "num_bag_folds": 4,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 4, "num_gpus": 1},
                # "resources_per_model": {"num_cpus": 4, "num_gpus": 1},
                # FIXME: Above is commented out in v1.4 to fix bug in sequential.
                #  But this creates inconsistency with Parallel logic. Resolve in v1.5.
                "resources_per_model": {"num_cpus": 1, "num_gpus": 0},
            },
        }
    ),
    "hpo_and_sequential_bagging_with_total_resources_and_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "ag_args_fit": {"num_cpus": 4, "num_gpus": 1},
            "num_trials": 2,
            "num_bag_folds": 4,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "expected_answer": {
                # TODO: This is incorrect but doesn't cause big enough issue...
                # hpo resource calculator needs to know which folding strategy will be used, which is only being inferred later...
                # we calculate parallel folding for now...
                "resources_per_trial": {"num_cpus": 8, "num_gpus": 2},
                "resources_per_model": {"num_cpus": 4, "num_gpus": 1},
            },
        }
    ),
    "hpo_and_sequential_bagging_with_total_resources": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "model_default_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "num_trials": 2,
            "num_bag_folds": 4,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "expected_answer": {
                # TODO: This is incorrect but doesn't cause big enough issue...
                # hpo resource calculator needs to know which folding strategy will be used, which is only being inferred later...
                # we calculate parallel folding for now...
                "resources_per_trial": {"num_cpus": 8, "num_gpus": 2},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
            },
        }
    ),
    "custom_hpo_and_bagging_with_total_resources_and_ag_args_ensemble_and_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 0.5},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "num_bag_folds": 4,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 4, "num_gpus": 1},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
                "model_in_parallel": 2,
            },
        }
    ),
    "custom_hpo_and_bagging_with_total_resources_and_ag_args_ensemble": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "num_bag_folds": 4,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 4, "num_gpus": 1},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
                "model_in_parallel": 2,
            },
        }
    ),
    "custom_hpo_and_bagging_with_total_resources_and_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "ag_args_fit": {"num_cpus": 4, "num_gpus": 1},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "num_bag_folds": 4,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 8, "num_gpus": 2},
                "resources_per_model": {"num_cpus": 4, "num_gpus": 1},
                "model_in_parallel": 2,
            },
        }
    ),
    "custom_hpo_and_bagging_with_total_resources": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "num_bag_folds": 4,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 8, "num_gpus": 2},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
                "model_in_parallel": 4,
            },
        }
    ),
    "custom_hpo_and_sequential_bagging_with_total_resources_and_ag_args_ensemble_and_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "ag_args_fit": {"num_cpus": 2, "num_gpus": 0.5},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "num_bag_folds": 4,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 4, "num_gpus": 1},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
            },
        }
    ),
    "custom_hpo_and_sequential_bagging_with_total_resources_and_ag_args_ensemble": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "num_bag_folds": 4,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "expected_answer": {
                "resources_per_trial": {"num_cpus": 4, "num_gpus": 1},
                # "resources_per_model": {"num_cpus": 4, "num_gpus": 1},
                # FIXME: Above is commented out in v1.4 to fix bug in sequential.
                #  But this creates inconsistency with Parallel logic. Resolve in v1.5.
                "resources_per_model": {"num_cpus": 1, "num_gpus": 0},
            },
        }
    ),
    "custom_hpo_and_sequential_bagging_with_total_resources_and_ag_args_fit": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "ag_args_fit": {"num_cpus": 4, "num_gpus": 1},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "num_bag_folds": 4,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "expected_answer": {
                # TODO: This is incorrect but doesn't cause big enough issue...
                # hpo resource calculator needs to know which folding strategy will be used to determine if need to consider num_folds
                # but folding strategy is only being decided later...
                # we calculate parallel folding for now...
                "resources_per_trial": {"num_cpus": 8, "num_gpus": 2},
                "resources_per_model": {"num_cpus": 4, "num_gpus": 1},
            },
        }
    ),
    "custom_hpo_and_sequential_bagging_with_total_resources": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "model_minimum_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "model_default_resources": {"num_cpus": 2, "num_gpus": 0.5},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "num_bag_folds": 4,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "expected_answer": {
                # TODO: This is incorrect but doesn't cause big enough issue...
                # hpo resource calculator needs to know which folding strategy will be used to determine if need to consider num_folds
                # but folding strategy is only being decided later...
                # we calculate parallel folding for now...
                "resources_per_trial": {"num_cpus": 8, "num_gpus": 2},
                "resources_per_model": {"num_cpus": 2, "num_gpus": 0.5},
            },
        }
    ),
    # Should raise staring now
    "hpo_and_bagging_invalid_ag_args_ensemble_more_than_total": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_ensemble": {"num_cpus": 99, "num_gpus": 99},
            "num_trials": 2,
            "num_bag_folds": 2,
            "should_raise": True,
        }
    ),
    "hpo_and_bagging_invalid_ag_args_fit_more_than_total": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_fit": {"num_cpus": 99, "num_gpus": 99},
            "num_trials": 2,
            "num_bag_folds": 2,
            "should_raise": True,
        }
    ),
    "bagging_invalid_ag_args_ensemble_more_than_total": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_ensemble": {"num_cpus": 99, "num_gpus": 99},
            "num_bag_folds": 2,
            "should_raise": True,
        }
    ),
    "bagging_invalid_ag_args_fit_more_than_total": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_fit": {"num_cpus": 99, "num_gpus": 1},
            "num_bag_folds": 2,
            "should_raise": True,
        }
    ),
    "hpo_invalid_ag_args_fit_more_than_total": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_fit": {"num_cpus": 99, "num_gpus": 1},
            "num_trials": 2,
            "should_raise": True,
        }
    ),
    "custom_hpo_invalid_ag_args_fit_more_than_total": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_fit": {"num_cpus": 99, "num_gpus": 1},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "should_raise": True,
        }
    ),
    "sequential_bagging_invalid_ag_args_ensemble_more_than_total": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_ensemble": {"num_cpus": 99, "num_gpus": 99},
            "num_bag_folds": 2,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "should_raise": True,
        }
    ),
    "sequential_bagging_invalid_ag_args_fit_more_than_total": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_fit": {"num_cpus": 99, "num_gpus": 99},
            "num_bag_folds": 2,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "should_raise": True,
        }
    ),
    "invalid_ag_args_fit_more_than_total": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "total_resources": {"num_cpus": 8, "num_gpus": 2},
            "ag_args_fit": {"num_cpus": 99, "num_gpus": 99},
            "should_raise": True,
        }
    ),
    "hpo_and_bagging_invalid_ag_args_ensemble_less_than_min": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "num_trials": 2,
            "num_bag_folds": 2,
            "should_raise": True,
        }
    ),
    "hpo_and_bagging_invalid_ag_args_fit_less_than_min": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_fit": {"num_cpus": 4, "num_gpus": 1},
            "num_trials": 2,
            "num_bag_folds": 2,
            "should_raise": True,
        }
    ),
    "bagging_invalid_ag_args_ensemble_less_than_min": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "num_bag_folds": 2,
            "should_raise": True,
        }
    ),
    "bagging_invalid_ag_args_fit_less_than_min": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_fit": {"num_cpus": 4, "num_gpus": 1},
            "num_bag_folds": 2,
            "should_raise": True,
        }
    ),
    "sequential_bagging_invalid_ag_args_ensemble_less_than_min": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "num_bag_folds": 2,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "should_raise": True,
        }
    ),
    "sequential_bagging_invalid_ag_args_fit_less_than_min": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_fit": {"num_cpus": 4, "num_gpus": 1},
            "num_bag_folds": 2,
            "fold_strategy_cls": SequentialLocalFoldFittingStrategy,
            "should_raise": True,
        }
    ),
    "hpo_invalid_ag_args_ensemble_less_than_min": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "num_trials": 2,
            "should_raise": True,
        }
    ),
    "hpo_invalid_ag_args_fit_less_than_min": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_fit": {"num_cpus": 4, "num_gpus": 1},
            "num_trials": 2,
            "should_raise": True,
        }
    ),
    "custom_hpo_invalid_ag_args_ensemble_less_than_min": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "should_raise": True,
        }
    ),
    "custom_hpo_invalid_ag_args_fit_less_than_min": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_ensemble": {"num_cpus": 4, "num_gpus": 1},
            "num_trials": 2,
            "executor_cls": CustomHpoExecutor,
            "should_raise": True,
        }
    ),
    "invalid_ag_args_fit_less_than_min": (
        {
            "system_resources": {"num_cpus": 16, "num_gpus": 4},
            "model_minimum_resources": {"num_cpus": 16, "num_gpus": 4},
            "ag_args_fit": {"num_cpus": 4, "num_gpus": 1},
            "should_raise": True,
        }
    ),
}

ids = sorted(list(tests_dict.keys()))
tests = [tests_dict[id] for id in ids]


@pytest.mark.parametrize("test_args", tests, ids=ids)
def test_resource_allocation_combined_valid(mock_system_resources_ctx_mgr, test_args):
    should_raise = test_args.get("should_raise", False)
    if should_raise:
        with pytest.raises(AssertionError) as e:
            _test_functionality(mock_system_resources_ctx_mgr=mock_system_resources_ctx_mgr, test_args=test_args)
    else:
        _test_functionality(mock_system_resources_ctx_mgr=mock_system_resources_ctx_mgr, test_args=test_args)
