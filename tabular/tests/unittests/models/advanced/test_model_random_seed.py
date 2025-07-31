"""
Unit tests to ensure correctness random seed logic
"""

import os
import uuid
from copy import deepcopy
from autogluon.core.utils import generate_train_test_split_combined
from autogluon.tabular import TabularPredictor
from autogluon.tabular.testing import FitHelper
import pytest

TEST_CASES = [
    ({"GBM": {"num_boost_round": 10}}, {}, list(range(3))),
    ({"GBM": {"num_boost_round": 10}}, {"vary_seed_across_folds": False}, [0] * 3),
    ({"GBM": {"num_boost_round": 10}}, {"model_random_seed": 42}, [42, 43, 44]),
    ({"GBM": {"num_boost_round": 10}}, {"vary_seed_across_folds": False, "model_random_seed": 42}, [42] * 3),
    ({"GBM": {"num_boost_round": 10, "ag_args_ensemble": {"vary_seed_across_folds": False, "model_random_seed": 42}}}, {}, [42] * 3),
]

@pytest.mark.parametrize(
    "hyperparameters, ag_args_ensemble, expected_random_seeds",
    TEST_CASES,
)
def test_bagged_random_seed(hyperparameters, ag_args_ensemble, expected_random_seeds):
    """
    Tests that the random seeds for bagged models are correct.

    Tests 4 fit types:
    sequential fit + sequential bag
    sequential fit + parallel bag
    parallel fit + parallel bag
    parallel fit + sequential bag
    """
    sample_size = 50
    dataset_name = "adult"
    directory_prefix = "./datasets/"
    train_data, _, dataset_info = FitHelper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)
    label = dataset_info["label"]
    init_args = dict(
        label=label,
        eval_metric="log_loss",
        problem_type=dataset_info["problem_type"]
    )
    save_path = os.path.join(directory_prefix, dataset_name, f"AutogluonOutput_{uuid.uuid4()}")
    train_data, _ = generate_train_test_split_combined(
        data=train_data,
        label=init_args["label"],
        problem_type=init_args["problem_type"],
        train_size=sample_size,
    )

    fit_args = dict(
        train_data=train_data,
        num_bag_folds=3,
        num_stack_levels=0,
        calibrate=False,  # ensure calibration is also deterministic
        dynamic_stacking=False,
        fit_weighted_ensemble=False,
    )


    for fit_strategy, fold_fitting_strategy in [
        ("sequential", "sequential_local"),
        ("sequential", None),
        ("parallel", "sequential_local"),
        ("parallel", None),
    ]:
        if fold_fitting_strategy is not  None:
            ag_args_ensemble = deepcopy(ag_args_ensemble)
            ag_args_ensemble["fold_fitting_strategy"] = fold_fitting_strategy

        predictor = TabularPredictor(path=save_path, **init_args).fit(
            hyperparameters=hyperparameters,
            fit_strategy=fit_strategy,
            ag_args_ensemble=ag_args_ensemble,
            **fit_args,
        )

        for model_name in predictor.model_names():
            model = predictor._trainer.load_model(model_name)
            for child_i, child_name in enumerate(model.models):
                child_model = model.load_child(child_name)
                expected_random_seed = expected_random_seeds[child_i]
                assert child_model.random_seed == expected_random_seed, (
                    f"Random seed for bagged model should be {expected_random_seed}, but got {child_model.random_seed}"
                )




