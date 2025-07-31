"""
Unit tests to ensure correctness random seed logic
"""

import os
import uuid
from copy import deepcopy
from autogluon.tabular import TabularPredictor
from autogluon.tabular.testing import FitHelper
import pytest

TEST_CASES = []

for model_key, model_hps in [
    ("GBM", {"num_boost_round": 10}),
    # The below is only for full extended tests only for sanity check, not the CI
    # ("CAT", {"iterations": 10}),
    # ("FASTAI", {"epochs": 10}),
]:
    # Vary default
    TEST_CASES.append(({model_key: model_hps}, {"vary_seed_across_folds": True}, list(range(3))))
    # No vary default
    TEST_CASES.append(
        ({model_key: model_hps}, {"vary_seed_across_folds": False}, [0] * 3),
    )
    # Different fixed seed
    TEST_CASES.append(({model_key: model_hps}, {"vary_seed_across_folds": False, "model_random_seed": 42}, [42] * 3))
    # Two models, two different sets of seeds
    model_hps_2nd_model = deepcopy(model_hps)
    model_hps_2nd_model["ag_args_ensemble"] = {"model_random_seed": 42}
    TEST_CASES.append(
        ({model_key: [model_hps, model_hps_2nd_model]}, {"vary_seed_across_folds": True}, ([0, 1, 2], [42, 43, 44]))
    )
    # Vary via model HPs instead of fit args
    model_hps = deepcopy(model_hps)
    model_hps["ag_args_ensemble"] = {"vary_seed_across_folds": False, "model_random_seed": 42}
    TEST_CASES.append(({model_key: model_hps}, {}, [42] * 3))


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
    directory_prefix = "./datasets/"
    dataset_name = "toy_binary"
    train_data, _, dataset_info = FitHelper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)
    label = dataset_info["label"]
    init_args = dict(label=label, eval_metric="log_loss", problem_type=dataset_info["problem_type"])
    save_path = os.path.join(directory_prefix, dataset_name, f"AutogluonOutput_{uuid.uuid4()}")

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
        if fold_fitting_strategy is not None:
            ag_args_ensemble = deepcopy(ag_args_ensemble)
            ag_args_ensemble["fold_fitting_strategy"] = fold_fitting_strategy

        predictor = TabularPredictor(path=save_path, **init_args).fit(
            hyperparameters=hyperparameters,
            fit_strategy=fit_strategy,
            ag_args_ensemble=ag_args_ensemble,
            **fit_args,
        )

        for model_i, model_name in enumerate(predictor.model_names()):
            model = predictor._trainer.load_model(model_name)
            _expected_random_seeds = (
                expected_random_seeds[model_i] if isinstance(expected_random_seeds, tuple) else expected_random_seeds
            )
            for child_i, child_name in enumerate(model.models):
                child_model = model.load_child(child_name)
                expected_random_seed = _expected_random_seeds[child_i]
                assert child_model.random_seed == expected_random_seed, (
                    f"Random seed for bagged model should be {expected_random_seed}, but got {child_model.random_seed}"
                )
