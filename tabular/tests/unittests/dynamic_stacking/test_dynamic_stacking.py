import shutil

import pytest

from autogluon.core.constants import BINARY
from autogluon.core.metrics import METRICS
from autogluon.tabular.testing import FitHelper
from autogluon.tabular.testing.fit_helper import stacked_overfitting_assert

DS_ARGS_TEST_DEFAULTS = dict(
    validation_procedure="holdout",
    detection_time_frac=1 / 4,
    holdout_frac=1 / 9,
    n_folds=2,
    n_repeats=1,
    memory_safe_fits=True,
    clean_up_fits=True,
    holdout_data=None,
)


def test_spot_and_avoid_stacked_overfitting():
    """Tests that dynamic stacking works."""
    fit_args = dict(
        hyperparameters={"RF": {}, "GBM": {}},
        fit_weighted_ensemble=False,
        dynamic_stacking=True,
        num_stack_levels=1,
        num_bag_folds=2,
        num_bag_sets=1,
        time_limit=None,
        ds_args=DS_ARGS_TEST_DEFAULTS,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        extra_metrics=extra_metrics,
        expected_model_count=2,
        refit_full=False,
        allowed_dataset_features=["age"],
        expected_stacked_overfitting_at_test=False,
        expected_stacked_overfitting_at_val=True,
    )


def test_dynamic_stacking_hps():
    """Tests dynamic stacking arguments."""
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        fit_weighted_ensemble=False,
        dynamic_stacking=True,
        num_stack_levels=1,
        num_bag_folds=2,
        num_bag_sets=1,
        time_limit=None,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )

    # Get custom val data (the test data)
    train_data, test_data, dataset_info = FitHelper.load_dataset(name="adult", directory_prefix="./datasets/")
    label = dataset_info["label"]
    allowed_cols = ["age", label]
    train_data = train_data[allowed_cols]
    test_data = test_data[allowed_cols]
    n_test_data = len(test_data)

    for ds_args_update in [
        dict(validation_procedure="holdout", holdout_frac=1 / 5),  # holdout
        dict(validation_procedure="cv"),  # 2-fold CV
        dict(validation_procedure="cv", n_repeats=2),  # 2-repeated 2-fold CV
        dict(memory_safe_fits=False, clean_up_fits=False),  # fit options False
        dict(holdout_data=test_data),
        dict(holdout_data=test_data, validation_procedure="cv", expect_raise=ValueError),
    ]:
        expect_raise = ds_args_update.pop("expect_raise", None)
        tmp_ds_args = DS_ARGS_TEST_DEFAULTS.copy()
        if ds_args_update is not None:
            tmp_ds_args.update(ds_args_update)
        tmp_fit_args = fit_args.copy()
        tmp_fit_args["ds_args"] = tmp_ds_args
        if expect_raise is None:
            predictor = FitHelper.fit_dataset(
                train_data=train_data, init_args=dict(label=label), fit_args=tmp_fit_args, sample_size=1000
            )
            if ("holdout_data" in ds_args_update) and (ds_args_update["holdout_data"] is not None):
                n_expected = 1000 + n_test_data
                assert len(predictor.predict_oof()) == n_expected, "Verify that holdout data was used for training"
            lb = predictor.leaderboard(test_data, extra_info=True)
            stacked_overfitting_assert(lb, predictor, False, False)
            shutil.rmtree(predictor.path)
        else:
            with pytest.raises(expect_raise):
                FitHelper.fit_dataset(
                    train_data=train_data, init_args=dict(label=label), fit_args=tmp_fit_args, sample_size=1000
                )


def test_no_dynamic_stacking():
    """Tests that dynamic stacking does not run if stacking is disabled."""
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        dynamic_stacking=True,
        fit_weighted_ensemble=False,
        num_stack_levels=0,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        extra_metrics=extra_metrics,
        expected_model_count=1,
        refit_full=False,
    )
    assert predictor._stacked_overfitting_occurred is None


def test_dynamic_stacking_fit_extra():
    """Tests that fit_extra works after dynamic stacking."""
    fit_args = dict(
        hyperparameters={"RF": {}},
        dynamic_stacking=True,
        fit_weighted_ensemble=False,
        num_bag_folds=2,
        num_bag_sets=1,
        num_stack_levels=1,
        ds_args=DS_ARGS_TEST_DEFAULTS,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        extra_metrics=extra_metrics,
        expected_model_count=1,
        refit_full=False,
        delete_directory=False,
        allowed_dataset_features=["age"],
        expected_stacked_overfitting_at_test=False,
        # This also check that we only consider something to be stacked overfitting if the dynamic stacking holdout score gets worse.
        expected_stacked_overfitting_at_val=True,
    )

    fit_extra_args = dict(
        hyperparameters={"GBM": {}},
        fit_weighted_ensemble=False,
    )

    predictor.fit_extra(**fit_extra_args)

    assert len(predictor.model_names()) == 2
    shutil.rmtree(predictor.path, ignore_errors=True)


def test_dynamic_stacking_with_time_limit():
    """Tests that dynamic stacking does not run if stacking is disabled."""
    ds_args = DS_ARGS_TEST_DEFAULTS.copy()
    ds_args["holdout_frac"] = 0.5
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        dynamic_stacking=True,
        fit_weighted_ensemble=False,
        num_bag_folds=2,
        num_bag_sets=1,
        num_stack_levels=1,
        time_limit=60,  # won't take 60s, but we need a number here instead of None.
        ds_args=ds_args,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        extra_metrics=extra_metrics,
        expected_model_count=2,
        refit_full=False,
        delete_directory=False,
        allowed_dataset_features=["age"],
        expected_stacked_overfitting_at_test=False,
        expected_stacked_overfitting_at_val=False,
    )


@pytest.mark.timeout(
    120
)  # if running AutoGluon twice fails due to a multiprocessing bug, we want to hang up and crash.
def test_dynamic_stacking_run_twice_parallel_fold_fitting_strategy():
    """Tests that dynamic stacking memory save fit works."""
    ds_args = DS_ARGS_TEST_DEFAULTS.copy()
    ds_args["memory_safe_fits"] = True  # guarantee for sanity
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        fit_weighted_ensemble=False,
        dynamic_stacking=True,
        num_stack_levels=1,
        num_bag_folds=2,
        num_bag_sets=1,
        time_limit=None,
        ds_args=ds_args,
    )

    # Get custom val data (the test data)
    train_data, test_data, dataset_info = FitHelper.load_dataset(name="adult", directory_prefix="./datasets/")
    label = dataset_info["label"]
    allowed_cols = ["age", label]
    train_data = train_data[allowed_cols]
    test_data = test_data[allowed_cols]

    for _ in range(2):
        predictor = FitHelper.fit_dataset(
            train_data=train_data, init_args=dict(label=label), fit_args=fit_args, sample_size=1000
        )
        lb = predictor.leaderboard(test_data, extra_info=True)
        stacked_overfitting_assert(lb, predictor, False, False)
        shutil.rmtree(predictor.path)
