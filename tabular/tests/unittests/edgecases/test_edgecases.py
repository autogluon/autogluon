import shutil
from pathlib import Path

import pytest

from autogluon.core.constants import BINARY
from autogluon.core.metrics import METRICS
from autogluon.tabular.testing import FitHelper


def test_no_weighted_ensemble():
    """Tests that fit_weighted_ensemble=False works"""
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        fit_weighted_ensemble=False,
    )
    dataset_name = "toy_binary"
    extra_metrics = list(METRICS[BINARY])

    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, expected_model_count=1
    )


def test_no_full_last_level_weighted_ensemble():
    """Tests that fit_weighted_ensemble=False works"""
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        fit_weighted_ensemble=True,
        fit_full_last_level_weighted_ensemble=False,
        num_stack_levels=1,
        num_bag_folds=2,
        num_bag_sets=1,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "toy_binary"
    extra_metrics = list(METRICS[BINARY])

    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, expected_model_count=4
    )


def test_no_full_last_level_weighted_ensemble_additionally():
    """Tests that fit_weighted_ensemble=False works"""
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        fit_weighted_ensemble=True,
        fit_full_last_level_weighted_ensemble=False,
        full_weighted_ensemble_additionally=False,
        num_stack_levels=1,
        num_bag_folds=2,
        num_bag_sets=1,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "toy_binary"
    extra_metrics = list(METRICS[BINARY])

    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, expected_model_count=4
    )


def test_full_last_level_weighted_ensemble_additionally():
    """Tests that fit_weighted_ensemble=False works"""
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        fit_weighted_ensemble=True,
        fit_full_last_level_weighted_ensemble=True,
        full_weighted_ensemble_additionally=True,
        num_stack_levels=1,
        num_bag_folds=2,
        num_bag_sets=1,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "toy_binary"
    extra_metrics = list(METRICS[BINARY])

    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, expected_model_count=5
    )

    fit_args["num_stack_levels"] = 0
    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, expected_model_count=2
    )


def test_full_last_level_weighted_ensemble():
    """Tests that fit_weighted_ensemble=False works"""
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        fit_weighted_ensemble=True,
        fit_full_last_level_weighted_ensemble=True,
        num_stack_levels=1,
        num_bag_folds=2,
        num_bag_sets=1,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "toy_binary"
    extra_metrics = list(METRICS[BINARY])

    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, expected_model_count=4
    )


def test_max_sets():
    """Tests that max_sets works"""
    fit_args = dict(
        hyperparameters={"DUMMY": {"ag_args_ensemble": {"max_sets": 3}}},
        fit_weighted_ensemble=False,
        num_bag_folds=2,
        num_bag_sets=5,
    )
    dataset_name = "toy_binary"

    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        expected_model_count=1,
        refit_full=False,
        delete_directory=False,
    )
    leaderboard = predictor.leaderboard(extra_info=True)
    # 2 folds * 3 sets = 6
    assert leaderboard.iloc[0]["num_models"] == 6
    shutil.rmtree(predictor.path, ignore_errors=True)


def test_num_folds():
    """Tests that num_folds works"""
    fit_args = dict(
        hyperparameters={"DUMMY": {"ag_args_ensemble": {"num_folds": 3}}},
        fit_weighted_ensemble=False,
        num_bag_folds=7,
        num_bag_sets=2,
    )
    dataset_name = "toy_binary"

    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        expected_model_count=1,
        refit_full=False,
        delete_directory=False,
    )
    leaderboard = predictor.leaderboard(extra_info=True)
    # 3 folds * 2 sets = 6
    assert leaderboard.iloc[0]["num_models"] == 6
    shutil.rmtree(predictor.path, ignore_errors=True)


def test_num_folds_hpo():
    """Tests that num_folds works"""
    fit_args = dict(
        hyperparameters={"GBM": {"ag_args_ensemble": {"num_folds": 2}}},
        fit_weighted_ensemble=False,
        num_bag_folds=5,
        num_bag_sets=2,
        hyperparameter_tune_kwargs={
            "searcher": "random",
            "scheduler": "local",
            "num_trials": 2,
        },
    )
    dataset_name = "toy_binary"

    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        expected_model_count=2,
        refit_full=False,
        delete_directory=False,
    )
    leaderboard = predictor.leaderboard(extra_info=True)
    # 2 folds * 2 sets = 4
    assert leaderboard.iloc[0]["num_models"] == 4
    assert leaderboard.iloc[1]["num_models"] == 4
    shutil.rmtree(predictor.path, ignore_errors=True)


def test_use_bag_holdout_calibrate():
    """
    Test that use_bag_holdout=True works for calibration
    Ensures the bug is fixed in https://github.com/autogluon/autogluon/issues/2674
    """
    init_args = dict(eval_metric="log_loss")

    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        num_bag_folds=2,
        use_bag_holdout=True,
        calibrate=True,
    )

    dataset_name = "adult"
    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        init_args=init_args,
        fit_args=fit_args,
        expected_model_count=2,
        refit_full=False,
    )


def test_num_folds_parallel(capsys):
    """Tests that num_folds_parallel equal to 1 works"""
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        fit_weighted_ensemble=False,
        num_bag_folds=2,
        num_bag_sets=1,
        ag_args_ensemble=dict(num_folds_parallel=1),
    )
    dataset_name = "toy_binary"

    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        expected_model_count=1,
        refit_full=False,
        delete_directory=False,
    )
    leaderboard = predictor.leaderboard(extra_info=True)
    assert leaderboard.iloc[0]["num_models"] == 2
    shutil.rmtree(predictor.path, ignore_errors=True)


def test_raises_num_cpus_float():
    """Tests that num_cpus specified as a float raises a TypeError"""
    fit_args = dict(num_cpus=1.0)
    dataset_name = "toy_binary"
    with pytest.raises(TypeError, match=r"`num_cpus` must be an int or 'auto'. Found: .*"):
        FitHelper.fit_and_validate_dataset(
            dataset_name=dataset_name,
            fit_args=fit_args,
            expected_model_count=None,
            delete_directory=True,
        )


def test_raises_num_cpus_zero():
    """Tests that num_cpus=0 raises a ValueError"""
    fit_args = dict(num_cpus=0)
    dataset_name = "toy_binary"
    with pytest.raises(ValueError, match=r"`num_cpus` must be greater than or equal to 1. .*"):
        FitHelper.fit_and_validate_dataset(
            dataset_name=dataset_name,
            fit_args=fit_args,
            expected_model_count=None,
            delete_directory=True,
        )


def test_raises_num_gpus_neg():
    """Tests that num_gpus<0 raises a ValueError"""
    fit_args = dict(num_gpus=-1)
    dataset_name = "toy_binary"
    with pytest.raises(ValueError, match=r"`num_gpus` must be greater than or equal to 0. .*"):
        FitHelper.fit_and_validate_dataset(
            dataset_name=dataset_name,
            fit_args=fit_args,
            expected_model_count=None,
            delete_directory=True,
        )


@pytest.mark.parametrize("delay_bag_sets", [True, False])
def test_delay_bag_sets(delay_bag_sets):
    """Tests that max_sets works"""
    fit_args = dict(
        hyperparameters={"DUMMY": [{}, {}]},
        fit_weighted_ensemble=False,
        num_bag_folds=2,
        num_bag_sets=2,
        time_limit=30,  # has no impact, but otherwise `delay_bag_sets` is ignored.
        delay_bag_sets=delay_bag_sets,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "toy_binary"

    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        expected_model_count=2,
        refit_full=False,
        delete_directory=False,
    )

    # Verify fit order is correct.
    model_1 = predictor._trainer.load_model("Dummy_BAG_L1")
    max_model_times_1 = max([(Path(model_1.path) / bm).stat().st_mtime_ns for bm in model_1.models])

    model_2 = predictor._trainer.load_model("Dummy_2_BAG_L1")
    min_model_times_2 = min([(Path(model_2.path) / bm).stat().st_mtime_ns for bm in model_2.models])

    if delay_bag_sets:
        # Last model of model 1 should be trained after fist model of model 2
        assert max_model_times_1 > min_model_times_2
    else:
        # Last model of model 1 should be trained before fist model of model 2
        assert max_model_times_1 <= min_model_times_2

    shutil.rmtree(predictor.path, ignore_errors=True)
