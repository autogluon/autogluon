import shutil

from autogluon.core.constants import BINARY
from autogluon.core.metrics import METRICS


def test_no_weighted_ensemble(fit_helper):
    """Tests that fit_weighted_ensemble=False works"""
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        fit_weighted_ensemble=False,
    )
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, expected_model_count=1)


def test_no_full_last_level_weighted_ensemble(fit_helper):
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
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, expected_model_count=4)


def test_no_full_last_level_weighted_ensemble_additionally(fit_helper):
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
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, expected_model_count=4)


def test_full_last_level_weighted_ensemble_additionally(fit_helper):
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
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, expected_model_count=5)

    fit_args["num_stack_levels"] = 0
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, expected_model_count=2)


def test_full_last_level_weighted_ensemble(fit_helper):
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
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, expected_model_count=4)


def test_max_sets(fit_helper):
    """Tests that max_sets works"""
    fit_args = dict(
        hyperparameters={"DUMMY": {"ag_args_ensemble": {"max_sets": 3}}},
        fit_weighted_ensemble=False,
        num_bag_folds=2,
        num_bag_sets=5,
    )
    dataset_name = "adult"

    predictor = fit_helper.fit_and_validate_dataset(
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


def test_num_folds(fit_helper):
    """Tests that num_folds works"""
    fit_args = dict(
        hyperparameters={"DUMMY": {"ag_args_ensemble": {"num_folds": 3}}},
        fit_weighted_ensemble=False,
        num_bag_folds=7,
        num_bag_sets=2,
    )
    dataset_name = "adult"

    predictor = fit_helper.fit_and_validate_dataset(
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


def test_num_folds_hpo(fit_helper):
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
    dataset_name = "adult"

    predictor = fit_helper.fit_and_validate_dataset(
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


def test_use_bag_holdout_calibrate(fit_helper):
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
    fit_helper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        init_args=init_args,
        fit_args=fit_args,
        expected_model_count=2,
        refit_full=False,
    )
