
import shutil

from autogluon.core.constants import BINARY
from autogluon.core.metrics import METRICS


def test_no_weighted_ensemble(fit_helper):
    """Tests that fit_weighted_ensemble=False works"""
    fit_args = dict(
        hyperparameters={'GBM': {}},
        fit_weighted_ensemble=False,
    )
    dataset_name = 'adult'
    extra_metrics = list(METRICS[BINARY])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name,
                                        fit_args=fit_args,
                                        extra_metrics=extra_metrics,
                                        expected_model_count=1)


def test_max_sets(fit_helper):
    """Tests that max_sets works"""
    fit_args = dict(
        hyperparameters={'GBM': {'ag_args_ensemble': {'max_sets': 3}}},
        fit_weighted_ensemble=False,
        num_bag_folds=2,
        num_bag_sets=5,
    )
    dataset_name = 'adult'

    predictor = fit_helper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        expected_model_count=1,
        refit_full=False,
        delete_directory=False,
    )
    leaderboard = predictor.leaderboard(extra_info=True)
    # 2 folds * 3 sets = 6
    assert leaderboard.iloc[0]['num_models'] == 6
    shutil.rmtree(predictor.path, ignore_errors=True)


def test_num_folds(fit_helper):
    """Tests that num_folds works"""
    fit_args = dict(
        hyperparameters={'GBM': {'ag_args_ensemble': {'num_folds': 3}}},
        fit_weighted_ensemble=False,
        num_bag_folds=7,
        num_bag_sets=2,
    )
    dataset_name = 'adult'

    predictor = fit_helper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        expected_model_count=1,
        refit_full=False,
        delete_directory=False,
    )
    leaderboard = predictor.leaderboard(extra_info=True)
    # 3 folds * 2 sets = 6
    assert leaderboard.iloc[0]['num_models'] == 6
    shutil.rmtree(predictor.path, ignore_errors=True)


def test_num_folds_hpo(fit_helper):
    """Tests that num_folds works"""
    fit_args = dict(
        hyperparameters={'GBM': {'ag_args_ensemble': {'num_folds': 2}}},
        fit_weighted_ensemble=False,
        num_bag_folds=5,
        num_bag_sets=2,
        hyperparameter_tune_kwargs={
            'searcher': 'random',
            'scheduler': 'local',
            'num_trials': 2,
        },
    )
    dataset_name = 'adult'

    predictor = fit_helper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        expected_model_count=2,
        refit_full=False,
        delete_directory=False,
    )
    leaderboard = predictor.leaderboard(extra_info=True)
    # 2 folds * 2 sets = 4
    assert leaderboard.iloc[0]['num_models'] == 4
    assert leaderboard.iloc[1]['num_models'] == 4
    shutil.rmtree(predictor.path, ignore_errors=True)
