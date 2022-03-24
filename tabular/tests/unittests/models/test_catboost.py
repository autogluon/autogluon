
from autogluon.tabular.models.catboost.catboost_model import CatBoostModel


def test_catboost_binary(fit_helper):
    fit_args = dict(
        hyperparameters={CatBoostModel: {}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_catboost_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={CatBoostModel: {}},
    )
    dataset_name = 'covertype_small'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_catboost_regression(fit_helper):
    fit_args = dict(
        hyperparameters={CatBoostModel: {}},
        time_limit=10,  # CatBoost trains for a very long time on ames (many iterations)
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
