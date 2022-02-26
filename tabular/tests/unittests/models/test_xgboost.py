
from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel


def test_xgboost_binary(fit_helper):
    fit_args = dict(
        hyperparameters={XGBoostModel: {}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_xgboost_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={XGBoostModel: {}},
    )
    dataset_name = 'covertype_small'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_xgboost_regression(fit_helper):
    fit_args = dict(
        hyperparameters={XGBoostModel: {}},
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
