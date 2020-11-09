
from autogluon.tabular.models.lgb.lgb_model import LGBModel


def test_lightgbm_binary(fit_helper):
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_lightgbm_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    dataset_name = 'covertype'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_lightgbm_regression(fit_helper):
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
