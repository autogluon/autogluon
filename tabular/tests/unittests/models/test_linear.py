
from autogluon.tabular.models.lr.lr_model import LinearModel


def test_linear_binary(fit_helper):
    fit_args = dict(
        hyperparameters={LinearModel: {}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_linear_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={LinearModel: {}},
    )
    dataset_name = 'covertype_small'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_linear_regression(fit_helper):
    fit_args = dict(
        hyperparameters={LinearModel: {}},
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
