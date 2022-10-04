
from autogluon.tabular.models.pecos_tabular.pecos_model import PecosModel


def test_pecos_binary(fit_helper):
    fit_args = dict(
        hyperparameters={PecosModel: {}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_pecos_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={PecosModel: {}},
    )
    dataset_name = 'covertype_small'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_pecos_regression(fit_helper):
    fit_args = dict(
        hyperparameters={PecosModel: {}},
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
