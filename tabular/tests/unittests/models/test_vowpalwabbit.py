
from autogluon.tabular.models.vowpalwabbit.vowpalwabbit_model import VowpalWabbitModel


def test_vowpalwabbit_binary(fit_helper):
    fit_args = dict(
        hyperparameters={VowpalWabbitModel: {}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_vowpalwabbit_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={VowpalWabbitModel: {}},
    )
    dataset_name = 'covertype_small'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_vowpalwabbit_regression(fit_helper):
    fit_args = dict(
        hyperparameters={VowpalWabbitModel: {}},
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
