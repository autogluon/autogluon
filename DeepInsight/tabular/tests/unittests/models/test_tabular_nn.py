
from autogluon.tabular.models.tabular_nn.tabular_nn_model import TabularNeuralNetModel


def test_tabular_nn_binary(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetModel: {}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabular_nn_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetModel: {}},
    )
    dataset_name = 'covertype'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabular_nn_regression(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetModel: {}},
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
