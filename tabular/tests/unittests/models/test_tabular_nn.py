
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel


def test_tabular_nn_binary(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabular_nn_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
    )
    dataset_name = 'covertype_small'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabular_nn_regression(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
        time_limit=10,  # TabularNN trains for a long time on ames
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
