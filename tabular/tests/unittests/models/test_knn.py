
from autogluon.tabular.models.knn.knn_model import KNNModel


def test_knn_binary(fit_helper):
    fit_args = dict(
        hyperparameters={KNNModel: {}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_knn_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={KNNModel: {}},
    )
    dataset_name = 'covertype_small'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_knn_regression(fit_helper):
    fit_args = dict(
        hyperparameters={KNNModel: {}},
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
