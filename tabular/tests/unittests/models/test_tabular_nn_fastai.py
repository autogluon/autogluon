
import pytest

from autogluon.tabular.models.fastainn.tabular_nn_fastai import NNFastAiTabularModel


# TODO: Add torch and fastai to CI to enable tests
@pytest.mark.skip(reason="torch and fastai are not installed during tests.")
def test_tabular_nn_fastai_binary(fit_helper):
    fit_args = dict(
        hyperparameters={NNFastAiTabularModel: {}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


@pytest.mark.skip(reason="torch and fastai are not installed during tests.")
def test_tabular_nn_fastai_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={NNFastAiTabularModel: {}},
    )
    dataset_name = 'covertype'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


@pytest.mark.skip(reason="torch and fastai are not installed during tests.")
def test_tabular_nn_fastai_regression(fit_helper):
    fit_args = dict(
        hyperparameters={NNFastAiTabularModel: {}},
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
