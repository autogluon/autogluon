import sys

import pytest

from autogluon.tabular.models.vowpalwabbit.vowpalwabbit_model import VowpalWabbitModel


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="vowpalwabbit doesn't support python 3.11 and above yet")
def test_vowpalwabbit_binary(fit_helper):
    fit_args = dict(
        hyperparameters={VowpalWabbitModel: {}},
    )
    dataset_name = "adult"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="vowpalwabbit doesn't support python 3.11 and above yet")
def test_vowpalwabbit_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={VowpalWabbitModel: {}},
    )
    dataset_name = "covertype_small"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="vowpalwabbit doesn't support python 3.11 and above yet")
def test_vowpalwabbit_regression(fit_helper):
    fit_args = dict(
        hyperparameters={VowpalWabbitModel: {}},
    )
    dataset_name = "ames"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
