import sys

import pytest

from autogluon.tabular.models.vowpalwabbit.vowpalwabbit_model import VowpalWabbitModel
from autogluon.tabular.testing import FitHelper


@pytest.mark.skipif(sys.version_info >= (3, 11) or sys.platform=="darwin", reason="vowpalwabbit doesn't support python 3.11 and above or MacOS yet.")
def test_vowpalwabbit_binary():
    fit_args = dict(
        hyperparameters={VowpalWabbitModel: {}},
    )
    dataset_name = "adult"
    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


@pytest.mark.skipif(sys.version_info >= (3, 11) or sys.platform=="darwin", reason="vowpalwabbit doesn't support python 3.11 and above or MacOS yet.")
def test_vowpalwabbit_multiclass():
    fit_args = dict(
        hyperparameters={VowpalWabbitModel: {}},
    )
    dataset_name = "covertype_small"
    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


@pytest.mark.skipif(sys.version_info >= (3, 11) or sys.platform=="darwin", reason="vowpalwabbit doesn't support python 3.11 and above or MacOS yet.")
def test_vowpalwabbit_regression():
    fit_args = dict(
        hyperparameters={VowpalWabbitModel: {}},
    )
    dataset_name = "ames"
    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
