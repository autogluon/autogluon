import sys

import pytest

from autogluon.tabular.models.interpret.ebm_model import EBMModel


def test_ebm_binary(fit_helper):
    fit_args = dict(
        hyperparameters={EBMModel: {}},
    )
    dataset_name = "adult"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_ebm_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={EBMModel: {}},
    )
    dataset_name = "covertype_small"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_ebm_regression(fit_helper):
    fit_args = dict(
        hyperparameters={EBMModel: {}},
    )
    dataset_name = "ames"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)

