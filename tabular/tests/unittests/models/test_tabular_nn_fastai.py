import numpy as np
import pandas as pd
from autogluon.tabular.models.fastainn.tabular_nn_fastai import NNFastAiTabularModel


def test_tabular_nn_fastai_binary(fit_helper):
    fit_args = dict(
        hyperparameters={NNFastAiTabularModel: {}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabular_nn_fastai_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={NNFastAiTabularModel: {}},
    )
    dataset_name = 'covertype'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabular_nn_fastai_regression(fit_helper):
    fit_args = dict(
        hyperparameters={NNFastAiTabularModel: {}},
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_batch_size():
    model = NNFastAiTabularModel()
    model.params['bs'] = 'adaptive'
    assert model._get_bs(pd.Series(np.arange(200000))) == 512
    assert model._get_bs(pd.Series(np.arange(199999))) == 256
    assert model._get_bs(pd.Series(np.arange(42))) == 42

    model.params['bs'] = 99
    assert model._get_bs(pd.Series(np.arange(200000))) == 99
    assert model._get_bs(pd.Series(np.arange(199999))) == 99
    assert model._get_bs(pd.Series(np.arange(42))) == 42
