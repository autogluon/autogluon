from autogluon.tabular.models import EBMModel
from autogluon.tabular.testing import FitHelper


def test_ebm_binary():
    fit_args = dict(
        hyperparameters={EBMModel: {}},
    )
    dataset_name = "toy_binary"
    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, refit_full=False)

def test_ebm_multiclass():
    fit_args = dict(
        hyperparameters={EBMModel: {}},
    )
    dataset_name = "covertype_small"
    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, refit_full=False)

def test_ebm_regression():
    fit_args = dict(
        hyperparameters={EBMModel: {}},
    )
    dataset_name = "ames"
    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, refit_full=False)

