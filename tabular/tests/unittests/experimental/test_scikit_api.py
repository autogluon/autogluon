from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.metrics import METRICS
from autogluon.tabular.models.lgb.lgb_model import LGBModel
from autogluon.tabular.testing import FitHelper


def test_scikit_api_binary():
    """Additionally tests that all binary metrics work"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    dataset_name = "toy_binary"
    extra_metrics = list(METRICS[BINARY])

    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, scikit_api=True
    )


def test_scikit_api_multiclass():
    """Additionally tests that all multiclass metrics work"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    extra_metrics = list(METRICS[MULTICLASS])

    dataset_name = "toy_multiclass"
    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, scikit_api=True
    )


def test_scikit_api_regression():
    """Additionally tests that all regression metrics work"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    extra_metrics = list(METRICS[REGRESSION])

    dataset_name = "toy_regression"
    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics, scikit_api=True
    )
