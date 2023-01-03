
from autogluon.tabular.models.lgb.lgb_model import LGBModel
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.metrics import METRICS


def test_lightgbm_binary(fit_helper):
    """Additionally tests that all binary metrics work"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    dataset_name = 'adult'
    extra_metrics = list(METRICS[BINARY])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_lightgbm_multiclass(fit_helper):
    """Additionally tests that all multiclass metrics work"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    extra_metrics = list(METRICS[MULTICLASS])

    dataset_name = 'covertype_small'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_lightgbm_regression(fit_helper):
    """Additionally tests that all regression metrics work"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    extra_metrics = list(METRICS[REGRESSION])

    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_lightgbm_binary_model(model_fit_helper):
    fit_args = dict()
    dataset_name = 'adult'
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=LGBModel(), fit_args=fit_args)


def test_lightgbm_multiclass_model(model_fit_helper):
    fit_args = dict()
    dataset_name = 'covertype_small'
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=LGBModel(), fit_args=fit_args)


def test_lightgbm_regression_model(model_fit_helper):
    fit_args = dict()
    dataset_name = 'ames'
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=LGBModel(), fit_args=fit_args)
