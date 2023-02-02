
from autogluon.core.models.dummy.dummy_model import DummyModel
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.metrics import METRICS


def test_dummy_binary(fit_helper):
    """Additionally tests that all binary metrics work"""
    fit_args = dict(
        hyperparameters={DummyModel: {}},
    )
    dataset_name = 'adult'
    extra_metrics = list(METRICS[BINARY])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_dummy_multiclass(fit_helper):
    """Additionally tests that all multiclass metrics work"""
    fit_args = dict(
        hyperparameters={DummyModel: {}},
    )
    extra_metrics = list(METRICS[MULTICLASS])

    dataset_name = 'covertype_small'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_dummy_regression(fit_helper):
    """Additionally tests that all regression metrics work"""
    fit_args = dict(
        hyperparameters={DummyModel: {}},
    )
    extra_metrics = list(METRICS[REGRESSION])

    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_dummy_quantile(fit_helper):
    fit_args = dict(
        hyperparameters={'DUMMY': {}},
    )
    dataset_name = 'ames'
    init_args = dict(problem_type='quantile', quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, init_args=init_args)


def test_dummy_binary_model(model_fit_helper):
    fit_args = dict()
    dataset_name = 'adult'
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=DummyModel(), fit_args=fit_args)


def test_dummy_multiclass_model(model_fit_helper):
    fit_args = dict()
    dataset_name = 'covertype_small'
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=DummyModel(), fit_args=fit_args)


def test_dummy_regression_model(model_fit_helper):
    fit_args = dict()
    dataset_name = 'ames'
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=DummyModel(), fit_args=fit_args)
