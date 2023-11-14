import os
from pathlib import Path

import pytest

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.metrics import METRICS
from autogluon.core.models.dummy.dummy_model import DummyModel


def test_no_models(fit_helper, dataset_loader_helper):
    """Tests that logic works properly when no models are trained"""
    fit_args = dict(
        hyperparameters={},
    )

    dataset_name = "adult"
    directory_prefix = "./datasets/"
    train_data, test_data, dataset_info = dataset_loader_helper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)

    predictor = fit_helper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)

    assert not predictor.model_names()
    with pytest.raises(AssertionError):
        predictor.predict(test_data)
    assert len(predictor.leaderboard()) == 0
    assert len(predictor.leaderboard(test_data)) == 0
    assert len(predictor.model_failures()) == 0


def test_no_models_raise(fit_helper, dataset_loader_helper):
    """Tests that logic works properly when no models are trained, and tests predictor.model_failures()"""

    expected_exc_str = "Test Error Message"

    # Force DummyModel to raise an exception when fit.
    fit_args = dict(
        hyperparameters={DummyModel: {"raise": ValueError, "raise_msg": expected_exc_str}},
    )

    dataset_name = "adult"
    directory_prefix = "./datasets/"
    train_data, test_data, dataset_info = dataset_loader_helper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)

    predictor = fit_helper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)

    assert not predictor.model_names()
    with pytest.raises(AssertionError):
        predictor.predict(test_data)
    assert len(predictor.leaderboard()) == 0
    assert len(predictor.leaderboard(test_data)) == 0

    model_failures = predictor.model_failures()
    assert len(model_failures) == 1
    model_failures_dict = model_failures.iloc[0].to_dict()
    assert model_failures_dict["model"] == "DummyModel"
    assert model_failures_dict["model_type"] == "DummyModel"
    assert model_failures_dict["exc_type"] == "ValueError"
    assert model_failures_dict["exc_str"] == expected_exc_str


def test_dummy_binary(fit_helper):
    """Additionally tests that all binary metrics work"""
    fit_args = dict(
        hyperparameters={DummyModel: {}},
    )
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_dummy_multiclass(fit_helper):
    """Additionally tests that all multiclass metrics work"""
    fit_args = dict(
        hyperparameters={DummyModel: {}},
    )
    extra_metrics = list(METRICS[MULTICLASS])

    dataset_name = "covertype_small"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_dummy_regression(fit_helper):
    """Additionally tests that all regression metrics work"""
    fit_args = dict(
        hyperparameters={DummyModel: {}},
    )
    extra_metrics = list(METRICS[REGRESSION])

    dataset_name = "ames"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_dummy_quantile(fit_helper):
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
    )
    dataset_name = "ames"
    init_args = dict(problem_type="quantile", quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, init_args=init_args)


def test_dummy_binary_model(model_fit_helper):
    fit_args = dict()
    dataset_name = "adult"
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=DummyModel(), fit_args=fit_args)


def test_dummy_multiclass_model(model_fit_helper):
    fit_args = dict()
    dataset_name = "covertype_small"
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=DummyModel(), fit_args=fit_args)


def test_dummy_regression_model(model_fit_helper):
    fit_args = dict()
    dataset_name = "ames"
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=DummyModel(), fit_args=fit_args)


def test_dummy_binary_absolute_path(fit_helper):
    """Test that absolute path works"""
    fit_args = dict(
        hyperparameters={DummyModel: {}},
    )
    path = Path(".") / "AG_test"
    path = str(path.resolve())
    init_args = dict(path=path)

    dataset_name = "adult"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, init_args=init_args, fit_args=fit_args)


def test_dummy_binary_absolute_path_stack(fit_helper):
    """Test that absolute path works"""
    fit_args = dict(
        hyperparameters={DummyModel: {}},
        num_bag_folds=2,
        num_bag_sets=2,
        num_stack_levels=1,
    )

    dataset_name = "adult"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, expected_model_count=4, path_as_absolute=True)


def test_dummy_binary_model_absolute_path(model_fit_helper):
    """Test that absolute path works"""
    fit_args = dict()
    path = Path(".") / "AG_test"
    path = str(path.resolve())
    model = DummyModel(path=path)
    dataset_name = "adult"
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=model, fit_args=fit_args)
