import os
from pathlib import Path

import pytest

from autogluon.core.models.dummy.dummy_model import DummyModel
from autogluon.tabular.testing import FitHelper, ModelFitHelper


def test_no_models_will_raise():
    """Tests that RuntimeError is raised when no models fit"""
    fit_args = dict(
        hyperparameters={},
    )

    dataset_name = "toy_binary"
    train_data, test_data, dataset_info = FitHelper.load_dataset(name=dataset_name)

    with pytest.raises(RuntimeError):
        FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)


def test_no_models():
    """Tests that logic works properly when no models are trained and raise_on_no_models_fitted=False"""
    fit_args = dict(
        hyperparameters={},
        raise_on_no_models_fitted=False,
    )

    dataset_name = "toy_binary"
    train_data, test_data, dataset_info = FitHelper.load_dataset(name=dataset_name)

    predictor = FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)

    assert not predictor.model_names()
    with pytest.raises(AssertionError):
        predictor.predict(test_data)
    assert len(predictor.leaderboard()) == 0
    assert len(predictor.leaderboard(test_data)) == 0
    assert len(predictor.model_failures()) == 0


def test_no_models_raise():
    """Tests that logic works properly when no models are trained, and tests predictor.model_failures() and raise_on_no_models_fitted=False"""

    expected_exc_str = "Test Error Message"

    # Force DummyModel to raise an exception when fit.
    fit_args = dict(
        hyperparameters={DummyModel: {"raise": ValueError, "raise_msg": expected_exc_str}},
        raise_on_no_models_fitted=False,
    )

    dataset_name = "toy_binary"
    train_data, test_data, dataset_info = FitHelper.load_dataset(name=dataset_name)

    predictor = FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)

    assert not predictor.model_names()
    with pytest.raises(AssertionError):
        predictor.predict(test_data)
    assert len(predictor.leaderboard()) == 0
    assert len(predictor.leaderboard(test_data)) == 0

    model_failures = predictor.model_failures()
    assert len(model_failures) == 1
    model_failures_dict = model_failures.iloc[0].to_dict()
    assert model_failures_dict["model"] == "Dummy"
    assert model_failures_dict["model_type"] == "DummyModel"
    assert model_failures_dict["exc_type"] == "ValueError"
    assert model_failures_dict["exc_str"] == expected_exc_str


def test_raise_on_model_failure():
    """Tests that logic works properly when model raises exception and raise_on_model_failure=True"""

    expected_exc_str = "Test Error Message"

    train_data, test_data, dataset_info = FitHelper.load_dataset(name="toy_binary")

    # Force DummyModel to raise an exception when fit.
    fit_args = dict(
        hyperparameters={DummyModel: {"raise": ValueError, "raise_msg": expected_exc_str}},
        raise_on_model_failure=True,
        feature_generator=None,
    )

    with pytest.raises(ValueError) as excinfo:
        FitHelper.fit_dataset(train_data=train_data, init_args=dict(label=dataset_info["label"]), fit_args=fit_args)
    assert str(excinfo.value) == "Test Error Message"


def test_dummy():
    model_cls = DummyModel
    model_hyperparameters = {}

    """Additionally tests that all metrics work"""
    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters, extra_metrics=True)


def test_dummy_binary_model():
    fit_args = dict()
    dataset_name = "toy_binary"
    ModelFitHelper.fit_and_validate_dataset(dataset_name=dataset_name, model=DummyModel(), fit_args=fit_args)


def test_dummy_multiclass_model():
    fit_args = dict()
    dataset_name = "toy_multiclass"
    ModelFitHelper.fit_and_validate_dataset(dataset_name=dataset_name, model=DummyModel(), fit_args=fit_args)


def test_dummy_regression_model():
    fit_args = dict()
    dataset_name = "toy_regression"
    ModelFitHelper.fit_and_validate_dataset(dataset_name=dataset_name, model=DummyModel(), fit_args=fit_args)


def test_dummy_binary_absolute_path():
    """Test that absolute path works"""
    fit_args = dict(
        hyperparameters={DummyModel: {}},
    )
    path = Path(".") / "AG_test"
    path = str(path.resolve())
    init_args = dict(path=path)

    dataset_name = "toy_binary"

    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, init_args=init_args, fit_args=fit_args)


def test_dummy_binary_absolute_path_stack():
    """Test that absolute path works"""
    fit_args = dict(
        hyperparameters={DummyModel: {}},
        num_bag_folds=2,
        num_bag_sets=2,
        num_stack_levels=1,
    )

    dataset_name = "toy_binary"
    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, expected_model_count=4, path_as_absolute=True)


def test_dummy_binary_model_absolute_path():
    """Test that absolute path works"""
    fit_args = dict()
    path = Path(".") / "AG_test"
    path = str(path.resolve())
    model = DummyModel(path=path)
    dataset_name = "toy_binary"
    ModelFitHelper.fit_and_validate_dataset(dataset_name=dataset_name, model=model, fit_args=fit_args)
