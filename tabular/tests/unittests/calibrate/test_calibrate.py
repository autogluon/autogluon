import numpy as np
import pandas as pd
import pytest

from autogluon.core.calibrate.logistic_calibration import LogisticCalibrator
from autogluon.tabular import TabularPredictor
from autogluon.tabular.testing import FitHelper


def test_calibrate_binary():
    """Tests that calibrate=True doesn't crash in binary"""
    fit_args = dict(
        hyperparameters={"GBM": {}},
        calibrate=True,
    )
    dataset_name = "toy_binary"

    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_calibrate_binary_bag():
    """Tests that calibrate=True doesn't crash in binary w/ bagging"""
    fit_args = dict(
        hyperparameters={"GBM": {"ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"}}},
        calibrate=True,
        num_bag_folds=3,
    )
    dataset_name = "toy_binary"

    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_calibrate_multiclass():
    """Tests that calibrate=True doesn't crash in multiclass"""
    fit_args = dict(
        hyperparameters={"GBM": {}},
        calibrate=True,
    )
    dataset_name = "toy_multiclass"

    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_calibrate_multiclass_bag():
    """Tests that calibrate=True doesn't crash in multiclass w/ bagging"""
    fit_args = dict(
        hyperparameters={"GBM": {"ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"}}},
        calibrate=True,
        num_bag_folds=3,
    )
    dataset_name = "toy_multiclass"

    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_calibrate_quantile():
    """Tests that calibrate=True doesn't crash in quantile"""
    fit_args = dict(
        hyperparameters={"RF": {}},
        calibrate=True,
    )
    dataset_name = "toy_quantile"

    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_calibrate_quantile_bag():
    """Tests that calibrate=True doesn't crash in quantile w/ bagging"""
    fit_args = dict(
        hyperparameters={"RF": {"ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"}}},
        calibrate=True,
        num_bag_folds=3,
    )
    dataset_name = "toy_quantile"

    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


@pytest.mark.parametrize("dataset_name", ["toy_binary", "toy_multiclass"])
@pytest.mark.parametrize("num_bag_folds", [0, 3])
def test_calibrate_logistic(dataset_name, num_bag_folds):
    """Tests that calibration_method="logistic" doesn't crash, with and without bagging"""
    fit_args = dict(
        hyperparameters={"GBM": {"ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"}}},
        calibrate=True,
        calibration_method="logistic",
        num_bag_folds=num_bag_folds,
    )
    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def _knn_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    logits = np.stack([df["a"], df["b"], -df["a"] - df["b"]], axis=1)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    df["label"] = [rng.choice(3, p=row) for row in probs]
    return df


def test_calibrate_logistic_improves_a_miscalibrated_model_and_survives_load(tmp_path):
    """A 3-neighbor KNN predicts probabilities in steps of 1/3, including 0 and 1; logistic calibration repairs its log_loss."""
    train, test = _knn_data(2000, seed=0), _knn_data(1000, seed=1)
    fit_kwargs = dict(hyperparameters={"KNN": {"n_neighbors": 3}}, fit_weighted_ensemble=False, calibrate=True)

    uncalibrated = TabularPredictor(label="label", eval_metric="log_loss", path=str(tmp_path / "t"), verbosity=0)
    uncalibrated.fit(train, **fit_kwargs)
    predictor = TabularPredictor(label="label", eval_metric="log_loss", path=str(tmp_path / "l"), verbosity=0)
    predictor.fit(train, calibration_method="logistic", **fit_kwargs)

    model = predictor._trainer.load_model(predictor.model_best)
    assert isinstance(model.calibrator, LogisticCalibrator)
    assert model.temperature_scalar is None
    proba = predictor.predict_proba(test)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert predictor.evaluate(test)["log_loss"] > uncalibrated.evaluate(test)["log_loss"]

    loaded = TabularPredictor.load(predictor.path)
    pd.testing.assert_frame_equal(loaded.predict_proba(test), proba)


def test_calibration_method_is_validated(tmp_path):
    with pytest.raises(ValueError, match="calibration_method"):
        TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
            _knn_data(100, seed=0), hyperparameters={"DUMMY": {}}, calibration_method="platt"
        )
