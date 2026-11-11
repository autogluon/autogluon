import numpy as np
import pandas as pd
import pytest

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.metrics import METRICS
from autogluon.tabular.experimental import TabularClassifier, TabularRegressor
from autogluon.tabular.models.lgb.lgb_model import LGBModel
from autogluon.tabular.testing import FitHelper


@pytest.mark.parametrize("estimator_type", [TabularClassifier, TabularRegressor])
@pytest.mark.parametrize("feature_type", ["numeric", "categorical", "datetime"])
def test_scikit_predictions_preserve_dataframe_features(estimator_type, feature_type, tmp_path):
    X = pd.DataFrame({"numeric_feature": np.arange(20, dtype=float), "other_feature": np.arange(20) % 3})
    if feature_type == "categorical":
        X["other_feature"] = pd.Series(["a", "b"] * 10, dtype="category")
        X.loc[0, "numeric_feature"] = np.nan
        X.loc[1, "other_feature"] = np.nan
    elif feature_type == "datetime":
        X["other_feature"] = pd.date_range("2020-01-01", periods=len(X))
    y = np.arange(20) % 2
    estimator = estimator_type(
        path=str(tmp_path / "predictor"),
        verbosity=0,
        hyperparameters={"DUMMY": {}},
        fit_args={"fit_weighted_ensemble": False},
    ).fit(X, y)

    np.testing.assert_array_equal(estimator.predict(X), estimator.predictor_.predict(X).to_numpy())
    if estimator_type is TabularClassifier:
        np.testing.assert_allclose(estimator.predict_proba(X), estimator.predictor_.predict_proba(X).to_numpy())
    with pytest.raises(ValueError, match="Inconsistent number of features"):
        estimator.predict(X.iloc[:, :1])
    with pytest.raises(ValueError, match="at least one sample"):
        estimator.predict(X.iloc[:0])


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
