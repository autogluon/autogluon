import numpy as np
import pandas as pd
import pytest

from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import RealTabPFNv2Model
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_estimators": 1}


def test_tabpfnv2():
    model_cls = RealTabPFNv2Model
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
        # TabPFN returns different predictions when predicting on an individual sample
        verify_single_prediction_equivalent_to_multi=False,
    )


def _make_regression_data(n: int = 30):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    # Skewed target so that mean and median of the predictive distribution differ
    y = pd.Series(np.exp(X["a"]) + rng.normal(scale=0.1, size=n), name="label")
    return X, y


def test_tabpfnv2_regression_output_type_median():
    X, y = _make_regression_data()
    model = RealTabPFNv2Model(
        problem_type="regression",
        hyperparameters={**toy_model_params, "regression_output_type": "median"},
    )
    model.fit(X=X, y=y)

    y_pred = model.predict(X)

    X_inner = model.preprocess(X)
    y_pred_median = model.model.predict(X_inner, output_type="median")
    y_pred_mean = model.model.predict(X_inner, output_type="mean")

    assert np.allclose(y_pred, y_pred_median)
    assert not np.allclose(y_pred, y_pred_mean)


def test_tabpfnv2_invalid_regression_output_type():
    X, y = _make_regression_data()
    model = RealTabPFNv2Model(
        problem_type="regression",
        hyperparameters={**toy_model_params, "regression_output_type": "maximum"},
    )
    with pytest.raises(ValueError, match="regression_output_type"):
        model.fit(X=X, y=y)
