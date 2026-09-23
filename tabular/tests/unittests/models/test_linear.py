from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import Lasso, LogisticRegression, Ridge

from autogluon.tabular.models.lr.lr_model import LinearModel
from autogluon.tabular.models.lr.lr_rapids_model import LinearRapidsModel
from autogluon.tabular.testing import FitHelper


@pytest.mark.parametrize("model_hyperparameters", [{}, {"penalty": "L1"}])
def test_linear(model_hyperparameters):
    FitHelper.verify_model(model_cls=LinearModel, model_hyperparameters=model_hyperparameters)


@pytest.fixture(params=["binary", "multiclass"])
def classification_data(request):
    problem_type = request.param
    X, y = make_classification(
        n_samples=150,
        n_features=10,
        n_informative=4,
        n_redundant=0,
        n_classes=2 if problem_type == "binary" else 3,
        random_state=0,
    )
    return problem_type, pd.DataFrame(X), pd.Series(y)


@pytest.mark.parametrize("penalty", [None, "L1", "L2"])
def test_linear_classification_penalty(tmp_path, classification_data, penalty):
    problem_type, X, y = classification_data
    hyperparameters = {"C": 0.1, "proc.skew_threshold": None, "random_state": 0}
    if penalty is not None:
        hyperparameters["penalty"] = penalty
    model = LinearModel(path=str(tmp_path), problem_type=problem_type, hyperparameters=hyperparameters)
    model.fit(X=X, y=y)

    assert isinstance(model.model, LogisticRegression)
    assert model.model.solver == ("saga" if penalty == "L1" else "lbfgs")
    # L1 must produce sparse coefficients on this dataset; L2 keeps every coefficient.
    if penalty == "L1":
        assert np.any(model.model.coef_ == 0)
        assert np.any(model.model.coef_ != 0)
    else:
        assert np.all(model.model.coef_ != 0)
    assert np.isfinite(model.predict_proba(X)).all()


@pytest.mark.parametrize("penalty, model_cls", [("L1", Lasso), ("L2", Ridge)])
def test_linear_regression_penalty(tmp_path, penalty, model_cls):
    X, y = make_regression(n_samples=100, n_features=10, n_informative=3, random_state=0)
    X, y = pd.DataFrame(X), pd.Series(y)
    model = LinearModel(path=str(tmp_path), problem_type="regression", hyperparameters={"penalty": penalty, "C": 0.1})
    model.fit(X=X, y=y)

    assert isinstance(model.model, model_cls)
    assert model.model.alpha == 10
    assert np.isfinite(model.predict(X)).all()


@pytest.mark.parametrize("penalty, solver", [("L1", "saga"), ("L2", "saga"), ("L1", "lbfgs")])
def test_linear_explicit_solver(tmp_path, classification_data, penalty, solver):
    problem_type, X, y = classification_data
    model = LinearModel(
        path=str(tmp_path), problem_type=problem_type, hyperparameters={"penalty": penalty, "solver": solver}
    )
    if penalty == "L1" and solver == "lbfgs":
        with pytest.raises(ValueError, match="lbfgs"):
            model.fit(X=X, y=y)
    else:
        model.fit(X=X, y=y)
        assert model.model.solver == solver


def test_linear_invalid_classification_penalty(tmp_path, classification_data):
    problem_type, X, y = classification_data
    model = LinearModel(path=str(tmp_path), problem_type=problem_type, hyperparameters={"penalty": "invalid"})
    with pytest.raises(ValueError, match="Unknown value for penalty"):
        model.fit(X=X, y=y)


def test_linear_invalid_regression_penalty(tmp_path):
    X, y = make_regression(n_samples=50, n_features=5, random_state=0)
    model = LinearModel(path=str(tmp_path), problem_type="regression", hyperparameters={"penalty": "invalid"})
    with pytest.raises(ValueError, match="Unknown value for penalty"):
        model.fit(X=pd.DataFrame(X), y=pd.Series(y))


@pytest.mark.parametrize("penalty", ["L1", "L2"])
@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_linear_rapids_penalty_forwarding(tmp_path, monkeypatch, penalty, problem_type):
    # Check the cuML constructor arguments without requiring a GPU installation.
    model = LinearRapidsModel(
        path=str(tmp_path), problem_type=problem_type, hyperparameters={"penalty": penalty, "C": 0.1}
    )
    model._init_params()
    X, y = pd.DataFrame({"x": [0.0, 1.0, 2.0]}), pd.Series([0, 1, 0])
    backend_cls = Mock()
    monkeypatch.setattr(model, "_get_model_type", lambda: backend_cls)
    monkeypatch.setattr(model, "preprocess", lambda X, **kwargs: X)
    model._fit(X=X, y=y)

    params = backend_cls.call_args.kwargs
    if problem_type == "regression":
        assert "penalty" not in params
        assert "C" not in params
        assert params["alpha"] == 10
    else:
        assert params["penalty"] == penalty.lower()
        assert params["solver"] == "qn"
        assert params["C"] == 0.1
    backend_cls.return_value.fit.assert_called_once()
