from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge

from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.lr.lr_model import LinearModel
from autogluon.tabular.models.lr.lr_rapids_model import LinearRapidsModel
from autogluon.tabular.testing import FitHelper


@pytest.mark.parametrize("model_hyperparameters", [{}, {"penalty": "L1"}, {"penalty": "elasticnet"}])
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


@pytest.mark.parametrize("penalty", [None, "L1", "l1", "L2", "l2", "elasticnet", "ELASTICNET"])
def test_linear_classification_penalty(tmp_path, classification_data, penalty):
    problem_type, X, y = classification_data
    hyperparameters = {"C": 0.1, "proc.skew_threshold": None, "random_state": 0}
    if penalty is not None:
        hyperparameters["penalty"] = penalty
    model = LinearModel(path=str(tmp_path), problem_type=problem_type, hyperparameters=hyperparameters)
    model.fit(X=X, y=y)

    assert isinstance(model.model, LogisticRegression)
    norm_penalty = penalty.upper() if penalty is not None else "L2"
    assert model.model.solver == ("saga" if norm_penalty in ("L1", "ELASTICNET") else "lbfgs")
    if norm_penalty == "ELASTICNET":
        assert getattr(model.model, "l1_ratio", None) == 0.5
        if hasattr(model.model, "penalty"):
            assert model.model.penalty == "elasticnet"
    elif norm_penalty == "L1":
        # L1 must produce sparse coefficients on this dataset; L2 keeps every coefficient.
        assert np.any(model.model.coef_ == 0)
        assert np.any(model.model.coef_ != 0)
    else:
        assert np.all(model.model.coef_ != 0)
    assert np.isfinite(model.predict_proba(X)).all()


@pytest.mark.parametrize(
    "penalty, model_cls",
    [
        ("L1", Lasso),
        ("l1", Lasso),
        ("L2", Ridge),
        ("l2", Ridge),
        ("elasticnet", ElasticNet),
        ("ELASTICNET", ElasticNet),
    ],
)
def test_linear_regression_penalty(tmp_path, penalty, model_cls):
    X, y = make_regression(n_samples=100, n_features=10, n_informative=3, random_state=0)
    X, y = pd.DataFrame(X), pd.Series(y)
    model = LinearModel(path=str(tmp_path), problem_type="regression", hyperparameters={"penalty": penalty, "C": 0.1})
    model.fit(X=X, y=y)

    assert isinstance(model.model, model_cls)
    assert model.model.alpha == 10
    if penalty.upper() == "ELASTICNET":
        assert model.model.l1_ratio == 0.5
    assert np.isfinite(model.predict(X)).all()


def test_linear_custom_l1_ratio(tmp_path, classification_data):
    problem_type, X, y = classification_data
    model = LinearModel(
        path=str(tmp_path),
        problem_type=problem_type,
        hyperparameters={"penalty": "elasticnet", "l1_ratio": 0.75, "C": 0.1, "random_state": 0},
    )
    model.fit(X=X, y=y)
    assert getattr(model.model, "l1_ratio", None) == 0.75

    reg_X, reg_y = make_regression(n_samples=100, n_features=10, random_state=0)
    reg_model = LinearModel(
        path=str(tmp_path),
        problem_type="regression",
        hyperparameters={"penalty": "elasticnet", "l1_ratio": 0.25, "C": 0.1},
    )
    reg_model.fit(X=pd.DataFrame(reg_X), y=pd.Series(reg_y))
    assert isinstance(reg_model.model, ElasticNet)
    assert reg_model.model.l1_ratio == 0.25


@pytest.mark.parametrize(
    "penalty, solver",
    [
        ("L1", "saga"),
        ("L2", "saga"),
        ("L1", "lbfgs"),
        ("elasticnet", "saga"),
        ("elasticnet", "lbfgs"),
    ],
)
def test_linear_explicit_solver(tmp_path, classification_data, penalty, solver):
    problem_type, X, y = classification_data
    model = LinearModel(
        path=str(tmp_path), problem_type=problem_type, hyperparameters={"penalty": penalty, "solver": solver}
    )
    if penalty.upper() in ("L1", "ELASTICNET") and solver == "lbfgs":
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


@pytest.mark.parametrize("penalty", ["L1", "L2", "elasticnet"])
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
        if penalty == "elasticnet":
            assert params["l1_ratio"] == 0.5
    else:
        assert params["penalty"] == penalty.lower()
        assert params["solver"] == "qn"
        assert params["C"] == 0.1
        if penalty == "elasticnet":
            assert params["l1_ratio"] == 0.5
    backend_cls.return_value.fit.assert_called_once()


def test_linear_tabular_predictor_elasticnet(tmp_path):
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature2": [2, 4, 6, 8, 10, 12, 14, 16],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    predictor = TabularPredictor(label="target", problem_type="binary", path=str(tmp_path)).fit(
        train_data=data,
        hyperparameters={
            "LR": {
                "penalty": "elasticnet",
                "solver": "saga",
                "l1_ratio": 0.5,
            }
        },
    )
    model = predictor._trainer.load_model("LinearModel")
    assert getattr(model.model, "penalty", None) == "elasticnet" or getattr(model.model, "l1_ratio", None) == 0.5
