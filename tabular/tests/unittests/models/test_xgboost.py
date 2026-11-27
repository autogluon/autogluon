import numpy as np
import pandas as pd

from autogluon.core.metrics import get_metric, make_scorer
from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_estimators": 10}


def test_xgboost():
    model_cls = XGBoostModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)


def test_xgboost_binary_enable_categorical():
    fit_args = dict(
        hyperparameters={XGBoostModel: {"enable_categorical": True}},
    )
    dataset_name = "toy_binary"
    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, refit_full=False)


def test_xgboost_enable_categorical_predict_time_categories():
    """Prediction with enable_categorical=True must handle categories that fit never observed.

    xgboost >= 3.3 records the fit-time categories and rejects predict-time values outside
    them, so the model's ordinal encoding must put fit and predict frames in one shared
    category space: declared-but-unobserved levels stay predictable, and truly novel
    categories or NaN are treated as missing.
    """
    rng = np.random.default_rng(0)
    n = 100
    levels = ["a", "b", "c", "rare"]
    X_train = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            # "rare" is a declared level that never occurs in the training rows
            "cat": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=levels),
        }
    )
    y_train = pd.Series(rng.integers(0, 2, size=n))
    model = XGBoostModel(
        name="XGB",
        path="",
        problem_type="binary",
        eval_metric=get_metric("log_loss", problem_type="binary"),
        hyperparameters={"enable_categorical": True, "n_estimators": 10},
    )
    model.fit(X=X_train, y=y_train)

    X_test = pd.DataFrame(
        {
            "num": rng.normal(size=4),
            "cat": pd.Categorical(["a", "rare", None, "b"], categories=levels),
        }
    )
    y_pred_proba = model.predict_proba(X_test)
    assert len(y_pred_proba) == len(X_test)

    # a category the fit never declared is treated as missing rather than raising
    X_test_novel = X_test.copy()
    X_test_novel["cat"] = pd.Categorical(["z"] * len(X_test))
    y_pred_proba_novel = model.predict_proba(X_test_novel)
    assert len(y_pred_proba_novel) == len(X_test_novel)


def test_xgboost_custom_metric_with_sample_weights():
    """Verify custom evaluation metrics requiring sample_weight work with XGBoost."""
    rng = np.random.default_rng(42)
    n = 60
    X_train = pd.DataFrame({"feat1": rng.normal(size=n), "feat2": rng.normal(size=n)})
    y_train = pd.Series(rng.normal(size=n))
    weights = pd.Series(rng.uniform(0.5, 1.5, size=n))

    X_val = pd.DataFrame({"feat1": rng.normal(size=20), "feat2": rng.normal(size=20)})
    y_val = pd.Series(rng.normal(size=20))
    weights_val = pd.Series(rng.uniform(0.5, 1.5, size=20))

    def weighted_mae(y_true, y_pred, sample_weight):
        return float(np.average(np.abs(y_true - y_pred), weights=sample_weight))

    custom_metric = make_scorer(
        name="weighted_mae",
        score_func=weighted_mae,
        optimum=0,
        greater_is_better=False,
    )

    model = XGBoostModel(
        name="XGB",
        problem_type="regression",
        eval_metric=custom_metric,
        hyperparameters={"n_estimators": 5},
    )
    model.fit(
        X=X_train,
        y=y_train,
        X_val=X_val,
        y_val=y_val,
        sample_weight=weights,
        sample_weight_val=weights_val,
    )
    score = model.score(X=X_val, y=y_val, sample_weight=weights_val)
    assert not np.isnan(score)
