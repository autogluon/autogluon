import numpy as np
import pandas as pd

from autogluon.core.metrics import get_metric
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
