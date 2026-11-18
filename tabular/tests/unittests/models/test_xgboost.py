import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("enable_categorical", [True, False])
def test_xgboost_special_feature_names(tmp_path, problem_type, enable_categorical):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a[1]": rng.normal(size=60),
            "a1": rng.normal(size=60),
            "b<2": rng.normal(size=60),
            "c]": pd.Categorical(["a", "b", "c"] * 20),
        }
    )
    original_columns = X.columns.tolist()
    y = pd.Series(
        rng.normal(size=60)
        if problem_type == "regression"
        else np.arange(60) % (3 if problem_type == "multiclass" else 2)
    )
    X_train, X_val = X.iloc[:48].copy(), X.iloc[48:].copy()
    y_train, y_val = y.iloc[:48], y.iloc[48:]
    model = XGBoostModel(
        name="XGB",
        path=str(tmp_path / "model"),
        problem_type=problem_type,
        eval_metric="root_mean_squared_error" if problem_type == "regression" else "log_loss",
        hyperparameters={"enable_categorical": enable_categorical, "n_estimators": 5, "seed": 0},
    )
    model.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val, num_cpus=1)
    assert X_train.columns.tolist() == original_columns
    assert X_val.columns.tolist() == original_columns
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)
    assert predictions.shape == (len(X_val),)
    assert probabilities.shape == ((len(X_val), 3) if problem_type == "multiclass" else (len(X_val),))
    reordered = X_val[original_columns[::-1]]
    np.testing.assert_allclose(model.predict_proba(reordered), probabilities)
    assert reordered.columns.tolist() == original_columns[::-1]
    assert X_val.columns.tolist() == original_columns
    assert model.features == original_columns
    importance = model.compute_feature_importance(X_val, y_val, num_shuffle_sets=1, silent=True)
    assert set(importance.index) == set(original_columns)
    path = model.save()
    loaded = XGBoostModel.load(path)
    assert loaded.features == original_columns
    np.testing.assert_allclose(loaded.predict_proba(X_val), probabilities)


def test_xgboost_safe_feature_names_legacy_state(tmp_path):
    X = pd.DataFrame({"first": np.arange(40, dtype=float), "second": np.arange(40) % 3})
    y = pd.Series(np.arange(40) % 2)
    model = XGBoostModel(
        name="XGB",
        path=str(tmp_path / "model"),
        problem_type="binary",
        eval_metric="log_loss",
        hyperparameters={"enable_categorical": True, "n_estimators": 3},
    )
    model.fit(X=X, y=y, num_cpus=1)
    predictions = model.predict_proba(X)
    assert model.model.get_booster().feature_names == list(X.columns)
    # Models saved before the name remapping was added have no new wrapper state.
    model.__dict__.pop("_feature_names_internal", None)
    np.testing.assert_allclose(model.predict_proba(X), predictions)
