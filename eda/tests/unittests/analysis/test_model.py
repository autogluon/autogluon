import os.path
import tempfile

import pandas as pd

import autogluon.eda.analysis as eda
import autogluon.eda.auto as auto
from autogluon.tabular import TabularPredictor

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))


def fit_model(path, df_train, target, fit_args=None):
    if fit_args is None:
        fit_args = {"hyperparameters": {"RF": {}}}

    predictor = TabularPredictor(path=path, label=target, verbosity=0).fit(
        df_train,
        **fit_args,
    )
    return predictor


def test_AutoGluonModelEvaluator_regression():
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "houses", "train_data.csv")).sample(100, random_state=0)
    df_test = pd.read_csv(os.path.join(RESOURCE_PATH, "houses", "test_data.csv")).sample(50, random_state=0)
    target_col = "SalePrice"

    with tempfile.TemporaryDirectory() as path:
        predictor = fit_model(path, df_train, target_col)

        state = auto.analyze(
            model=predictor,
            val_data=df_test,
            return_state=True,
            anlz_facets=[eda.model.AutoGluonModelEvaluator(normalize="true")],
        )

    assert state.model_evaluation.problem_type == "regression"
    assert len(state.model_evaluation.y_true) == len(df_test)
    assert len(state.model_evaluation.y_pred) == len(df_test)
    expected = [c for c in df_train.columns if c not in ["Street", "Utilities", "SalePrice"]]
    assert sorted(state.model_evaluation.importance.index.to_list()) == sorted(expected)
    assert state.model_evaluation.importance.columns.to_list() == [
        "importance",
        "stddev",
        "p_value",
        "n",
        "p99_high",
        "p99_low",
    ]
    assert state.model_evaluation.confusion_matrix is None
    assert state.model_evaluation.confusion_matrix_normalized is None


def test_AutoGluonModelEvaluator_classification():
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)
    df_test = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "test_data.csv")).sample(50, random_state=0)
    target_col = "class"

    with tempfile.TemporaryDirectory() as path:
        predictor = fit_model(path, df_train, target_col)

        state = auto.analyze(
            model=predictor,
            val_data=df_test,
            return_state=True,
            anlz_facets=[eda.model.AutoGluonModelEvaluator(normalize="true")],
        )

    assert state.model_evaluation.problem_type == "binary"
    assert len(state.model_evaluation.y_true) == len(df_test)
    assert len(state.model_evaluation.y_pred) == len(df_test)
    expected = [c for c in df_train.columns if c not in ["class"]]
    assert sorted(state.model_evaluation.importance.index.to_list()) == sorted(expected)
    assert state.model_evaluation.importance.columns.to_list() == [
        "importance",
        "stddev",
        "p_value",
        "n",
        "p99_high",
        "p99_low",
    ]
