import os.path
import tempfile

import pandas as pd
import pytest

import autogluon.eda.analysis as eda
import autogluon.eda.auto as auto
from autogluon.core.constants import REGRESSION
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
    df_val = pd.read_csv(os.path.join(RESOURCE_PATH, "houses", "test_data.csv")).sample(50, random_state=0)
    target_col = "SalePrice"

    with tempfile.TemporaryDirectory() as path:
        predictor = fit_model(path, df_train, target_col)

        state = auto.analyze(
            model=predictor,
            train_data=df_train,
            val_data=df_val,
            return_state=True,
            anlz_facets=[eda.model.AutoGluonModelEvaluator(normalize="true")],
        )

    assert state.model_evaluation.problem_type == REGRESSION
    assert len(state.model_evaluation.y_true_val) == len(df_val)
    assert len(state.model_evaluation.y_pred_val) == len(df_val)
    expected = [c for c in df_train.columns if c not in ["Street", "Utilities", "SalePrice", "PoolQC"]]
    assert sorted(state.model_evaluation.importance.index.to_list()) == sorted(expected)
    _assert_importance_is_present(state)
    assert state.model_evaluation.confusion_matrix is None
    assert state.model_evaluation.confusion_matrix_normalized is None
    assert state.model_evaluation.y_true_test is None
    assert state.model_evaluation.y_pred_test is None
    assert state.model_evaluation.y_true_train is None
    assert state.model_evaluation.y_pred_train is None


def test_AutoGluonModelEvaluator_regression__with_test_data():
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "houses", "train_data.csv")).sample(100, random_state=0)
    df_val = pd.read_csv(os.path.join(RESOURCE_PATH, "houses", "test_data.csv"))[:50]
    df_test = pd.read_csv(os.path.join(RESOURCE_PATH, "houses", "test_data.csv"))[50:101]
    target_col = "SalePrice"

    with tempfile.TemporaryDirectory() as path:
        predictor = fit_model(path, df_train, target_col)

        state = auto.analyze(
            model=predictor,
            train_data=df_train,
            val_data=df_val,
            test_data=df_test,
            return_state=True,
            anlz_facets=[eda.model.AutoGluonModelEvaluator(normalize="true")],
        )

    assert state.model_evaluation.problem_type == REGRESSION
    assert len(state.model_evaluation.y_true_test) == len(df_test)
    assert len(state.model_evaluation.y_pred_test) == len(df_test)
    assert len(state.model_evaluation.y_true_val) == len(df_val)
    assert len(state.model_evaluation.y_pred_val) == len(df_val)
    expected = [c for c in df_train.columns if c not in ["Street", "Utilities", "SalePrice", "PoolQC"]]
    assert sorted(state.model_evaluation.importance.index.to_list()) == sorted(expected)
    _assert_importance_is_present(state)
    assert state.model_evaluation.confusion_matrix is None
    assert state.model_evaluation.confusion_matrix_normalized is None


def test_AutoGluonModelEvaluator_classification():
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)
    df_val = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "test_data.csv")).sample(50, random_state=0)
    target_col = "class"

    with tempfile.TemporaryDirectory() as path:
        predictor = fit_model(path, df_train, target_col)

        state = auto.analyze(
            model=predictor,
            train_data=df_train,
            val_data=df_val,
            return_state=True,
            anlz_facets=[eda.model.AutoGluonModelEvaluator(normalize="true")],
        )

    assert state.model_evaluation.problem_type == "binary"
    assert len(state.model_evaluation.y_true_val) == len(df_val)
    assert len(state.model_evaluation.y_pred_val) == len(df_val)
    expected = [c for c in df_train.columns if c not in ["class"]]
    assert sorted(state.model_evaluation.importance.index.to_list()) == sorted(expected)
    _assert_importance_is_present(state)


@pytest.mark.parametrize("save_model_to_state", [True, False])
def test_AutoGluonModelQuickFit(save_model_to_state):
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)
    target_col = "class"

    with tempfile.TemporaryDirectory() as path:
        state = auto.analyze(
            train_data=df_train,
            label=target_col,
            return_state=True,
            anlz_facets=[
                eda.dataset.ProblemTypeControl(),
                eda.dataset.TrainValidationSplit(
                    children=[
                        eda.model.AutoGluonModelQuickFit(
                            save_model_to_state=save_model_to_state,
                            estimator_args=dict(path=path),
                            verbosity=0,
                            hyperparameters={
                                "RF": {
                                    "criterion": "entropy",
                                    "max_depth": 15,
                                    "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]},
                                }
                            },
                            children=[eda.model.AutoGluonModelEvaluator()],
                        )
                    ]
                ),
            ],
        )

    assert state.model_evaluation.problem_type == "binary"
    assert len(state.model_evaluation.y_true_val) == int(len(df_train) * 0.3)
    assert len(state.model_evaluation.y_pred_val) == int(len(df_train) * 0.3)
    expected = [c for c in df_train.columns if c not in ["class"]]
    assert sorted(state.model_evaluation.importance.index.to_list()) == sorted(expected)
    _assert_importance_is_present(state)
    if save_model_to_state:
        assert str(state.model.__class__) == "<class 'autogluon.tabular.predictor.predictor.TabularPredictor'>"
    else:
        assert "model" not in state


def test_AutoGluonModelQuickFit__constructor_defaults():
    assert eda.model.AutoGluonModelQuickFit().estimator_args == {}


def _assert_importance_is_present(state):
    assert state.model_evaluation.importance.columns.to_list() == [
        "importance",
        "stddev",
        "p_value",
        "n",
        "p99_high",
        "p99_low",
    ]
