from unittest.mock import ANY, MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from autogluon.eda import AnalysisState
from autogluon.eda.visualization import ConfusionMatrix, FeatureImportance, ModelLeaderboard, RegressionEvaluation


@pytest.mark.parametrize("confusion_matrix_normalized,expected_fmt", [(True, ",.2%"), (False, "d")])
def test_ConfusionMatrix(monkeypatch, confusion_matrix_normalized, expected_fmt):
    state = AnalysisState(
        {
            "model_evaluation": {
                "problem_type": "binary",
                "y_true": pd.Series([0, 1, 0, 1]),
                "y_pred": pd.Series([1, 0, 1, 0]),
                "importance": pd.DataFrame(
                    {
                        "importance": [0.1, 0.2, 0.3],
                        "stddev": [0.01, 0.02, 0.03],
                        "p_value": [0.1, 0.2, 0.3],
                        "n": [5, 5, 5],
                        "p99_high": [0.02, 0.03, 0.04],
                        "p99_low": [-0.02, -0.03, -0.04],
                    }
                ),
                "confusion_matrix_normalized": confusion_matrix_normalized,
                "confusion_matrix": np.array([[148, 20], [32, 68]]),
            }
        }
    )

    call_plt_subplots = MagicMock(return_value=("fig", "ax"))
    call_plt_show = MagicMock()
    call_sns_heatmap = MagicMock()
    call_render_markdown = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(plt, "subplots", call_plt_subplots)
        m.setattr(plt, "show", call_plt_show)
        m.setattr(sns, "heatmap", call_sns_heatmap)
        viz = ConfusionMatrix(fig_args=dict(a=1, b=2), headers=True, some_kwarg=123)
        viz.render_markdown = call_render_markdown
        viz.render(state)
    call_plt_subplots.assert_called_with(a=1, b=2, figsize=(2, 2))
    call_plt_show.assert_called_with("fig")
    call_sns_heatmap.assert_called_with(
        ANY,
        ax="ax",
        cmap="Blues",
        annot=True,
        linewidths=0.5,
        linecolor="lightgrey",
        fmt=expected_fmt,
        cbar=False,
        some_kwarg=123,
    )
    call_render_markdown.assert_called_with("**Confusion Matrix**")


def test_ConfusionMatrix__can_handle():
    state = AnalysisState({"model_evaluation": {"confusion_matrix": np.array([])}})
    assert ConfusionMatrix().can_handle(state) is True
    assert ConfusionMatrix().can_handle(AnalysisState({"some_args": {}})) is False


def test_RegressionEvaluation(monkeypatch):
    state = AnalysisState(
        {
            "model_evaluation": {
                "problem_type": "regression",
                "y_true_train": pd.Series([0, 1, 0, 0]),
                "y_pred_train": pd.Series([1, 0, 1, 1]),
                "y_true_val": pd.Series([0, 1, 0, 1]),
                "y_pred_val": pd.Series([1, 0, 1, 0]),
            }
        }
    )

    call_plt_subplots = MagicMock(return_value=("fig", "ax"))
    call_plt_show = MagicMock()
    call_residuals_plot = MagicMock()
    call_render_markdown = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(plt, "subplots", call_plt_subplots)
        m.setattr(plt, "show", call_plt_show)
        m.setattr("autogluon.eda.visualization.model.residuals_plot", call_residuals_plot)
        viz = RegressionEvaluation(headers=True, fig_args=dict(a=1, b=2), some_kwarg=123)
        viz.residuals_plot = call_residuals_plot
        viz.render_markdown = call_render_markdown
        viz.render(state)
    call_plt_subplots.assert_called_with(figsize=(12, 6), a=1, b=2)
    call_plt_show.assert_called_with("fig")
    call_residuals_plot.assert_called_with(
        ANY,
        state.model_evaluation.y_pred_train,
        state.model_evaluation.y_true_train,
        state.model_evaluation.y_pred_val,
        state.model_evaluation.y_true_val,
        show=False,
        ax="ax",
        hist=False,
        qqplot=True,
    )
    call_render_markdown.assert_called_with("**Prediction vs Target**")


@pytest.mark.parametrize(
    "train_present, val_present, test_present, expected",
    [
        (True, True, True, ("pred_train", "true_train", "pred_test", "true_test")),
        (False, True, True, ("pred_val", "true_val", "pred_test", "true_test")),
        (True, False, True, ("pred_train", "true_train", "pred_test", "true_test")),
        (False, False, True, (None, None, "pred_test", "true_test")),
        (True, True, False, ("pred_train", "true_train", "pred_val", "true_val")),
        (False, True, False, ("pred_val", "true_val", None, None)),
        (True, False, False, ("pred_train", "true_train", None, None)),
        (False, False, False, (None, None, None, None)),
    ],
)
def test_RegressionEvaluation__repack_parameters(train_present, val_present, test_present, expected):
    s = {}
    if train_present:
        s["y_true_train"] = "true_train"
        s["y_pred_train"] = "pred_train"
    if val_present:
        s["y_true_val"] = "true_val"
        s["y_pred_val"] = "pred_val"
    if test_present:
        s["y_true_test"] = "true_test"
        s["y_pred_test"] = "pred_test"

    state = AnalysisState(model_evaluation=s)
    assert RegressionEvaluation._repack_parameters(state.model_evaluation) == expected


def test_RegressionEvaluation__can_handle():
    assert (
        RegressionEvaluation().can_handle(
            AnalysisState(
                {
                    "model_evaluation": {
                        "problem_type": "regression",
                    }
                }
            )
        )
        is True
    )
    assert (
        RegressionEvaluation().can_handle(
            AnalysisState(
                {
                    "model_evaluation": {
                        "problem_type": "binary",
                    }
                }
            )
        )
        is False
    )
    assert RegressionEvaluation().can_handle(AnalysisState({"some_args": {}})) is False


def test_RegressionEvaluation__handle_None_fig_args():
    assert RegressionEvaluation(fig_args={"abc": 1}).fig_args == {"abc": 1, "figsize": (12, 6)}
    assert RegressionEvaluation(fig_args=None).fig_args == {"figsize": (12, 6)}
    assert RegressionEvaluation(fig_args={"figsize": (6, 6)}).fig_args == {"figsize": (6, 6)}


def test_FeatureImportance__can_handle():
    assert (
        FeatureImportance().can_handle(
            AnalysisState(
                {
                    "model_evaluation": {
                        "importance": "something",
                    }
                }
            )
        )
        is True
    )
    assert FeatureImportance().can_handle(AnalysisState({"model_evaluation": {}})) is False
    assert FeatureImportance().can_handle(AnalysisState({"some_args": {}})) is False


def test_FeatureImportance__handle_None_fig_args():
    assert FeatureImportance(fig_args={"abc": 1}).fig_args == {"abc": 1}
    assert FeatureImportance(fig_args=None).fig_args == {}


@pytest.mark.parametrize(
    "show_barplots",
    [
        True,
        False,
    ],
)
def test_FeatureImportance(monkeypatch, show_barplots):
    state = AnalysisState(
        {
            "model_evaluation": {
                "importance": pd.DataFrame(
                    {
                        "importance": [0.1, 0.2, 0.3],
                        "stddev": [0.01, 0.02, 0.03],
                        "p_value": [0.1, 0.2, 0.3],
                        "n": [5, 5, 5],
                        "p99_high": [0.02, 0.03, 0.04],
                        "p99_low": [-0.02, -0.03, -0.04],
                    }
                ),
            }
        }
    )

    call_plt_subplots = MagicMock(return_value=("fig", "ax"))
    call_plt_show = MagicMock()
    call_sns_barplot = MagicMock()
    call_display_obj = MagicMock()
    call_render_markdown = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(plt, "subplots", call_plt_subplots)
        m.setattr(plt, "show", call_plt_show)
        m.setattr(sns, "barplot", call_sns_barplot)
        viz = FeatureImportance(headers=True, show_barplots=show_barplots, fig_args=dict(a=1, b=2), some_kwarg=123)
        viz.display_obj = call_display_obj
        viz.render_markdown = call_render_markdown
        viz.render(state)
    call_display_obj.assert_called_once()
    call_render_markdown.assert_called_with("**Feature Importance**")
    if show_barplots:
        call_plt_subplots.assert_called_with(a=1, b=2, figsize=(12, 0.75))
        call_plt_show.assert_called_with("fig")
        call_sns_barplot.assert_called_with(ax="ax", data=ANY, y="index", x="importance", some_kwarg=123)
    else:
        call_plt_subplots.assert_not_called()
        call_plt_show.assert_not_called()
        call_sns_barplot.assert_not_called()


def test_ModelLeaderboard():
    assert ModelLeaderboard().can_handle(state=AnalysisState(model_evaluation={})) is False

    state = AnalysisState(model_evaluation={"leaderboard": "some_leaderboard"})

    viz = ModelLeaderboard(headers=True)
    assert viz.can_handle(state=state) is True

    viz.render_markdown = MagicMock()
    viz.display_obj = MagicMock()
    viz.render(state)
    viz.render_markdown.assert_called_with("**Model Leaderboard**")
    viz.display_obj.assert_called_with("some_leaderboard")
