from unittest.mock import MagicMock, ANY

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from autogluon.eda import AnalysisState
from autogluon.eda.visualization import ConfusionMatrix, RegressionEvaluation, FeatureImportance


@pytest.mark.parametrize(
    "confusion_matrix_normalized,expected_fmt", [
        (True, ",.2%"),
        (False, "d")
    ]
)
def test_ConfusionMatrix(monkeypatch, confusion_matrix_normalized, expected_fmt):
    state = AnalysisState({
        'model_evaluation': {
            'problem_type': 'binary',
            'y_true': pd.Series([0, 1, 0, 1]),
            'y_pred': pd.Series([1, 0, 1, 0]),
            'importance': pd.DataFrame({
                'importance': [0.1, 0.2, 0.3],
                'stddev': [0.01, 0.02, 0.03],
                'p_value': [0.1, 0.2, 0.3],
                'n': [5, 5, 5],
                'p99_high': [0.02, 0.03, 0.04],
                'p99_low': [-0.02, -0.03, -0.04],
            }),
            'confusion_matrix_normalized': confusion_matrix_normalized,
            'confusion_matrix': np.array([[148, 20],
                                          [32, 68]])
        }
    })

    call_plt_subplots = MagicMock(return_value=("fig", "ax"))
    call_plt_show = MagicMock()
    call_sns_heatmap = MagicMock()
    call_render_text = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(plt, "subplots", call_plt_subplots)
        m.setattr(plt, "show", call_plt_show)
        m.setattr(sns, "heatmap", call_sns_heatmap)
        viz = ConfusionMatrix(fig_args=dict(a=1, b=2), headers=True, some_kwarg=123)
        viz.render_text = call_render_text
        viz.render(state)
    call_plt_subplots.assert_called_with(a=1, b=2)
    call_plt_show.assert_called_with("fig")
    call_sns_heatmap.assert_called_with(ANY, ax='ax', cmap='Blues', annot=True, fmt=expected_fmt, cbar=False, some_kwarg=123)
    call_render_text.assert_called_with('Confusion Matrix', text_type="h3")


def test_ConfusionMatrix__can_handle():
    state = AnalysisState({
        'model_evaluation': {
            'confusion_matrix': np.array([])
        }
    })
    assert ConfusionMatrix().can_handle(state) is True
    assert ConfusionMatrix().can_handle(AnalysisState({'some_args': {}})) is False


def test_RegressionEvaluation(monkeypatch):
    state = AnalysisState({
        'model_evaluation': {
            'problem_type': 'regression',
            'y_true': pd.Series([0, 1, 0, 1]),
            'y_pred': pd.Series([1, 0, 1, 0]),
        }
    })

    call_plt_subplots = MagicMock(return_value=("fig", "ax"))
    call_plt_show = MagicMock()
    call_sns_regplot = MagicMock()
    call_render_text = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(plt, "subplots", call_plt_subplots)
        m.setattr(plt, "show", call_plt_show)
        m.setattr(sns, "regplot", call_sns_regplot)
        viz = RegressionEvaluation(headers=True, fig_args=dict(a=1, b=2), some_kwarg=123)
        viz.render_text = call_render_text
        viz.render(state)
    call_plt_subplots.assert_called_with(a=1, b=2)
    call_plt_show.assert_called_with("fig")
    call_sns_regplot.assert_called_with(ax='ax', data=ANY, x="y_true", y="y_pred", some_kwarg=123)
    call_render_text.assert_called_with('Prediction vs Target', text_type="h3")


def test_RegressionEvaluation__can_handle():
    assert RegressionEvaluation().can_handle(
        AnalysisState({
            'model_evaluation': {
                'problem_type': 'regression',
            }
        })) is True
    assert RegressionEvaluation().can_handle(
        AnalysisState({
            'model_evaluation': {
                'problem_type': 'binary',
            }
        })) is False
    assert RegressionEvaluation().can_handle(AnalysisState({'some_args': {}})) is False


def test_ConfusionMatrix__handle_None_fig_args():
    assert RegressionEvaluation(fig_args={'abc': 1}).fig_args == {'abc': 1}
    assert RegressionEvaluation(fig_args=None).fig_args == {}


def test_FeatureImportance__can_handle():
    assert FeatureImportance().can_handle(
        AnalysisState({
            'model_evaluation': {
                'importance': 'something',
            }
        })) is True
    assert FeatureImportance().can_handle(
        AnalysisState({
            'model_evaluation': {
            }
        })) is False
    assert FeatureImportance().can_handle(AnalysisState({'some_args': {}})) is False


def test_FeatureImportance__handle_None_fig_args():
    assert FeatureImportance(fig_args={'abc': 1}).fig_args == {'abc': 1}
    assert FeatureImportance(fig_args=None).fig_args == {}


@pytest.mark.parametrize(
    "show_barplots", [
        True,
        False,
    ]
)
def test_FeatureImportance(monkeypatch, show_barplots):
    state = AnalysisState({
        'model_evaluation': {
            'importance': pd.DataFrame({
                'importance': [0.1, 0.2, 0.3],
                'stddev': [0.01, 0.02, 0.03],
                'p_value': [0.1, 0.2, 0.3],
                'n': [5, 5, 5],
                'p99_high': [0.02, 0.03, 0.04],
                'p99_low': [-0.02, -0.03, -0.04],
            }),
        }
    })

    call_plt_subplots = MagicMock(return_value=("fig", "ax"))
    call_plt_show = MagicMock()
    call_sns_barplot = MagicMock()
    call_display_obj = MagicMock()
    call_render_text = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(plt, "subplots", call_plt_subplots)
        m.setattr(plt, "show", call_plt_show)
        m.setattr(sns, "barplot", call_sns_barplot)
        viz = FeatureImportance(headers=True, show_barplots=show_barplots, fig_args=dict(a=1, b=2), some_kwarg=123)
        viz.display_obj = call_display_obj
        viz.render_text = call_render_text
        viz.render(state)
    call_display_obj.assert_called_once()
    call_render_text.assert_called_with('Feature Importance', text_type="h3")
    if show_barplots:
        call_plt_subplots.assert_called_with(a=1, b=2)
        call_plt_show.assert_called_with("fig")
        call_sns_barplot.assert_called_with(ax='ax', data=ANY, y="index", x="importance", some_kwarg=123)
    else:
        call_plt_subplots.assert_not_called()
        call_plt_show.assert_not_called()
        call_sns_barplot.assert_not_called()
