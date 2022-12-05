from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import autogluon.eda.auto as auto
from autogluon.eda import AnalysisState
from autogluon.eda.visualization import CorrelationVisualization, CorrelationSignificanceVisualization


def test_CorrelationVisualization_single_value(monkeypatch):
    state = AnalysisState(
        {
            "correlations": {"train_data": pd.DataFrame(index=["a"], data={"a": [1.00]})},
            "correlations_method": "kendall",
        }
    )
    call_subplots = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(plt, "subplots", call_subplots)
        auto.analyze(state=state, viz_facets=[(CorrelationVisualization())])
    call_subplots.assert_not_called()


def test_CorrelationVisualization(monkeypatch):
    state = AnalysisState({"correlations": {"train_data": __get_train_data()}, "correlations_method": "kendall"})
    heatmap_args = dict(vmin=-1, vmax=1, center=0, cmap="Spectral")
    __test_internal(monkeypatch, "correlations", state, heatmap_args, CorrelationVisualization)


def test_CorrelationVisualization_phik(monkeypatch):
    state = AnalysisState({"correlations": {"train_data": __get_train_data()}, "correlations_method": "phik"})
    heatmap_args = dict(vmin=0, vmax=1, center=0, cmap="Spectral")
    __test_internal(
        monkeypatch,
        "correlations",
        state,
        heatmap_args,
        CorrelationVisualization,
        "train_data - phik correlation matrix",
        headers=True,
    )


def test_CorrelationVisualization_focus(monkeypatch):
    state = AnalysisState(
        {
            "correlations": {"train_data": __get_train_data()},
            "correlations_method": "spearman",
            "correlations_focus_field": "c",
            "correlations_focus_field_threshold": 0.6,
            "correlations_focus_high_corr": {"train_data": pd.DataFrame(index=list("ad"), data={"c": [0.88, -1.00]})},
        }
    )
    heatmap_args = dict(vmin=-1, vmax=1, center=0, cmap="Spectral")
    __test_internal(
        monkeypatch,
        "correlations",
        state,
        heatmap_args,
        CorrelationVisualization,
        "train_data - spearman correlation matrix; focus: absolute correlation for c >= 0.6",
        headers=True,
    )


def test_CorrelationVisualization__can_handle():
    assert (
        CorrelationVisualization().can_handle(AnalysisState({"correlations": 123, "something": "something"})) is True
    )
    assert CorrelationVisualization().can_handle(AnalysisState({"something": "something"})) is False


def test_CorrelationSignificanceVisualization__can_handle():
    assert (
        CorrelationSignificanceVisualization().can_handle(
            AnalysisState({"significance_matrix": 123, "something": "something"})
        )
        is True
    )
    assert CorrelationSignificanceVisualization().can_handle(AnalysisState({"something": "something"})) is False


def test_CorrelationSignificanceVisualization(monkeypatch):
    state = AnalysisState(
        {
            "correlations": {"train_data": __get_train_data()},
            "significance_matrix": {
                "train_data": __get_train_data(),
            },
            "correlations_method": "spearman",
            "correlations_focus_field": "c",
            "correlations_focus_field_threshold": 0.6,
            "correlations_focus_high_corr": {"train_data": pd.DataFrame(index=list("ad"), data={"c": [0.88, -1.00]})},
        }
    )
    heatmap_args = dict(center=3, vmax=5, cmap="Spectral", robust=True)
    __test_internal(monkeypatch, "significance_matrix", state, heatmap_args, CorrelationSignificanceVisualization)


def __get_train_data():
    return pd.DataFrame(
        index=list("abcd"),
        data={
            "a": [1.00, 0.00, 0.77, -0.77],
            "b": [0.00, 1.00, 0.36, -0.36],
            "c": [0.77, 0.36, 1.00, -1.00],
            "d": [-0.77, -0.36, -1.00, 1.00],
        },
    )


def __test_internal(monkeypatch, render_key, state, heatmap_args, facet_cls, text_render=None, **kwargs):
    call_heatmap = MagicMock()
    call_show = MagicMock()
    call_yticks = MagicMock()
    call_subplots = MagicMock(return_value=("fig", "ax"))
    call_render_text = MagicMock()

    analysis = facet_cls(fig_args=dict(some_arg=123), **kwargs)
    analysis.render_text = call_render_text

    with monkeypatch.context() as m:
        m.setattr(sns, "heatmap", call_heatmap)
        m.setattr(plt, "show", call_show)
        m.setattr(plt, "yticks", call_yticks)
        m.setattr(plt, "subplots", call_subplots)
        auto.analyze(state=state, viz_facets=[analysis])

    call_yticks.assert_called_with(rotation=0)
    call_subplots.assert_called_with(some_arg=123)
    call_show.assert_called_with("fig")

    call_heatmap.assert_called_with(
        state[render_key].train_data,
        annot=True,
        ax="ax",
        linewidths=0.9,
        linecolor="white",
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.5},
        **heatmap_args,
    )
    if text_render is not None:
        call_render_text.assert_called_with(text_render, text_type="h3")
    else:
        call_render_text.assert_not_called()
