from typing import List
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from autogluon.eda import AnalysisState
from autogluon.eda.visualization import AnomalyScoresVisualization


def test_AnomalyScoresVisualization__init():
    chart_args = {
        "normal.color": "grey",
        "anomaly.color": "orange",
    }
    viz = AnomalyScoresVisualization(**chart_args)
    assert viz.threshold_stds == 3
    assert viz.headers is False
    assert viz.fig_args == {}
    assert viz.chart_args == {"anomaly": {"color": "orange"}, "normal": {"color": "grey"}}


def test_AnomalyScoresVisualization(monkeypatch):
    train_data = [0.13, 0.01, 0.08, 0.76]
    test_data = [0.60, 0.20, 0.91, 0.60]
    state = AnalysisState(
        {
            "anomaly_detection": {
                "scores": {
                    "train_data": pd.Series(train_data, name="score"),
                    "test_data": pd.Series(test_data, name="score"),
                }
            }
        }
    )

    chart_args = {
        "normal.color": "grey",
        "anomaly.color": "orange",
    }

    call_ax = MagicMock()
    call_plt_subplots = MagicMock(return_value=("fig", call_ax))
    call_plt_show = MagicMock()
    call_plt_tight_layout = MagicMock()
    call_sns_scatterplot = MagicMock()
    call_render_markdown = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(plt, "subplots", call_plt_subplots)
        m.setattr(plt, "show", call_plt_show)
        m.setattr(plt, "tight_layout", call_plt_tight_layout)
        m.setattr(sns, "scatterplot", call_sns_scatterplot)

        viz = AnomalyScoresVisualization(threshold_stds=2, headers=True, **chart_args)
        viz.render_markdown = call_render_markdown

        viz.render(state)

    call_plt_tight_layout.assert_called_with(h_pad=0.3, w_pad=0.5)
    call_plt_show.assert_called_with("fig")
    call_ax.axhline.assert_called_with(y=0.693685807840985, color="r", linestyle="--")
    call_ax.text.assert_called_with(
        x=0,
        y=0.693685807840985,
        s="0.6937",
        color="red",
        rotation="vertical",
        horizontalalignment="right",
        verticalalignment="top",
    )

    calls: List[dict] = [call.kwargs for call in call_sns_scatterplot.call_args_list]
    for call in calls:
        call["data"] = call["data"].to_dict()

    assert calls == [
        {
            "data": {"index": {0: 0, 1: 1, 2: 2}, "score": {0: 0.13, 1: 0.01, 2: 0.08}},
            "ax": call_ax,
            "color": "grey",
            "x": "index",
            "y": "score",
        },
        {"data": {"index": {3: 3}, "score": {3: 0.76}}, "ax": call_ax, "color": "orange", "x": "index", "y": "score"},
        {
            "data": {"index": {0: 0, 1: 1, 3: 3}, "score": {0: 0.6, 1: 0.2, 3: 0.6}},
            "ax": call_ax,
            "color": "grey",
            "x": "index",
            "y": "score",
        },
        {"data": {"index": {2: 2}, "score": {2: 0.91}}, "ax": call_ax, "color": "orange", "x": "index", "y": "score"},
    ]
