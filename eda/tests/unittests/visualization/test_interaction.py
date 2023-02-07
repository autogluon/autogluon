from unittest.mock import ANY, MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy
import seaborn as sns
from hamcrest.library.integration import match_equality
from pandas import DataFrame

import autogluon.eda.auto as auto
from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT, R_OBJECT
from autogluon.eda import AnalysisState
from autogluon.eda.visualization import (
    CorrelationSignificanceVisualization,
    CorrelationVisualization,
    FeatureInteractionVisualization,
)
from autogluon.eda.visualization.interaction import FeatureDistanceAnalysisVisualization


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
        "**`train_data` - `phik` correlation matrix**",
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
        "**`train_data` - `spearman` correlation matrix; focus: absolute correlation for `c` >= `0.6`**",
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
    call_render_markdown = MagicMock()

    analysis = facet_cls(fig_args=dict(some_arg=123), **kwargs)
    analysis.render_markdown = call_render_markdown

    with monkeypatch.context() as m:
        m.setattr(sns, "heatmap", call_heatmap)
        m.setattr(plt, "show", call_show)
        m.setattr(plt, "yticks", call_yticks)
        m.setattr(plt, "subplots", call_subplots)
        auto.analyze(state=state, viz_facets=[analysis])

    call_yticks.assert_called_with(rotation=0)
    call_subplots.assert_called_with(some_arg=123, figsize=(4, 4))
    call_show.assert_called_with("fig")

    call_heatmap.assert_called_with(
        state[render_key].train_data,
        annot=True,
        ax="ax",
        linewidths=0.5,
        linecolor="lightgrey",
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.5},
        **heatmap_args,
    )
    if text_render is not None:
        call_render_markdown.assert_called_with(text_render)
    else:
        call_render_markdown.assert_not_called()


def test_FeatureInteractionVisualization__happy_path(monkeypatch):
    state = __get_feature_interaction_state()
    call_render = MagicMock()
    call_show = MagicMock()
    call_subplots = MagicMock(return_value=("fig", "ax"))

    viz = FeatureInteractionVisualization(
        key="abc", numeric_as_categorical_threshold=2, some_chart_arg=123, fig_args=dict(key="value")
    )

    with monkeypatch.context() as m:
        m.setattr(plt, "subplots", call_subplots)
        m.setattr(plt, "show", call_show)
        m.setattr(viz._RegPlotRenderer, "_render", call_render)
        auto.analyze(state=state, viz_facets=[viz])

    call_render.assert_called_with(
        match_equality(state),  # noqa
        "train_data",
        ("a", "b", None),
        ("numeric", "numeric", None),
        "ax",
        match_equality(state.interactions.train_data.abc.data),
        dict(x="a", y="b", some_chart_arg=123),
    )
    call_show.assert_called_with("fig")
    call_subplots.assert_called_with(key="value", figsize=(12, 6))


@pytest.mark.parametrize("is_single, header", [(True, True), (False, True), (True, False), (False, False)])
def test_FeatureInteractionVisualization__headers(monkeypatch, is_single, header):
    state = __get_feature_interaction_state()
    if is_single:
        state.interactions.train_data.abc.features.pop("y")
    call_render = MagicMock()
    call_show = MagicMock()
    call_subplots = MagicMock(return_value=("fig", "ax"))

    viz = FeatureInteractionVisualization(key="abc", numeric_as_categorical_threshold=2, headers=header)
    viz.render_markdown = MagicMock()

    with monkeypatch.context() as m:
        m.setattr(plt, "subplots", call_subplots)
        m.setattr(plt, "show", call_show)
        if is_single:
            m.setattr(viz._HistPlotRenderer, "_render", call_render)
        else:
            m.setattr(viz._RegPlotRenderer, "_render", call_render)
        auto.analyze(state=state, viz_facets=[viz])

    if not header:
        viz.render_markdown.assert_not_called()
    elif is_single:
        viz.render_markdown.assert_called_with("**`a` in `train_data`**")
    else:
        viz.render_markdown.assert_called_with("**Feature interaction between `a`/`b` in `train_data`**")


def test_FeatureInteractionVisualization__state_different_key(monkeypatch):
    state = __get_feature_interaction_state()
    call_render = MagicMock()
    call_show = MagicMock()
    call_subplots = MagicMock(return_value=("fig", "ax"))

    viz = FeatureInteractionVisualization(
        key="not_present", numeric_as_categorical_threshold=2, some_chart_arg=123, fig_args=dict(key="value")
    )

    with monkeypatch.context() as m:
        m.setattr(plt, "subplots", call_subplots)
        m.setattr(plt, "show", call_show)
        m.setattr(viz._RegPlotRenderer, "_render", call_render)
        auto.analyze(state=state, viz_facets=[viz])

    call_render.assert_not_called()
    call_show.assert_not_called()
    call_subplots.assert_not_called()


def test_FeatureInteractionVisualization__no_renderer(monkeypatch):
    state = __get_feature_interaction_state()
    state.interactions.train_data.abc.features = {}
    call_show = MagicMock()
    call_subplots = MagicMock(return_value=("fig", "ax"))

    viz = FeatureInteractionVisualization(key="abc")

    with monkeypatch.context() as m:
        m.setattr(plt, "subplots", call_subplots)
        m.setattr(plt, "show", call_show)
        auto.analyze(state=state, viz_facets=[viz])

    call_show.assert_not_called()
    call_subplots.assert_not_called()


def test_FeatureInteractionVisualization__convert_categoricals_to_objects():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [True, False, True],
            "c": [0.1, 0.2, 0.1],
        }
    )
    assert [str(t).replace("64", "").replace("32", "") for t in df.dtypes.to_list()] == ["int", "bool", "float"]
    FeatureInteractionVisualization(key="abc")._convert_categoricals_to_objects(
        df, "a", "category", "b", "category", "c", "category"
    )
    assert [str(t).replace("64", "").replace("32", "") for t in df.dtypes.to_list()] == ["object", "object", "object"]


@pytest.mark.parametrize(
    "x, x_type, y, y_type, hue, expected_is_single, restrict_to_col",
    [
        ("aa", "numeric", "bb", "numeric", None, False, None),
        (None, None, "bb", "numeric", None, True, "bb"),
        ("aa", "category", "bb", "numeric", None, False, None),
        (None, None, "bb", "numeric", None, True, "bb"),
        ("aa", "numeric", None, None, None, True, "aa"),
        (None, None, None, None, None, False, None),
        ("aa", "category", None, None, None, True, None),
        (None, None, None, None, None, False, None),
        ("aa", "numeric", "bb", "category", None, False, None),
        (None, None, "bb", "category", None, True, None),
        ("aa", "category", "bb", "category", None, False, None),
        (None, None, "bb", "category", None, True, None),
        ("aa", "numeric", None, None, None, True, "aa"),
        (None, None, None, None, None, False, None),
        ("aa", "category", None, None, None, True, None),
        (None, None, None, None, None, False, None),
        ("aa", "numeric", "bb", "numeric", "cc", False, None),
        (None, None, "bb", "numeric", "cc", False, None),
        ("aa", "category", "bb", "numeric", "cc", False, None),
        (None, None, "bb", "numeric", "cc", False, None),
        ("aa", "numeric", None, None, "cc", False, None),
        (None, None, None, None, "cc", False, None),
        ("aa", "category", None, None, "cc", False, None),
        (None, None, None, None, "cc", False, None),
        ("aa", "numeric", "bb", "category", "cc", False, None),
        (None, None, "bb", "category", "cc", False, None),
        ("aa", "category", "bb", "category", "cc", False, None),
        (None, None, "bb", "category", "cc", False, None),
        ("aa", "numeric", None, None, "cc", False, None),
        (None, None, None, None, "cc", False, None),
        ("aa", "category", None, None, "cc", False, None),
        (None, None, None, None, "cc", False, None),
    ],
)
def test_FeatureInteractionVisualization__prepare_chart_args__single_var(
    x, x_type, y, y_type, hue, expected_is_single, restrict_to_col
):
    df = pd.DataFrame(
        {
            "aa": [1, 2, 3],
            "bb": [0.1, 0.2, 0.1],
        }
    )

    viz = FeatureInteractionVisualization(key="abc", some_kwarg=123)
    chart_args, data, is_single_var = viz._prepare_chart_args(df, x=x, x_type=x_type, y=y, y_type=y_type, hue=hue)

    assert is_single_var is expected_is_single
    assert chart_args["some_kwarg"] == 123
    if restrict_to_col is None:
        assert data.equals(df)
        for var, col, val in [(x, "x", "aa"), (y, "y", "bb"), (hue, "hue", "cc")]:
            if var is not None:
                assert chart_args[col] == val
            else:
                assert col not in chart_args
    else:
        assert data.equals(df[restrict_to_col])


@pytest.mark.parametrize(
    "x_type, y_type, hue_type, swapped",
    [
        (None, "category", "~~", False),
        (None, "other~~~", "~~", False),
        (None, "category", None, False),
        (None, "other~~~", None, False),
        ("~~", "other~~~", "~~", False),
        ("~~", "other~~~", None, False),
        ("~~", "category", "~~", False),
        ("~~", "category", None, True),
    ],
)
def test_FeatureInteractionVisualization__fig_args(x_type, y_type, hue_type, swapped):
    y = 2
    hue = 3
    _y, _y_type, _hue, _hue_type = FeatureInteractionVisualization(key="abc")._swap_y_and_hue_if_necessary(
        x_type, y, y_type, hue, hue_type
    )
    if swapped:
        assert _hue == y
        assert _hue_type == y_type
        assert _y is None
        assert _y_type is None
    else:
        assert _y == y
        assert _y_type == y_type
        assert _hue == hue
        assert _hue_type == hue_type


@pytest.mark.parametrize(
    "col, raw_type, numeric_as_categorical_threshold, expected_type",
    [
        (None, R_INT, 20, None),
        ("a", R_INT, 20, "category"),
        ("a", R_INT, 2, "numeric"),
        ("a", R_FLOAT, 2, "numeric"),
        ("a", R_OBJECT, 2, "category"),
        ("a", R_CATEGORY, 2, "category"),
        ("a", R_BOOL, 2, "category"),
        ("a", "some_type", 2, None),
    ],
)
def test_FeatureInteractionVisualization__map_raw_type_to_feature_type(
    monkeypatch, col, raw_type, numeric_as_categorical_threshold, expected_type
):
    df = pd.DataFrame({"a": [1, 2, 3]})
    v = FeatureInteractionVisualization(key="abc")
    actual_type = v._map_raw_type_to_feature_type(col, raw_type, df, numeric_as_categorical_threshold)
    if expected_type is None:
        assert actual_type is None
    else:
        assert actual_type == expected_type


def __get_feature_interaction_state():
    class EqualDataFrames(DataFrame):
        def __eq__(self, other):
            return self.equals(other)

    state = AnalysisState(
        {
            "interactions": {
                "train_data": {
                    "abc": {
                        "data": EqualDataFrames(data={"a": [1, 2, 3], "b": ["a", "b", "c"], "c": [0.1, 0.2, 0.3]}),
                        "features": {"x": "a", "y": "b"},
                    }
                }
            },
            "raw_type": {"train_data": {"a": "int", "b": "int", "c": "int", "d": "int", "e": "object", "f": "object"}},
        }
    )
    return state


@pytest.mark.parametrize("has_dist_fit", [True, False])
def test_FeatureInteractionVisualization__HistPlotRenderer(monkeypatch, has_dist_fit):
    r = FeatureInteractionVisualization._HistPlotRenderer()

    state = AnalysisState(
        {
            "distributions_fit": {
                "train_data": {
                    "aaa": {
                        "fisk": {"param": (11, -65, 94), "statistic": 0.04, "pvalue": 0.17},
                        "lognorm": {"param": (0.12, -82, 111), "statistic": 0.05, "pvalue": 0.021},
                    }
                }
            }
        }
    )
    if not has_dist_fit:
        state.distributions_fit.train_data.aaa = None

    params = ("aaa", None, None)
    param_types = ("numeric", None, None)
    chart_args = dict(some_chart_arg=123)

    ax = MagicMock()
    ax.get_xlim = MagicMock(return_value=(0, 1))
    call_sns_histplot = MagicMock()
    call_plt_legend = MagicMock()

    with monkeypatch.context() as m:
        m.setattr(sns, "histplot", call_sns_histplot)
        m.setattr(plt, "legend", call_plt_legend)
        r._render(state, "train_data", params, param_types, ax, "data", chart_args, num_point_to_fit=3)

    if has_dist_fit:
        call_sns_histplot.assert_called_with(ax=ax, data="data", some_chart_arg=123, stat="density")
        ax.get_xlim.assert_called()
        ax.plot.assert_called_with(
            ANY,
            ANY,
            ls="--",
            label="lognorm: pvalue 0.02",
        )
        ax.set_xlim.assert_called()
        call_plt_legend.assert_called()
    else:
        call_sns_histplot.assert_called_with(ax=ax, data="data", some_chart_arg=123)
        ax.get_xlim.assert_not_called()
        ax.plot.assert_not_called()
        ax.set_xlim.assert_not_called()
        call_plt_legend.assert_not_called()


@pytest.mark.parametrize("has_near_duplicates", [True, False])
def test_FeatureDistanceAnalysisVisualization__happy_path(monkeypatch, has_near_duplicates):
    state = AnalysisState(
        {
            "feature_distance": {
                "columns": [
                    "age",
                    "fnlwgt",
                    "education-num",
                    "sex",
                    "capital-gain",
                    "capital-loss",
                    "hours-per-week",
                    "workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "native-country",
                ],
                "linkage": np.array(
                    [
                        [9.0, 11.0, 0.6113, 2.0],
                        [3.0, 10.0, 0.7652, 2.0],
                        [7.0, 15.0, 0.7905, 3.0],
                        [2.0, 6.0, 0.8097, 2.0],
                        [0.0, 4.0, 0.8306, 2.0],
                        [17.0, 18.0, 0.86785, 4.0],
                        [12.0, 13.0, 0.8872, 2.0],
                        [8.0, 20.0, 0.9333, 3.0],
                        [1.0, 5.0, 0.946, 2.0],
                        [16.0, 19.0, 0.94793333, 7.0],
                        [21.0, 23.0, 0.9676619, 10.0],
                        [22.0, 24.0, 0.981165, 12.0],
                        [14.0, 25.0, 1.13729167, 14.0],
                    ]
                ),
                "near_duplicates": [],
                "near_duplicates_threshold": 0.85,
            }
        }
    )

    if has_near_duplicates:
        state.feature_distance["near_duplicates"] = [
            {"distance": 0.6113, "nodes": ["marital-status", "relationship"]},
            {"distance": 0.7905, "nodes": ["occupation", "sex", "workclass"]},
            {"distance": 0.8097, "nodes": ["education-num", "hours-per-week"]},
            {"distance": 0.8306, "nodes": ["age", "capital-gain"]},
        ]

    ax = MagicMock()
    call_subplots = MagicMock(return_value=("fig", ax))
    call_dendrogram = MagicMock()
    call_show = MagicMock()
    call_render_markdown = MagicMock()

    viz = FeatureDistanceAnalysisVisualization(some_chart_arg=123, fig_args=dict(key="value"))
    viz.render_markdown = call_render_markdown

    with monkeypatch.context() as m:
        m.setattr(plt, "subplots", call_subplots)
        m.setattr(plt, "show", call_show)
        m.setattr(scipy.cluster.hierarchy, "dendrogram", call_dendrogram)
        auto.analyze(state=state, viz_facets=[viz])

    call_dendrogram.assert_called_with(
        ax=ax, Z=ANY, labels=state.feature_distance.columns, leaf_font_size=10, orientation="left", some_chart_arg=123
    )
    np.testing.assert_array_equal(call_dendrogram.call_args[1]["Z"], state.feature_distance.linkage)
    call_show.assert_called_with("fig")
    call_subplots.assert_called_with(key="value", figsize=(12, 3.5))
    if has_near_duplicates:
        call_render_markdown.assert_called_with(
            "**The following feature groups are considered as near-duplicates**:\n\nDistance threshold: <= `0.85`. "
            "Consider keeping only some of the columns within each group:\n\n"
            " - `marital-status`, `relationship` - distance `0.61`\n"
            " - `occupation`, `sex`, `workclass` - distance `0.79`\n"
            " - `education-num`, `hours-per-week` - distance `0.81`\n"
            " - `age`, `capital-gain` - distance `0.83`"
        )
    else:
        call_render_markdown.assert_not_called()


def test_FeatureInteractionVisualization__no_fig_args():
    assert FeatureDistanceAnalysisVisualization().fig_args == {}
