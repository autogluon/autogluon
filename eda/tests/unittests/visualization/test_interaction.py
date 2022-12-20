from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import seaborn as sns
from hamcrest.library.integration import match_equality
from pandas import DataFrame

import autogluon.eda.auto as auto
from autogluon.eda import AnalysisState
from autogluon.eda.visualization import (
    CorrelationVisualization,
    CorrelationSignificanceVisualization,
    FeatureInteractionVisualization,
)


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
    call_subplots.assert_called_with(key="value")


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


def test_FeatureInteractionVisualization__fig_args():
    assert FeatureInteractionVisualization(key="abc").fig_args == {}


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
