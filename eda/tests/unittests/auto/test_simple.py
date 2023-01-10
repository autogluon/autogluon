import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from autogluon.eda import AnalysisState
from autogluon.eda.analysis import Namespace
from autogluon.eda.analysis.base import BaseAnalysis
from autogluon.eda.auto import analyze, analyze_interaction, covariate_shift_detection, dataset_overview, quick_fit
from autogluon.eda.auto.simple import (
    _is_single_numeric_variable,
    get_default_estimator_if_not_specified,
    get_empty_dict_if_none,
)
from autogluon.eda.visualization import (
    ConfusionMatrix,
    DatasetStatistics,
    DatasetTypeMismatch,
    FeatureImportance,
    FeatureInteractionVisualization,
    MarkdownSectionComponent,
    ModelLeaderboard,
    RegressionEvaluation,
    XShiftSummary,
)
from autogluon.eda.visualization.base import AbstractVisualization
from autogluon.eda.visualization.interaction import FeatureDistanceAnalysisVisualization

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))


class SomeAnalysis(BaseAnalysis):
    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.args = args.copy()


class SomeVisualization(AbstractVisualization):
    def can_handle(self, state: AnalysisState) -> bool:
        return "required_key" in state

    def _render(self, state: AnalysisState) -> None:
        pass


def test_analyze():
    df_train = pd.DataFrame(np.random.randint(0, 100, size=(10, 2)), columns=list("AB"))
    df_test = pd.DataFrame(np.random.randint(0, 100, size=(11, 2)), columns=list("CD"))
    df_val = pd.DataFrame(np.random.randint(0, 100, size=(12, 2)), columns=list("EF"))
    state = {"some_previous_state": {"arg": 1}}

    anlz = SomeAnalysis()
    anlz.can_handle = MagicMock(return_value=True)

    viz = SomeVisualization(namespace="ns1")
    viz._render = MagicMock()
    viz.can_handle = MagicMock(wraps=viz.can_handle)

    state = analyze(
        train_data=df_train,
        test_data=df_test,
        val_data=df_val,
        model="model",
        label="label",
        state=state,
        sample=5,
        return_state=True,
        anlz_facets=[Namespace(namespace="ns1", children=[anlz])],
        viz_facets=[viz],
    )
    assert state.sample_size == 5
    assert state.some_previous_state == {"arg": 1}
    assert state.ns1.args.train_data.shape == (5, 2)
    assert state.ns1.args.test_data.shape == (5, 2)
    assert state.ns1.args.val_data.shape == (5, 2)
    assert state.ns1.args.model == "model"
    assert state.ns1.args.label == "label"


def test_analyze_return_state():
    state = {"some_previous_state": {"arg": 1}}
    assert analyze(state=state) is None
    assert analyze(state=state, return_state=True) == state


def test_analyze_None_state():
    state = None
    assert analyze(state=state, return_state=True) == {}


def test_analyze_state_dict_convert():
    state = {"some_previous_state": {"arg": 1}}
    assert not isinstance(state, AnalysisState)
    _state = analyze(state=state, return_state=True)
    assert _state == state
    assert _state is not state
    assert isinstance(_state, AnalysisState)


def test_analyze_state_no_AnalysisState_convert():
    state = AnalysisState({"some_previous_state": {"arg": 1}})
    assert isinstance(state, AnalysisState)
    _state = analyze(state=state, return_state=True)
    assert _state == state
    assert _state is state
    assert isinstance(_state, AnalysisState)


def test_quick_fit(monkeypatch):
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)

    call_md_render = MagicMock()
    call_cm_render = MagicMock()
    call_reg_render = MagicMock()
    call_ldr_render = MagicMock()
    call_fi_render = MagicMock()

    with monkeypatch.context() as m:
        m.setattr(MarkdownSectionComponent, "render", call_md_render)
        m.setattr(ConfusionMatrix, "render", call_cm_render)
        m.setattr(RegressionEvaluation, "render", call_reg_render)
        m.setattr(ModelLeaderboard, "render", call_ldr_render)
        m.setattr(FeatureImportance, "render", call_fi_render)

        with tempfile.TemporaryDirectory() as path:
            quick_fit(path=path, train_data=df_train, label="class")

    assert call_md_render.call_count == 3
    call_cm_render.assert_called_once()
    call_reg_render.assert_called_once()
    call_ldr_render.assert_called_once()
    call_fi_render.assert_called_once()


def test_dataset_overview(monkeypatch):
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)

    call_ds_render = MagicMock()
    call_dtm_render = MagicMock()
    call_md_render = MagicMock()
    call_fdav_render = MagicMock()

    with monkeypatch.context() as m:
        m.setattr(DatasetStatistics, "render", call_ds_render)
        m.setattr(DatasetTypeMismatch, "render", call_dtm_render)
        m.setattr(MarkdownSectionComponent, "render", call_md_render)
        m.setattr(FeatureDistanceAnalysisVisualization, "render", call_fdav_render)

        dataset_overview(train_data=df_train, label="class")

    call_ds_render.assert_called_once()
    call_dtm_render.assert_called_once()
    call_md_render.assert_called_once()
    call_fdav_render.assert_called_once()


def test_covariate_shift_detection(monkeypatch):
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)
    df_train["shift_col"] = np.random.rand(len(df_train))
    df_test = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "test_data.csv")).sample(100, random_state=0)
    df_test["shift_col"] = np.random.rand(len(df_test)) + 2

    call_xss_render = MagicMock()
    call_md_render = MagicMock()
    call_fiv_render = MagicMock()
    with monkeypatch.context() as m:
        with tempfile.TemporaryDirectory() as path:
            m.setattr(XShiftSummary, "render", call_xss_render)
            m.setattr(MarkdownSectionComponent, "render_markdown", call_md_render)
            m.setattr(FeatureInteractionVisualization, "render", call_fiv_render)
            state = covariate_shift_detection(
                path=path, train_data=df_train, test_data=df_test, label="class", return_state=True, verbosity=2
            )

    call_xss_render.assert_called_once()
    assert state.xshift_results.detection_status is True
    assert state.xshift_results.test_statistic > 0.99
    assert state.xshift_results.pvalue < 0.01
    assert state.xshift_results.feature_importance.iloc[0].name == "shift_col"
    call_fiv_render.assert_called_once()
    call_md_render.assert_called_once_with("#### `shift_col` values distribution between datasets; p-value: `0.0000`")


def test_get_empty_dict_if_none():
    assert get_empty_dict_if_none(None) == {}
    assert get_empty_dict_if_none({"q"}) == {"q"}


@pytest.mark.parametrize(
    "hyperparameters_present, presets_present",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_get_default_estimator_if_not_specified(hyperparameters_present, presets_present):
    fit_args = {}
    if hyperparameters_present:
        fit_args["hyperparameters"] = "some_params"
    if presets_present:
        fit_args["presets"] = "some_presets"

    if (not hyperparameters_present) and (not presets_present):
        assert "RF" in get_default_estimator_if_not_specified(fit_args)["hyperparameters"]
    else:
        assert get_default_estimator_if_not_specified(fit_args) == fit_args


@pytest.mark.parametrize(
    "fit_distributions, expected_dist",
    [
        (["exponpow", "nakagami", "beta", "gamma", "lognorm"], ["exponpow", "nakagami", "beta", "lognorm", "gamma"]),
        ("lognorm", ["lognorm"]),
        (True, ["exponpow", "nakagami", "gompertz", "foldnorm", "genpareto"]),
    ],
)
def test_analyze_interaction__with_distribution(monkeypatch, fit_distributions, expected_dist):
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)

    call_fiv_render = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(FeatureInteractionVisualization, "render", call_fiv_render)
        state = analyze_interaction(
            train_data=df_train,
            x="age",
            fit_distributions=fit_distributions,
            return_state=True,
        )
    assert list(state.distributions_fit.train_data.age.keys()) == expected_dist
    call_fiv_render.assert_called_once()


def test_analyze_interaction__do_not_fit(monkeypatch):
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)

    call_fiv_render = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(FeatureInteractionVisualization, "render", call_fiv_render)
        state = analyze_interaction(
            train_data=df_train,
            x="age",
            return_state=True,
        )
    assert state.distributions_fit is None
    call_fiv_render.assert_called_once()


@pytest.mark.parametrize(
    "x, y, hue, x_type, expected",
    [
        ("x", "y", "hue", "numeric", False),
        ("x", "y", "hue", "category", False),
        ("x", "y", None, "numeric", False),
        ("x", "y", None, "category", False),
        ("x", None, "hue", "numeric", False),
        ("x", None, "hue", "category", False),
        ("x", None, None, "numeric", True),
        ("x", None, None, "category", False),
    ],
)
def test_analyze_interaction__is_single_numeric_variable(x, y, hue, x_type, expected):
    assert _is_single_numeric_variable(x, y, hue, x_type) is expected
