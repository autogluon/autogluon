import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from autogluon.eda import AnalysisState
from autogluon.eda.analysis import Namespace
from autogluon.eda.analysis.base import BaseAnalysis
from autogluon.eda.auto import analyze, quick_fit
from autogluon.eda.visualization import (
    ConfusionMatrix,
    FeatureImportance,
    MarkdownSectionComponent,
    ModelLeaderboard,
    RegressionEvaluation,
)
from autogluon.eda.visualization.base import AbstractVisualization

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
