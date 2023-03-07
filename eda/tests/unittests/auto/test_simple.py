import os
import re
import tempfile
from sys import platform
from unittest.mock import ANY, MagicMock, call

import numpy as np
import pandas as pd
import pytest

import autogluon.eda as eda
from autogluon.eda import AnalysisState
from autogluon.eda.analysis import Namespace
from autogluon.eda.analysis.base import BaseAnalysis
from autogluon.eda.auto import (
    analyze,
    analyze_interaction,
    covariate_shift_detection,
    dataset_overview,
    explain_rows,
    quick_fit,
    target_analysis,
)
from autogluon.eda.auto.simple import (
    _is_single_numeric_variable,
    get_default_estimator_if_not_specified,
    get_empty_dict_if_none,
)
from autogluon.eda.visualization import (
    ConfusionMatrix,
    CorrelationVisualization,
    DatasetStatistics,
    DatasetTypeMismatch,
    FeatureImportance,
    FeatureInteractionVisualization,
    LabelInsightsVisualization,
    MarkdownSectionComponent,
    ModelLeaderboard,
    PropertyRendererComponent,
    RegressionEvaluation,
    XShiftSummary,
)
from autogluon.eda.visualization.base import AbstractVisualization
from autogluon.eda.visualization.interaction import FeatureDistanceAnalysisVisualization
from autogluon.eda.visualization.jupyter import JupyterMixin

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
    call_prc_render = MagicMock()

    with monkeypatch.context() as m:
        m.setattr(MarkdownSectionComponent, "render", call_md_render)
        m.setattr(ConfusionMatrix, "render", call_cm_render)
        m.setattr(RegressionEvaluation, "render", call_reg_render)
        m.setattr(ModelLeaderboard, "render", call_ldr_render)
        m.setattr(FeatureImportance, "render", call_fi_render)
        m.setattr(PropertyRendererComponent, "render", call_prc_render)
        _force_using_rf_if_on_mac(m)

        with tempfile.TemporaryDirectory() as path:
            quick_fit(path=path, train_data=df_train, label="class")

    assert call_md_render.call_count == 7
    assert call_prc_render.call_count == 2
    call_cm_render.assert_called_once()
    call_reg_render.assert_called_once()
    call_ldr_render.assert_called_once()
    call_fi_render.assert_called_once()


def test_dataset_overview(monkeypatch):
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)
    df_train["education-num"] = df_train["education-num"] + np.random.rand(len(df_train)) / 100
    df_train["near_duplicate"] = df_train["education-num"] + 0.1

    call_ds_render = MagicMock()
    call_dtm_render = MagicMock()
    call_md_render = MagicMock()
    call_fdav_render = MagicMock()
    call_fiv_render = MagicMock()

    with monkeypatch.context() as m:
        m.setattr(DatasetStatistics, "render", call_ds_render)
        m.setattr(DatasetTypeMismatch, "render", call_dtm_render)
        m.setattr(MarkdownSectionComponent, "render_markdown", call_md_render)
        m.setattr(FeatureDistanceAnalysisVisualization, "render", call_fdav_render)
        m.setattr(FeatureInteractionVisualization, "render", call_fiv_render)

        dataset_overview(train_data=df_train, label="class")

    call_md_render.assert_has_calls(
        [
            call("### Feature Distance"),
            call("Feature interaction between `education-num`/`near_duplicate`"),
        ]
    )
    call_ds_render.assert_called_once()
    call_dtm_render.assert_called_once()
    call_fdav_render.assert_called_once()
    call_fiv_render.assert_called_once()


def test_covariate_shift_detection(monkeypatch):
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)
    df_train["shift_col"] = np.random.rand(len(df_train))
    df_test = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "test_data.csv")).sample(100, random_state=0)
    df_test["shift_col"] = np.random.rand(len(df_test)) + 2

    call_xss_render = MagicMock()
    call_md_render = MagicMock()
    call_fiv_render = MagicMock()
    with monkeypatch.context() as m:
        _force_using_rf_if_on_mac(m)
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
    call_md_render.assert_called_once_with("**`shift_col` values distribution between datasets; p-value: `0.0000`**")


def _force_using_rf_if_on_mac(m):
    if platform == "darwin":
        # Stability - use RF instead of LightGBM can on mac for tests
        call_is_lightgbm_available = MagicMock(return_value=False)
        m.setattr(eda.auto.simple, "_is_lightgbm_available", call_is_lightgbm_available)


def test_get_empty_dict_if_none():
    assert get_empty_dict_if_none(None) == {}
    assert get_empty_dict_if_none({"q"}) == {"q"}


@pytest.mark.parametrize(
    "hyperparameters_present, lgbm_present, presets_present",
    [
        (True, False, True),
        (True, False, False),
        (False, False, True),
        (False, False, False),
        (True, True, True),
        (True, True, False),
        (False, True, True),
        (False, True, False),
    ],
)
def test_get_default_estimator_if_not_specified(monkeypatch, hyperparameters_present, lgbm_present, presets_present):
    fit_args = {}
    if hyperparameters_present:
        fit_args["hyperparameters"] = "some_params"
    if presets_present:
        fit_args["presets"] = "some_presets"

    with monkeypatch.context() as m:
        m.setattr(eda.auto.simple, "_is_lightgbm_available", lambda: lgbm_present)

        if (not hyperparameters_present) and (not presets_present):
            if lgbm_present:
                assert "GBM" in get_default_estimator_if_not_specified(fit_args)["hyperparameters"]
            else:
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


def test_target_analysis__classification(monkeypatch):
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)

    call_md_render = MagicMock()
    call_ds_render = MagicMock()
    call_cv_render = MagicMock()
    call_fiv_render = MagicMock()
    call_liv_render = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(MarkdownSectionComponent, "render_markdown", call_md_render)
        m.setattr(DatasetStatistics, "render", call_ds_render)
        m.setattr(CorrelationVisualization, "render", call_cv_render)
        m.setattr(FeatureInteractionVisualization, "render", call_fiv_render)
        m.setattr(LabelInsightsVisualization, "render", call_liv_render)

        state = target_analysis(train_data=df_train, label="class", return_state=True)

    call_md_render.assert_has_calls(
        [
            call("## Target variable analysis"),
            call("### Label Insights"),
        ]
    )
    call_ds_render.assert_called_once()
    call_cv_render.assert_called_once()
    call_liv_render.assert_called_once()
    assert call_fiv_render.call_count == 2
    assert sorted(set(state.keys())) == [
        "correlations",
        "correlations_focus_field",
        "correlations_focus_field_threshold",
        "correlations_focus_high_corr",
        "correlations_method",
        "dataset_stats",
        "interactions",
        "label_insights",
        "missing_statistics",
        "problem_type",
        "raw_type",
        "special_types",
        "variable_type",
    ]

    assert sorted(set(state.interactions.train_data.keys())) == ["__analysis__", "relationship:class"]
    assert sorted(state.correlations.train_data.columns.tolist()) == ["class", "relationship"]
    assert state.correlations_focus_high_corr.train_data.index.tolist() == ["relationship"]


def test_target_analysis__regression(monkeypatch):
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)

    call_md_render = MagicMock()
    call_ds_render = MagicMock()
    call_cv_render = MagicMock()
    call_fiv_render = MagicMock()
    call_liv_render = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(MarkdownSectionComponent, "render_markdown", call_md_render)
        m.setattr(DatasetStatistics, "render", call_ds_render)
        m.setattr(CorrelationVisualization, "render", call_cv_render)
        m.setattr(FeatureInteractionVisualization, "render", call_fiv_render)
        m.setattr(LabelInsightsVisualization, "render", call_liv_render)

        state = target_analysis(train_data=df_train, label="fnlwgt", return_state=True)

    assert call_md_render.call_count == 3
    calls = [re.sub(r"[.][0-9]{4,}", ".xx", c[0][0]) for c in call_md_render.call_args_list]
    assert calls == [
        "## Target variable analysis",
        "\n".join(
            [
                "### Distribution fits for target variable",
                " - [kstwobign](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstwobign.html)",
                "   - p-value: 0.976",
                "   - Parameters: (loc: -134163.xx, scale: 377621.xx)",
                " - [gumbel_r](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html)",
                "   - p-value: 0.966",
                "   - Parameters: (loc: 149399.xx, scale: 79111.xx)",
                " - [nakagami](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nakagami.html)",
                "   - p-value: 0.965",
                "   - Parameters: (nu: 0.xx, loc: 28236.xx, scale: 192280.xx)",
                " - [skewnorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html)",
                "   - p-value: 0.963",
                "   - Parameters: (a: 3.xx, loc: 78497.xx, scale: 150470.xx)",
                " - [weibull_min](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html)",
                "   - p-value: 0.963",
                "   - Parameters: (c: 1.xx, loc: 16324.xx, scale: 200669.xx)",
            ]
        ),
        "### Target variable correlations\n - ⚠️ no fields with absolute correlation greater than `0.5` found for target variable `fnlwgt`.",
    ]

    call_ds_render.assert_called_once()
    call_cv_render.assert_called_once()
    call_fiv_render.assert_called_once()
    call_liv_render.assert_called_once()
    assert sorted(set(state.keys())) == [
        "correlations",
        "correlations_focus_field",
        "correlations_focus_field_threshold",
        "correlations_focus_high_corr",
        "correlations_method",
        "dataset_stats",
        "distributions_fit",
        "distributions_fit_pvalue_min",
        "interactions",
        "missing_statistics",
        "problem_type",
        "raw_type",
        "special_types",
        "variable_type",
    ]

    assert sorted(set(state.interactions.train_data.keys())) == ["__analysis__"]
    assert sorted(state.correlations.train_data.columns.tolist()) == ["fnlwgt"]
    assert state.correlations_focus_high_corr.train_data.index.tolist() == []
    assert list(state.distributions_fit.train_data.fnlwgt.keys()) == [
        "kstwobign",
        "gumbel_r",
        "nakagami",
        "skewnorm",
        "weibull_min",
    ]


@pytest.mark.parametrize(
    "label,expected_task_type",
    [
        ("class", "binary"),
        ("fnlwgt", "regression"),
    ],
)
def test_explain_rows(label, expected_task_type, monkeypatch):
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)
    with tempfile.TemporaryDirectory() as path:
        state = quick_fit(
            estimator_args=dict(path=path),
            train_data=df_train,
            label=label,
            return_state=True,
            save_model_to_state=True,
            render_analysis=False,
        )

        assert state.model.problem_type == expected_task_type

        with monkeypatch.context() as m:
            call_shap_force_plot = MagicMock()
            call_shap_waterfall_plot = MagicMock()
            m.setattr("shap.force_plot", call_shap_force_plot)
            m.setattr(eda.auto.simple, "waterfall_plot", call_shap_waterfall_plot)

            # --- force_plot

            call_display_obj = MagicMock()
            m.setattr(JupyterMixin, "display_obj", call_display_obj)
            explain_rows(
                train_data=df_train,
                model=state.model,
                rows=state.model_evaluation.highest_error[:2],
                text_rotation=40,
                display_rows=True,
                other_arg="a",
            )
            expected_calls = [
                call(ANY, ANY, ANY, text_rotation=40, matplotlib=True, other_arg="a"),
                call(ANY, ANY, ANY, text_rotation=40, matplotlib=True, other_arg="a"),
            ]

            expected_rows = [state.model_evaluation.highest_error.iloc[i][df_train.columns] for i in range(2)]

            assert call_display_obj.call_count == 2
            call_shap_force_plot.assert_has_calls(expected_calls)

            for c, er in zip(call_shap_force_plot.call_args_list, expected_rows):
                assert er.equals(c.args[2])

            # --- waterfall_plot

            call_display_obj = MagicMock()
            m.setattr(JupyterMixin, "display_obj", call_display_obj)
            explain_rows(
                train_data=df_train,
                model=state.model,
                rows=state.model_evaluation.highest_error[2:4],
                waterfall=True,
            )

            expected_rows = [state.model_evaluation.highest_error.iloc[i][df_train.columns] for i in range(2, 4)]

            call_display_obj.assert_not_called()
            assert call_shap_waterfall_plot.call_count == 2

            for c, er in zip(call_shap_waterfall_plot.call_args_list, expected_rows):
                assert er.equals(c.args[2])
