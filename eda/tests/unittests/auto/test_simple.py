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
from autogluon.eda.analysis import AnomalyDetectorAnalysis, Namespace, ShapAnalysis
from autogluon.eda.analysis.base import BaseAnalysis
from autogluon.eda.auto import (
    analyze,
    analyze_interaction,
    covariate_shift_detection,
    dataset_overview,
    detect_anomalies,
    explain_rows,
    partial_dependence_plots,
    quick_fit,
    target_analysis,
)
from autogluon.eda.auto.simple import (
    _is_single_numeric_variable,
    _prepare_pdp_data,
    _validate_and_normalize_pdp_args,
    get_default_estimator_if_not_specified,
    get_empty_dict_if_none,
)
from autogluon.eda.visualization import (
    AnomalyScoresVisualization,
    ConfusionMatrix,
    CorrelationVisualization,
    DatasetStatistics,
    DatasetTypeMismatch,
    ExplainForcePlot,
    ExplainWaterfallPlot,
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
from autogluon.eda.visualization.interaction import FeatureDistanceAnalysisVisualization, PDPInteractions

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
    assert state.sample_size == {"test_data": 5, "train_data": 5, "val_data": 5}
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

    assert call_md_render.call_count == 9
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
    "plot",
    [
        ("force"),
        ("waterfall"),
        (None),
    ],
)
def test_explain_rows(plot, monkeypatch):
    train_data = MagicMock()
    model = MagicMock()
    rows = MagicMock()
    with monkeypatch.context() as m:
        call_analyze = MagicMock(return_value="result")
        m.setattr(eda.auto.simple, "analyze", call_analyze)

        result = explain_rows(
            train_data=train_data,
            model=model,
            rows=rows,
            plot=plot,
            return_state=True,
            baseline_sample=300,
            other_arg="other_arg",
        )
        assert result == "result"

        call_analyze.assert_called_with(
            train_data=train_data[model.original_features],
            model=model,
            return_state=True,
            anlz_facets=ANY,
            viz_facets=ANY,
        )

        anlz_facet = call_analyze.call_args.kwargs["anlz_facets"][0]
        assert type(anlz_facet) is ShapAnalysis
        assert anlz_facet.rows == rows
        assert anlz_facet.baseline_sample == 300

        expected_plot = {
            "force": ExplainForcePlot,
            "waterfall": ExplainWaterfallPlot,
        }.get(plot, None)
        viz_facet = call_analyze.call_args.kwargs["viz_facets"]
        if expected_plot is None:
            assert viz_facet is None
        else:
            assert type(viz_facet[0]) is expected_plot


def test_partial_dependence_plots__prepare_pdp_data():
    train_data = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)
    train_data = train_data.drop(
        columns=["marital-status", "education", "occupation", "relationship", "workclass", "race"]
    )
    assert sorted(train_data["native-country"].unique().tolist()) == [
        " ?",
        " Canada",
        " Cuba",
        " El-Salvador",
        " Haiti",
        " Mexico",
        " Philippines",
        " Puerto-Rico",
        " United-States",
    ]
    pdp_data, state, id_to_category_mappings = _prepare_pdp_data(train_data, label="class", sample=1000)
    assert id_to_category_mappings == {
        "native-country": {0: " ?", 1: " Mexico", 2: " United-States"},
    }
    assert state.pdp_train_data is not None
    assert sorted(pdp_data["native-country"].unique().tolist()) == [-1, 0, 1, 2]


def test_partial_dependence_plots__prepare_pdp_data__features_specified():
    train_data = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)
    pdp_data, state, id_to_category_mappings = _prepare_pdp_data(
        train_data, label="class", features=["native-country"], sample=1000
    )
    assert id_to_category_mappings == {
        "native-country": {0: " ?", 1: " Mexico", 2: " United-States"},
    }


@pytest.mark.parametrize("col_number_warning", [5, 20])
def test_partial_dependence_plots__validate_and_normalize_pdp_args__none_args(col_number_warning):
    train_data = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv"))
    chart_args, fig_args, features = _validate_and_normalize_pdp_args(
        train_data, features=None, fig_args=None, chart_args=None, col_number_warning=col_number_warning
    )
    assert chart_args == {}
    assert fig_args == {}
    assert features is None


def test_partial_dependence_plots__validate_and_normalize_pdp_args__single_feature():
    train_data = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv"))
    chart_args, fig_args, features = _validate_and_normalize_pdp_args(train_data, features="education")
    assert chart_args == {}
    assert fig_args == {}
    assert features == ["education"]


def test_partial_dependence_plots__validate_and_normalize_pdp_args__wrong_features():
    train_data = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv"))
    with pytest.raises(AssertionError):
        _validate_and_normalize_pdp_args(train_data, features=["not_present_1", "not_present_1"])


def test_partial_dependence_plots__validate_and_normalize_pdp_args():
    train_data = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv"))
    chart_args, fig_args, features = _validate_and_normalize_pdp_args(
        train_data,
        features=["education", "occupation"],
        fig_args={"fig_arg": 1},
        chart_args={"chart_arg": 2},
    )
    assert chart_args == {"chart_arg": 2}
    assert fig_args == {"fig_arg": 1}
    assert features == ["education", "occupation"]


def test_partial_dependence_plots(monkeypatch):
    train_data = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(100, random_state=0)

    call_md_render = MagicMock()
    call_pdp_interaction_render = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(MarkdownSectionComponent, "render", call_md_render)
        m.setattr(PDPInteractions, "render", call_pdp_interaction_render)

        with tempfile.TemporaryDirectory() as path:
            state = partial_dependence_plots(
                train_data, label="class", features=["education", "native-country"], return_state=True, path=path
            )

            assert state == {
                "pdp_id_to_category_mappings": {
                    "education": {
                        0: " 10th",
                        1: " 11th",
                        2: " 12th",
                        3: " 7th-8th",
                        4: " Assoc-acdm",
                        5: " Assoc-voc",
                        6: " Bachelors",
                        7: " HS-grad",
                        8: " Masters",
                        9: " Prof-school",
                        10: " Some-college",
                    },
                    "native-country": {0: " ?", 1: " Mexico", 2: " United-States"},
                }
            }

    call_pdp_interaction_render.assert_called_once()
    call_md_render.assert_called()


@pytest.mark.parametrize(
    "add_explainability",
    [
        True,
        False,
    ],
)
def test_detect_anomalies(monkeypatch, add_explainability):
    df_train = pd.DataFrame({"A": np.arange(4), "B": np.arange(4)})
    df_test = pd.DataFrame({"A": np.arange(5), "B": np.arange(5)})

    train_data_scores = pd.Series([0.13, 0.01, 0.08, 0.76], name="score")
    test_data_scores = pd.Series([0.60, 0.20, 0.91, 0.60, 0.23], name="score")
    assert len(df_train) == len(train_data_scores)
    assert len(df_test) == len(test_data_scores)

    call_md_render = MagicMock()
    call_df_render = MagicMock()
    call_anomaly_viz_render = MagicMock()

    def verify_explain_fn(anomaly_idx):
        return {"called_vals": list(anomaly_idx)}

    def call_anomaly_anlz_fit_mock(state: AnalysisState, args: AnalysisState, **fit_kwargs):
        assert state == {"problem_type": "regression"}
        assert args.train_data.equals(df_train)
        assert args.test_data.equals(df_test)
        assert args.label == "B"
        assert args.feature_generator is True

        state.anomaly_detection = {
            "scores": {"train_data": train_data_scores, "test_data": test_data_scores},
            "explain_rows_fns": {
                "train_data": verify_explain_fn,
                "test_data": verify_explain_fn,
            },
        }

    call_anomaly_anlz_fit = MagicMock(side_effect=call_anomaly_anlz_fit_mock)
    call_explain_rows = MagicMock()

    with monkeypatch.context() as m:
        m.setattr(AnomalyDetectorAnalysis, "_fit", call_anomaly_anlz_fit)
        m.setattr(MarkdownSectionComponent, "render_markdown", call_md_render)
        m.setattr(PropertyRendererComponent, "display_obj", call_df_render)
        m.setattr(AnomalyScoresVisualization, "render", call_anomaly_viz_render)
        m.setattr("autogluon.eda.auto.simple.explain_rows", call_explain_rows)
        state = detect_anomalies(
            train_data=df_train,
            test_data=df_test,
            label="B",
            threshold_stds=2,
            explain_top_n_anomalies=1 if add_explainability else None,
            show_help_text=False,
            return_state=True,
        )

    assert list(state.anomaly_detection.scores.keys()) == ["train_data", "test_data"]
    assert state.anomaly_detection.scores.train_data.equals(train_data_scores)
    assert state.anomaly_detection.scores.test_data.equals(test_data_scores)
    state.anomaly_detection.pop("scores")

    assert list(state.anomaly_detection.anomalies.keys()) == ["train_data", "test_data"]
    assert state.anomaly_detection.anomalies.train_data.equals(
        pd.DataFrame({"A": [3], "B": [3], "score": 0.76}, index=[3])
    )
    assert state.anomaly_detection.anomalies.test_data.equals(
        pd.DataFrame({"A": [2], "B": [2], "score": 0.91}, index=[2])
    )
    state.anomaly_detection.pop("anomalies")

    assert state == {
        "problem_type": "regression",
        "anomaly_detection": {
            "anomaly_score_threshold": 0.693685807840985,
            "explain_rows_fns": {
                "train_data": verify_explain_fn,
                "test_data": verify_explain_fn,
            },
        },
    }

    # Markdown rendering
    calls = [c[0][0] for c in call_md_render.call_args_list]
    expected_calls = [
        "### Anomaly Detection Report",
        # train_data
        "**Top-10 `train_data` anomalies (total: 1)**",
        # test_data
        "**Top-10 `test_data` anomalies (total: 1)**",
    ]
    if add_explainability:
        expected_calls.insert(
            2,
            "⚠️ Please note that the feature values shown on the charts below are transformed "
            "into an internal representation; they may be encoded or modified based on internal preprocessing. "
            "Refer to the original datasets for the actual feature values.",
        )
        expected_calls.insert(
            3,
            "⚠️ The detector has seen this dataset; the may result in overly optimistic estimates. "
            "Although the anomaly score in the explanation might not match, the magnitude of the feature scores "
            "can still be utilized to evaluate the impact of the feature on the anomaly score.",
        )
        expected_calls.append(
            "⚠️ Please note that the feature values shown on the charts below are transformed "
            "into an internal representation; they may be encoded or modified based on internal preprocessing. "
            "Refer to the original datasets for the actual feature values."
        )
    assert calls == expected_calls

    # Tables rendering
    calls = [c[0][0].to_dict() for c in call_df_render.call_args_list]
    assert calls == [
        {"A": {3: 3}, "B": {3: 3}, "score": {3: 0.76}},
        {"A": {2: 2}, "B": {2: 2}, "score": {2: 0.91}},
    ]

    # Explainability data
    if add_explainability:
        calls = [c.kwargs for c in call_explain_rows.call_args_list]
        assert calls == [
            {"called_vals": [3], "plot": "waterfall"},
            {"called_vals": [2], "plot": "waterfall"},
        ]
    else:
        call_explain_rows.assert_not_called()
