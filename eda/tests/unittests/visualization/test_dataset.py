import os
from unittest.mock import ANY, MagicMock

import numpy as np
import pandas as pd
import pytest
from numpy import dtype

from autogluon.eda import AnalysisState
from autogluon.eda.analysis import MissingValuesAnalysis, RawTypesAnalysis, SpecialTypesAnalysis, VariableTypeAnalysis
from autogluon.eda.analysis.dataset import DatasetSummary
from autogluon.eda.auto import analyze
from autogluon.eda.visualization import DatasetStatistics, DatasetTypeMismatch, LabelInsightsVisualization

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))


def test_DatasetStatistics():
    train_data = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv"))[["education", "class"]]

    viz = DatasetStatistics()
    viz.display_obj = MagicMock()

    analyze(
        train_data=train_data,
        label="class",
        anlz_facets=[
            DatasetSummary(),
            RawTypesAnalysis(),
            VariableTypeAnalysis(),
            SpecialTypesAnalysis(),
            MissingValuesAnalysis(),
        ],
        viz_facets=[viz],
    )

    assert viz.display_obj.call_args.args[0].to_dict() == {
        "count": {"class": 200, "education": 200},
        "dtypes": {"class": dtype("O"), "education": dtype("O")},
        "freq": {"class": 154, "education": 59},
        "missing_count": {"class": "", "education": ""},
        "missing_ratio": {"class": "", "education": ""},
        "raw_type": {"class": "object", "education": "object"},
        "special_types": {"class": "", "education": ""},
        "top": {"class": " <=50K", "education": " HS-grad"},
        "unique": {"class": 2, "education": 14},
        "variable_type": {"class": "category", "education": "category"},
    }


@pytest.mark.parametrize(
    "state_present,expected",
    [
        ("dataset_stats", True),
        ("missing_statistics", True),
        ("raw_type", True),
        ("special_types", True),
        ("unknown_type", False),
    ],
)
def test_DatasetStatistics__can_handle(state_present, expected):
    assert DatasetStatistics().can_handle(AnalysisState({state_present: ""})) is expected


@pytest.mark.parametrize("field", ["dataset_stats", "raw_type", "variable_type", "special_types"])
def test__merge_analysis_facets__single_values(field):
    expected_result = {"some_stat": "value"}
    state = AnalysisState({field: {"ds": expected_result}})
    if field == "dataset_stats":
        assert DatasetStatistics._merge_analysis_facets("ds", state) == expected_result
    else:
        assert DatasetStatistics._merge_analysis_facets("ds", state) == {field: expected_result}


def test__merge_analysis_facets__single_values__missing_statistics():
    state = AnalysisState(
        {"missing_statistics": {"ds": {"count": [1, 2], "ratio": [0.1, 0.2], "some_field": ["a", "b"]}}}
    )
    assert DatasetStatistics._merge_analysis_facets("ds", state) == {
        "missing_count": [1, 2],
        "missing_ratio": [0.1, 0.2],
    }


def test__merge_analysis_facets__multiple_values():
    state = AnalysisState(
        {
            "dataset_stats": {"ds": {"dataset_stats": "value"}},
            "missing_statistics": {"ds": {"count": [1, 2], "ratio": [0.1, 0.2], "some_field": ["a", "b"]}},
            "raw_type": {"ds": {"raw_type": "value"}},
            "variable_type": {"ds": {"variable_type": "value"}},
            "special_types": {"ds": {"special_types": "value"}},
        }
    )
    assert DatasetStatistics._merge_analysis_facets("ds", state) == {
        "dataset_stats": "value",
        "missing_count": [1, 2],
        "missing_ratio": [0.1, 0.2],
        "raw_type": {"raw_type": "value"},
        "special_types": {"special_types": "value"},
        "variable_type": {"variable_type": "value"},
    }


def test__fix_counts():
    df = pd.DataFrame(
        {
            "a": [1.0, np.NaN],
            "b": [1.0, np.NaN],
            "c": [1, np.NaN],
            "d": [1, 2],
        }
    )
    expected_out = {"a": {0: 1, 1: ""}, "b": {0: 1.0, 1: "--NA--"}, "c": {0: 1, 1: ""}, "d": {0: 1, 1: 2}}
    assert DatasetStatistics._fix_counts(df, cols=["a", "c", "d"]).fillna("--NA--").to_dict() == expected_out


def test_DatasetTypeMismatch():
    df_train = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    df_test = pd.DataFrame({"x": ["1", "2", "3"], "y": [4, 5, 6]})

    viz = DatasetTypeMismatch()
    viz.render_header_if_needed = MagicMock()
    viz.display_obj = MagicMock()

    analyze(train_data=df_train, test_data=df_test, anlz_facets=[RawTypesAnalysis()], viz_facets=[viz])

    viz.render_header_if_needed.assert_called_with(ANY, "Types warnings summary")
    assert viz.display_obj.call_args.args[0].to_dict() == {
        "test_data": {"x": "object"},
        "train_data": {"x": "int"},
        "warnings": {"x": "warning"},
    }


def test_DatasetTypeMismatch__no_warnings():
    df_train = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    viz = DatasetTypeMismatch()
    viz.render_header_if_needed = MagicMock()
    viz.display_obj = MagicMock()
    analyze(train_data=df_train, test_data=df_train, anlz_facets=[RawTypesAnalysis()], viz_facets=[viz])

    viz.render_header_if_needed.assert_not_called()
    viz.display_obj.assert_not_called()


def test_LabelInsightsVisualization():
    state = AnalysisState(
        {
            "label_insights": {
                "low_cardinality_classes": {
                    "instances": {"A": 10, "B": 20},
                    "threshold": 50,
                },
                "not_present_in_train": [1, "2", True],
                "ood": {
                    "count": 100,
                    "train_range": [10, 100],
                    "test_range": [10, 50],
                },
            }
        }
    )
    viz = LabelInsightsVisualization()
    viz.render_header_if_needed = MagicMock()
    viz.render_markdown = MagicMock()

    viz.render(state)

    viz.render_header_if_needed.assert_called_with(state, "Label insights")
    viz.render_markdown.assert_called_with(
        " - Low-cardinality classes are detected. It is recommended to have at least "
        "`50` instances per class. Consider adding more data to cover the classes or "
        "remove such rows.\n"
        "   - class `A`: `10` instances\n"
        "   - class `B`: `20` instances\n"
        " - the following classes are found in `test_data`, but not present in "
        "`train_data`: `1`, `2`, `True`. Consider either removing the rows with "
        "classes not covered or adding more training data covering the classes.\n"
        " - Rows with out-of-domain labels were found. Consider removing rows with "
        "labels outside of this range or expand training data since some algorithms "
        "(i.e. trees) are unable to extrapolate beyond data present in the training "
        "data.\n"
        "   - `100` rows\n"
        "   - `train_data` values range `[10, 100]`\n"
        "   - `test_data` values range `[10, 50]`"
    )


def test_LabelInsightsVisualization__no_state():
    state = AnalysisState()

    viz = LabelInsightsVisualization()
    viz.render_header_if_needed = MagicMock()
    viz.render_markdown = MagicMock()

    viz.render(state)

    viz.render_header_if_needed.assert_not_called()
    viz.render_markdown.assert_not_called()
