from unittest.mock import MagicMock, call

import missingno as msno
import numpy as np
import pandas as pd
import pytest

from autogluon.eda import AnalysisState
from autogluon.eda.visualization import MissingValues


def test_MissingValues():
    state = __prepare_test_data()
    viz = MissingValues(graph_type="matrix", headers=True, abc=123)
    viz._internal_render = MagicMock()
    viz.render_markdown = MagicMock()

    viz.render(state)

    assert viz._internal_render.call_count == 2

    assert viz.render_markdown.call_count == 2
    viz.render_markdown.assert_has_calls(
        calls=[
            call("**`train_data` missing values analysis**"),
            call("**`test_data` missing values analysis**"),
        ]
    )

    viz._internal_render.assert_has_calls(
        calls=[
            call(msno.matrix, state.missing_statistics.train_data.data, abc=123),
            call(msno.matrix, state.missing_statistics.test_data.data, abc=123),
        ]
    )


def test_MissingValues__no_headers():
    state = __prepare_test_data()
    viz = MissingValues(headers=False)
    viz._internal_render = MagicMock()
    viz.render_text = MagicMock()

    viz.render(state)

    assert viz._internal_render.call_count == 2
    assert viz.render_text.call_count == 0


@pytest.mark.parametrize(
    "input_type,expected",
    [("matrix", msno.matrix), ("bar", msno.bar), ("heatmap", msno.heatmap), ("dendrogram", msno.dendrogram)],
)
def test_get_operation(input_type, expected):
    assert MissingValues()._get_operation(input_type) is expected


@pytest.mark.parametrize(
    "cols_number,expected",
    [
        (0, False),
        (1, False),
        (50, False),
        (51, True),
    ],
)
def test_has_too_many_variables_for_matrix(cols_number, expected):
    cols = [f"col{i}" for i in range(cols_number)]
    df_test = pd.DataFrame((np.arange(100))[:, None].repeat([len(cols)], axis=1), columns=cols)
    s = {
        "missing_statistics": {
            "test_data": {
                "data": df_test,
            }
        }
    }
    assert MissingValues()._has_too_many_variables_for_matrix(AnalysisState(s)) is expected


def __prepare_test_data():
    cols = list("AB")
    df_train = pd.DataFrame((np.arange(100))[:, None].repeat([len(cols)], axis=1), columns=cols)
    df_test = pd.DataFrame((np.arange(200))[:, None].repeat([len(cols)], axis=1), columns=cols)
    for df in [df_train, df_test]:
        df["A"] = (df["A"] % 4).replace(2, np.NaN)
    s = {
        "missing_statistics": {
            "train_data": {
                "data": df_train,
            },
            "test_data": {
                "data": df_test,
            },
        }
    }
    state = AnalysisState(s)
    return state
