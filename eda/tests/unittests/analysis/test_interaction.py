import pandas as pd

import autogluon.eda.auto as auto
from autogluon.eda import AnalysisState
from autogluon.eda.analysis import Correlation, CorrelationSignificance, FeatureInteraction
from autogluon.eda.analysis.dataset import RawTypesAnalysis


def test_Correlation_spearman():
    state = auto.analyze(train_data=__create_test_df(), return_state=True, anlz_facets=[Correlation()])
    state.correlations.train_data = state.correlations.train_data.round(2)
    expected = AnalysisState(
        {
            "correlations": {
                "train_data": pd.DataFrame(
                    index=list("abcd"),
                    data={
                        "a": [1.00, 0.0, 0.88, -0.88],
                        "b": [0.00, 1.0, 0.40, -0.40],
                        "c": [0.88, 0.4, 1.00, -1.00],
                        "d": [-0.88, -0.4, -1.00, 1.00],
                    },
                )
            },
            "correlations_method": "spearman",
        }
    )
    __compare_outputs(expected, state)


def test_Correlation_pearson():
    state = auto.analyze(train_data=__create_test_df(), return_state=True, anlz_facets=[Correlation(method="pearson")])
    state.correlations.train_data = state.correlations.train_data.round(2)
    expected = AnalysisState(
        {
            "correlations": {
                "train_data": pd.DataFrame(
                    index=list("abcd"),
                    data={
                        "a": [1.00, 0.03, 0.88, -0.88],
                        "b": [0.03, 1.00, 0.43, -0.43],
                        "c": [0.88, 0.43, 1.00, -1.00],
                        "d": [-0.88, -0.43, -1.00, 1.00],
                    },
                )
            },
            "correlations_method": "pearson",
        }
    )
    __compare_outputs(expected, state)


def test_Correlation_kendall():
    state = auto.analyze(train_data=__create_test_df(), return_state=True, anlz_facets=[Correlation(method="kendall")])
    state.correlations.train_data = state.correlations.train_data.round(2)
    expected = AnalysisState(
        {
            "correlations": {
                "train_data": pd.DataFrame(
                    index=list("abcd"),
                    data={
                        "a": [1.00, 0.00, 0.77, -0.77],
                        "b": [0.00, 1.00, 0.36, -0.36],
                        "c": [0.77, 0.36, 1.00, -1.00],
                        "d": [-0.77, -0.36, -1.00, 1.00],
                    },
                )
            },
            "correlations_method": "kendall",
        }
    )
    __compare_outputs(expected, state)


def test_Correlation_phik():
    state = auto.analyze(train_data=__create_test_df(), return_state=True, anlz_facets=[Correlation(method="phik")])
    state.correlations.train_data = state.correlations.train_data.round(2)
    expected = AnalysisState(
        {
            "correlations": {
                "train_data": pd.DataFrame(
                    index=list("abcdef"),
                    data={
                        "a": [1.0, 1.0, 1.00, 1.00, 1.00, 1.00],
                        "b": [1.0, 1.0, 0.00, 0.00, 0.00, 0.00],
                        "c": [1.0, 0.0, 1.00, 0.79, 0.79, 0.79],
                        "d": [1.0, 0.0, 0.79, 1.00, 0.79, 0.79],
                        "e": [1.0, 0.0, 0.79, 0.79, 1.00, 0.79],
                        "f": [1.0, 0.0, 0.79, 0.79, 0.79, 1.00],
                    },
                )
            },
            "correlations_method": "phik",
        }
    )
    __compare_outputs(expected, state)


def test_Correlation_focus():
    actual = auto.analyze(
        train_data=(__create_test_df()),
        return_state=True,
        anlz_facets=[Correlation(focus_field="c", focus_field_threshold=0.5)],
    )
    print(actual)
    actual.correlations.train_data = actual.correlations.train_data.round(2)
    actual.correlations_focus_high_corr.train_data = actual.correlations_focus_high_corr.train_data.round(2)
    expected = AnalysisState(
        {
            "correlations": {
                "train_data": pd.DataFrame(
                    index=list("acd"),
                    data={
                        "a": [1.00, 0.88, -0.88],
                        "c": [0.88, 1.00, -1.00],
                        "d": [-0.88, -1.00, 1.00],
                    },
                )
            },
            "correlations_method": "spearman",
            "correlations_focus_field": "c",
            "correlations_focus_field_threshold": 0.5,
            "correlations_focus_high_corr": {"train_data": pd.DataFrame(index=list("ad"), data={"c": [0.88, -1.00]})},
        }
    )
    assert actual.correlations_focus_high_corr.train_data.equals(
        expected.correlations_focus_high_corr.train_data
    )  # noqa
    actual.correlations_focus_high_corr.train_data = "--"
    expected.correlations_focus_high_corr.train_data = "--"
    __compare_outputs(expected, actual)


def test_CorrelationSignificance__can_handle():
    args = AnalysisState()
    assert (
        CorrelationSignificance().can_handle(
            AnalysisState({"correlations": 123, "correlations_method": "something"}), args
        )
        is True
    )
    assert (
        CorrelationSignificance().can_handle(
            AnalysisState({"something": 123, "correlations_method": "something"}), args
        )
        is False
    )
    assert (
        CorrelationSignificance().can_handle(AnalysisState({"correlations": 123, "something": "something"}), args)
        is False
    )


def test_CorrelationSignificance():
    state = AnalysisState(
        {
            "correlations": {
                "train_data": pd.DataFrame(
                    index=list("acd"),
                    data={
                        "a": [1.00, 0.88, -0.88],
                        "c": [0.88, 1.00, -1.00],
                        "d": [-0.88, -1.00, 1.00],
                    },
                )
            },
            "correlations_method": "spearman",
            "correlations_focus_field": "c",
            "correlations_focus_field_threshold": 0.5,
            "correlations_focus_high_corr": {"train_data": pd.DataFrame(index=list("ad"), data={"c": [0.88, -1.00]})},
        }
    )

    result = auto.analyze(
        train_data=__create_test_df(), state=state, return_state=True, anlz_facets=[CorrelationSignificance()]
    )

    assert result.significance_matrix.train_data.shape == (3, 3)


def __create_test_df():
    data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [5, 3, 2, 6, 7, 2],
            "c": [0, 0, 0, 1, 1, 1],
            "d": [1, 1, 1, 0, 0, 0],
            "e": ["a", "a", "a", "b", "b", "b"],
            "f": ["b", "b", "b", "a", "a", "a"],
        }
    )
    return data


def __compare_outputs(expected: AnalysisState, actual: AnalysisState):
    assert actual.correlations.train_data.equals(expected.correlations.train_data)  # noqa
    actual.correlations.train_data = "--"
    expected.correlations.train_data = "--"
    assert actual == expected


def test_FeatureInteraction():
    df = __create_test_df()
    state = auto.analyze(
        train_data=df,
        return_state=True,
        anlz_facets=[
            RawTypesAnalysis(),
            FeatureInteraction(x="a", y="b", hue="c", key="abc"),
            FeatureInteraction(x="a", y="b", hue="d"),
            FeatureInteraction(x="a", y="b", hue="q", key='missing_col'),
        ],
    )

    assert sorted(state.interactions.train_data.keys()) == ["abc", "x:a|y:b|hue:d"]
    assert state.interactions.train_data.abc.data.equals(df[["a", "b", "c"]])
    assert state.interactions.train_data.abc.features == {"hue": "c", "x": "a", "y": "b"}
    assert state.interactions.train_data["x:a|y:b|hue:d"].data.equals(df[["a", "b", "d"]])
    assert state.interactions.train_data["x:a|y:b|hue:d"].features == {"hue": "d", "x": "a", "y": "b"}


def test_FeatureInteraction__can_handle():
    args = AnalysisState()
    assert FeatureInteraction().can_handle(AnalysisState({"raw_type": "something"}), args) is True
    assert FeatureInteraction().can_handle(AnalysisState({"something": 123}), args) is False
