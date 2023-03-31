import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import autogluon.eda.auto as auto
from autogluon.eda import AnalysisState
from autogluon.eda.analysis import (
    ApplyFeatureGenerator,
    Correlation,
    CorrelationSignificance,
    DistributionFit,
    FeatureInteraction,
)
from autogluon.eda.analysis.dataset import RawTypesAnalysis
from autogluon.eda.analysis.interaction import FeatureDistanceAnalysis

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))

SAMPLE_SIZE = 200


def load_adult_data():
    train_data = os.path.join(RESOURCE_PATH, "adult", "train_data.csv")
    test_data = os.path.join(RESOURCE_PATH, "adult", "test_data.csv")
    train = pd.read_csv(train_data).sample(SAMPLE_SIZE, random_state=0)
    test = pd.read_csv(test_data).sample(SAMPLE_SIZE, random_state=0)
    data = (train, test)
    return data


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
            "sample_size": 10000,
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
            "sample_size": 10000,
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
            "sample_size": 10000,
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
            "sample_size": 10000,
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
            "sample_size": 10000,
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
            FeatureInteraction(x="a", y="b", hue="q", key="missing_col"),
        ],
    )

    assert sorted(state.interactions.train_data.keys()) == ["abc", "x:a|y:b|hue:d"]
    assert state.interactions.train_data.abc.data.equals(df[["a", "b", "c"]])
    assert state.interactions.train_data.abc.features == {"hue": "c", "x": "a", "y": "b"}
    assert state.interactions.train_data["x:a|y:b|hue:d"].data.equals(df[["a", "b", "d"]])
    assert state.interactions.train_data["x:a|y:b|hue:d"].features == {"hue": "d", "x": "a", "y": "b"}


def test_FeatureInteraction__key_provided():
    assert (
        FeatureInteraction()._generate_key_if_not_provided(key="abc", cols={"x": "a", "y": "b", "hue": "c"}) == "abc"
    )


@pytest.mark.parametrize(
    "cols, expected",
    [
        ({"x": "a"}, "x:a"),
        ({"y": "b"}, "y:b"),
        ({"hue": "c"}, "hue:c"),
        ({"x": "a", "y": "b"}, "x:a|y:b"),
        ({"x": "a", "hue": "c"}, "x:a|hue:c"),
        ({"y": "b", "hue": "c"}, "y:b|hue:c"),
        ({"x": "a", "y": "b", "hue": "c"}, "x:a|y:b|hue:c"),
        ({"x": "a", "y": None, "hue": None}, "x:a"),
        ({"x": None, "y": "b", "hue": None}, "y:b"),
    ],
)
def test_FeatureInteraction__generate_key_if_not_provided(cols, expected):
    assert FeatureInteraction()._generate_key_if_not_provided(key=None, cols=cols) == expected


def test_FeatureInteraction__can_handle():
    args = AnalysisState()
    assert FeatureInteraction().can_handle(AnalysisState({"raw_type": "something"}), args) is True
    assert FeatureInteraction().can_handle(AnalysisState({"something": 123}), args) is False


def test_DistributionFit__happy_path():
    df = __create_test_df()
    df["e"] = [1000, 100, 10, 1, 0.1, 0.001]
    state = auto.analyze(
        train_data=df,
        return_state=True,
        anlz_facets=[
            DistributionFit(
                columns=["a", "unknown"],
                keep_top_n=3,
                distributions_to_fit=["dweibull", "dgamma", "logistic", "lognorm"],
            )
        ],
    )
    assert state.distributions_fit.train_data.a == {
        "dgamma": {
            "param": (2.7071122240167984, 3.4999828973430622, 0.5541010748597517),
            "pvalue": 0.9997989026208239,
            "shapes": ["a", "loc", "scale"],
            "statistic": 0.12375823666401586,
        },
        "dweibull": {
            "param": (1.9212313611673846, 3.5000360875942134, 1.6939405342565321),
            "pvalue": 0.9998695812922455,
            "shapes": ["c", "loc", "scale"],
            "statistic": 0.12094344045455918,
        },
        "logistic": {
            "param": (3.5, 1.0421523470240495),
            "pvalue": 0.9981811820582718,
            "shapes": ["loc", "scale"],
            "statistic": 0.14168404087671216,
        },
    }

    a = DistributionFit(
        columns=["e"],
        keep_top_n=3,
        pvalue_min=0.9,
        distributions_to_fit=["dweibull", "dgamma", "logistic", "lognorm"],
    )

    state = auto.analyze(
        train_data=df,
        return_state=True,
        anlz_facets=[a],
    )
    assert state.distributions_fit.train_data.e is None


def test_DistributionFit__constructor_defaults():
    a = DistributionFit("a")
    assert a.distributions_to_fit == a.AVAILABLE_DISTRIBUTIONS
    assert a.keep_top_n == 3
    assert a.columns == ["a"]

    a = DistributionFit(["a"], distributions_to_fit="lognorm")
    assert a.columns == ["a"]
    assert a.distributions_to_fit == ["lognorm"]

    a = DistributionFit(["a"], distributions_to_fit=["lognorm", "gamma"])
    assert a.columns == ["a"]
    assert a.distributions_to_fit == ["lognorm", "gamma"]


def test_DistributionFit__constructor_unsupported_dist():
    with pytest.raises(ValueError) as exc_info:
        DistributionFit("a", distributions_to_fit="unknown")
    assert exc_info.value.args[0].startswith(
        "The following distributions are not supported: ['unknown']. Supported distributions are"
    )


def test_DistributionFit__non_numeric_col():
    df = __create_test_df()
    a = DistributionFit(
        columns=["a", "f"],
        distributions_to_fit=["dweibull", "dgamma", "logistic", "lognorm"],
    )
    a.logger.warning = MagicMock()
    state = auto.analyze(
        train_data=df,
        return_state=True,
        anlz_facets=[a],
    )
    assert len(state.distributions_fit.train_data.a) == 4
    assert state.distributions_fit.train_data.f is None
    a.logger.warning.assert_called_with("f: distribution cannot be fit; only numeric columns are supported")


def test_FeatureDistanceAnalysis__happy_path():
    df, _ = load_adult_data()
    label = "class"

    state = auto.analyze(
        train_data=df,
        label=label,
        return_state=True,
        anlz_facets=[
            ApplyFeatureGenerator(
                category_to_numbers=True, children=[FeatureDistanceAnalysis(near_duplicates_threshold=0.85)]
            ),
        ],
    )

    assert np.allclose(
        state.feature_distance.linkage,
        np.array(
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
    )

    state.feature_distance.pop("linkage")

    assert state == {
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
            "near_duplicates": [
                {"distance": 0.6113, "nodes": ["marital-status", "relationship"]},
                {"distance": 0.7905, "nodes": ["occupation", "sex", "workclass"]},
                {"distance": 0.8097, "nodes": ["education-num", "hours-per-week"]},
                {"distance": 0.8306, "nodes": ["age", "capital-gain"]},
            ],
            "near_duplicates_threshold": 0.85,
        },
        "sample_size": 10000,
    }
