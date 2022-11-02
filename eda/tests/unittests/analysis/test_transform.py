import numpy as np
import pandas as pd

from autogluon.eda import AnalysisState
from autogluon.eda.analysis import Namespace
from autogluon.eda.analysis.base import BaseAnalysis
from autogluon.eda.analysis.transform import ApplyFeatureGenerator


class SomeAnalysis(BaseAnalysis):
    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.args = args.copy()


def test_ApplyFeatureGenerator():
    df_train = pd.DataFrame((np.arange(10))[:, None].repeat([4], axis=1), columns=list("ABCD"))
    df_test = pd.DataFrame((np.arange(20))[:, None].repeat([4], axis=1), columns=list("ABCD"))
    for df in [df_train, df_test]:
        df["A"] = (df["A"] % 4).map({0: "a", 1: "b", 2: "c", 3: "d"})
        df["B"] = df["B"] % 5
        df["C"] = (df["C"] % 3).map({0: "a", 1: "b", 2: "c"})
        df["D"] = df["D"] % 5
    assert df_train.shape == (10, 4)
    assert df_test.shape == (20, 4)

    analysis = BaseAnalysis(
        train_data=df_train,
        test_data=df_test,
        label="D",
        children=[
            Namespace(
                namespace="feature_generator_numbers",
                children=[ApplyFeatureGenerator(category_to_numbers=True, children=[SomeAnalysis()])],
            ),
            Namespace(
                namespace="feature_generator_default", children=[ApplyFeatureGenerator(children=[SomeAnalysis()])]
            ),
            Namespace(namespace="no_wrapper", children=[SomeAnalysis()]),
        ],
    )

    state = analysis.fit()
    assert list(df_train.dtypes.apply(str).to_numpy()) == ["object", "int64", "object", "int64"]
    assert list(df_test.dtypes.apply(str).to_numpy()) == ["object", "int64", "object", "int64"]

    assert list(state.no_wrapper.args.train_data[df_train.columns].dtypes.apply(str).to_numpy()) == [
        "object",
        "int64",
        "object",
        "int64",
    ]
    assert list(state.no_wrapper.args.test_data[df_test.columns].dtypes.apply(str).to_numpy()) == [
        "object",
        "int64",
        "object",
        "int64",
    ]

    assert list(state.feature_generator_default.args.train_data[df_train.columns].dtypes.apply(str).to_numpy()) == [
        "category",
        "int64",
        "category",
        "int64",
    ]
    assert list(state.feature_generator_default.args.test_data[df_test.columns].dtypes.apply(str).to_numpy()) == [
        "category",
        "int64",
        "category",
        "int64",
    ]

    assert list(state.feature_generator_numbers.args.train_data[df_train.columns].dtypes.apply(str).to_numpy()) == [
        "int8",
        "int64",
        "int8",
        "int64",
    ]
    assert list(state.feature_generator_numbers.args.test_data[df_test.columns].dtypes.apply(str).to_numpy()) == [
        "int8",
        "int64",
        "int8",
        "int64",
    ]
