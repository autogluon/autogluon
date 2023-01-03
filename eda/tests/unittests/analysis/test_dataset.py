import numpy as np
import pandas as pd
import pytest

import autogluon.eda.auto as auto
from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT, R_OBJECT
from autogluon.core.constants import MULTICLASS, REGRESSION
from autogluon.eda import AnalysisState
from autogluon.eda.analysis import Namespace, Sampler, TrainValidationSplit
from autogluon.eda.analysis.base import BaseAnalysis
from autogluon.eda.analysis.dataset import DatasetSummary, RawTypesAnalysis, SpecialTypesAnalysis, VariableTypeAnalysis


class SomeAnalysis(BaseAnalysis):
    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.args = args.copy()


def test_Sampler():
    df_train = pd.DataFrame(np.random.randint(0, 100, size=(10, 4)), columns=list("ABCD"))
    df_test = pd.DataFrame(np.random.randint(0, 100, size=(20, 4)), columns=list("EFGH"))
    assert df_train.shape == (10, 4)
    assert df_test.shape == (20, 4)

    analysis = BaseAnalysis(
        train_data=df_train,
        test_data=df_test,
        children=[
            Namespace(namespace="ns_sampler", children=[Sampler(sample=5, children=[SomeAnalysis()])]),
            Namespace(namespace="ns_sampler_none", children=[Sampler(sample=None, children=[SomeAnalysis()])]),
            Namespace(namespace="ns_no_sampler", children=[SomeAnalysis()]),
        ],
    )

    state = analysis.fit()
    assert state.ns_sampler.args.train_data.shape == (5, 4)
    assert state.ns_sampler.args.test_data.shape == (5, 4)
    assert state.ns_sampler.sample_size == 5

    assert state.ns_sampler_none.args.train_data.shape == (10, 4)
    assert state.ns_sampler_none.args.test_data.shape == (20, 4)
    assert state.ns_sampler_none.sample_size is None

    assert state.ns_no_sampler.args.train_data.shape == (10, 4)
    assert state.ns_no_sampler.args.test_data.shape == (20, 4)
    assert state.ns_no_sampler.sample_size is None


def test_Sampler_frac():
    df_train = pd.DataFrame(np.random.randint(0, 100, size=(10, 4)), columns=list("ABCD"))
    df_test = pd.DataFrame(np.random.randint(0, 100, size=(20, 4)), columns=list("EFGH"))
    assert df_train.shape == (10, 4)
    assert df_test.shape == (20, 4)

    analysis = BaseAnalysis(
        train_data=df_train, test_data=df_test, children=[Sampler(sample=0.5, children=[SomeAnalysis()])]
    )

    state = analysis.fit()
    assert state.sample_size == 0.5
    assert state.args.train_data.shape == (5, 4)
    assert state.args.test_data.shape == (10, 4)


def test_TrainValidationSplit():
    df_train, _ = __get_dataset_summary_test_datasets()
    analysis = BaseAnalysis(
        train_data=df_train,
        label="D",
        children=[
            Namespace(
                namespace="ns_val_split_specified",
                children=[TrainValidationSplit(val_size=0.4, children=[SomeAnalysis()])],
            ),
            Namespace(
                namespace="ns_val_split_default",
                children=[TrainValidationSplit(problem_type=REGRESSION, children=[SomeAnalysis()])],
            ),
            Namespace(namespace="ns_no_split", children=[SomeAnalysis()]),
        ],
    )

    state = analysis.fit()
    assert state.ns_val_split_specified.args.train_data.shape == (60, 7)
    assert state.ns_val_split_specified.args.val_data.shape == (40, 7)
    assert state.ns_val_split_specified.problem_type == MULTICLASS

    assert state.ns_val_split_default.args.train_data.shape == (70, 7)
    assert state.ns_val_split_default.args.val_data.shape == (30, 7)
    assert state.ns_val_split_default.problem_type == REGRESSION

    assert state.ns_no_split.args.train_data.shape == (100, 7)
    assert state.ns_no_split.args.val_data is None
    assert state.ns_no_split.problem_type is None


def __get_dataset_summary_test_datasets():
    cols = list("ABCDEFG")
    df_train = pd.DataFrame((np.arange(100))[:, None].repeat([len(cols)], axis=1), columns=cols)
    df_test = pd.DataFrame((np.arange(200))[:, None].repeat([len(cols)], axis=1), columns=cols)
    str_mappings = {i: "Lorem ipsum " * (i + 1) for i in range(10)}
    for df in [df_train, df_test]:
        df["A"] = (df["A"] % 4).map({0: "a", 1: "b", 2: "c", 3: "d"})
        df["B"] = (df["B"] % 2).map({0: False, 1: True})
        df["C"] = df["C"] % 2
        df["D"] = df["D"] % 3
        df["E"] = (df["E"] % len(str_mappings.keys())).map(str_mappings)
        df["F"] = df["F"] * 0.1
    return df_train, df_test


def test_DatasetSummary():
    df_train, df_test = __get_dataset_summary_test_datasets()
    state = auto.analyze(train_data=df_train, test_data=df_test, return_state=True, anlz_facets=[DatasetSummary()])
    expected_cols = ["25%", "50%", "75%", "count", "dtypes", "freq", "max", "mean", "min", "std", "top", "unique"]
    expected_fields = list("ABCDEFG")
    assert sorted(state.dataset_stats.train_data.keys()) == expected_cols
    assert sorted(pd.DataFrame(state.dataset_stats.train_data).index) == expected_fields
    assert sorted(state.dataset_stats.test_data.keys()) == expected_cols
    assert sorted(pd.DataFrame(state.dataset_stats.test_data).index) == expected_fields


def test_RawTypesAnalysis():
    df_train, df_test = __get_dataset_summary_test_datasets()
    state = auto.analyze(train_data=df_train, test_data=df_test, return_state=True, anlz_facets=[RawTypesAnalysis()])
    expected_types = {"A": "object", "B": "bool", "C": "int", "D": "int", "E": "object", "F": "float", "G": "int"}
    assert state.raw_type.train_data == expected_types
    assert state.raw_type.test_data == expected_types


def test_VariableTypeAnalysis_can_handle():
    df_train, _ = __get_dataset_summary_test_datasets()
    assert VariableTypeAnalysis().can_handle(state=AnalysisState(), args=AnalysisState()) is False
    assert (
        VariableTypeAnalysis().can_handle(state=AnalysisState({"raw_type": "some_state"}), args=AnalysisState())
        is True
    )


def test_VariableTypeAnalysis__map_raw_type_to_feature_type__special_cases():
    df = pd.DataFrame({"a": np.arange(20) % 10})
    f = VariableTypeAnalysis().map_raw_type_to_feature_type
    assert f(col=None, raw_type="", df=df) is None
    assert f(col="a", df=df, raw_type="", numeric_as_categorical_threshold=3) is None
    assert f(col="a", df=df, raw_type="", numeric_as_categorical_threshold=100) == "category"


@pytest.mark.parametrize(
    "test_type,expected",
    [(R_INT, "numeric"), (R_FLOAT, "numeric"), (R_OBJECT, "category"), (R_CATEGORY, "category"), (R_BOOL, "category")],
)
def test_VariableTypeAnalysis__map_raw_type_to_feature_type__regular_cases(test_type, expected):
    df = pd.DataFrame({"a": np.arange(20) % 10})
    f = VariableTypeAnalysis().map_raw_type_to_feature_type
    args = dict(col="a", df=df, numeric_as_categorical_threshold=3)
    assert f(**{**args, **dict(raw_type=test_type)}) == expected


def test_VariableTypeAnalysis():
    df_train, df_test = __get_dataset_summary_test_datasets()
    state = auto.analyze(
        train_data=df_train,
        test_data=df_test,
        return_state=True,
        anlz_facets=[RawTypesAnalysis(), VariableTypeAnalysis()],
    )

    expected_types = {
        "A": "category",
        "B": "category",
        "C": "category",
        "D": "category",
        "E": "category",
        "F": "numeric",
        "G": "numeric",
    }
    assert state.variable_type.train_data == expected_types
    assert state.variable_type.test_data == expected_types


def test_SpecialTypesAnalysis():
    df_train, df_test = __get_dataset_summary_test_datasets()
    state = auto.analyze(
        train_data=df_train, test_data=df_test, return_state=True, anlz_facets=[SpecialTypesAnalysis()]
    )

    expected_types = {"E": "text"}
    assert state.special_types.train_data == expected_types
    assert state.special_types.test_data == expected_types
