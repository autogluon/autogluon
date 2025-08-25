import numpy as np
import pandas as pd
import pytest
from numpy.core.fromnumeric import sort

from autogluon.core.utils.feature_selection import *
from autogluon.core.utils.utils import unevaluated_fi_df_template


def evaluated_fi_df_template(features, importance=None, n=None):
    rng = np.random.default_rng(0)
    importance_df = pd.DataFrame({"name": features})
    importance_df["importance"] = rng.standard_normal(len(features)) if importance is None else importance
    importance_df["stddev"] = rng.standard_normal(len(features))
    importance_df["p_value"] = None
    importance_df["n"] = 5 if n is None else n
    importance_df.set_index("name", inplace=True)
    importance_df.index.name = None
    return importance_df


@pytest.fixture
def sample_features():
    return ["a", "b", "c", "d", "e"]


@pytest.fixture
def sample_importance_df_1(sample_features):
    return evaluated_fi_df_template(sample_features, importance=[0.2, 0.2, None, 1.0, None], n=[10, 5, 0, 5, 0])


@pytest.fixture
def sample_importance_df_2(sample_features):
    return evaluated_fi_df_template(sample_features, importance=[-0.1, -0.1, 0.1, None, None], n=[5, 10, 10, 0, 0])


def test_spurious_change_for_ci_test():
    assert True


def test_add_noise_column_df():
    # test noise columns are appended to input dataframe and feature_metadata
    X = pd.DataFrame({"a": [1, 2]})
    args = {"rng": np.random.default_rng(0), "count": 2}
    X_noised, noise_columns = add_noise_column(X, **args)
    expected_features = X.columns.tolist() + noise_columns
    assert expected_features == X_noised.columns.tolist()


def test_merge_importance_dfs_base(sample_features):
    # test the scenario when previous feature importance df is none
    prev_df, curr_df = None, unevaluated_fi_df_template(sample_features)
    assert merge_importance_dfs(prev_df, curr_df, using_prev_fit_fi=set()) is curr_df


def test_merge_importance_dfs_same_model(sample_features, sample_importance_df_1, sample_importance_df_2):
    # test the scenario where previous feature importance df exists and its importance estimates come from the same fitted model
    prev_df, curr_df = sample_importance_df_1, sample_importance_df_2
    result_df = merge_importance_dfs(prev_df, curr_df, using_prev_fit_fi=set())
    assert [score if score == score else None for score in result_df["importance"].tolist()] == [0.0, 0.1, 0.1, 1.0, None]
    assert result_df["n"].tolist() == [15, 15, 10, 5, 0]


def test_merge_importance_dfs_different_model(sample_features, sample_importance_df_1, sample_importance_df_2):
    # test the scenario where previous feature importance df exists and its importance estimates come from a different fitted model
    prev_df, curr_df = sample_importance_df_1, sample_importance_df_2
    using_prev_fit_fi = set(sample_features)
    result_df = merge_importance_dfs(prev_df, curr_df, using_prev_fit_fi=using_prev_fit_fi).sort_index()
    assert len(using_prev_fit_fi) == 2
    assert [score if score == score else None for score in result_df["importance"].tolist()] == [-0.1, -0.1, 0.1, 1.0, None]
    assert result_df["n"].tolist() == [5, 10, 10, 5, 0]


def test_merge_importance_dfs_all(sample_features, sample_importance_df_1, sample_importance_df_2):
    # test the scenario where previous feature importance df exists and its importance estimates come from both same and different fitted models
    prev_df, curr_df = sample_importance_df_1, sample_importance_df_2
    using_prev_fit_fi = set([sample_features[0]])
    result_df = merge_importance_dfs(prev_df, curr_df, using_prev_fit_fi=using_prev_fit_fi).sort_index()
    assert [score if score == score else None for score in result_df["importance"].tolist()] == [-0.1, 0.0, 0.1, 1.0, None]
    assert result_df["n"].tolist() == [5, 15, 10, 5, 0]
    assert using_prev_fit_fi == set()


def test_sort_features_by_priority_base(sample_features):
    # test the ordering of feature importance computation when no prior feature importance computation was done
    sorted_features = sort_features_by_priority(features=sample_features, prev_importance_df=None, using_prev_fit_fi=set())
    assert sorted_features == sample_features


def test_sort_features_by_priority_same_model(sample_features):
    # test the ordering of feature importance computation when prior feature importance computation from the same fitted model was done
    prev_importance_df = evaluated_fi_df_template(sample_features)
    sorted_features = sort_features_by_priority(features=sample_features, prev_importance_df=prev_importance_df, using_prev_fit_fi=set())
    assert sorted_features == prev_importance_df.sort_values("importance").index.tolist()


def test_sort_features_by_priority_different_model(sample_features):
    # test the ordering of feature importance computation when prior feature importance computation from a different fitted model was done
    prev_importance_df = evaluated_fi_df_template(sample_features)
    using_prev_fit_fi = sample_features[-2:]
    sorted_features = sort_features_by_priority(features=sample_features, prev_importance_df=prev_importance_df, using_prev_fit_fi=using_prev_fit_fi)
    sorted_prev_fit_features = prev_importance_df[prev_importance_df.index.isin(using_prev_fit_fi)].sort_values("importance").index.tolist()
    sorted_curr_fit_features = prev_importance_df[~prev_importance_df.index.isin(using_prev_fit_fi)].sort_values("importance").index.tolist()
    expected_features = sorted_prev_fit_features + sorted_curr_fit_features
    assert sorted_features == expected_features


def test_sort_features_by_priority_all(sample_features):
    # test the ordering of feature importance computation when feature impotance computation comes from mix of current and previous fit models,
    # and some feature are unevaluated
    length = len(sample_features)
    using_prev_fit_fi = set(sample_features[: length // 3])
    evaluated_rows, unevaluated_rows = evaluated_fi_df_template(sample_features[: length // 2]), unevaluated_fi_df_template(sample_features[length // 2 :])
    prev_importance_df = pd.concat([evaluated_rows, unevaluated_rows])
    sorted_features = sort_features_by_priority(features=sample_features, prev_importance_df=prev_importance_df, using_prev_fit_fi=using_prev_fit_fi)
    unevaluated_features = unevaluated_rows.index.tolist()
    sorted_prev_fit_features = (
        evaluated_rows[(~evaluated_rows.index.isin(sample_features[length // 2 :])) & (evaluated_rows.index.isin(using_prev_fit_fi))]
        .sort_values("importance")
        .index.tolist()
    )
    sorted_curr_fit_features = (
        evaluated_rows[(~evaluated_rows.index.isin(sample_features[length // 2 :])) & (~evaluated_rows.index.isin(using_prev_fit_fi))]
        .sort_values("importance")
        .index.tolist()
    )
    expected_features = unevaluated_features + sorted_prev_fit_features + sorted_curr_fit_features
    assert sorted_features == expected_features
