from autogluon.core.utils.feature_selection import *
from autogluon.core.utils.utils import unevaluated_fi_df_template
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd


def evaluated_fi_df_template(features, importance=None, n=None):
    rng = np.random.default_rng(0)
    importance_df = pd.DataFrame({'name': features})
    importance_df['importance'] = rng.standard_normal(len(features)) if importance is None else importance
    importance_df['stddev'] = rng.standard_normal(len(features))
    importance_df['p_value'] = None
    importance_df['n'] = 5 if n is None else n
    importance_df.set_index('name', inplace=True)
    importance_df.index.name = None
    return importance_df


def test_add_noise_column():
    X = None
    args = {'prefix': 'noise_prefix', 'rng': np.random.default_rng(0), 'count': 2}
    assert add_noise_column(X, **args) is None
    X = pd.DataFrame({'a': [1, 2]})
    feature_metadata = FeatureMetadata.from_df(X)
    X_noised = add_noise_column(X, **args, feature_metadata=feature_metadata)
    expected_features = X.columns.tolist() + ['noise_prefix_1', 'noise_prefix_2']
    assert expected_features == X_noised.columns.tolist()
    assert expected_features == feature_metadata.get_features()


def test_merge_importance_dfs():
    # test first time
    features = ['a', 'b', 'c', 'd', 'e']
    prev_fit_estimates = set()
    df1, df2 = None, unevaluated_fi_df_template(features)
    assert merge_importance_dfs(df1, df2, prev_fit_estimates) is df2
    # test subsequent time where no estimates are from previous runs
    df1 = evaluated_fi_df_template(features, importance=[0.2, 0.2, None, 1., None], n=[10, 5, 0, 5, 0])
    df2 = evaluated_fi_df_template(features, importance=[-0.1, -0.1, 0.1, None, None], n=[5, 10, 10, 0, 0])
    result_df = merge_importance_dfs(df1, df2, prev_fit_estimates)
    assert [score if score == score else None for score in result_df['importance'].tolist()] == [0., 0.1, 0.1, 1., None]
    assert result_df['n'].tolist() == [15, 15, 10, 5, 0]
    # test subsequent time where some estimates are from previous runs
    prev_fit_estimates = set(['a'])
    result_df = merge_importance_dfs(df1, df2, prev_fit_estimates)
    assert [score if score == score else None for score in result_df['importance'].tolist()] == [-0.1, 0., 0.1, 1., None]
    assert result_df['n'].tolist() == [5, 15, 10, 5, 0]
    assert prev_fit_estimates == set()


def test_sort_features_by_priority():
    # test first time with no prioritized features
    features = ['a', 'b', 'c', 'd']
    prioritized = set()
    prev_importance_df = unevaluated_fi_df_template(features)
    prev_fit_estimates = set()
    sorted_features = sort_features_by_priority(features, prioritized, prev_importance_df, prev_fit_estimates)
    assert sorted_features == ['a', 'b', 'c', 'd']
    # test first time with prioritized features
    prioritized = set(['d'])
    sorted_features = sort_features_by_priority(features, prioritized, prev_importance_df, prev_fit_estimates)
    assert sorted_features == ['d', 'a', 'b', 'c']
    # test subsequent time with all features evaluated
    prioritized = set()
    prev_importance_df = evaluated_fi_df_template(features)
    sorted_features = sort_features_by_priority(features, prioritized, prev_importance_df, prev_fit_estimates)
    assert sorted_features == prev_importance_df.sort_values('importance').index.tolist()
    # test subsequent time with prioritized features and all features evaluated
    prioritized = set(['d'])
    prev_importance_df = evaluated_fi_df_template(features)
    sorted_features = sort_features_by_priority(features, prioritized, prev_importance_df, prev_fit_estimates)
    assert sorted_features == list(prioritized) + prev_importance_df.drop(prioritized).sort_values('importance').index.tolist()
    # test subsequent time with no features evaluated
    prioritized = set()
    prev_importance_df = unevaluated_fi_df_template(features)
    sorted_features = sort_features_by_priority(features, prioritized, prev_importance_df, prev_fit_estimates)
    assert sorted_features == features
    # test subsequent time with some features evaluated
    unevaluated_df, evaluated_df = unevaluated_fi_df_template(['a', 'b']), evaluated_fi_df_template(['c', 'd'])
    prev_importance_df = pd.concat([unevaluated_df, evaluated_df])
    sorted_features = sort_features_by_priority(features, prioritized, prev_importance_df, prev_fit_estimates)
    assert sorted_features == ['a', 'b'] + evaluated_df.sort_values('importance').index.tolist()
    # test subsequent time with some features evaluated, some of which are from previous runs
    prev_fit_estimates = set(['d'])
    unevaluated_df, evaluated_df = unevaluated_fi_df_template(['a', 'b']), evaluated_fi_df_template(['c', 'd'])
    prev_importance_df = pd.concat([unevaluated_df, evaluated_df])
    sorted_features = sort_features_by_priority(features, prioritized, prev_importance_df, prev_fit_estimates)
    assert sorted_features == ['a', 'b', 'd', 'c']
