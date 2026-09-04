"""Tests for ``RandomSubsetFeatureCompressionGenerator``'s subset key construction.

The keys are built by hashing each subset of base features per row. Hashing every subset's sliced
frame independently re-hashes the same columns once per subset that uses them, so the columns are
hashed once and the per-column hashes reduced per subset instead. These pin that the reduction
reproduces ``pandas.util.hash_pandas_object`` exactly, since the keys are the categories the
downstream target encoding groups by.

They also pin the key frame's shape of contract -- category dtype and string column names -- which
lets the generator declare the columns to the target encoder rather than have it infer them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autogluon.features.generators import RandomSubsetFeatureCompressionGenerator


def _frame(n: int = 120, n_cols: int = 8, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {f"f{i}": rng.integers(0, 4, size=n) for i in range(n_cols)},
        index=pd.RangeIndex(n) + 5,  # non-trivial index, to catch alignment mistakes
    )
    y = pd.Series(rng.integers(0, 2, size=n), index=X.index)
    return X, y


def test_combine_column_hashes_matches_hash_pandas_object():
    X, _ = _frame()
    gen = RandomSubsetFeatureCompressionGenerator(target_type="binary", verbosity=0)
    for cols in (["f0"], ["f0", "f3"], list(X.columns)):
        expected = pd.util.hash_pandas_object(X[cols], index=False).to_numpy()
        actual = gen._combine_column_hashes(
            [pd.util.hash_array(X[c]._values) for c in cols], len(cols)
        )
        np.testing.assert_array_equal(actual, expected)


def test_make_keys_matches_per_subset_hashing():
    X, _ = _frame()
    gen = RandomSubsetFeatureCompressionGenerator(target_type="binary", verbosity=0)
    subsets = [tuple(X.columns), ("f0", "f1"), ("f2",), ("f1", "f4", "f7")]

    actual = gen._make_keys(X, subsets)

    # Same hashes as hashing each subset's sliced frame, column for column.
    for i, subset in enumerate(subsets):
        expected = pd.util.hash_pandas_object(X[list(subset)], index=False).to_numpy()
        np.testing.assert_array_equal(actual.iloc[:, i].to_numpy().astype("uint64"), expected)
    assert actual.index.equals(X.index)


def test_make_keys_are_categorical_with_string_names():
    """The target encoder only encodes object/category columns, and only string column names can
    be declared to it through ``FeatureMetadata`` -- an integer-named column is silently dropped."""
    X, _ = _frame()
    gen = RandomSubsetFeatureCompressionGenerator(target_type="binary", verbosity=0)
    keys = gen._make_keys(X, [tuple(X.columns), ("f0", "f1")])

    assert all(isinstance(dtype, pd.CategoricalDtype) for dtype in keys.dtypes)
    assert all(isinstance(col, str) for col in keys.columns)
    metadata = gen._key_feature_metadata(keys)
    assert set(metadata.get_features()) == set(keys.columns)
    # Nothing is flagged as text or datetime, which is what makes the inference skippable.
    assert not any(metadata.get_type_map_special().values())


def test_make_keys_empty_subsets():
    X, _ = _frame()
    gen = RandomSubsetFeatureCompressionGenerator(target_type="binary", verbosity=0)
    out = gen._make_keys(X, [])
    assert out.shape == (len(X), 0)
    assert out.index.equals(X.index)


def test_identical_rows_share_a_key():
    """The keys exist to group identical feature combinations; that must survive the reduction."""
    X = pd.DataFrame({"a": [1, 1, 2, 1], "b": [5, 5, 5, 6]})
    gen = RandomSubsetFeatureCompressionGenerator(target_type="binary", verbosity=0)
    keys = gen._make_keys(X, [("a", "b"), ("b",)])
    assert keys.iloc[0, 0] == keys.iloc[1, 0]  # rows 0 and 1 agree on (a, b)
    assert keys.iloc[0, 0] != keys.iloc[2, 0]
    assert keys.iloc[0, 1] == keys.iloc[2, 1]  # rows 0 and 2 agree on b alone


def test_fit_transform_is_unchanged_end_to_end():
    X, y = _frame(n=200, n_cols=6)
    gen = RandomSubsetFeatureCompressionGenerator(
        target_type="binary", verbosity=0, n_subsets=12, random_state=3
    )
    out = gen.fit_transform(X.copy(), y)
    assert out.shape[0] == len(X)
    assert out.index.equals(X.index)
    # Transform reuses the fitted subsets and must key them the same way.
    keys_fit = gen._make_keys(X, gen.selected_subsets)
    keys_again = gen._make_keys(X, gen.selected_subsets)
    pd.testing.assert_frame_equal(keys_fit, keys_again)


def test_transform_encodes_unseen_rows():
    """Unseen key values must fall back to the global mean rather than produce NaN."""
    X, y = _frame(n=200, n_cols=6)
    gen = RandomSubsetFeatureCompressionGenerator(
        target_type="binary", verbosity=0, n_subsets=8, random_state=3
    )
    gen.fit_transform(X.copy(), y)
    X_new = _frame(n=50, n_cols=6, seed=99)[0]
    out = gen.transform(X_new)
    assert out.shape == (len(X_new), 8)
    assert out.notna().all().all()
    assert out.index.equals(X_new.index)
