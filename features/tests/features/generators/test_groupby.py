"""Regression tests for ``GroupByFeatureGenerator`` and all-NaN grouping categoricals.

A categorical column with no non-null values is selected as a grouping categorical (``nunique()``
ignores NaN, so an all-NaN column has 0 unique values < ``num_as_cat_cardinality_thresh``), but
``groupby(observed=True)`` then yields zero groups. The old code built an empty group table and
``_transform`` raised ``IndexError: cannot do a non-empty take from an empty axes`` when indexing
into it. This happens in practice on datasets with (near-)all-missing categorical columns — e.g.
the ``kick`` dataset, where such a column lands entirely NaN in the 1000-row subsample AutoGluon
uses for memory estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autogluon.features.generators.groupby import GroupByFeatureGenerator


def _frame_with_all_nan_categorical(n: int = 40) -> pd.DataFrame:
    """A frame with one all-NaN object column plus a real categorical and numeric features."""
    return pd.DataFrame(
        {
            "all_nan_cat": pd.Series([np.nan] * n, dtype="object"),
            "real_cat": pd.Series((["a", "b", "c"] * n)[:n], dtype="object"),
            "num1": np.arange(n, dtype=float),
            "num2": np.linspace(0.0, 1.0, n),
        }
    )


def test_all_nan_categorical_does_not_crash():
    """fit_transform must not raise when a grouping categorical is entirely NaN (the kick bug)."""
    X = _frame_with_all_nan_categorical()
    gen = GroupByFeatureGenerator()

    out = gen.fit_transform(X)  # previously raised IndexError on the all-NaN column

    assert len(out) == len(X)
    # The all-NaN column must be dropped from the grouping categoricals (zero-group, no signal).
    assert "all_nan_cat" not in gen.categorical_features
    # No (cat, num) pair may reference the all-NaN column.
    assert all(cat != "all_nan_cat" for cat, _num in gen.pairs_)
    # The real categorical is still used for grouping features.
    assert any(cat == "real_cat" for cat, _num in gen.pairs_)


def test_normal_grouping_still_produces_finite_features():
    """A real categorical + numeric still yields finite group-by features (happy path intact)."""
    n = 60
    X = pd.DataFrame(
        {
            "cat": pd.Series((["x", "y"] * n)[:n], dtype="object"),
            # >= min_num_cardinality_thresh (10) unique values so it is treated as numeric.
            "num": np.arange(n, dtype=float),
        }
    )
    gen = GroupByFeatureGenerator()

    out = gen.fit_transform(X)

    assert len(out) == n
    assert len(gen.pairs_) > 0
    assert list(out.columns) == gen.output_columns_
    assert np.isfinite(out.to_numpy()).all()


def test_only_categorical_is_all_nan_yields_empty_output():
    """When the sole categorical is all-NaN, no grouping is possible -> empty (no crash)."""
    n = 20
    X = pd.DataFrame(
        {
            "all_nan_cat": pd.Series([np.nan] * n, dtype="object"),
            "num1": np.arange(n, dtype=float),
            "num2": np.linspace(0.0, 1.0, n),
        }
    )
    gen = GroupByFeatureGenerator()

    out = gen.fit_transform(X)

    assert len(out) == n
    assert gen.pairs_ == []
    assert list(out.columns) == []


def test_transform_is_robust_to_empty_group_table():
    """Defensive guard: an emptied group table in a fitted pair must not crash ``_transform``.

    Directly reproduces the original crash condition (empty group index + empty value arrays while
    the aggregation dict still has keys) on a fitted generator; the ``len(idx) > 0`` guard routes
    it to the NaN-fill path instead of ``vals.take`` on an empty array.
    """
    n = 30
    X = pd.DataFrame(
        {
            "cat": pd.Series((["a", "b", "c"] * n)[:n], dtype="object"),
            "num": np.arange(n, dtype=float),
        }
    )
    gen = GroupByFeatureGenerator()
    gen.fit_transform(X)
    assert gen.pairs_, "expected at least one fitted (cat, num) pair"

    pair = gen.pairs_[0]
    gen.group_index_[pair] = pd.Index([])
    gen.group_values_[pair] = {agg: np.array([], dtype=float) for agg in gen.group_values_[pair]}

    out = gen._transform(X)  # previously raised IndexError; now routes to the fill path
    assert len(out) == n
    assert list(out.columns) == gen.output_columns_
