"""Tests for ``SpearmanFeatureSelector``.

Spearman correlation is undefined for a constant column; scipy returns NaN and emits a
``ConstantInputWarning``. The selector already drops the resulting NaN correlations, so a constant
column is never selected -- this just verifies it happens without the spurious warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from autogluon.features.generators.selection import SpearmanFeatureSelector


def test_spearman_selector_excludes_constant_columns_without_warning():
    n = 50
    X = pd.DataFrame(
        {
            "const": np.ones(n),  # zero-variance -> undefined Spearman correlation
            "all_nan": pd.Series([np.nan] * n, dtype=float),
            "varying": np.arange(n, dtype=float),
        }
    )
    y = pd.Series(np.arange(n, dtype=float))
    gen = SpearmanFeatureSelector(max_features=10)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gen._fit(X, y)
    messages = [str(w.message).lower() for w in caught]
    assert not any("constant" in m or "not defined" in m for m in messages), messages

    # Constant / all-NaN columns are not selected; the informative column is.
    assert "const" not in gen.selected_features_
    assert "all_nan" not in gen.selected_features_
    assert "varying" in gen.selected_features_


def test_spearman_selector_happy_path_still_selects_correlated_features():
    n = 40
    base = np.arange(n, dtype=float)
    X = pd.DataFrame({"pos": base, "neg": -base, "noise": (base * 0) + np.tile([0.0, 1.0], n // 2)})
    y = pd.Series(base)
    gen = SpearmanFeatureSelector(max_features=2)

    gen._fit(X, y)

    # Perfectly (anti-)correlated features rank first.
    assert set(gen.selected_features_) == {"pos", "neg"}


def _reference_abs_spearman(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """The straightforward pandas computation the bulk implementation replaces."""
    return X.loc[:, X.nunique(dropna=True) > 1].corrwith(y, method="spearman").abs()


def _frame_with_nulls(n: int = 200, n_cols: int = 40, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.normal(size=n))
    data = {}
    for i in range(n_cols):
        col = y * rng.normal() + rng.normal(size=n) * rng.uniform(0.5, 5)
        # Varied null patterns, including some shared between columns and some columns with none.
        if i % 3:
            col = col.mask(rng.random(n) < 0.1 * (1 + i % 4))
        data[f"f{i}"] = col
    return pd.DataFrame(data), y


def test_spearman_matches_pandas_corrwith_with_nulls():
    X, y = _frame_with_nulls()
    expected = _reference_abs_spearman(X, y)
    actual = SpearmanFeatureSelector._abs_spearman(X, y)
    pd.testing.assert_series_equal(actual, expected, check_names=False, atol=1e-12, rtol=0)


def test_spearman_selection_matches_pandas_corrwith():
    X, y = _frame_with_nulls(n_cols=40)
    expected = _reference_abs_spearman(X, y).sort_values(ascending=False).dropna()
    for max_features in (5, 20):
        gen = SpearmanFeatureSelector(max_features=max_features, preserve_order=False)
        gen._fit(X, y)
        assert gen.selected_features_ == expected.head(max_features).index.tolist()


def test_preserve_order_keeps_input_order_and_same_set():
    X, y = _frame_with_nulls(n_cols=30)
    sorted_gen = SpearmanFeatureSelector(max_features=10, preserve_order=False)
    ordered_gen = SpearmanFeatureSelector(max_features=10, preserve_order=True)
    sorted_gen._fit(X, y)
    ordered_gen._fit(X, y)

    assert set(ordered_gen.selected_features_) == set(sorted_gen.selected_features_)
    # Input order, not correlation order.
    assert ordered_gen.selected_features_ == [c for c in X.columns if c in set(sorted_gen.selected_features_)]
    assert ordered_gen.selected_features_ != sorted_gen.selected_features_


def test_preserve_order_is_the_default():
    X, y = _frame_with_nulls(n_cols=30)
    gen = SpearmanFeatureSelector(max_features=10)
    gen._fit(X, y)
    assert gen.selected_features_ == [c for c in X.columns if c in set(gen.selected_features_)]


def test_skips_correlation_when_nothing_can_be_dropped(monkeypatch):
    """With no threshold and fewer columns than the cap, the correlation is never computed."""
    X, y = _frame_with_nulls(n_cols=10)

    def explode(*args, **kwargs):
        raise AssertionError("correlation should not be computed when nothing can be dropped")

    monkeypatch.setattr(SpearmanFeatureSelector, "_abs_spearman", staticmethod(explode))
    gen = SpearmanFeatureSelector(max_features=100)
    gen._fit(X, y)
    assert gen.selected_features_ == list(X.columns)

    # The skip must not apply when a threshold can still drop columns, or when sorting is asked for.
    for kwargs in ({"threshold": 0.1}, {"preserve_order": False}):
        gen = SpearmanFeatureSelector(max_features=100, **kwargs)
        with pytest.raises(AssertionError, match="should not be computed"):
            gen._fit(X, y)


def test_skip_still_drops_constant_columns():
    n = 50
    X = pd.DataFrame(
        {
            "const": np.ones(n),
            "all_nan": pd.Series([np.nan] * n, dtype=float),
            "varying": np.arange(n, dtype=float),
        }
    )
    y = pd.Series(np.arange(n, dtype=float))
    gen = SpearmanFeatureSelector(max_features=100)  # cap exceeds the column count -> skip path
    gen._fit(X, y)
    assert gen.selected_features_ == ["varying"]


def test_non_numeric_columns_delegate_to_pandas():
    """Non-numeric input takes the pandas path, behaving exactly as it did before.

    ``corrwith(method="spearman")`` cannot rank a categorical column and raises; the bulk
    implementation must not paper over that with a different error (or a silent wrong answer),
    so the boundary is pinned here.
    """
    n = 30
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=n))
    X = pd.DataFrame(
        {"num": y * 2 + rng.normal(size=n), "cat": pd.Categorical(["a", "b"] * (n // 2))}
    )
    with pytest.raises(ValueError, match="could not convert string to float"):
        X.loc[:, X.nunique(dropna=True) > 1].corrwith(y, method="spearman")
    with pytest.raises(ValueError, match="could not convert string to float"):
        SpearmanFeatureSelector._abs_spearman(X, y)

    # The constant filter does support non-numeric input, and keeps both columns here.
    assert SpearmanFeatureSelector._is_non_constant(X).tolist() == [True, True]
