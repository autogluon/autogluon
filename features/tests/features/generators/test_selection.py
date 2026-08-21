"""Tests for ``SpearmanFeatureSelector``.

Spearman correlation is undefined for a constant column; scipy returns NaN and emits a
``ConstantInputWarning``. The selector already drops the resulting NaN correlations, so a constant
column is never selected -- this just verifies it happens without the spurious warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

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
