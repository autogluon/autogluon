import numpy as np
import pandas as pd
import pytest

from autogluon.common.features.infer_types import get_bool_true_val


@pytest.mark.parametrize(
    "uniques,expected",
    [
        (np.array([0, 1]), 1),
        (np.array([1, 0]), 1),  # reversed order, still sorted
        (np.array(["no", "yes"]), "yes"),
        (pd.Index([0, 1]), 1),
    ],
)
def test_when_sortable_uniques_then_returns_sorted_second_value(uniques, expected):
    assert get_bool_true_val(uniques) == expected


def test_when_numpy_array_with_nan_then_nan_not_chosen_as_true():
    uniques = np.array([1.0, np.nan])
    assert get_bool_true_val(uniques) == 1.0


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1, 0, 1, 0], dtype="category"),
        pd.Series(["yes", "no", "yes", "no"], dtype="category"),
    ],
)
def test_when_pandas_categorical_then_no_attribute_error(series):
    """Regression test: Categorical objects don't have .sort() method."""
    uniques = series.unique()
    result = get_bool_true_val(uniques)
    assert result in list(series.unique())


@pytest.mark.parametrize(
    "series,expected",
    [
        (pd.Series(["yes", None], dtype="string"), "yes"),
        (pd.Series([True, None], dtype="boolean"), True),
        (pd.Series([1, None], dtype="Int64"), 1),
        (pd.Series([1.5, None], dtype="Float64"), 1.5),
    ],
)
def test_when_nullable_dtype_with_pd_na_then_na_not_chosen_as_true(series, expected):
    """Regression test: pd.NA must not be chosen as the True value, and must not raise.

    np.isnan(pd.NA) returns pd.NA instead of raising, so `if is_nan:` raised
    "TypeError: boolean value of NA is ambiguous" for any nullable-dtype column whose two unique
    values include pd.NA (e.g. a `string` column holding one value plus nulls).
    """
    uniques = series.unique()
    assert len(uniques) == 2
    assert get_bool_true_val(uniques) == expected
