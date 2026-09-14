import logging
import math
import sys
from functools import wraps

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from ..features.infer_types import get_type_map_raw
from ..features.types import R_CATEGORY, R_FLOAT, R_INT

logger = logging.getLogger(__name__)


def _suspend_logging_for_package(package_name):
    def _suspend_logging(func):
        """hides any logs within the called func that are below warnings"""

        @wraps(func)
        def inner(*args, **kwargs):
            package_logger = logging.getLogger(package_name)
            previous_log_level = package_logger.getEffectiveLevel()
            try:
                package_logger.setLevel(max(30, previous_log_level))
                return func(*args, **kwargs)
            finally:
                package_logger.setLevel(previous_log_level)

        return inner

    return _suspend_logging


def _object_column_mem_usage(values, num_rows: int, sample_ratio: float) -> int:
    """Memory of one object column, counting each distinct Python object once.

    ``memory_usage(deep=True)`` adds ``sys.getsizeof(obj)`` for every cell, so a value shared by
    many cells is charged once per cell instead of once. Sharing is the common case rather than a
    corner case: pandas' C ``read_csv`` parser reuses a single ``str`` object for every repeat of a
    value, so a low-cardinality string column read from disk is fully shared. The overstatement is
    ``sys.getsizeof(value) / 8`` per column, which is a few-fold for short strings and unbounded
    for long ones.

    Shaped like the category branch above: fixed-width storage for every row, plus the distinct
    values, extrapolated from the sample.
    """
    unique_objects = {id(obj): obj for obj in values}
    unique_bytes = sum(map(sys.getsizeof, unique_objects.values()))
    return int(values.itemsize * num_rows + unique_bytes / sample_ratio)


_OBJECT_DTYPE = np.dtype(object)


def get_two_valued_columns(df: DataFrame, columns: list | None = None) -> dict:
    """The columns of `df` (or of `columns`) holding exactly two distinct values, mapped to those values.

    The values come in order of first appearance and with the column's dtype, as ``df[column].unique()``
    returns them; missing counts as a value, as with ``unique``. numpy numeric and bool columns are tested
    block-wise per dtype, categorical columns block-wise through their codes, the other columns through
    ``unique``.
    """
    if columns is None:
        columns = list(df.columns)
    if len(df) < 2:
        return {}
    if df.columns.has_duplicates:
        return _two_valued_through_unique(df, columns)
    dtypes = dict(zip(df.columns, df.dtypes))
    by_dtype: dict = {}
    categorical = []
    other = []
    for column in columns:
        dtype = dtypes[column]
        if isinstance(dtype, np.dtype) and dtype.kind in "biuf":
            by_dtype.setdefault(dtype, []).append(column)
        elif isinstance(dtype, pd.CategoricalDtype):
            categorical.append(column)
        else:
            other.append(column)
    two_valued: dict = {}
    for dtype, dtype_columns in by_dtype.items():
        values = df[dtype_columns].to_numpy()
        is_first = _equal_or_both_missing(values, values[0], dtype)
        # the first row that differs from the first value, per column (0 where none does)
        second_position = np.argmax(~is_first, axis=0)
        second = values[second_position, np.arange(values.shape[1])]
        is_second = _equal_or_both_missing(values, second, dtype)
        has_second = ~is_first.all(axis=0)
        exactly_two = has_second & (is_first | is_second).all(axis=0)
        for column, first, second_value, is_two in zip(dtype_columns, values[0], second, exactly_two):
            if is_two:
                two_valued[column] = np.array([first, second_value], dtype=dtype)
    two_valued.update(_two_valued_categorical(df, categorical))
    two_valued.update(_two_valued_through_unique(df, other))
    return {column: two_valued[column] for column in columns if column in two_valued}


def _equal_or_both_missing(values: np.ndarray, reference: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Element-wise equality against a per-column reference, with NaN equal to NaN on float blocks."""
    equal = values == reference
    if dtype.kind == "f":
        equal |= np.isnan(values) & np.isnan(reference)
    return equal


def _two_valued_through_unique(df: DataFrame, columns: list) -> dict:
    two_valued = {}
    for column in columns:
        uniques = df[column].unique()
        if len(uniques) == 2:
            two_valued[column] = uniques
    return two_valued


def _categorical_codes(df: DataFrame, columns: list) -> np.ndarray:
    """The codes of categorical `columns` as one (rows x columns) integer array; missing is -1."""
    return np.column_stack([df[column].array.codes for column in columns])


def _first_two_codes(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per column of `codes`: whether every code equals the first row's, whether exactly two distinct codes
    occur, and the first code that differs from the first row's (the first row's code where none does).

    Missing (code -1) counts as a value. Comparing against the first and second value is linear in the rows,
    where sorting the codes to count distinct values is not.
    """
    is_first = codes == codes[0]
    second_position = np.argmax(~is_first, axis=0)
    second = codes[second_position, np.arange(codes.shape[1])]
    constant = is_first.all(axis=0)
    exactly_two = ~constant & (is_first | (codes == second)).all(axis=0)
    return constant, exactly_two, second


def _two_valued_categorical(df: DataFrame, columns: list) -> dict:
    """`_two_valued_through_unique` for categorical columns, from their codes in one array.

    The values come back as `Series.unique()` returns them for a categorical: a Categorical of the
    column's dtype holding the two values in order of first appearance, missing included.
    """
    if not columns:
        return {}
    codes = _categorical_codes(df, columns)
    _, exactly_two, second = _first_two_codes(codes)
    two_valued = {}
    for position in np.flatnonzero(exactly_two):
        column = columns[position]
        two_valued[column] = pd.Categorical.from_codes([codes[0, position], second[position]], dtype=df[column].dtype)
    return two_valued


def get_constant_columns(df: DataFrame, columns: list | None = None) -> list:
    """The columns of `df` (or of `columns`) holding a single distinct value: ``len(df[column].unique()) == 1``.

    All-missing counts as one value, ``-0.0`` equals ``0.0``, and an empty frame has no constant column, as
    with ``unique``. numpy numeric and bool columns are tested block-wise per dtype, categorical columns through
    their codes; every other column is asked through ``unique`` as before.
    """
    if columns is None:
        columns = list(df.columns)
    num_rows = len(df)
    if num_rows == 0:
        return []
    if df.columns.has_duplicates:
        return [column for column in columns if len(df[column].unique()) == 1]
    dtypes = dict(zip(df.columns, df.dtypes))
    by_dtype: dict = {}
    categorical = []
    other = []
    for column in columns:
        dtype = dtypes[column]
        if isinstance(dtype, np.dtype) and dtype.kind in "biuf":
            by_dtype.setdefault(dtype, []).append(column)
        elif isinstance(dtype, pd.CategoricalDtype):
            categorical.append(column)
        else:
            other.append(column)
    constant = set()
    if categorical:
        is_constant, _, _ = _first_two_codes(_categorical_codes(df, categorical))
        constant.update(column for column, flag in zip(categorical, is_constant) if flag)
    for dtype, dtype_columns in by_dtype.items():
        values = df[dtype_columns].to_numpy()
        same_as_first = (values == values[0]).all(axis=0)
        if dtype.kind == "f":
            missing = np.isnan(values)
            all_missing = missing.all(axis=0)
            same_as_first = np.where(missing.any(axis=0), all_missing, same_as_first)
        constant.update(column for column, is_constant in zip(dtype_columns, same_as_first) if is_constant)
    constant.update(column for column in other if len(df[column].unique()) == 1)
    return [column for column in columns if column in constant]


def _memory_usage_shallow(df: DataFrame) -> Series:
    """`df.memory_usage()` (index included, not deep) without building a Series per column.

    A numpy-dtype column's shallow usage is `itemsize * len`, which is what pandas reports for it; the
    extension-dtype columns are asked individually, as pandas would.
    """
    num_rows = len(df)
    values = [
        dtype.itemsize * num_rows
        if isinstance(dtype, np.dtype)
        # What `Series.memory_usage(index=False)` computes, read from the array so no Series is built
        # per column: a categorical's codes plus its categories, any other extension array's `nbytes`.
        else (df[column].array.memory_usage() if isinstance(dtype, pd.CategoricalDtype) else df[column].array.nbytes)
        for column, dtype in zip(df.columns, df.dtypes)
    ]
    index_usage = Series([df.index.memory_usage()], index=["Index"], dtype=np.intp)
    return pd.concat([index_usage, Series(values, index=df.columns, dtype=np.intp)])


# suspend_logging to hide the Pandas log of NumExpr initialization
@_suspend_logging_for_package("pandas")
def get_approximate_df_mem_usage(df: DataFrame, sample_ratio=0.2):
    num_rows = len(df)
    dtypes = dict(zip(df.columns, df.dtypes))
    if sample_ratio >= 1 or num_rows == 0:
        memory_usage = df.memory_usage(deep=True)
        for column in df:
            if dtypes[column] == _OBJECT_DTYPE:
                memory_usage[column] = _object_column_mem_usage(df[column].to_numpy(), num_rows, 1.0)
        return memory_usage
    else:
        num_rows_sample = math.ceil(sample_ratio * num_rows)
        sample_ratio = num_rows_sample / num_rows
        dtypes_raw = get_type_map_raw(df)
        columns_category = [column for column in df if dtypes_raw[column] == R_CATEGORY]
        columns_inexact = [column for column in df if dtypes_raw[column] not in [R_INT, R_FLOAT, R_CATEGORY]]
        # Object columns need per-object accounting, the rest extrapolate from a deep sample.
        columns_object = [column for column in columns_inexact if dtypes[column] == _OBJECT_DTYPE]
        columns_inexact = [column for column in columns_inexact if dtypes[column] != _OBJECT_DTYPE]
        memory_usage = _memory_usage_shallow(df)
        # One value per column, assigned to the Series in one go: a per-column `Series.__setitem__`
        # and the `.cat` accessor (a Series per call) cost more than the estimate itself on a wide
        # table with thousands of categorical columns.
        exact: dict = {}
        for column in columns_category:
            categorical = df[column].array
            categories = categorical.categories
            num_categories = max(len(categories), 1)
            num_categories_sample = math.ceil(sample_ratio * num_categories)
            sample_ratio_cat = num_categories_sample / num_categories
            exact[column] = int(
                categorical.codes.dtype.itemsize * num_rows
                + categories[:num_categories_sample].memory_usage(deep=True) / sample_ratio_cat
            )
        for column in columns_object:
            exact[column] = _object_column_mem_usage(df[column].to_numpy()[:num_rows_sample], num_rows, sample_ratio)
        if exact:
            memory_usage[list(exact)] = list(exact.values())
        if columns_inexact:
            # this line causes NumExpr log, suspend_logging is used to hide the log.
            memory_usage_inexact = (
                df[columns_inexact].head(num_rows_sample).memory_usage(deep=True)[columns_inexact] / sample_ratio
            )
            memory_usage = memory_usage_inexact.combine_first(memory_usage)
        return memory_usage
