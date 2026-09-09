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
    block-wise per dtype, the other columns through ``unique``.
    """
    if columns is None:
        columns = list(df.columns)
    if len(df) < 2:
        return {}
    if df.columns.has_duplicates:
        return _two_valued_through_unique(df, columns)
    dtypes = dict(zip(df.columns, df.dtypes))
    by_dtype: dict = {}
    other = []
    for column in columns:
        dtype = dtypes[column]
        if isinstance(dtype, np.dtype) and dtype.kind in "biuf":
            by_dtype.setdefault(dtype, []).append(column)
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


def get_constant_columns(df: DataFrame, columns: list | None = None) -> list:
    """The columns of `df` (or of `columns`) holding a single distinct value: ``len(df[column].unique()) == 1``.

    All-missing counts as one value, ``-0.0`` equals ``0.0``, and an empty frame has no constant column, as
    with ``unique``. numpy numeric and bool columns are tested block-wise per dtype; every other column is
    asked through ``unique`` as before.
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
    other = []
    for column in columns:
        dtype = dtypes[column]
        if isinstance(dtype, np.dtype) and dtype.kind in "biuf":
            by_dtype.setdefault(dtype, []).append(column)
        else:
            other.append(column)
    constant = set()
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
        dtype.itemsize * num_rows if isinstance(dtype, np.dtype) else df[column].memory_usage(index=False)
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
        if columns_category:
            for column in columns_category:
                num_categories = max(len(df[column].cat.categories), 1)
                num_categories_sample = math.ceil(sample_ratio * num_categories)
                sample_ratio_cat = num_categories_sample / num_categories
                memory_usage[column] = int(
                    df[column].cat.codes.dtype.itemsize * num_rows
                    + df[column].cat.categories[:num_categories_sample].memory_usage(deep=True) / sample_ratio_cat
                )
        if columns_object:
            for column in columns_object:
                memory_usage[column] = _object_column_mem_usage(
                    df[column].to_numpy()[:num_rows_sample], num_rows, sample_ratio
                )
        if columns_inexact:
            # this line causes NumExpr log, suspend_logging is used to hide the log.
            memory_usage_inexact = (
                df[columns_inexact].head(num_rows_sample).memory_usage(deep=True)[columns_inexact] / sample_ratio
            )
            memory_usage = memory_usage_inexact.combine_first(memory_usage)
        return memory_usage
