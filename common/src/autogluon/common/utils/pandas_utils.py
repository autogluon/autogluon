import logging
import math
import sys
from functools import wraps

from pandas import DataFrame

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


# suspend_logging to hide the Pandas log of NumExpr initialization
@_suspend_logging_for_package("pandas")
def get_approximate_df_mem_usage(df: DataFrame, sample_ratio=0.2):
    num_rows = len(df)
    if sample_ratio >= 1 or num_rows == 0:
        memory_usage = df.memory_usage(deep=True)
        for column in df:
            if df[column].dtype == object:
                memory_usage[column] = _object_column_mem_usage(df[column].to_numpy(), num_rows, 1.0)
        return memory_usage
    else:
        num_rows_sample = math.ceil(sample_ratio * num_rows)
        sample_ratio = num_rows_sample / num_rows
        dtypes_raw = get_type_map_raw(df)
        columns_category = [column for column in df if dtypes_raw[column] == R_CATEGORY]
        columns_inexact = [column for column in df if dtypes_raw[column] not in [R_INT, R_FLOAT, R_CATEGORY]]
        # Object columns need per-object accounting, the rest extrapolate from a deep sample.
        columns_object = [column for column in columns_inexact if df[column].dtype == object]
        columns_inexact = [column for column in columns_inexact if df[column].dtype != object]
        memory_usage = df.memory_usage()
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
