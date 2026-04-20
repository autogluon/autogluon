import logging
import math
from functools import wraps

import pandas as pd
from packaging import version
from pandas import DataFrame

from ..features.infer_types import get_type_map_raw
from ..features.types import R_CATEGORY, R_FLOAT, R_INT

PANDAS_VERSION = version.parse(pd.__version__)
PANDAS_V2_OR_NEWER = PANDAS_VERSION >= version.parse("2.0.0")
PANDAS_V2_1_OR_NEWER = PANDAS_VERSION >= version.parse("2.1.0")
PANDAS_V3_OR_NEWER = PANDAS_VERSION >= version.parse("3.0.0")

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


# suspend_logging to hide the Pandas log of NumExpr initialization
@_suspend_logging_for_package("pandas")
def get_approximate_df_mem_usage(df: DataFrame, sample_ratio=0.2):
    if sample_ratio >= 1:
        return df.memory_usage(deep=True)
    else:
        num_rows = len(df)
        num_rows_sample = math.ceil(sample_ratio * num_rows)
        sample_ratio = num_rows_sample / num_rows
        dtypes_raw = get_type_map_raw(df)
        columns_category = [column for column in df if dtypes_raw[column] == R_CATEGORY]
        columns_inexact = [column for column in df if dtypes_raw[column] not in [R_INT, R_FLOAT, R_CATEGORY]]
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
        if columns_inexact:
            # this line causes NumExpr log, suspend_logging is used to hide the log.
            memory_usage_inexact = (
                df[columns_inexact].head(num_rows_sample).memory_usage(deep=True)[columns_inexact] / sample_ratio
            )
            memory_usage = memory_usage_inexact.combine_first(memory_usage)
        return memory_usage


def pandas_freq_alias(freq: str) -> str:
    """Translate deprecated pandas frequency aliases to their new equivalents."""
    if not PANDAS_V3_OR_NEWER or not isinstance(freq, str):
        return freq
    mapping = {
        "M": "ME",
        "Q": "QE",
        "Y": "YE",
        "A": "YE",
        "H": "h",
        "T": "min",
        "S": "s",
        "L": "ms",
        "U": "us",
        "N": "ns",
    }
    if freq in mapping:
        return mapping[freq]

    # regex to replace M, Q, Y, A at the end of the string if preceded by a number or nothing
    import re

    for old, new in mapping.items():
        # Using a simple replacement for common cases like '2M' -> '2ME'
        if re.match(r"^\d*" + old + r"$", freq):
            return freq.replace(old, new)

    return freq
