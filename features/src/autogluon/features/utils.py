import logging

import numpy as np
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


def clip_and_astype(df: DataFrame, columns: list = None, clip_min=0, clip_max=255, dtype: str = "uint8") -> DataFrame:
    """
    Clips columns in a DataFrame to min and max values, and then converts dtype.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    columns : list, optional
        Column subset of df to apply the clip_and_astype logic to. If not specified, all columns of df are used.
    clip_min : int or float, default 0
        Minimum value to clip column values to. All values less than this will be set to clip_min.
    clip_max : int or float, default 255
        Maximum value to clip column values to. All values greater than this will be set to clip_max.
    dtype : dtype, default 'uint8'
        Data type to force after clipping is applied.

    Returns
    -------
    df_clipped : DataFrame
        clipped and astyped version of the input df.
    """
    if columns is None:
        df = np.clip(df, clip_min, clip_max).astype(dtype)
    elif columns:
        df[columns] = np.clip(df[columns], clip_min, clip_max).astype(dtype)
    return df


# TODO: Consider NaN values as a separate value?
def is_useless_feature(X: Series) -> bool:
    """If a feature has the same value for every row, it carries no useful information"""
    return len(X.unique()) <= 1


def get_smallest_valid_dtype_int(min_val: int, max_val: int):
    """Based on the minimum and maximum values of a feature, returns the smallest valid dtype to represent the feature."""
    if min_val < 0:
        dtypes_to_check = [np.int8, np.int16, np.int32, np.int64]
    else:
        dtypes_to_check = [np.uint8, np.uint16, np.uint32, np.uint64]
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if max_val <= np.iinfo(dtype).max and min_val >= np.iinfo(dtype).min:
            return dtype
    raise ValueError(
        f"Value is not able to be represented by {dtypes_to_check[-1].__name__}. (min_val, max_val): ({min_val}, {max_val})"
    )
