import logging
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


def get_type_family_raw(dtype) -> str:
    """From dtype, gets the dtype family."""
    try:
        if isinstance(dtype, pd.SparseDtype):
            dtype = dtype.subtype
        if dtype.name == "category":
            return "category"
        if "datetime" in dtype.name:
            return "datetime"
        if "string" in dtype.name:
            return "object"
        elif np.issubdtype(dtype, np.integer):
            return "int"
        elif np.issubdtype(dtype, np.floating):
            return "float"
    except Exception as err:
        logger.error(
            f"Warning: dtype {dtype} is not recognized as a valid dtype by numpy! "
            f"AutoGluon may incorrectly handle this feature..."
        )
        logger.error(err)

    if dtype.name in ["bool", "bool_"]:
        return "bool"
    elif dtype.name in ["str", "string", "object"]:
        return "object"
    else:
        return dtype.name


# Real dtypes
def get_type_map_real(df: DataFrame) -> dict:
    features_types = df.dtypes.to_dict()
    return {k: v.name for k, v in features_types.items()}


# Raw dtypes (Real dtypes family)
def get_type_map_raw(df: DataFrame) -> dict:
    features_types = df.dtypes.to_dict()
    return {k: get_type_family_raw(v) for k, v in features_types.items()}


def get_type_map_special(X: DataFrame) -> dict:
    type_map_special = {}
    for column in X:
        types_special = get_types_special(X[column])
        if types_special:
            type_map_special[column] = types_special
    return type_map_special


def get_types_special(X: Series) -> List[str]:
    types_special = []
    if isinstance(X.dtype, pd.SparseDtype):
        types_special.append("sparse")
    if check_if_datetime_as_object_feature(X):
        types_special.append("datetime_as_object")
    elif check_if_nlp_feature(X):
        types_special.append("text")
    return types_special


def get_type_group_map(type_map: dict) -> defaultdict:
    type_group_map = defaultdict(list)
    for key, val in type_map.items():
        if isinstance(val, list):
            for feature_type in val:
                type_group_map[feature_type].append(key)
        else:
            type_group_map[val].append(key)
    return type_group_map


def get_type_group_map_real(df: DataFrame) -> defaultdict:
    type_map_real = get_type_map_real(df)
    return get_type_group_map(type_map_real)


def get_type_group_map_raw(df: DataFrame) -> defaultdict:
    type_map_raw = get_type_map_raw(df)
    return get_type_group_map(type_map_raw)


def get_type_group_map_special(df: DataFrame) -> defaultdict:
    type_map_special = get_type_map_special(df)
    return get_type_group_map(type_map_special)


# TODO: Expand to int64 -> date features (milli from epoch etc)
# TODO: This takes a surprisingly long time to run, ~30 seconds a laptop for
#  50,000 rows of datetime_as_object for a single column. Try to optimize.
def check_if_datetime_as_object_feature(X: Series) -> bool:
    type_family = get_type_family_raw(X.dtype)
    # TODO: Check if low numeric numbers, could be categorical encoding!
    # TODO: If low numeric, potentially it is just numeric instead of date
    if X.isnull().all():
        return False
    if type_family != "object":  # TODO: seconds from epoch support
        return False
    try:
        # TODO: pd.Series(['20170204','20170205','20170206']) is incorrectly not detected as datetime_as_object
        #  But we don't want pd.Series(['184','822828','20170206']) to be detected as datetime_as_object
        #  Need some smart logic (check min/max values?, check last 2 values don't go >31?)
        pd.to_numeric(X)
    except (ValueError, TypeError):
        try:
            if len(X) > 500:
                # Sample to speed-up type inference
                X = X.sample(n=500, random_state=0)
            result = pd.to_datetime(X, errors="coerce", format="mixed")
            if result.isnull().mean() > 0.8:  # If over 80% of the rows are NaN
                return False
            return True
        except (ValueError, TypeError):
            return False
    else:
        return False


def check_if_nlp_feature(X: Series) -> bool:
    type_family = get_type_family_raw(X.dtype)
    if type_family != "object":
        return False
    if len(X) > 5000:
        # Sample to speed-up type inference
        X = X.sample(n=5000, random_state=0)
    X_unique = X.unique()
    num_unique = len(X_unique)
    num_rows = len(X)
    unique_ratio = num_unique / num_rows
    if unique_ratio <= 0.01:
        return False
    try:
        avg_words = Series(X_unique.astype(str)).str.split().str.len().mean()
    except AttributeError:
        return False
    if avg_words < 3:
        return False

    return True


def get_bool_true_val(uniques):
    """
    From a pandas series with `uniques = series.unique()`, get the replace_val to convert to boolean when calling:
    series_bool = series == replace_val

    Therefore, any value other than `replace_val` will be set to `False` when converting to boolean.

    series must have exactly 2 unique values

    We make the assumption that the value chosen as `True` between the two options is mostly arbitrary,
    with the exception that np.nan will not be considered `True`.
    When possible, we try to sort the values so that (0, 1) will choose 1 as True, however this decision
    should ideally not impact downstream models much.
    Any new unseen values (including nan) at inference time will be mapped to `False` automatically.

    In this code, 0 and 0.0 (int and float) are treated as the same value. Similarly with any other integer and float (such as 1 and 1.0).

    """
    # This is a safety net in case the unique types are mixed (such as string and int). In this scenario, an exception is raised
    # and therefore we use the unsorted values.
    try:
        # Sort the values to avoid relying on row-order when determining which value is mapped to `True`.
        uniques.sort()
    except (ValueError, TypeError):
        pass
    replace_val = uniques[1]
    try:
        # This is to ensure that we don't map np.nan to `True` in the boolean.
        is_nan = np.isnan(replace_val)
    except (ValueError, TypeError):
        if replace_val is None:
            is_nan = True
        else:
            is_nan = False
    if is_nan:
        replace_val = uniques[0]
    return replace_val
