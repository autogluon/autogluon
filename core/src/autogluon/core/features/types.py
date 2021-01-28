import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


def get_type_family_raw(dtype) -> str:
    """From dtype, gets the dtype family."""
    try:
        if dtype.name == 'category':
            return 'category'
        if 'datetime' in dtype.name:
            return 'datetime'
        elif np.issubdtype(dtype, np.integer):
            return 'int'
        elif np.issubdtype(dtype, np.floating):
            return 'float'
    except Exception as err:
        logger.exception(f'Warning: dtype {dtype} is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...')
        logger.exception(err)

    if dtype.name in ['bool', 'bool_']:
        return 'bool'
    elif dtype.name in ['str', 'string', 'object']:
        return 'object'
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
        type_special = get_type_special(X[column])
        if type_special is not None:
            type_map_special[column] = type_special
    return type_map_special


def get_type_special(X: Series) -> str:
    if check_if_datetime_as_object_feature(X):
        type_special = 'datetime_as_object'
    elif check_if_nlp_feature(X):
        type_special = 'text'
    else:
        type_special = None
    return type_special


def get_type_group_map(type_map: dict) -> defaultdict:
    type_group_map = defaultdict(list)
    for key, val in type_map.items():
        type_group_map[val].append(key)
    return type_group_map


def get_type_group_map_real(df: DataFrame) -> defaultdict:
    type_map_real = get_type_map_real(df)
    return get_type_group_map(type_map_real)


def get_type_group_map_raw(df: DataFrame) -> defaultdict:
    type_map_raw = get_type_map_raw(df)
    return get_type_group_map(type_map_raw)


# TODO: Expand to enable multiple special types per feature
def get_type_group_map_special(df: DataFrame) -> defaultdict:
    type_map_special = get_type_map_special(df)
    return get_type_group_map(type_map_special)


# TODO: Expand to int64 -> date features (milli from epoch etc)
# TODO: This takes a surprisingly long time to run, ~30 seconds a laptop for 50,000 rows of datetime_as_object for a single column. Try to optimize.
def check_if_datetime_as_object_feature(X: Series) -> bool:
    type_family = get_type_family_raw(X.dtype)
    # TODO: Check if low numeric numbers, could be categorical encoding!
    # TODO: If low numeric, potentially it is just numeric instead of date
    if X.isnull().all():
        return False
    if type_family != 'object':  # TODO: seconds from epoch support
        return False
    try:
        # TODO: pd.Series(['20170204','20170205','20170206']) is incorrectly not detected as datetime_as_object
        #  But we don't want pd.Series(['184','822828','20170206']) to be detected as datetime_as_object
        #  Need some smart logic (check min/max values?, check last 2 values don't go >31?)
        X.apply(pd.to_numeric)
    except:
        try:
            X.apply(pd.to_datetime)
            return True
        except:
            return False
    else:
        return False


def check_if_nlp_feature(X: Series) -> bool:
    type_family = get_type_family_raw(X.dtype)
    if type_family != 'object':
        return False
    X_unique = X.unique()
    num_unique = len(X_unique)
    num_rows = len(X)
    unique_ratio = num_unique / num_rows
    if unique_ratio <= 0.01:
        return False
    try:
        avg_words = Series(X_unique).str.split().str.len().mean()
    except AttributeError:
        return False
    if avg_words < 3:
        return False

    return True
