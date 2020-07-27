import logging
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


def get_type_family_raw(dtype) -> str:
    """From dtype, gets the dtype family."""
    try:
        if dtype.name is 'category':
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
    return {column: get_type_family_special(X[column]) for column in X}


def get_type_family_special(X: Series) -> str:
    type_family = get_type_family_raw(X.dtype)
    if check_if_datetime_feature(X):
        type_family = 'datetime'
    elif check_if_nlp_feature(X):
        type_family = 'text'
    return type_family


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


def get_type_group_map_special(df: DataFrame) -> defaultdict:
    type_map_special = get_type_map_special(df)
    return get_type_group_map(type_map_special)


# TODO: Expand to int64 -> date features (milli from epoch etc)
def check_if_datetime_feature(X: Series) -> bool:
    type_family = get_type_family_raw(X.dtype)
    # TODO: Check if low numeric numbers, could be categorical encoding!
    # TODO: If low numeric, potentially it is just numeric instead of date
    if X.isnull().all():
        return False
    if type_family == 'datetime':
        return True
    if type_family != 'object':  # TODO: seconds from epoch support
        return False
    try:
        X.apply(pd.to_datetime)
        return True
    except:
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
    avg_words = np.mean([len(re.sub(' +', ' ', value).split(' ')) if isinstance(value, str) else 0 for value in X_unique])
    if avg_words < 3:
        return False

    return True
