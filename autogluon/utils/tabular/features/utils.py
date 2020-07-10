import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def get_type_family(dtype):
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


def get_type_groups_df(df):
    features_types = df.dtypes.to_dict()

    features_type_groups = defaultdict(list)
    features_types_tmp = {k: v.name for k, v in features_types.items()}
    for key, val in features_types_tmp.items():
        features_type_groups[val].append(key)
    return features_type_groups


def get_type_family_groups_df(df):
    features_types = df.dtypes.to_dict()

    features_type_groups = defaultdict(list)
    features_types_tmp = {k: get_type_family(v) for k, v in features_types.items()}
    for key, val in features_types_tmp.items():
        features_type_groups[val].append(key)
    return features_type_groups
