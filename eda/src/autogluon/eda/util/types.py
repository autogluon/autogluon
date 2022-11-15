from typing import Union

from pandas import DataFrame

from autogluon.common.features.types import *


def map_raw_type_to_feature_type(col: str, raw_type: str, series: DataFrame, numeric_as_categorical_threshold: int = 20) -> Union[None, str]:
    if col is None:
        return None
    elif series[col].nunique() <= numeric_as_categorical_threshold:
        return 'category'
    elif raw_type in [R_INT, R_FLOAT]:
        return 'numeric'
    elif raw_type in [R_OBJECT, R_CATEGORY, R_BOOL]:
        return 'category'
    else:
        return None
