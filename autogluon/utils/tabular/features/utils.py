import logging

import numpy as np
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


# TODO: Add documentation
def clip_and_astype(df: DataFrame, columns: list, clip_min=0, clip_max=255, dtype: str = 'uint8') -> DataFrame:
    if columns:
        df[columns] = np.clip(df[columns], clip_min, clip_max).astype(dtype)
    return df


def check_if_useless_feature(X: Series) -> bool:
    if len(X.unique()) <= 1:
        return True
    else:
        return False
