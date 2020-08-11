import logging

import numpy as np
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


# TODO: Add documentation
def clip_and_astype(df: DataFrame, columns: list = None, clip_min=0, clip_max=255, dtype: str = 'uint8') -> DataFrame:
    if columns is None:
        df = np.clip(df, clip_min, clip_max).astype(dtype)
    elif columns:
        df[columns] = np.clip(df[columns], clip_min, clip_max).astype(dtype)
    return df


def is_useless_feature(X: Series) -> bool:
    if len(X.unique()) <= 1:
        return True
    else:
        return False
