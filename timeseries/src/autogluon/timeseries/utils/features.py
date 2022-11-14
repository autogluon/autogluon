from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.features.types import R_FLOAT, R_INT
from autogluon.features.generators import (
    AsTypeFeatureGenerator,
    CategoryFeatureGenerator,
    IdentityFeatureGenerator,
    PipelineFeatureGenerator,
)


class ContinuousAndCategoricalFeatureGenerator(PipelineFeatureGenerator):
    """Generates categorical and continuous features for time series models."""

    def __init__(self, feature_metadata: Optional[FeatureMetadata] = None, verbosity: int = 0, **kwargs):
        generators = [
            CategoryFeatureGenerator(minimum_cat_count=1),
            IdentityFeatureGenerator(infer_features_in_args={"valid_raw_types": [R_INT, R_FLOAT]}),
        ]
        super().__init__(
            generators=[generators],
            post_generators=[],
            pre_generators=[AsTypeFeatureGenerator(convert_bool=False)],
            feature_metadata_in=feature_metadata,
            pre_enforce_types=False,
            pre_drop_useless=False,
            verbosity=verbosity,
            **kwargs,
        )


def get_categorical_and_continuous_features(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split categorical and continuous columns of a dataframe into two separate dataframes.

    Columns that have neither categorical nor numerical dtypes are ignored.
    """
    categorical_column_names = []
    continuous_column_names = []

    for column_name, column_values in dataframe.items():
        if is_categorical_dtype(column_values):
            categorical_column_names.append(column_name)
        elif is_numeric_dtype(column_values):
            continuous_column_names.append(column_name)

    return dataframe[categorical_column_names], dataframe[continuous_column_names]


def convert_numerical_features_to_float(dataframe: pd.DataFrame, float_dtype=np.float64):
    """In-place convert the dtype of all numerical (float or int) columns to the given float dtype."""
    for col_name in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[col_name]):
            dataframe[col_name] = dataframe[col_name].astype(float_dtype)
    return dataframe
