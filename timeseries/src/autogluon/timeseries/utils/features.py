from typing import Optional, Tuple

import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.features.types import R_FLOAT, R_INT
from autogluon.features.generators import CategoryFeatureGenerator, IdentityFeatureGenerator, PipelineFeatureGenerator


class ContinuousAndCategoricalFeatureGenerator(PipelineFeatureGenerator):
    """Generates categorical and continuous features for time series models."""

    def __init__(self, feature_metadata: Optional[FeatureMetadata] = None):
        # TODO: Ensure that feature_metadata only contains numeric and categorical dtypes
        generators = [
            CategoryFeatureGenerator(minimum_cat_count=1),
            IdentityFeatureGenerator(infer_features_in_args={"valid_raw_types": [R_INT, R_FLOAT]}),
        ]
        super().__init__(generators=[generators], post_generators=[], feature_metadata_in=feature_metadata)


def get_categorical_and_continuous_features(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split categorical and continuous columns of a dataframe into two separate dataframes."""
    categorical_column_names = []
    continuous_column_names = []

    for column_name, column_values in dataframe.iteritems():
        if is_categorical_dtype(column_values):
            categorical_column_names.append(column_name)
        elif is_numeric_dtype(column_values):
            continuous_column_names.append(column_name)

    return dataframe[categorical_column_names], dataframe[continuous_column_names]
