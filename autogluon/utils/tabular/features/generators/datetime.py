import logging

import pandas as pd
from pandas import DataFrame, Series

from .abstract import AbstractFeatureGenerator
from ..feature_metadata import FeatureMetadata

logger = logging.getLogger(__name__)


# TODO: Add verbose descriptions of each special dtype this generator can create.
class DatetimeFeatureGenerator(AbstractFeatureGenerator):
    def _fit_transform(self, X: DataFrame, y: Series = None, feature_metadata_in: FeatureMetadata = None) -> (DataFrame, dict):
        X_out = self._transform(X)
        type_family_groups_special = dict(
            datetime_as_int=list(X_out.columns)
        )
        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._generate_features_datetime(X)

    def _infer_features_in_from_metadata(self, X, y=None, feature_metadata_in: FeatureMetadata = None) -> list:
        datetime_features = feature_metadata_in.type_group_map_special['datetime_as_object'] + feature_metadata_in.type_group_map_raw['datetime']
        return datetime_features

    def _generate_features_datetime(self, X: DataFrame) -> DataFrame:
        X_datetime = pd.DataFrame(index=X.index)
        for datetime_feature in self.features_in:
            # TODO: Be aware: When converted to float32 by downstream models, the seconds value will be up to 3 seconds off the true time due to rounding error. If seconds matter, find a separate way to generate (Possibly subtract smallest datetime from all values).
            X_datetime[datetime_feature] = pd.to_datetime(X[datetime_feature])
            X_datetime[datetime_feature] = pd.to_numeric(X_datetime[datetime_feature])  # TODO: Use actual date info
            # X_datetime[datetime_feature] = pd.to_timedelta(X_datetime[datetime_feature]).dt.total_seconds()
            # TODO: Add fastai date features
        return X_datetime
