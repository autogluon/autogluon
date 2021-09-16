import logging

import pandas as pd
from pandas import DataFrame

from autogluon.core.features.types import R_DATETIME, S_DATETIME_AS_OBJECT

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class DatetimeFeatureGenerator(AbstractFeatureGenerator):
    """Transforms datetime features into numeric features."""
    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        self._fillna_map = self._compute_fillna_map(X)
        X_out = self._transform(X)
        type_family_groups_special = dict(
            datetime_as_int=list(X_out.columns)
        )
        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._generate_features_datetime(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(required_raw_special_pairs=[
            (R_DATETIME, None),
            (None, [S_DATETIME_AS_OBJECT])
        ])

    def _compute_fillna_map(self, X: DataFrame):
        fillna_map = dict()
        for datetime_feature in self.features_in:
            datetime_series = pd.to_datetime(X[datetime_feature], errors='coerce')

            # Best guess is currently to fill by the mean.
            fillna_datetime = datetime_series.mean()
            fillna_map[datetime_feature] = fillna_datetime
        return fillna_map

    # TODO: Improve handling of missing datetimes
    def _generate_features_datetime(self, X: DataFrame) -> DataFrame:
        X_datetime = DataFrame(index=X.index)
        for datetime_feature in self.features_in:
            # TODO: Be aware: When converted to float32 by downstream models, the seconds value will be up to 3 seconds off the true time due to rounding error. If seconds matter, find a separate way to generate (Possibly subtract smallest datetime from all values).
            # TODO: could also return an extra boolean column is_nan which could provide predictive signal.
            X_datetime[datetime_feature] = pd.to_datetime(X[datetime_feature], errors='coerce').fillna(self._fillna_map[datetime_feature])
            X_datetime[datetime_feature] = pd.to_numeric(X_datetime[datetime_feature])  # TODO: Use actual date info
            # X_datetime[datetime_feature] = pd.to_timedelta(X_datetime[datetime_feature]).dt.total_seconds()
            # TODO: Add fastai date features
        return X_datetime

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self._fillna_map:
            for feature in features:
                if feature in self._fillna_map:
                    self._fillna_map.pop(feature)
