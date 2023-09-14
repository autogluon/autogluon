import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

from autogluon.common.features.types import R_DATETIME, S_DATETIME_AS_OBJECT

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class DatetimeFeatureGenerator(AbstractFeatureGenerator):
    """Transforms datetime features into numeric features.

    Parameters
    ----------
    features : list, optional
        A list of datetime features to parse out of dates.
        For a full list of options see the methods inside pandas.Series.dt at https://pandas.pydata.org/docs/reference/api/pandas.Series.html
    """

    def __init__(self, features: list = ["year", "month", "day", "dayofweek"], **kwargs):
        super().__init__(**kwargs)
        self.features = features

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        self._fillna_map = dict()
        X_out = self._transform(X, is_fit=True)
        type_family_groups_special = dict(datetime_as_int=list(X_out.columns))
        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame, is_fit=False) -> DataFrame:
        return self._generate_features_datetime(X, is_fit=is_fit)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(required_raw_special_pairs=[(R_DATETIME, None), (None, [S_DATETIME_AS_OBJECT])])

    def normalize_timeseries(self, X: pd.DataFrame, feature: str, is_fit: bool) -> pd.Series:
        # TODO: Be aware: When converted to float32 by downstream models, the seconds value will be up to 3 seconds off the true time due to rounding error.
        #  If seconds matter, find a separate way to generate (Possibly subtract smallest datetime from all values).
        # TODO: could also return an extra boolean column is_nan which could provide predictive signal.
        # Note: The .replace call is required to handle the obnoxious edge-case of:
        #   NaN, empty string, datetime without timezone, and datetime with timezone, all as an object type, all being present in the same column.
        #   I don't know why, but in this specific situation (and not otherwise), NaN will be filled by .fillna, but empty string will be converted to NaT
        #   and refuses to be filled by .fillna, requiring a dedicated replace call. (NaT is filled by .fillna in every other situation...)
        series = pd.to_datetime(X[feature].copy(), utc=True, errors="coerce", format="mixed")
        broken_idx = series[(series == "NaT") | series.isna() | series.isnull()].index
        bad_rows = series.iloc[broken_idx]
        if is_fit:
            good_rows = series[~series.isin(bad_rows)].astype(np.int64)
            self._fillna_map[feature] = pd.to_datetime(int(good_rows.mean()), utc=True, format="mixed")
        series[broken_idx] = self._fillna_map[feature]
        return series

    # TODO: Improve handling of missing datetimes
    def _generate_features_datetime(self, X: DataFrame, is_fit: bool) -> DataFrame:
        X_datetime = DataFrame(index=X.index)
        for datetime_feature in self.features_in:
            X_datetime[datetime_feature] = self.normalize_timeseries(X, datetime_feature, is_fit=is_fit)
            for feature in self.features:
                X_datetime[datetime_feature + "." + feature] = getattr(X_datetime[datetime_feature].dt, feature).astype(np.int64)
            X_datetime[datetime_feature] = pd.to_numeric(X_datetime[datetime_feature])
        return X_datetime

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self._fillna_map:
            for feature in features:
                if feature in self._fillna_map:
                    self._fillna_map.pop(feature)
