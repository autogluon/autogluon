import logging

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

    def __init__(self,
                 features: list = ['year', 'month', 'day', 'dayofweek'],
                 **kwargs):
        super().__init__(**kwargs)
        self.features = features

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
            try:
                fillna_datetime = datetime_series.mean()
            except TypeError:
                # Strange datetime object, try removing timezone info
                fillna_datetime = self._remove_timezones(datetime_series).mean()
            fillna_map[datetime_feature] = fillna_datetime
        return fillna_map

    # TODO: Improve handling of missing datetimes
    def _generate_features_datetime(self, X: DataFrame) -> DataFrame:
        X_datetime = DataFrame(index=X.index)
        for datetime_feature in self.features_in:
            # TODO: Be aware: When converted to float32 by downstream models, the seconds value will be up to 3 seconds off the true time due to rounding error.
            #  If seconds matter, find a separate way to generate (Possibly subtract smallest datetime from all values).
            # TODO: could also return an extra boolean column is_nan which could provide predictive signal.
            # Note: The .replace call is required to handle the obnoxious edge-case of:
            #   NaN, empty string, datetime without timezone, and datetime with timezone, all as an object type, all being present in the same column.
            #   I don't know why, but in this specific situation (and not otherwise), NaN will be filled by .fillna, but empty string will be converted to NaT
            #   and refuses to be filled by .fillna, requiring a dedicated replace call. (NaT is filled by .fillna in every other situation...)
            X_datetime[datetime_feature] = pd.to_datetime(X[datetime_feature], errors='coerce')\
                .fillna(self._fillna_map[datetime_feature])\
                .replace({'NaT': self._fillna_map[datetime_feature]})
            # X_datetime[datetime_feature] = pd.to_timedelta(X_datetime[datetime_feature]).dt.total_seconds()
            # Parse the date into lots of derived fields.
            # Most of the pandas Series.dt properties are here, a few are omitted (e.g. is_month_start) if they can be inferred
            # from other features.
            for feature in self.features:
                try:
                    X_datetime[datetime_feature + '.' + feature] = getattr(X_datetime[datetime_feature].dt, feature).astype(int)
                except AttributeError:
                    # Strange datetime object, try removing timezone info
                    X_datetime[datetime_feature + '.' + feature] = getattr(self._remove_timezones(X_datetime[datetime_feature]).dt, feature).astype(int)

            try:
                X_datetime[datetime_feature] = pd.to_numeric(X_datetime[datetime_feature])
            except TypeError:
                # Strange datetime object, try removing timezone info
                X_datetime[datetime_feature] = pd.to_numeric(self._remove_timezones(X_datetime[datetime_feature]))
        return X_datetime

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self._fillna_map:
            for feature in features:
                if feature in self._fillna_map:
                    self._fillna_map.pop(feature)

    # FIXME: This drops timezone information without converting to UTC first,
    #  it is possible that this leads to worse performance.
    #  I wasn't able to figure out how to convert to UTC properly when implementing this code,
    #  which is why it simply drops the timezone instead.
    # TODO: Currently identifies presence of timezones via try/except, although this is suboptimal in terms of speed.
    @staticmethod
    def _remove_timezones(datetime_as_object: pd.Series) -> pd.Series:
        """
        Fixes issue with datetime objects in pandas when they have timezones.
        Timezones cause the dtype of pandas columns to remain as "object" type.
        This means that logic such as `.dt`, `.mean`, etc. do not work.
        This method removes the timezone information from the values, converting the series to datetime pandas type.
        """
        def _remove_timezone_if_valid(datetime):
            try:
                # datetime
                return datetime.replace(tzinfo=None)
            except:
                # NaN, NaT, Timestamp
                return datetime

        return datetime_as_object.apply(lambda x: _remove_timezone_if_valid(x))
