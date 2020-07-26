import logging

import pandas as pd
from pandas import DataFrame

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Add verbose descriptions of each special dtype this generator can create.
class DatetimeFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X):
        self.fit_transform(X)

    def _fit_transform(self, X):
        X_out = self._transform(X)
        type_family_groups_special = dict(
            datetime=list(X_out.columns)
        )
        return X_out, type_family_groups_special

    def _transform(self, X):
        return self._generate_features_datetime(X)

    def _generate_features_datetime(self, X: DataFrame):
        X_datatime = pd.DataFrame(index=X.index)
        for datetime_feature in self.features_in:
            X_datatime[datetime_feature] = pd.to_datetime(X[datetime_feature])
            X_datatime[datetime_feature] = pd.to_numeric(X_datatime[datetime_feature])  # TODO: Use actual date info
            # TODO: Add fastai date features
        return X_datatime
