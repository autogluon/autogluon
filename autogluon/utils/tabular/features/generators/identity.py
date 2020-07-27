import logging

from pandas import DataFrame

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class IdentityFeatureGenerator(AbstractFeatureGenerator):
    def fit(self, X, y=None):
        self.fit_transform(X, y=y)

    def _fit_transform(self, X, y=None):
        X_out = self._transform(X)
        return X_out, None

    def _transform(self, X):
        return self._generate_features_identity(X)

    def _generate_features_identity(self, X: DataFrame):
        return X[self.features_in]
