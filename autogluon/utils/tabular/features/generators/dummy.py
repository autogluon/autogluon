import logging

from pandas import DataFrame

from .identity import IdentityFeatureGenerator

logger = logging.getLogger(__name__)


class DummyFeatureGenerator(IdentityFeatureGenerator):
    def __init__(self, features_in='empty', **kwargs):
        if features_in == 'empty':
            features_in = []
        super().__init__(features_in=features_in, **kwargs)

    def _transform(self, X):
        return self._generate_features_dummy(X)

    @staticmethod
    def _generate_features_dummy(X: DataFrame):
        X_out = DataFrame(index=X.index)
        X_out['__dummy__'] = 0
        return X_out
