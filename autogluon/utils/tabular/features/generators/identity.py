import logging

from pandas import DataFrame

from .abstract import AbstractFeatureGenerator
from ..feature_metadata import R_INT, R_FLOAT

logger = logging.getLogger(__name__)


class IdentityFeatureGenerator(AbstractFeatureGenerator):
    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        X_out = self._transform(X)
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X):
        return X

    def _infer_features_in(self, X, y=None) -> list:
        identity_features = []
        valid_raw_types = {R_INT, R_FLOAT}
        features = self.feature_metadata_in.get_features()
        for feature in features:
            feature_type_raw = self.feature_metadata_in.get_feature_type_raw(feature)
            if feature_type_raw in valid_raw_types:
                identity_features.append(feature)
        return identity_features
