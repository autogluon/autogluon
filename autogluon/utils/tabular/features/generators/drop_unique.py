import logging

from pandas import DataFrame

from .abstract import AbstractFeatureGenerator
from ..feature_metadata import FeatureMetadata

logger = logging.getLogger(__name__)


# TODO: Not necessary to exist after fitting, can just update outer context feature_out/feature_in and then delete this
class DropUniqueFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, max_unique_ratio=0.99, **kwargs):
        super().__init__(**kwargs)
        self.max_unique_ratio = max_unique_ratio

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        features_to_drop = self._drop_unique_features(X, self.feature_metadata_in, max_unique_ratio=self.max_unique_ratio)
        self._remove_features_in(features_to_drop)
        X_out = X[self.features_in]
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X):
        return X

    # TODO: Consider NaN?
    @staticmethod
    def _drop_unique_features(X: DataFrame, feature_metadata: FeatureMetadata, max_unique_ratio) -> list:
        features_to_drop = []
        X_len = len(X)
        max_unique_value_count = X_len * max_unique_ratio
        for column in X:
            unique_value_count = len(X[column].unique())
            if unique_value_count == 1:
                features_to_drop.append(column)
            elif feature_metadata.get_feature_type_raw(column) in ['category', 'object'] and (unique_value_count > max_unique_value_count):
                features_to_drop.append(column)
        return features_to_drop
