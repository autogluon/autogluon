import logging

from pandas import DataFrame

from autogluon.common.features.feature_metadata import FeatureMetadata

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class IdentityFeatureGenerator(AbstractFeatureGenerator):
    """IdentityFeatureGenerator simply passes the data along without alterations."""

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        X_out = self._transform(X)
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    def _more_tags(self):
        return {"feature_interactions": False}

    def estimate_output_feature_metadata(self, feature_metadata_in: FeatureMetadata, **kwargs) -> FeatureMetadata:
        features_to_remove = feature_metadata_in.get_features(**self._infer_features_in_args)
        return feature_metadata_in.keep_features(features_to_remove, inplace=False)
