import logging

from pandas import DataFrame

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
        return {'feature_interactions': False}
