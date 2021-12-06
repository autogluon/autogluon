import copy
import logging

from pandas import DataFrame

from autogluon.common.features.types import R_CATEGORY, S_TEXT_AS_CATEGORY

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: LabelEncoderTransformer
class LabelEncoderFeatureGenerator(AbstractFeatureGenerator):
    """Converts category features to int features by mapping to the category codes."""
    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        X_out = self._transform(X)
        feature_metadata_out_type_group_map_special = copy.deepcopy(self.feature_metadata_in.type_group_map_special)
        if S_TEXT_AS_CATEGORY in feature_metadata_out_type_group_map_special:
            feature_metadata_out_type_group_map_special.pop(S_TEXT_AS_CATEGORY)
        return X_out, feature_metadata_out_type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self.convert_category_to_int(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_CATEGORY])

    @staticmethod
    def convert_category_to_int(X: DataFrame) -> DataFrame:
        # TODO: add inplace option?
        X = X.apply(lambda x: x.cat.codes)
        return X

    def _more_tags(self):
        return {'feature_interactions': False}
