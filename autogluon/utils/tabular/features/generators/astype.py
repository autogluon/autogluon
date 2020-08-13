import logging

from pandas import DataFrame, Series

from .abstract import AbstractFeatureGenerator
from ..feature_metadata import FeatureMetadata
from ..types import get_type_map_real

logger = logging.getLogger(__name__)


# TODO: Add int fillna input value options: 0, set value, mean, mode, median
class AsTypeFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_metadata_in_real: FeatureMetadata = None  # FeatureMetadata object based on the original input features real dtypes (will contain dtypes such as 'int16' and 'float32' instead of 'int' and 'float').
        # self.inplace = inplace  # TODO, also add check if dtypes are same as expected and skip .astype

    # TODO: consider returning self._transform(X) if we allow users to specify real dtypes as input
    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        return X, self.feature_metadata_in.type_group_map_special

    def _transform(self, X):
        int_features = self.feature_metadata_in.type_group_map_raw['int']
        if int_features:
            null_count = X[int_features].isnull().sum()
            with_null = null_count[null_count != 0]
            # If int feature contains null during inference but not during fit.
            if len(with_null) > 0:
                # TODO: Consider imputing to mode? This is tricky because training data had no missing values.
                # TODO: Add unit test for this situation, to confirm it is handled properly.
                with_null_features = list(with_null.index)
                logger.warning(f'WARNING: Int features without null values at train time contain null values at inference time! Imputing nulls to 0. To avoid this, pass the features as floats during fit!')
                logger.warning(f'WARNING: Int features with nulls: {with_null_features}')
                X[with_null_features] = X[with_null_features].fillna(0)
        if self._feature_metadata_in_real.type_map_raw:
            # TODO: Confirm this works with sparse and other feature types!
            X = X.astype(self._feature_metadata_in_real.type_map_raw)
        return X

    def _infer_features_in_full(self, X: DataFrame, y: Series = None, feature_metadata_in: FeatureMetadata = None):
        super()._infer_features_in_full(X=X, y=y, feature_metadata_in=feature_metadata_in)
        type_map_real = get_type_map_real(X[self.feature_metadata_in.get_features()])
        self._feature_metadata_in_real = FeatureMetadata(type_map_raw=type_map_real, type_group_map_special=self.feature_metadata_in.type_group_map_raw)

    def _remove_features_in(self, features):
        super()._remove_features_in(features)
        if features:
            self._feature_metadata_in_real = self._feature_metadata_in_real.remove_features(features=features)

    def print_feature_metadata_info(self):
        logger.log(20, 'Original Features (exact raw dtype, raw dtype):')
        self._feature_metadata_in_real.print_feature_metadata_full('\t', print_only_one_special=True)
        super().print_feature_metadata_info()
