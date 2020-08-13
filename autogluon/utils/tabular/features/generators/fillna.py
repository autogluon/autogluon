import logging

import numpy as np
from pandas import DataFrame

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Add fillna_special_map, fillna_combined_map to increase options
# TODO: Add options to specify mean/median/mode for int/float
# TODO: Add fillna_features for feature specific fill values
class FillNaFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, fillna_map=None, fillna_default=np.nan, inplace=False, **kwargs):
        super().__init__(**kwargs)
        if fillna_map is None:
            fillna_map = {'object': ''}
        self.fillna_map = fillna_map
        self.fillna_default = fillna_default
        self._fillna_feature_map = None
        self.inplace = inplace

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        self._fillna_feature_map = {feature: self.fillna_map.get(self.feature_metadata_in.get_feature_type_raw(feature), self.fillna_default) for feature in self.feature_metadata_in.get_features()}
        for feature in self._fillna_feature_map:
            if self.feature_metadata_in.get_feature_type_raw(feature) == 'object' and (isinstance(self._fillna_feature_map[feature], int) or isinstance(self._fillna_feature_map[feature], float)):
                logger.warning(f'Warning: Feature {feature} has raw type of object but has {type(self._fillna_feature_map[feature])} fillna value of {self._fillna_feature_map[feature]}'
                               f', If the feature could be coerced to {type(self._fillna_feature_map[feature])}, it will be if NaN values are present, which may cause defective behavior. Please set the fill value to a string value to prevent this behavior.')
        return self._transform(X), self.feature_metadata_in.type_group_map_special

    def _transform(self, X):
        null_count = X.isnull().sum()
        with_null = null_count[null_count != 0]
        with_null_features = list(with_null.index)
        if with_null_features:
            fillna_feature_map = {feature: self._fillna_feature_map[feature] for feature in with_null_features}
            # TODO: WARNING: .fillna() will convert an object type of integers to int64 type if it could be converted to int after fillna.
            #  This is why fillna_feature_map must be used instead of self._fillna_feature_map.
            #  Furthermore, if a user specifies a fillna_map which sets integer/float values to nan's of objects, it could still convert it to int/float even with this protection.
            #  One solution is to keep track of original types and do .astype post-fillna, but this may have a significant performance hit.
            if self.inplace:
                X.fillna(fillna_feature_map, inplace=True)
            else:
                X = X.fillna(fillna_feature_map, inplace=False)
        return X
