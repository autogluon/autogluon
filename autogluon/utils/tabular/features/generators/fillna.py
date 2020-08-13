import logging

import numpy as np
from pandas import DataFrame

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Add fillna_special_map, fillna_combined_map to increase options
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
        return self._transform(X), self.feature_metadata_in.type_group_map_special

    def _transform(self, X):
        if self.inplace:
            X.fillna(self._fillna_feature_map, inplace=True)
        else:
            X = X.fillna(self._fillna_feature_map, inplace=False)
        return X
