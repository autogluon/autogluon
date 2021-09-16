import logging

import numpy as np
from pandas import DataFrame

from autogluon.core.features.types import R_OBJECT

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Add fillna_special_map, fillna_combined_map to increase options
# TODO: Add options to specify mean/median/mode for int/float
# TODO: Add fillna_features for feature specific fill values
class FillNaFeatureGenerator(AbstractFeatureGenerator):
    """
    Fills missing values in the data.

    Parameters
    ----------
    fillna_map : dict, default {'object': ''}
        Map which dictates the fill values of NaNs.
        Keys are the raw types of the features as in self.feature_metadata_in.type_map_raw.
        If a feature's raw type is not present in fillna_map, its NaN values are filled to fillna_default.
    fillna_default, default np.nan
        The default fillna value if the feature's raw type is not present in fillna_map.
        Be careful about setting this to anything other than np.nan, as not all raw types can handle int, float, or string values.
    inplace : bool, default False
        If True, then the NaN values are filled inplace without copying the input data.
        This will alter the input data outside of the scope of this function.
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """
    def __init__(self, fillna_map=None, fillna_default=np.nan, inplace=False, **kwargs):
        super().__init__(**kwargs)
        if fillna_map is None:
            fillna_map = {R_OBJECT: ''}
        self.fillna_map = fillna_map
        self.fillna_default = fillna_default
        self._fillna_feature_map = None
        self.inplace = inplace

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        features = self.feature_metadata_in.get_features()
        self._fillna_feature_map = dict()
        for feature in features:
            feature_raw_type = self.feature_metadata_in.get_feature_type_raw(feature)
            feature_fillna_val = self.fillna_map.get(feature_raw_type, self.fillna_default)
            if feature_fillna_val is not np.nan:
                self._fillna_feature_map[feature] = feature_fillna_val
        return self._transform(X), self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        if self._fillna_feature_map:
            if self.inplace:
                X.fillna(self._fillna_feature_map, inplace=True, downcast=False)
            else:
                X = X.fillna(self._fillna_feature_map, inplace=False, downcast=False)
        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    def _remove_features_in(self, features):
        super()._remove_features_in(features)
        if features:
            for feature in features:
                self._fillna_feature_map.pop(feature, None)

    def _more_tags(self):
        return {'feature_interactions': False}
