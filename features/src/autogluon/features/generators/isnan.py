import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

from autogluon.core.features.types import R_OBJECT, S_BOOL

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class IsNanFeatureGenerator(AbstractFeatureGenerator):
    """
    Transforms features into isnull flags.

    Parameters
    ----------
    null_map : dict, default {'object': ''}
        Map which dictates the values to consider as NaN.
        Keys are the raw types of the features as in self.feature_metadata_in.type_map_raw.
        If a feature's raw type is not present in null_map, np.nan is treated as NaN.
        If a value other than np.nan is specified, np.nan is not considered NaN.
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """
    def __init__(self, null_map=None, **kwargs):
        super().__init__(**kwargs)
        if null_map is None:
            null_map = {R_OBJECT: ''}
        self.null_map = null_map
        self._null_feature_map = None

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        features = self.feature_metadata_in.get_features()
        self._null_feature_map = dict()
        for feature in features:
            feature_raw_type = self.feature_metadata_in.get_feature_type_raw(feature)
            if feature_raw_type in self.null_map:
                self._null_feature_map[feature] = self.null_map[feature_raw_type]
        X_out = self._transform(X)
        type_family_groups_special = {S_BOOL: list(X_out.columns)}
        return X_out, type_family_groups_special

    # TODO: Try returning bool type instead of uint8
    def _transform(self, X: DataFrame) -> DataFrame:
        is_nan_features = dict()
        for feature in self.features_in:
            if feature in self._null_feature_map:
                null_val = self._null_feature_map[feature]
                is_nan_features['__nan__.' + feature] = (X[feature] == null_val).astype(np.uint8)
            else:
                is_nan_features['__nan__.' + feature] = X[feature].isnull().astype(np.uint8)
        return pd.DataFrame(is_nan_features, index=X.index)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self._null_feature_map:
            for feature in features:
                if feature in self._null_feature_map:
                    self._null_feature_map.pop(feature)

    def _more_tags(self):
        return {'feature_interactions': False}
