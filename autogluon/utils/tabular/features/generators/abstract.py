import logging

from ..types import get_type_map_raw
from ..feature_metadata import FeatureMetadata

logger = logging.getLogger(__name__)


class AbstractFeatureGenerator:
    def __init__(self, features_in=None):
        # TODO: Add name_prefix, name_suffix
        # TODO: Add post_generators
        # TODO: Add y to fit calls
        self._is_fit = False  # Whether the feature generator has been fit
        self.feature_metadata: FeatureMetadata = None  # FeatureMetadata object based on the processed features. Pass to models to enable advanced functionality.
        self.features_in = features_in  # Original features to use as input to feature generation
        self.features_out = None  # Final list of features after transformation

    def fit(self, X, y=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        if self._is_fit:
            raise AssertionError('FeatureGenerator is already fit.')
        if self.features_in is None:
            self.features_in = list(X.columns)
        X_out, type_family_groups_special = self._fit_transform(X[self.features_in], y=y)
        self.features_out = list(X_out.columns)
        type_map_raw = get_type_map_raw(X_out)
        self.feature_metadata = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_family_groups_special)
        self._is_fit = True
        return X_out

    def transform(self, X):
        if not self._is_fit:
            raise AssertionError('FeatureGenerator is not fit.')
        return self._transform(X[self.features_in])

    def _fit_transform(self, X, y=None):
        raise NotImplementedError

    def _transform(self, X):
        raise NotImplementedError
