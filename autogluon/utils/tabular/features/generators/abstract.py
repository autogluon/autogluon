import logging

from ..types import get_type_map_raw
from ..feature_metadata import FeatureMetadata

logger = logging.getLogger(__name__)


# TODO: Add documentation
# TODO: Add unit tests
class AbstractFeatureGenerator:
    def __init__(self, features_in=None, name_prefix=None, name_suffix=None):
        # TODO: Add post_generators
        self._is_fit = False  # Whether the feature generator has been fit
        self.feature_metadata: FeatureMetadata = None  # FeatureMetadata object based on the processed features. Pass to models to enable advanced functionality.
        self.features_in = features_in  # Original features to use as input to feature generation
        self.features_out = None  # Final list of features after transformation
        self.name_prefix = name_prefix  # Prefix added to all output feature names
        self.name_suffix = name_suffix  # Suffix added to all output feature names

        self._is_updated_name = False  # If feature names have been altered by name_prefix or name_suffix

    def fit(self, X, y=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        if self._is_fit:
            raise AssertionError('FeatureGenerator is already fit.')
        if self.features_in is None:
            self.features_in = list(X.columns)
        X_out, type_family_groups_special = self._fit_transform(X[self.features_in], y=y)
        X_out, type_family_groups_special = self._update_feature_names(X_out, type_family_groups_special)
        self.features_out = list(X_out.columns)
        type_map_raw = get_type_map_raw(X_out)
        self.feature_metadata = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_family_groups_special)
        self._is_fit = True
        return X_out

    def transform(self, X):
        if not self._is_fit:
            raise AssertionError('FeatureGenerator is not fit.')
        X_out = self._transform(X[self.features_in])
        if self._is_updated_name:
            X_out.columns = self.features_out
        return X_out

    def _fit_transform(self, X, y=None):
        raise NotImplementedError

    def _transform(self, X):
        raise NotImplementedError

    def _update_feature_names(self, X, type_family_groups):
        X_columns_orig = list(X.columns)
        if self.name_prefix:
            X.columns = [self.name_prefix + column for column in X.columns]
            if type_family_groups:
                for type in type_family_groups:
                    type_family_groups[type] = [self.name_prefix + feature for feature in type_family_groups[type]]
        if self.name_suffix:
            X.columns = [column + self.name_suffix for column in X.columns]
            if type_family_groups:
                for type in type_family_groups:
                    type_family_groups[type] = [feature + self.name_suffix for feature in type_family_groups[type]]
        if X_columns_orig != list(X.columns):
            self._is_updated_name = True
        return X, type_family_groups
