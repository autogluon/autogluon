import logging

from pandas import DataFrame, Series

from ..types import get_type_map_raw, get_type_map_real, get_type_group_map_special
from ..feature_metadata import FeatureMetadata
from ...utils.savers import save_pkl

logger = logging.getLogger(__name__)


# TODO: Add documentation
# TODO: Add unit tests
class AbstractFeatureGenerator:
    def __init__(self, features_in: list = None, feature_metadata_in: FeatureMetadata = None, name_prefix: str = None, name_suffix: str = None):
        # TODO: Add post_generators
        self._is_fit = False  # Whether the feature generator has been fit
        self.feature_metadata_in: FeatureMetadata = feature_metadata_in  # FeatureMetadata object based on the original input features.
        self.feature_metadata: FeatureMetadata = None  # FeatureMetadata object based on the processed features. Pass to models to enable advanced functionality.
        # TODO: Consider merging feature_metadata and feature_metadata_real, have FeatureMetadata contain exact dtypes, grouped raw dtypes, and special dtypes all at once.
        self.feature_metadata_real: FeatureMetadata = None  # FeatureMetadata object based on the processed features, containing the true raw dtype information (such as int32, float64, etc.). Pass to models to enable advanced functionality.
        self.features_in = features_in  # Original features to use as input to feature generation
        self.features_out = None  # Final list of features after transformation
        self.name_prefix = name_prefix  # Prefix added to all output feature names
        self.name_suffix = name_suffix  # Suffix added to all output feature names

        self._is_updated_name = False  # If feature names have been altered by name_prefix or name_suffix

    def fit(self, X: DataFrame, **kwargs):
        self.fit_transform(X, **kwargs)

    def fit_transform(self, X: DataFrame, y: Series = None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        if self._is_fit:
            raise AssertionError('FeatureGenerator is already fit.')
        self._infer_features_in_full(X=X, y=y, feature_metadata_in=feature_metadata_in)
        X_out, type_family_groups_special = self._fit_transform(X[self.features_in], y=y, **kwargs)
        X_out, type_family_groups_special = self._update_feature_names(X_out, type_family_groups_special)
        self.features_out = list(X_out.columns)
        type_map_real = get_type_map_real(X_out)
        type_map_raw = get_type_map_raw(X_out)
        self.feature_metadata = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_family_groups_special)
        self.feature_metadata_real = FeatureMetadata(type_map_raw=type_map_real, type_group_map_special=self.feature_metadata.type_group_map_raw)
        self._is_fit = True
        return X_out

    def transform(self, X: DataFrame) -> DataFrame:
        if not self._is_fit:
            raise AssertionError('FeatureGenerator is not fit.')
        try:
            X = X[self.features_in]
        except KeyError:
            missing_cols = []
            for col in self.features_in:
                if col not in X.columns:
                    missing_cols.append(col)
            raise KeyError(f'{len(missing_cols)} required columns are missing from the provided dataset. Missing columns: {missing_cols}')
        X_out = self._transform(X)
        if self._is_updated_name:
            X_out.columns = self.features_out
        return X_out

    # TODO: feature_metadata_in as parameter?
    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        raise NotImplementedError

    def _transform(self, X: DataFrame) -> DataFrame:
        raise NotImplementedError

    def _infer_features_in_full(self, X: DataFrame, y: Series = None, feature_metadata_in: FeatureMetadata = None):
        if self.feature_metadata_in is None:
            self.feature_metadata_in = feature_metadata_in
        elif feature_metadata_in is not None:
            logger.warning('Warning: feature_metadata_in passed as input to fit_transform, but self.feature_metadata_in was already set. Ignoring feature_metadata_in.')
        if self.feature_metadata_in is None:
            logger.log(20, f'feature_metadata_in was not set in {self.__class__.__name__}, inferring feature_metadata_in based on data. Specify feature_metadata_in to control the special dtypes of the input data.')
            self.feature_metadata_in = self._infer_feature_metadata_in(X=X, y=y)
        if self.features_in is None:
            self.features_in = self._infer_features_in(X, y=y)
        self.feature_metadata_in = self.feature_metadata_in.keep_features(features=self.features_in)

    # TODO: Find way to increase flexibility here, possibly through init args
    def _infer_features_in(self, X: DataFrame, y: Series = None) -> list:
        return list(X.columns)

    @staticmethod
    def _infer_feature_metadata_in(X: DataFrame, y: Series = None) -> FeatureMetadata:
        type_map_raw = get_type_map_raw(X)
        type_group_map_special = get_type_group_map_special(X)
        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    def _update_feature_names(self, X: DataFrame, type_family_groups: dict) -> (DataFrame, dict):
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

    def print_feature_metadata_info(self):
        logger.log(20, 'Original Features (raw dtype, special dtypes):')
        self.feature_metadata_in.print_feature_metadata_full('\t')
        logger.log(20, 'Processed Features (exact raw dtype, raw dtype):')
        self.feature_metadata_real.print_feature_metadata_full('\t')
        logger.log(20, 'Processed Features (raw dtype, special dtypes):')
        self.feature_metadata.print_feature_metadata_full('\t')

    def save(self, path):
        save_pkl.save(path=path, object=self)
