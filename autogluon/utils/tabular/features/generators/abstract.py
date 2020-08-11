import logging

from pandas import DataFrame, Series

from ..feature_metadata import FeatureMetadata
from ..types import get_type_map_raw, get_type_map_real, get_type_group_map_special
from ..utils import is_useless_feature
from ...utils.savers import save_pkl

logger = logging.getLogger(__name__)


# TODO: Add option to minimize memory usage of feature names by making them integers / strings of integers
# TODO: Make logging 20 for fitting/fitted.
# TODO: Add documentation
# TODO: Add unit tests
class AbstractFeatureGenerator:
    def __init__(self, features_in: list = None, feature_metadata_in: FeatureMetadata = None, post_generators: list = None, name_prefix: str = None, name_suffix: str = None, pre_drop_useless=False, post_drop_duplicates=False):
        # TODO: Add post_generators
        self._is_fit = False  # Whether the feature generator has been fit
        self.feature_metadata_in: FeatureMetadata = feature_metadata_in  # FeatureMetadata object based on the original input features.
        self.feature_metadata: FeatureMetadata = None  # FeatureMetadata object based on the processed features. Pass to models to enable advanced functionality.
        # TODO: Consider merging feature_metadata and feature_metadata_real, have FeatureMetadata contain exact dtypes, grouped raw dtypes, and special dtypes all at once.
        self.feature_metadata_real: FeatureMetadata = None  # FeatureMetadata object based on the processed features, containing the true raw dtype information (such as int32, float64, etc.). Pass to models to enable advanced functionality.
        self.features_in = features_in  # Original features to use as input to feature generation
        self.features_out = None  # Final list of features after transformation
        self._features_out_internal = None  # Final list of features after transformation, before the feature renaming from self._get_renamed_features() is applied
        self.name_prefix = name_prefix  # Prefix added to all output feature names
        self.name_suffix = name_suffix  # Suffix added to all output feature names

        if post_generators is None:
            post_generators = []
        elif not isinstance(post_generators, list):
            post_generators = [post_generators]
        self.post_generators: list = post_generators  # TODO: Description
        if post_drop_duplicates:
            from .drop_duplicates import DropDuplicatesFeatureGenerator
            self.post_generators.append(DropDuplicatesFeatureGenerator(post_drop_duplicates=False))

        self.pre_drop_useless = pre_drop_useless  # TODO: Description
        self._useless_features_in: list = None

        self._is_updated_name = False  # If feature names have been altered by name_prefix or name_suffix

    def fit(self, X: DataFrame, **kwargs):
        self.fit_transform(X, **kwargs)

    def fit_transform(self, X: DataFrame, y: Series = None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        logger.log(15, f'Fitting {self.__class__.__name__}...')
        if self._is_fit:
            raise AssertionError(f'{self.__class__.__name__} is already fit.')
        self._infer_features_in_full(X=X, y=y, feature_metadata_in=feature_metadata_in)
        if self.pre_drop_useless:
            self._useless_features_in = self._get_useless_features(X)
            if self._useless_features_in:
                self._remove_features_in(self._useless_features_in)

        X_out, type_family_groups_special = self._fit_transform(X[self.features_in], y=y, **kwargs)

        type_map_raw = get_type_map_raw(X_out)
        if self.post_generators:
            feature_metadata = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_family_groups_special)
            X_out, self.feature_metadata, self.post_generators = self._fit_post_generators(X=X_out, y=y, feature_metadata=feature_metadata, post_generators=self.post_generators, **kwargs)
        else:
            self.feature_metadata = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_family_groups_special)
        type_map_real = get_type_map_real(X_out)
        self.features_out = list(X_out.columns)
        self.feature_metadata_real = FeatureMetadata(type_map_raw=type_map_real, type_group_map_special=self.feature_metadata.type_group_map_raw)

        self._post_fit_cleanup()
        self._features_out_internal = self.features_out.copy()
        column_rename_map, self._is_updated_name = self._get_renamed_features(X_out)
        if self._is_updated_name:
            X_out.columns = [column_rename_map.get(col, col) for col in X_out.columns]
            self._rename_features_out(column_rename_map=column_rename_map)
        self._is_fit = True
        logger.log(15, f'Fitted {self.__class__.__name__}.')
        return X_out

    def transform(self, X: DataFrame) -> DataFrame:
        if not self._is_fit:
            raise AssertionError(f'{self.__class__.__name__} is not fit.')
        try:
            X = X[self.features_in]
        except KeyError:
            missing_cols = []
            for col in self.features_in:
                if col not in X.columns:
                    missing_cols.append(col)
            raise KeyError(f'{len(missing_cols)} required columns are missing from the provided dataset. Missing columns: {missing_cols}')
        X_out = self._transform(X)
        if self.post_generators:
            X_out = self._transform_post_generators(X=X_out, post_generators=self.post_generators)
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
        feature_metadata_in_features = self.feature_metadata_in.get_features()
        features_in = [feature for feature in X.columns if feature in feature_metadata_in_features]
        return features_in

    @staticmethod
    def _infer_feature_metadata_in(X: DataFrame, y: Series = None) -> FeatureMetadata:
        type_map_raw = get_type_map_raw(X)
        type_group_map_special = get_type_group_map_special(X)
        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    @staticmethod
    def _fit_post_generators(X, y, feature_metadata, post_generators: list, **kwargs):
        for post_generator in post_generators:
            X = post_generator.fit_transform(X=X, y=y, feature_metadata_in=feature_metadata, **kwargs)
            feature_metadata = post_generator.feature_metadata
        return X, feature_metadata, post_generators

    @staticmethod
    def _transform_post_generators(X, post_generators: list):
        for post_generator in post_generators:
            X = post_generator.transform(X=X)
        return X

    def _remove_features_in(self, features):
        self.feature_metadata_in = self.feature_metadata_in.remove_features(features=features)
        self.features_in = self.feature_metadata_in.get_features()

    def _rename_features_out(self, column_rename_map: dict):
        self.feature_metadata = self.feature_metadata.rename_features(column_rename_map)
        self.feature_metadata_real = self.feature_metadata_real.rename_features(column_rename_map)
        self.features_out = [column_rename_map.get(col, col) for col in self.features_out]

    def _get_renamed_features(self, X: DataFrame) -> (DataFrame, dict):
        X_columns_orig = list(X.columns)
        X_columns_new = list(X.columns)
        if self.name_prefix:
            X_columns_new = [self.name_prefix + column for column in X_columns_new]
        if self.name_suffix:
            X_columns_new = [column + self.name_suffix for column in X_columns_new]
        if X_columns_orig != X_columns_new:
            is_updated_name = True
        else:
            is_updated_name = False
        column_rename_map = {orig: new for orig, new in zip(X_columns_orig, X_columns_new)}
        return column_rename_map, is_updated_name

    def _post_fit_cleanup(self):
        """
        Any cleanup operations after all metadata objects have been constructed, but prior to feature renaming, should be done here
        This includes removing keys from internal lists and dictionaries of features which have been removed, and deletion of any temp variables.
        """
        pass

    @staticmethod
    def _get_useless_features(X: DataFrame) -> list:
        useless_features = []
        for column in X:
            if is_useless_feature(X[column]):
                useless_features.append(column)
        return useless_features

    def print_feature_metadata_info(self):
        logger.log(20, 'Original Features (raw dtype, special dtypes):')
        self.feature_metadata_in.print_feature_metadata_full('\t')
        logger.log(20, 'Processed Features (exact raw dtype, raw dtype):')
        self.feature_metadata_real.print_feature_metadata_full('\t', print_only_one_special=True)
        logger.log(20, 'Processed Features (raw dtype, special dtypes):')
        self.feature_metadata.print_feature_metadata_full('\t')

    def save(self, path):
        save_pkl.save(path=path, object=self)
