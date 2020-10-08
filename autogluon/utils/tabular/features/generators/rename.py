import copy
import logging

from pandas import DataFrame

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class RenameFeatureGenerator(AbstractFeatureGenerator):
    """
    RenameFeatureGenerator renames the columns without altering their values.
    This can be used to avoid column name collisions when transforming the same feature in multiple ways, or to highlight that a feature was derived from a particular pipeline.

    Parameters
    ----------
    name_prefix : str, default None
        Name prefix to add to all output feature names.
    name_suffix : str, default None
        Name suffix to add to all output feature names.
    inplace : bool, default False
        If True, then the column names are renamed inplace without copying the input data.
        This will alter the input data outside of the scope of this function.
    **kwargs :
        Refer to AbstractFeatureGenerator documentation for details on valid key word arguments.
    """
    def __init__(self, name_prefix=None, name_suffix=None, inplace=False, **kwargs):
        super().__init__(**kwargs)
        self._name_prefix = name_prefix
        self._name_suffix = name_suffix
        self.inplace = inplace
        self._is_updated_name = None

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        column_rename_map, self._is_updated_name = self._get_renamed_features(X)
        if not self.inplace:
            X = copy.deepcopy(X)
        X.columns = [column_rename_map.get(col, col) for col in X.columns]

        feature_metadata_out = self.feature_metadata_in.rename_features(column_rename_map)
        return X, feature_metadata_out.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        if self._is_updated_name:
            if not self.inplace:
                X = copy.deepcopy(X)
            X.columns = self.features_out
        return X

    def _get_renamed_features(self, X: DataFrame) -> (DataFrame, dict):
        X_columns_orig = list(X.columns)
        X_columns_new = list(X.columns)
        if self._name_prefix:
            X_columns_new = [self._name_prefix + column for column in X_columns_new]
        if self._name_suffix:
            X_columns_new = [column + self._name_suffix for column in X_columns_new]
        if X_columns_orig != X_columns_new:
            is_updated_name = True
        else:
            is_updated_name = False
        column_rename_map = {orig: new for orig, new in zip(X_columns_orig, X_columns_new)}
        return column_rename_map, is_updated_name

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    def _more_tags(self):
        return {
            'feature_interactions': False,
            'allow_post_generators': False,  # TODO: This might not be necessary anymore
        }
