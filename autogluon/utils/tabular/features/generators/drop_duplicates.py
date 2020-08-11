import logging

from pandas import DataFrame

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Not necessary to exist after fitting, can just update outer context feature_out/feature_in and then delete this
class DropDuplicatesFeatureGenerator(AbstractFeatureGenerator):
    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        features_to_drop = self._drop_duplicate_features(X)
        self._remove_features_in(features_to_drop)
        X_out = X[self.features_in]
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X):
        return X

    # TODO: optimize by not considering columns with unique sums/means
    # TODO: Multithread?
    @staticmethod
    def _drop_duplicate_features(X: DataFrame) -> list:
        X_without_dups = X.T.drop_duplicates().T
        logger.debug(f"X_without_dups.shape: {X_without_dups.shape}")

        columns_orig = X.columns.values
        columns_new = X_without_dups.columns.values
        columns_removed = [column for column in columns_orig if column not in columns_new]

        logger.log(15, 'Warning: duplicate columns removed ')
        logger.log(15, columns_removed)
        logger.log(15, f'Removed {len(columns_removed)} duplicate columns before training models')

        return columns_removed
