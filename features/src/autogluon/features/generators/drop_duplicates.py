import logging
from typing import Union
from collections import defaultdict

from pandas import DataFrame

from autogluon.core.features.types import R_INT, R_FLOAT

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Not necessary to exist after fitting, can just update outer context feature_out/feature_in and then delete this
class DropDuplicatesFeatureGenerator(AbstractFeatureGenerator):
    """
    Drops features which are exact duplicates of other features, leaving only one instance of the data.

    Parameters
    ----------
    sample_size_init : int, default 1000
        The number of rows to sample when doing an initial filter of duplicate feature candidates.
        Usually, the majority of features can be filtered out using this smaller amount of rows which greatly speeds up the computation of the final check.
        If None or greater than the number of rows, no initial filter will occur. This may increase the time to fit immensely for large datasets.
    sample_size_final : int, default 20000
        The number of rows to sample when doing the final filter to determine duplicate features.
        This theoretically can lead to features that are very nearly duplicates but not exact duplicates being removed, but should be near impossible in practice.
        If None or greater than the number of rows, will perform exact duplicate detection (most expensive).
        It is recommend to keep this value below 100000 to maintain reasonable fit times.
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """
    def __init__(self, sample_size_init=1000, sample_size_final=20000, **kwargs):
        super().__init__(**kwargs)
        self.sample_size_init = sample_size_init
        self.sample_size_final = sample_size_final

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        if self.sample_size_init is not None and len(X) > self.sample_size_init:
            features_to_check = self._drop_duplicate_features(X, self.feature_metadata_in, keep=False, sample_size=self.sample_size_init)
            X_candidates = X[features_to_check]
        else:
            X_candidates = X
        features_to_drop = self._drop_duplicate_features(X_candidates, self.feature_metadata_in, sample_size=self.sample_size_final)
        self._remove_features_in(features_to_drop)
        if features_to_drop:
            self._log(15, f'\t{len(features_to_drop)} duplicate columns removed: {features_to_drop}')
        X_out = X[self.features_in]
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    # TODO: optimize categorical/object handling
    @staticmethod
    def _drop_duplicate_features(X: DataFrame, feature_metadata_in, keep: Union[str, bool] = 'first', sample_size=None) -> list:
        if sample_size is not None and len(X) > sample_size:
            X = X.sample(sample_size, random_state=0)
        feature_sum_map = defaultdict(list)
        for feature in feature_metadata_in.get_features(valid_raw_types=[R_INT, R_FLOAT]):
            if feature in X:
                feature_sum_map[round(X[feature].sum(), 2)].append(feature)

        features_to_keep = []
        features_to_remove = []
        for key in feature_sum_map:
            if len(feature_sum_map[key]) <= 1:
                features_to_keep += feature_sum_map[key]
            else:
                features_to_keep += list(X[feature_sum_map[key]].T.drop_duplicates(keep=keep).T.columns)
                features_to_remove += [feature for feature in feature_sum_map[key] if feature not in features_to_keep]

        if features_to_keep or features_to_remove:
            X = X.drop(columns=features_to_keep + features_to_remove)
        if len(X.columns) > 0:
            X_without_dups = X.T.drop_duplicates(keep=keep).T
            columns_orig = X.columns.values
            columns_new = X_without_dups.columns.values
            features_to_remove += [column for column in columns_orig if column not in columns_new]

        return features_to_remove

    def _more_tags(self):
        return {'feature_interactions': False}
