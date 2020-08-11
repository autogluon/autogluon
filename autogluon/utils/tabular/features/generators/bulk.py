import logging
from typing import List

import pandas as pd
from pandas import DataFrame, Series

from .abstract import AbstractFeatureGenerator
from ..feature_metadata import FeatureMetadata

logger = logging.getLogger(__name__)


class BulkFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, generators: List[AbstractFeatureGenerator], **kwargs):
        super().__init__(**kwargs)
        self.generators = generators
        self._feature_metadata_in_unused: FeatureMetadata = None  # FeatureMetadata object based on the original input features that were unused by any feature generator.

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        feature_df_list = []
        for generator in self.generators:
            feature_df_list.append(generator.fit_transform(X, feature_metadata_in=self.feature_metadata_in, **kwargs))

        self.generators = [generator for i, generator in enumerate(self.generators) if feature_df_list[i] is not None and len(feature_df_list[i].columns) > 0]
        feature_df_list = [feature_df for feature_df in feature_df_list if feature_df is not None and len(feature_df.columns) > 0]

        if self.generators:
            feature_metadata_in_generators = FeatureMetadata.join_metadatas([generator.feature_metadata_in for generator in self.generators], shared_raw_features='error_if_diff')
        else:
            feature_metadata_in_generators = FeatureMetadata(type_map_raw=dict())

        used_features_in = feature_metadata_in_generators.get_features()
        unused_features_in = [feature for feature in self.feature_metadata_in.get_features() if feature not in used_features_in]
        self._feature_metadata_in_unused = self.feature_metadata_in.keep_features(features=unused_features_in)
        self._remove_features_in(features=unused_features_in)

        if self.generators:
            feature_metadata = FeatureMetadata.join_metadatas([generator.feature_metadata for generator in self.generators], shared_raw_features='error')
        else:
            feature_metadata = FeatureMetadata(type_map_raw=dict())

        if not feature_df_list:
            X_out = DataFrame(index=X.index)
        elif len(feature_df_list) == 1:
            X_out = feature_df_list[0]
        else:
            X_out = pd.concat(feature_df_list, axis=1, ignore_index=False, copy=False)

        return X_out, feature_metadata.type_group_map_special

    def _transform(self, X):
        feature_df_list = []
        for generator in self.generators:
            feature_df_list.append(generator.transform(X))

        if not feature_df_list:
            X_out = DataFrame(index=X.index)
        elif len(feature_df_list) == 1:
            X_out = feature_df_list[0]
        else:
            X_out = pd.concat(feature_df_list, axis=1, ignore_index=False, copy=False)

        return X_out
