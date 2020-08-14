import logging
from typing import List

import pandas as pd
from pandas import DataFrame, Series

from .abstract import AbstractFeatureGenerator
from ..feature_metadata import FeatureMetadata

logger = logging.getLogger(__name__)


# TODO: Add parameter to add prefix to each generator to guarantee no name collisions: 'G1_', 'G2_', etc.
class BulkFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, generators: List[List[AbstractFeatureGenerator]], pre_generators: List[AbstractFeatureGenerator] = None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(generators, list):
            generators = [[generators]]
        elif len(generators) == 0:
            raise AssertionError('generators must contain at least one AbstractFeatureGenerator.')
        generators = [generator_group if isinstance(generator_group, list) else [generator_group] for generator_group in generators]
        self.generators: List[List[AbstractFeatureGenerator]] = generators
        if pre_generators is None:
            pre_generators = []
        elif not isinstance(pre_generators, list):
            pre_generators = [pre_generators]
        pre_generators = [[pre_generator] for pre_generator in pre_generators]
        if self._post_generators is not None:
            post_generators = [[post_generator] for post_generator in self._post_generators]
            self._post_generators = []
        else:
            post_generators = []
        self.generators = pre_generators + self.generators + post_generators

        self._feature_metadata_in_unused: FeatureMetadata = None  # FeatureMetadata object based on the original input features that were unused by any feature generator.

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        feature_metadata = self.feature_metadata_in
        for i, generator_group in enumerate(self.generators):
            self.log(20, f'\tStage {i+1} Generators:')
            feature_df_list = []
            for generator in generator_group:
                generator.set_log_prefix(log_prefix=self.log_prefix + '\t\t', prepend=True)
                feature_df_list.append(generator.fit_transform(X, feature_metadata_in=feature_metadata, **kwargs))

            self.generators[i] = [generator for j, generator in enumerate(generator_group) if feature_df_list[j] is not None and len(feature_df_list[j].columns) > 0]
            feature_df_list = [feature_df for feature_df in feature_df_list if feature_df is not None and len(feature_df.columns) > 0]

            if self.generators[i]:
                feature_metadata_in_generators = FeatureMetadata.join_metadatas([generator.feature_metadata_in for generator in self.generators[i]], shared_raw_features='error_if_diff')
            else:
                feature_metadata_in_generators = FeatureMetadata(type_map_raw=dict())

            if i == 0:  # TODO: Improve this
                used_features_in = feature_metadata_in_generators.get_features()
                unused_features_in = [feature for feature in self.feature_metadata_in.get_features() if feature not in used_features_in]
                self._feature_metadata_in_unused = self.feature_metadata_in.keep_features(features=unused_features_in)
                self._remove_features_in(features=unused_features_in)

            if self.generators[i]:
                feature_metadata = FeatureMetadata.join_metadatas([generator.feature_metadata for generator in self.generators[i]], shared_raw_features='error')
            else:
                feature_metadata = FeatureMetadata(type_map_raw=dict())

            if not feature_df_list:
                X = DataFrame(index=X.index)
            elif len(feature_df_list) == 1:
                X = feature_df_list[0]
            else:
                X = pd.concat(feature_df_list, axis=1, ignore_index=False, copy=False)
        X_out = X

        return X_out, feature_metadata.type_group_map_special

    def _transform(self, X):
        for generator_group in self.generators:
            feature_df_list = []
            for generator in generator_group:
                feature_df_list.append(generator.transform(X))

            if not feature_df_list:
                X = DataFrame(index=X.index)
            elif len(feature_df_list) == 1:
                X = feature_df_list[0]
            else:
                X = pd.concat(feature_df_list, axis=1, ignore_index=False, copy=False)
        X_out = X

        return X_out
