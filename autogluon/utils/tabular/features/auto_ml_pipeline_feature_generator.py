import copy
import logging
from collections import defaultdict

import pandas as pd
from pandas import DataFrame, Series

from .abstract_pipeline_feature_generator import AbstractPipelineFeatureGenerator
from .feature_metadata import FeatureMetadata
from .generators.category import CategoryFeatureGenerator
from .generators.text_special import TextSpecialFeatureGenerator
from .generators.identity import IdentityFeatureGenerator
from .generators.datetime import DatetimeFeatureGenerator
from .generators.text_ngram import TextNgramFeatureGenerator

logger = logging.getLogger(__name__)


class AutoMLPipelineFeatureGenerator(AbstractPipelineFeatureGenerator):
    def __init__(self, generators=None, enable_text_ngram_features=True, enable_text_special_features=True,
                 enable_categorical_features=True, enable_raw_features=True, enable_datetime_features=True,
                 vectorizer=None):
        self.enable_nlp_features = enable_text_ngram_features
        self.enable_text_special_features = enable_text_special_features
        self.enable_categorical_features = enable_categorical_features
        self.enable_raw_features = enable_raw_features
        self.enable_datetime_features = enable_datetime_features
        if generators is None:
            generators = self._get_default_generators(vectorizer=vectorizer)
        super().__init__(generators=generators)

    # TODO: Move type strings into generators, or have generators determine the proper features at fit time
    def _get_default_generators(self, vectorizer=None):
        generators = [
            (IdentityFeatureGenerator(), 'raw'),
            (CategoryFeatureGenerator(), 'category'),
            (DatetimeFeatureGenerator(), 'datetime'),
            (TextSpecialFeatureGenerator(), 'text_special'),
            (TextNgramFeatureGenerator(vectorizer=vectorizer), 'text_ngram'),
        ]
        return generators

    # TODO: Change this or remove the need for it
    def _compute_feature_transformations(self):
        """Determines which features undergo which feature transformations."""
        feature_transformations = defaultdict(list)
        if self.enable_categorical_features:
            category_features = self._feature_metadata_in.type_group_map_raw['object']
            datetime_as_object_features = set(self._feature_metadata_in.type_group_map_special['datetime_as_object'])
            category_features = [feature for feature in category_features if feature not in datetime_as_object_features]
            feature_transformations['category'] += category_features

        text_features = self._feature_metadata_in.type_group_map_special['text']
        if self.enable_text_special_features:
            feature_transformations['text_special'] += text_features
        if self.enable_nlp_features:
            feature_transformations['text_ngram'] += text_features

        datetime_features = self._feature_metadata_in.type_group_map_special['datetime_as_object'] + self._feature_metadata_in.type_group_map_raw['datetime']
        if datetime_features and self.enable_datetime_features:
            feature_transformations['datetime'] += datetime_features

        if self.enable_raw_features:
            invalid_raw_types = {'object', 'datetime'}
            invalid_special_types = {'text', 'datetime_as_object'}

            features = self._feature_metadata_in.get_features()
            for feature in features:
                feature_type_raw = self._feature_metadata_in.get_feature_type_raw(feature)
                feature_types_special = self._feature_metadata_in.get_feature_types_special(feature)
                if feature_type_raw not in invalid_raw_types and invalid_special_types.isdisjoint(feature_types_special):
                    feature_transformations['raw'].append(feature)

        return feature_transformations

    def _generate_features(self, X: DataFrame):
        if not self._is_fit:
            feature_transformations = self._compute_feature_transformations()

        X = self._preprocess(X)

        feature_df_list = []
        for generator, transformation_key in self.generators:
            if not self._is_fit:
                X_out = generator.fit_transform(X[feature_transformations[transformation_key]])
            else:
                X_out = generator.transform(X)
            feature_df_list.append(X_out)

        if not self._is_fit:
            self.generators = [generator_tuple for i, generator_tuple in enumerate(self.generators) if feature_df_list[i] is not None and len(feature_df_list[i].columns) > 0]
            feature_df_list = [feature_df for feature_df in feature_df_list if feature_df is not None and len(feature_df.columns) > 0]

        if not feature_df_list:
            X_features = DataFrame(index=X.index)
        elif len(feature_df_list) == 1:
            X_features = feature_df_list[0]
        else:
            X_features = pd.concat(feature_df_list, axis=1, ignore_index=False, copy=False)

        # TODO: Remove the need for this
        if not self._is_fit:
            if self.generators:
                self.feature_metadata = FeatureMetadata.join_metadatas([generator.feature_metadata for generator, _ in self.generators])
            else:
                self.feature_metadata = FeatureMetadata(type_map_raw=dict())

            self._features_binned += self.feature_metadata.type_group_map_special['text_special']
            if self._feature_metadata_in.type_group_map_special['text']:
                self.feature_metadata.type_group_map_special['text_as_category'] += [feature for feature in self._feature_metadata_in.type_group_map_special['text'] if feature in self.feature_metadata.type_group_map_raw['category']]

        return X_features
