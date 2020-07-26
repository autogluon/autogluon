import copy
import logging
from collections import defaultdict

import pandas as pd
from pandas import DataFrame, Series

from .abstract_feature_generator import AbstractFeatureGenerator
from .feature_types_metadata import FeatureTypesMetadata
from .generators.category import CategoryFeatureGenerator
from .generators.text_special import TextSpecialFeatureGenerator
from .generators.identity import IdentityFeatureGenerator
from .generators.datetime import DatetimeFeatureGenerator
from .generators.text_ngram import TextNgramFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Add verbose descriptions of each special dtype this generator can create.
class AutoMLFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, enable_text_ngram_features=True, enable_text_special_features=True,
                 enable_categorical_features=True, enable_raw_features=True, enable_datetime_features=True,
                 vectorizer=None):
        self.enable_nlp_features = enable_text_ngram_features
        self.enable_text_special_features = enable_text_special_features
        self.enable_categorical_features = enable_categorical_features
        self.enable_raw_features = enable_raw_features
        self.enable_datetime_features = enable_datetime_features
        super().__init__(generators=self._get_default_generators(vectorizer=vectorizer))

    # TODO: FIXME
    def _get_default_generators(self, vectorizer=None):
        generators = [
            (IdentityFeatureGenerator(), 'raw'),
            (CategoryFeatureGenerator(), 'category'),
            (DatetimeFeatureGenerator(), 'datetime'),
            (TextSpecialFeatureGenerator(), 'text_special'),
            (TextNgramFeatureGenerator(vectorizer=vectorizer), 'text_ngram'),
        ]
        return generators

    def _compute_feature_transformations(self):
        """Determines which features undergo which feature transformations."""
        feature_transformations = defaultdict(list)
        if self.enable_categorical_features:
            if self.feature_type_family['object']:
                feature_transformations['category'] += self.feature_type_family['object']
            if self.feature_type_family['text']:
                feature_transformations['category'] += self.feature_type_family['text']

        if self.feature_type_family['text']:
            text_features = self.feature_type_family['text']
            if self.enable_text_special_features:
                feature_transformations['text_special'] += text_features
            if self.enable_nlp_features:
                feature_transformations['text_ngram'] += text_features

        if self.feature_type_family['datetime']:
            datetime_features = self.feature_type_family['datetime']
            if self.enable_datetime_features:
                feature_transformations['datetime'] += datetime_features

        if self.enable_raw_features:
            for type_family in self.feature_type_family:
                if type_family not in ['object', 'text', 'datetime']:
                    feature_transformations['raw'] += self.feature_type_family[type_family]

        return feature_transformations

    # TODO: Parallelize with decorator!
    def _generate_features(self, X: DataFrame):
        if not self._is_fit:
            self.feature_transformations = self._compute_feature_transformations()

        X = self._preprocess(X)

        feature_df_list = []
        for generator, transformation_key in self.generators:
            if not self._is_fit:
                X_out = generator.fit_transform(X[self.feature_transformations[transformation_key]])
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
                self.feature_types_metadata = FeatureTypesMetadata.join_metadatas([generator.feature_types_metadata for generator, _ in self.generators])
            else:
                self.feature_types_metadata = FeatureTypesMetadata(feature_types_raw=defaultdict(list))
            self.feature_type_family_generated = copy.deepcopy(self.feature_types_metadata.feature_types_special)

            self.features_binned += self.feature_types_metadata.feature_types_special['text_special']
            if self.feature_type_family['text']:
                self.feature_types_metadata.feature_types_special['text_as_category'] += [feature for feature in self.feature_type_family['text'] if feature in self.feature_types_metadata.feature_types_raw['category']]

        return X_features
