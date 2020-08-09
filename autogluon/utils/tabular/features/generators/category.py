import copy
import logging

from pandas import DataFrame, Series

from .identity import IdentityFeatureGenerator
from .memory_minimize import CategoryMemoryMinimizeFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Add stateful version
class CategoryFeatureGenerator(IdentityFeatureGenerator):
    def __init__(self, stateful_categories=True, minimize_memory=True, **kwargs):
        super().__init__(**kwargs)
        self._stateful_categories = stateful_categories
        self.minimize_memory = minimize_memory
        self._category_map = None

        if self.minimize_memory:
            self.post_generators = [CategoryMemoryMinimizeFeatureGenerator(inplace=True)] + self.post_generators

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        if self._stateful_categories:
            X_out, self._category_map = self._generate_category_map(X=X)
        else:
            X_out = self._transform(X)
        feature_metadata_out_type_group_map_special = copy.deepcopy(self.feature_metadata_in.type_group_map_special)
        if 'text' in feature_metadata_out_type_group_map_special:
            text_features = feature_metadata_out_type_group_map_special.pop('text')
            feature_metadata_out_type_group_map_special['text_as_category'] += [feature for feature in text_features if feature not in feature_metadata_out_type_group_map_special['text_as_category']]
        return X_out, feature_metadata_out_type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._generate_features_category(X)

    def _infer_features_in(self, X: DataFrame, y: Series = None) -> list:
        object_features = self.feature_metadata_in.type_group_map_raw['object'] + self.feature_metadata_in.type_group_map_raw['category']
        datetime_as_object_features = self.feature_metadata_in.type_group_map_special['datetime_as_object']
        object_features = [feature for feature in object_features if feature not in datetime_as_object_features]
        return object_features

    # TODO: Add stateful categorical generator, merge rare cases to an unknown value
    # TODO: What happens when training set has no unknown/rare values but test set does? What models can handle this?
    def _generate_features_category(self, X: DataFrame) -> DataFrame:
        if self.features_in:
            X_category = X.astype('category')
            if self._category_map is not None:
                X_category = copy.deepcopy(X_category)  # TODO: Add inplace version / parameter
                for column in self._category_map:
                    X_category[column].cat.set_categories(self._category_map[column], inplace=True)
        else:
            X_category = DataFrame(index=X.index)
        return X_category

    def _generate_category_map(self, X: DataFrame) -> (DataFrame, dict):
        if self.features_in:
            category_map = dict()
            X_category = X.astype('category')
            for column in X_category:
                category_map[column] = copy.deepcopy(X_category[column].cat.categories)
            return X_category, category_map
        else:
            return DataFrame(index=X.index), None
