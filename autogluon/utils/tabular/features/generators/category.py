import copy
import logging

from pandas import DataFrame, Series
from pandas.api.types import CategoricalDtype

from .identity import IdentityFeatureGenerator
from .memory_minimize import CategoryMemoryMinimizeFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Retroactively prune mappings if feature was removed downstream
# TODO: Have a concept of a 1:1 mapping, so you can safely remove features from features_in and features_out
#  Have a method remove_features() where each generator implements custom logic, default is just to only remove from features_out, but Category could remove from features_in + inner generators + category_map.
class CategoryFeatureGenerator(IdentityFeatureGenerator):
    def __init__(self, stateful_categories=True, minimize_memory=True, minimum_cat_count: int = None, maximum_num_cat: int = None, **kwargs):
        super().__init__(**kwargs)
        self._stateful_categories = stateful_categories
        if minimum_cat_count is not None and minimum_cat_count <= 1:
            minimum_cat_count = None
        self._minimum_cat_count = minimum_cat_count
        self._maximum_num_cat = maximum_num_cat
        self._category_map = None

        if minimize_memory:
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
                if self._minimum_cat_count is not None or self._maximum_num_cat is not None:
                    rank = X_category[column].value_counts().sort_values(ascending=True)
                    if self._minimum_cat_count is not None:
                        rank = rank[rank >= self._minimum_cat_count]
                    if self._maximum_num_cat is not None:
                        rank = rank[-self._maximum_num_cat:]
                    rank = rank.reset_index()

                    val_list = list(rank['index'].values)
                    X_category[column] = X_category[column].astype(CategoricalDtype(categories=val_list))  # TODO: Remove columns if all NaN after this?
                    # TODO: Check if this reorders column codes to most-frequent -> least-frequent instead of alphanumeric
                category_map[column] = copy.deepcopy(X_category[column].cat.categories)
            return X_category, category_map
        else:
            return DataFrame(index=X.index), None
