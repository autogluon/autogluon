import copy
import logging

from pandas import DataFrame
from pandas.api.types import CategoricalDtype

from .abstract import AbstractFeatureGenerator
from .memory_minimize import CategoryMemoryMinimizeFeatureGenerator
from ..feature_metadata import R_CATEGORY, R_OBJECT, S_DATETIME_AS_OBJECT, S_TEXT, S_TEXT_AS_CATEGORY

logger = logging.getLogger(__name__)


# TODO: Add hashing trick if minimize_memory=True to avoid storing full original mapping
# TODO: Retroactively prune mappings if feature was removed downstream
# TODO: Have a concept of a 1:1 mapping, so you can safely remove features from features_in and features_out
#  Have a method remove_features() where each generator implements custom logic, default is just to only remove from features_out, but Category could remove from features_in + inner generators + category_map.
class CategoryFeatureGenerator(AbstractFeatureGenerator):
    """
    CategoryFeatureGenerator is used to convert object types to category types, as well as remove rare categories and optimize memory usage.
    After fitting, previously unseen categories during transform are treated as missing values.

    Parameters
    ----------
    stateful_categories : bool, default True
        If True, categories from training are applied to transformed data, and any unknown categories from input data will be treated as missing values.
        It is recommended to keep this value as True to avoid strange downstream behaviour.
    minimize_memory : bool, default True
        If True, minimizes category memory usage by converting all category values to sequential integers.
        This replaces any string data present in the categories but does not alter the behavior of models when using the category as a feature so long as the original string values are not required downstream.
        It is recommended to keep this value as True to dramatically reduce memory usage with no cost to accuracy.
    cat_order : str, default 'original'
        Determines the order in which categories are stored.
        This is important when minimize_memory is True, as the order will determine which categories are converted to which integer values.
        Valid values:
            'original' : Keep the original order. If the feature was originally an object, this is equivalent to 'alphanumeric'.
            'alphanumeric' : Sort the categories alphanumerically.
            'count' : Sort the categories by frequency (Least frequent in front with code of 0)
    minimum_cat_count : int, default None
        The minimum number of occurrences a category must have in the training data to avoid being considered a rare category.
        Rare categories are removed and treated as missing values.
        If None, no minimum count is required. This includes categories that never occur in the data but are present in the category object as possible categories.
    maximum_num_cat : int, default None
        The maximum amount of categories that can be considered non-rare.
        Sorted by occurrence count, up to the N highest count categories will be kept if maximum_num_cat=N. All others will be considered rare categories.
    **kwargs :
        Refer to AbstractFeatureGenerator documentation for details on valid key word arguments.
    """
    def __init__(self, stateful_categories=True, minimize_memory=True, cat_order='original', minimum_cat_count: int = None, maximum_num_cat: int = None, **kwargs):
        super().__init__(**kwargs)
        self._stateful_categories = stateful_categories
        if minimum_cat_count is not None and minimum_cat_count <= 1:
            minimum_cat_count = None
        if cat_order not in ['original', 'alphanumeric', 'count']:
            raise ValueError(f"cat_order must be one of {['original', 'alphanumeric', 'count']}, but was: {cat_order}")
        self.cat_order = cat_order
        self._minimum_cat_count = minimum_cat_count
        self._maximum_num_cat = maximum_num_cat
        self.category_map = None

        if minimize_memory:
            self._post_generators = [CategoryMemoryMinimizeFeatureGenerator(inplace=True)] + self._post_generators

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        if self._stateful_categories:
            X_out, self.category_map = self._generate_category_map(X=X)
        else:
            X_out = self._transform(X)
        feature_metadata_out_type_group_map_special = copy.deepcopy(self.feature_metadata_in.type_group_map_special)
        if S_TEXT in feature_metadata_out_type_group_map_special:
            text_features = feature_metadata_out_type_group_map_special.pop(S_TEXT)
            feature_metadata_out_type_group_map_special[S_TEXT_AS_CATEGORY] += [feature for feature in text_features if feature not in feature_metadata_out_type_group_map_special[S_TEXT_AS_CATEGORY]]
        return X_out, feature_metadata_out_type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._generate_features_category(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(
            valid_raw_types=[R_OBJECT, R_CATEGORY],
            invalid_special_types=[S_DATETIME_AS_OBJECT]
        )

    def _generate_features_category(self, X: DataFrame) -> DataFrame:
        if self.features_in:
            X_category = X.astype('category')
            if self.category_map is not None:
                X_category = copy.deepcopy(X_category)  # TODO: Add inplace version / parameter
                for column in self.category_map:
                    X_category[column].cat.set_categories(self.category_map[column], inplace=True)
        else:
            X_category = DataFrame(index=X.index)
        return X_category

    def _generate_category_map(self, X: DataFrame) -> (DataFrame, dict):
        if self.features_in:
            category_map = dict()
            X_category = X.astype('category')
            for column in X_category:
                if self.cat_order == 'count' or self._minimum_cat_count is not None or self._maximum_num_cat is not None:
                    rank = X_category[column].value_counts().sort_values(ascending=True)
                    if self._minimum_cat_count is not None:
                        rank = rank[rank >= self._minimum_cat_count]
                    if self._maximum_num_cat is not None:
                        rank = rank[-self._maximum_num_cat:]
                    rank = rank.reset_index()

                    category_list = list(rank['index'].values)  # category_list in 'count' order
                    if len(category_list) > 1:
                        if self.cat_order == 'original':
                            original_cat_order = list(X_category[column].cat.categories)
                            category_list = [cat for cat in original_cat_order if cat in category_list]
                        elif self.cat_order == 'alphanumeric':
                            category_list.sort()
                    X_category[column] = X_category[column].astype(CategoricalDtype(categories=category_list))  # TODO: Remove columns if all NaN after this?
                    X_category[column] = X_category[column].cat.reorder_categories(category_list)
                elif self.cat_order == 'alphanumeric':
                    category_list = list(X_category[column].cat.categories)
                    category_list.sort()
                    X_category[column] = X_category[column].astype(CategoricalDtype(categories=category_list))
                    X_category[column] = X_category[column].cat.reorder_categories(category_list)
                category_map[column] = copy.deepcopy(X_category[column].cat.categories)
            return X_category, category_map
        else:
            return DataFrame(index=X.index), None

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self.category_map:
            for feature in features:
                if feature in self.category_map:
                    self.category_map.pop(feature)

    def _more_tags(self):
        return {'feature_interactions': False}
