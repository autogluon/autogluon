import logging

from pandas import DataFrame, Series

from .identity import IdentityFeatureGenerator
from ..feature_metadata import FeatureMetadata

logger = logging.getLogger(__name__)


# TODO: Add stateful version
class CategoryFeatureGenerator(IdentityFeatureGenerator):
    def _transform(self, X: DataFrame) -> DataFrame:
        return self._generate_features_category(X)

    def _infer_features_in_from_metadata(self, X: DataFrame, y: Series = None, feature_metadata_in: FeatureMetadata = None) -> list:
        object_features = feature_metadata_in.type_group_map_raw['object']
        datetime_as_object_features = feature_metadata_in.type_group_map_special['datetime_as_object']
        object_features = [feature for feature in object_features if feature not in datetime_as_object_features]
        return object_features

    # TODO: Add stateful categorical generator, merge rare cases to an unknown value
    # TODO: What happens when training set has no unknown/rare values but test set does? What models can handle this?
    def _generate_features_category(self, X: DataFrame) -> DataFrame:
        if self.features_in:
            X_category = X.astype('category')
        else:
            X_category = DataFrame(index=X.index)
        return X_category
