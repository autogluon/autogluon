import logging

from pandas import DataFrame

from .identity import IdentityFeatureGenerator

logger = logging.getLogger(__name__)


class CategoryFeatureGenerator(IdentityFeatureGenerator):
    def _transform(self, X):
        return self._generate_features_category(X)

    # TODO: Add stateful categorical generator, merge rare cases to an unknown value
    # TODO: What happens when training set has no unknown/rare values but test set does? What models can handle this?
    def _generate_features_category(self, X: DataFrame):
        if self.features_in:
            X_category = X.astype('category')
        else:
            X_category = DataFrame(index=X.index)
        return X_category
