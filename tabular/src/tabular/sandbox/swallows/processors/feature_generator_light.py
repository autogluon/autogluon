import pandas as pd
from pandas import DataFrame

from tabular.feature_generators.abstract_feature_generator import AbstractFeatureGenerator

# TODO: short_description, make new feature with site names removed. Will improve greatly as a categorical
# TODO: Metafeature - # of top ngrams found in row
# TODO: Metafeature - # of top ngrams / # of words
# TODO: Metafeature - # of words - # of top ngrams
# TODO: Metafeature - # of 1gram / 2gram / 3gram etc.

SHORT_DESCRIPTION = 'short_description'
DETAILS = 'details'
ID = 'ID'
ASSIGNED_DATE = 'assigned_date'
CREATE_DATE = 'create_date'
IMPACT = 'impact'


class FeatureGeneratorLight(AbstractFeatureGenerator):
    def __init__(self):
        super().__init__()

    # TODO: Parallelize with decorator!
    def generate_features(self, X: DataFrame):
        X_features = pd.DataFrame(index=X.index)

        for column in X.columns:
            if X[column].dtype.name == 'object':
                X[column].fillna('', inplace=True)
            else:
                X[column].fillna(-1, inplace=True)

        # TODO: Bin into 4-10 groups for the continous features, fit_transform gets mapping
        X_text_short_description_features = self.generate_text_features(X[SHORT_DESCRIPTION], SHORT_DESCRIPTION)
        X_text_details_features = self.generate_text_features(X[DETAILS], DETAILS)
        # if not self.fit:
        #     self.features_binned += list(X_text_short_description_features.columns)
        #     self.features_binned += list(X_text_details_features.columns)

        X_features = X_features.join(X_text_details_features)
        X_features = X_features.join(X_text_short_description_features)

        print('transformed...')

        # print(X_features.memory_usage())

        return X_features
