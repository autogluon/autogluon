
import pandas as pd
from pandas import DataFrame
from tabular.feature_generators.abstract_feature_generator import AbstractFeatureGenerator

ITEM_NAME = 'item_name'


class FeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, vectorizer):
        super().__init__()
        self.vectorizer_raw = vectorizer
        self.vectorizer = None

    def preprocess(self, X: DataFrame):
        X[ITEM_NAME] = X[ITEM_NAME].str.lower()
        return X

    def generate_features(self, X: DataFrame):
        X = X.reset_index(drop=True)
        X = X.fillna('')

        X_text_item_name_features = self.generate_text_features(X[ITEM_NAME], ITEM_NAME)

        X = self.preprocess(X)

        if not self.fit:
            text_list = (list(X[ITEM_NAME].values))
            self.vectorizer = self.train_vectorizer(text_list, self.vectorizer_raw)

        text_combined_names = self.vectorizer.get_feature_names()

        X_text_combined_features = pd.DataFrame(self.vectorizer.transform(X[ITEM_NAME].values).toarray())
        X_text_combined_features.columns = [ITEM_NAME + '.' + str(x) for x in text_combined_names]

        X_features = X_text_combined_features
        X_features = X_features.join(X_text_item_name_features)

        return X_features
