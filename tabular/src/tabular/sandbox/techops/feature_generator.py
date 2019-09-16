
import pandas as pd
from pandas import DataFrame
from tabular.feature_generators.abstract_feature_generator import AbstractFeatureGenerator
# from fastai.tabular.transform import add_datepart
# from f3_grail_monty.preprocessors.lemmatize import Lemmatize
# from f3_grail_monty.sandbox.mla import mla_preprocess_v2
import copy

SHORT_DESCRIPTION = 'short_description'
DETAILS = 'details'
ID = 'ID'
ASSIGNED_DATE = 'assigned_date'
CREATE_DATE = 'create_date'
IMPACT = 'impact'
ASSIGNED_TO_GROUP = 'assigned_to_group'
FIRST_ASSIGNED_TO_GROUP = 'first_assigned_group'


BANNED_FEATURES = [
    # 'create_date_Elapsed',
    # 'assigned_date_Elapsed',
    # 'create_date_Week',
    # 'assigned_date_Week',
    # 'create_date_Day',
    # 'assigned_date_Day',
    # 'create_date_Dayofweek',
    # 'assigned_date_Dayofweek',
    # 'create_date_Month',
    # 'assigned_date_Month',
    # 'create_date_Year',
    # 'assigned_date_Year',
    # 'create_date_Dayofyear',
    # 'assigned_date_Dayofyear',
    # 'create_to_assigned_lag_log',
    # 'create_to_assigned_lag_log_bin',
    # ''

]


# TODO: short_description, make new feature with site names removed. Will improve greatly as a categorical
# TODO: Metafeature - # of top ngrams found in row
# TODO: Metafeature - # of top ngrams / # of words
# TODO: Metafeature - # of words - # of top ngrams
# TODO: Metafeature - # of 1gram / 2gram / 3gram etc.

class FeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, vectorizer):
        super().__init__()
        self.features_to_remove += BANNED_FEATURES
        self.vectorizer_raw = vectorizer
        self.vectorizer = None
        self.vectorizer_details = None
        self.nlp_enabled = vectorizer is not None

    def preprocess(self, X: DataFrame):
        X[SHORT_DESCRIPTION] = X[SHORT_DESCRIPTION].str.lower()
        X[DETAILS] = X[DETAILS].str.lower()

        X[SHORT_DESCRIPTION] = [string.replace('.', '') for string in X[SHORT_DESCRIPTION].values]
        X[DETAILS] = [string.replace('.', '') for string in X[DETAILS].values]


        # X[DETAILS] = Lemmatize.process(X, DETAILS)
        # X[DETAILS] = mla_preprocess_v2.process_text(X[DETAILS].values)
        # X[SHORT_DESCRIPTION] = mla_preprocess_v2.process_text(X[SHORT_DESCRIPTION].values)

        return X

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
        if not self.fit:
            self.features_binned += list(X_text_short_description_features.columns)
            self.features_binned += list(X_text_details_features.columns)

        X = self.preprocess(X)





        print(self.features_categorical)

        if self.nlp_enabled:
            if not self.fit:
                text_list_short_description = (list(X[SHORT_DESCRIPTION].drop_duplicates().values))  # FIXME
                text_list_details = (list(X[DETAILS].drop_duplicates().values))  # FIXME
                print('shortening short description:', len(X), len(text_list_short_description))
                print('shortening details:', len(X), len(text_list_details))
                vectorizer_raw_2 = copy.deepcopy(self.vectorizer_raw)
                self.vectorizer, _ = self.train_vectorizer(text_list_short_description, self.vectorizer_raw)


                self.vectorizer_details, _ = self.train_vectorizer(text_list_details, vectorizer_raw_2)

            text_combined_names = self.vectorizer.get_feature_names()
            text_combined_names_details = self.vectorizer_details.get_feature_names()

            # TODO: Optimize
            # tf = self.vectorizer.transform(X[SHORT_DESCRIPTION].values)
            # X_text_combined_features = pd.DataFrame(index=X.index)
            # for i, col in enumerate(text_combined_names):
            #     print(i, col)
            #     x = tf[:, i].toarray().ravel()
            #     z = pd.SparseSeries(x, fill_value=0)
            #     X_text_combined_features[col] = z
            # print('hello world')

            # TODO: Sparse matrix instead of Dense!!!
            X_text_combined_features = pd.DataFrame(self.vectorizer.transform(X[SHORT_DESCRIPTION].values).toarray())  # FIXME
            X_text_combined_features.columns = ['sd' + '.' + str(x) for x in text_combined_names]
            X_text_combined_features['sd._total_'] = X_text_combined_features.gt(0).sum(axis=1)

            # X_text_details_features['sd._frac_'] = X_text_details_features['sd._total_'] / X_text_details_features[DETAILS + '.word_count']  # FIXME
            # X_text_short_description_features['sd._rare_words_'] = X_text_short_description_features[SHORT_DESCRIPTION + '.word_count'] - X_text_short_description_features['sd._total_']
            # X_text_details_features = X_text_details_features.fillna(0)

            X_text_combined_features_details = pd.DataFrame(self.vectorizer_details.transform(X[DETAILS].values).toarray())  # FIXME
            X_text_combined_features_details.columns = ['dt' + '.' + str(x) for x in text_combined_names_details]
            X_text_combined_features_details['dt._total_'] = X_text_combined_features_details.gt(0).sum(axis=1)


        X_categoricals = X[self.features_categorical]
        X_categoricals = X_categoricals.astype('category')

        X_features = X_features.join(X_categoricals)
        X_features = X_features.join(X[self.features_to_keep_raw])

        X_features['date_concat'] = [str(x) for x in list(zip(X_features[ASSIGNED_DATE], X_features[CREATE_DATE]))]
        X_features['date_concat'] = X_features['date_concat'].astype('category')

        X_features['date_lag'] = X_features[ASSIGNED_DATE] - X_features[CREATE_DATE]

        # X_features['assigned_group_is_first'] = [x[0] == x[1] for x in zip(X_features[ASSIGNED_TO_GROUP], X_features[FIRST_ASSIGNED_TO_GROUP])]

        # print(X_features['assigned_group_is_first'])

        X_features = X_features.join(X_text_details_features)
        X_features = X_features.join(X_text_short_description_features)

        if self.nlp_enabled:
            X_features = X_features.join(X_text_combined_features)
            X_features = X_features.join(X_text_combined_features_details)

        print('transformed...')

        # print(X_features.memory_usage())

        return X_features
