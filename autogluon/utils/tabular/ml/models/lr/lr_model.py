import logging
import re

import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS
from autogluon.utils.tabular.ml.models.abstract.abstract_model import AbstractModel
from autogluon.utils.tabular.ml.models.lr.lr_preprocessing_utils import NlpDataPreprocessor, OheFeaturesGenerator, get_one_hot_features, NumericDataPreprocessor

logger = logging.getLogger(__name__)


class LRModel(AbstractModel):

    def __init__(self, path: str, name: str, problem_type: str, objective_func, num_classes=None, hyperparameters=None, features=None,
                 feature_types_metadata=None, debug=0):
        super().__init__(path=path, name=name, problem_type=problem_type, objective_func=objective_func, hyperparameters=hyperparameters, features=features,
                         feature_types_metadata=feature_types_metadata, debug=debug)
        self.types_of_features = None
        self.pipeline = None
        self.cat_one_hot = None

    def tokenize(self, s):
        return re.split('[ ]+', s)

    def _get_types_of_features(self, df):
        """ Returns dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = ordered list of feature-names falling into each category.
            Each value is a list of feature-names corresponding to columns in original dataframe.
            TODO: ensure features with zero variance have already been removed before this function is called.
        """
        if self.types_of_features is not None:
            Warning("Attempting to _get_types_of_features for TabularNeuralNetModel, but previously already did this.")
        categorical_featnames = self.__get_feature_type_if_present('object') + self.__get_feature_type_if_present('bool')
        continuous_featnames = self.__get_feature_type_if_present('float') + self.__get_feature_type_if_present('int') + self.__get_feature_type_if_present(
            'datetime')
        language_featnames = self.feature_types_metadata['nlp']
        valid_features = categorical_featnames + continuous_featnames + language_featnames
        if len(categorical_featnames) + len(continuous_featnames) \
                + len(language_featnames) \
                != df.shape[1]:
            unknown_features = [feature for feature in df.columns if feature not in valid_features]
            df = df.drop(columns=unknown_features)
            self.features = list(df.columns)

        types_of_features = {'continuous': [], 'skewed': [], 'onehot': [], 'language': []}
        # continuous = numeric features to rescale
        # skewed = features to which we will apply power (ie. log / box-cox) transform before normalization
        # onehot = features to one-hot encode (unknown categories for these features encountered at test-time are encoded as all zeros). We one-hot encode any features encountered that only have two unique values.
        one_hot_threshold = 2 if len(language_featnames) > 0 else 10000
        for feature in self.features:
            feature_data = df[feature]
            num_unique_vals = len(feature_data.unique())
            if feature in language_featnames:
                types_of_features['language'].append(feature)
            elif feature in continuous_featnames:
                if np.abs(feature_data.skew()) > self.params['proc.skew_threshold']:
                    types_of_features['skewed'].append(feature)
                else:
                    types_of_features['continuous'].append(feature)
            elif (feature in categorical_featnames) and (num_unique_vals <= one_hot_threshold):
                types_of_features['onehot'].append(feature)
        return types_of_features

    def __get_feature_type_if_present(self, feature_type):
        """ Returns crude categorization of feature types """
        return self.feature_types_metadata[feature_type] if feature_type in self.feature_types_metadata else []

    def preprocess(self, X: DataFrame, is_train=False, vect_max_features=1000):
        X = X.copy()
        feature_types = self._get_types_of_features(X)
        logger.log(15, "Applying model-specific pre-processing")
        logger.log(15, " - input shape %s" % str(X.shape))
        if is_train:
            transformer_list = []

            if len(feature_types['language']) > 0:
                pipeline = Pipeline(steps=[
                    ("preparator", NlpDataPreprocessor(nlp_cols=feature_types['language'])),
                    ("vectorizer",
                     TfidfVectorizer(ngram_range=self.params['proc.ngram_range'], sublinear_tf=True, max_features=vect_max_features, tokenizer=self.tokenize))
                ])
                transformer_list.append(('vect', pipeline))

            if len(feature_types['onehot']) > 0:
                pipeline = Pipeline(steps=[
                    ('generator', OheFeaturesGenerator(cats_cols=feature_types['onehot'])),
                ])
                transformer_list.append(('cats', pipeline))

            if len(feature_types['continuous']) > 0:
                pipeline = Pipeline(steps=[
                    ('generator', NumericDataPreprocessor(cont_cols=feature_types['continuous'])),
                    ('imputer', SimpleImputer(strategy=self.params['proc.impute_strategy'])),
                    ('scaler', StandardScaler())
                ])
                transformer_list.append(('cont', pipeline))

            if len(feature_types['skewed']) > 0:
                pipeline = Pipeline(steps=[
                    ('generator', NumericDataPreprocessor(cont_cols=feature_types['skewed'])),
                    ('imputer', SimpleImputer(strategy=self.params['proc.impute_strategy'])),
                    ('quantile', QuantileTransformer(output_distribution='normal')),  # Or output_distribution = 'uniform'
                ])
                transformer_list.append(('skew', pipeline))

            self.pipeline = FeatureUnion(transformer_list=transformer_list)
            logger.log(15, " - fitting pre-processing pipeline")
            self.pipeline.fit(X)

        logger.log(15, " - transforming inputs using pre-processing pipeline")
        X = self.pipeline.transform(X)
        logger.log(15, " - output shape %s" % str(X.shape))

        return X

    def _set_default_params(self):
        default_params = {
            'model_type': 'LR',
            'C': 1,
            'vectorizer_dict_size': 75000,
            'proc.ngram_range': (1, 5),
            'proc.skew_threshold': 0.99,
            'proc.impute_strategy': 'median',  # strategy argument of sklearn.SimpleImputer() used to impute missing numeric values
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self, problem_type):
        spaces = {
            # 'C': Real(lower=1e-4, upper=1e5, default=1),
            # 'tokenizer': Categorical('split', 'sentencepiece')
        }
        return spaces

    def fit(self, X_train, Y_train, X_test=None, Y_test=None, time_limit=None, **kwargs):
        hyperparams = self.params.copy()

        self.cat_one_hot = get_one_hot_features(X_train)

        # See solver options here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        if self.problem_type == BINARY:
            Y_train = Y_train.astype(int).values
            solver = 'lbfgs'  # TODO use liblinear for smaller datasets
        elif self.problem_type == MULTICLASS:
            solver = 'saga'  # another option is lbfgs
        else:
            solver = 'lbfgs'

        X_train = self.preprocess(X_train, is_train=True, vect_max_features=hyperparams['vectorizer_dict_size'])

        # TODO: get seed from seeds provider
        self.model = LogisticRegression(C=hyperparams['C'], random_state=17, solver=solver, n_jobs=-1)
        self.model.fit(X_train, Y_train)

    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options=None, **kwargs):
        self.fit(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, **kwargs)
        hpo_model_performances = {self.name: self.score(X_test, Y_test)}
        hpo_results = {}
        self.save()
        hpo_models = {self.name: self.path}

        return hpo_models, hpo_model_performances, hpo_results
