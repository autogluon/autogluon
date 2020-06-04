import logging
import re

import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from .hyperparameters.parameters import get_param_baseline, get_model_params, get_default_params, INCLUDE, IGNORE, ONLY
from .hyperparameters.searchspaces import get_default_searchspace
from .lr_preprocessing_utils import NlpDataPreprocessor, OheFeaturesGenerator, NumericDataPreprocessor
from ...constants import BINARY, REGRESSION
from ....ml.models.abstract.abstract_model import AbstractModel

logger = logging.getLogger(__name__)


class LinearModel(AbstractModel):

    def __init__(self, path: str, name: str, problem_type: str, objective_func, hyperparameters=None, features=None, feature_types_metadata=None, debug=0, **kwargs):
        self.model_class, self.penalty, self.handle_text = get_model_params(problem_type, hyperparameters)
        super().__init__(path=path, name=name, problem_type=problem_type, objective_func=objective_func, hyperparameters=hyperparameters, features=features, feature_types_metadata=feature_types_metadata, debug=debug, **kwargs)

        self.types_of_features = None
        self.pipeline = None

        self.model_params, default_params = get_default_params(self.problem_type, self.penalty)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def tokenize(self, s):
        return re.split('[ ]+', s)

    def _get_types_of_features(self, df):
        """ Returns dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = ordered list of feature-names falling into each category.
            Each value is a list of feature-names corresponding to columns in original dataframe.
            TODO: ensure features with zero variance have already been removed before this function is called.
        """
        if self.types_of_features is not None:
            logger.warning("Attempting to _get_types_of_features for LRModel, but previously already did this.")
        categorical_featnames = self.__get_feature_type_if_present('object') + self.__get_feature_type_if_present('bool')
        continuous_featnames = self.__get_feature_type_if_present('float') + self.__get_feature_type_if_present('int') + self.__get_feature_type_if_present(
            'datetime')
        language_featnames = self.feature_types_metadata['nlp']
        valid_features = categorical_featnames + continuous_featnames + language_featnames
        if len(categorical_featnames) + len(continuous_featnames) + len(language_featnames) != df.shape[1]:
            unknown_features = [feature for feature in df.columns if feature not in valid_features]
            df = df.drop(columns=unknown_features)
        self.features = list(df.columns)

        types_of_features = {'continuous': [], 'skewed': [], 'onehot': [], 'language': []}
        return self._select_features(df, types_of_features, categorical_featnames, language_featnames, continuous_featnames)

    def _select_features(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
        features_seclector = {
            INCLUDE: self._select_features_handle_text_include,
            ONLY: self._select_features_handle_text_only,
            IGNORE: self._select_features_handle_text_ignore,
        }.get(self.handle_text, self._select_features_handle_text_ignore)
        return features_seclector(df, types_of_features, categorical_featnames, language_featnames, continuous_featnames)

    def __get_feature_type_if_present(self, feature_type):
        """ Returns crude categorization of feature types """
        return self.feature_types_metadata[feature_type] if feature_type in self.feature_types_metadata else []

    # TODO: handle collinear features - they will impact results quality
    def preprocess(self, X: DataFrame, is_train=False, vect_max_features=1000, model_specific_preprocessing=False):
        if model_specific_preprocessing:  # This is hack to work-around pre-processing caching in bagging/stacker models
            X = X.copy()
            if is_train:
                feature_types = self._get_types_of_features(X)
                self.preprocess_train(X, feature_types, vect_max_features)
            X = self.pipeline.transform(X)

        return X

    def preprocess_train(self, X, feature_types, vect_max_features):
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
        self.pipeline.fit(X)

    def _set_default_params(self):
        for param, val in get_param_baseline().items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self, problem_type):
        return get_default_searchspace(problem_type)

    # TODO: It could be possible to adaptively set max_iter [1] to approximately respect time_limit based on sample-size, feature-dimensionality, and the solver used.
    #  [1] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#examples-using-sklearn-linear-model-logisticregression
    def fit(self, X_train, Y_train, X_test=None, Y_test=None, time_limit=None, **kwargs):
        hyperparams = self.params.copy()

        if self.problem_type == BINARY:
            Y_train = Y_train.astype(int).values

        X_train = self.preprocess(X_train, is_train=True, vect_max_features=hyperparams['vectorizer_dict_size'], model_specific_preprocessing=True)

        params = {k: v for k, v in self.params.items() if k in self.model_params}

        # Ridge/Lasso are using alpha instead of C, which is C^-1
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
        if self.problem_type == REGRESSION:
            # For numerical reasons, using alpha = 0 with the Lasso object is not advised, so we add epsilon
            params['alpha'] = 1 / (params['C'] if params['C'] != 0 else 1e-8)
            params.pop('C', None)

        model = self.model_class(**params)

        logger.log(15, f'Training Model with the following hyperparameter settings:')
        logger.log(15, model)

        self.model = model.fit(X_train, Y_train)

    def predict_proba(self, X, preprocess=True):
        X = self.preprocess(X, is_train=False, model_specific_preprocessing=True)
        return super().predict_proba(X, preprocess=False)

    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options=None, **kwargs):
        self.fit(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, **kwargs)
        hpo_model_performances = {self.name: self.score(X_test, Y_test)}
        hpo_results = {}
        self.save()
        hpo_models = {self.name: self.path}

        return hpo_models, hpo_model_performances, hpo_results

    def get_info(self):
        # TODO: All AG-Tabular models now offer a get_info method:
        # https://github.com/awslabs/autogluon/blob/master/autogluon/utils/tabular/ml/models/abstract/abstract_model.py#L474
        # dict of weights?
        return super().get_info()

    def _select_features_handle_text_include(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
        # continuous = numeric features to rescale
        # skewed = features to which we will apply power (ie. log / box-cox) transform before normalization
        # onehot = features to one-hot encode (unknown categories for these features encountered at test-time are encoded as all zeros). We one-hot encode any features encountered that only have two unique values.
        one_hot_threshold = 10000  # FIXME research memory constraints
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

    def _select_features_handle_text_only(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
        for feature in self.features:
            if feature in language_featnames:
                types_of_features['language'].append(feature)
        return types_of_features

    def _select_features_handle_text_ignore(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
        # continuous = numeric features to rescale
        # skewed = features to which we will apply power (ie. log / box-cox) transform before normalization
        # onehot = features to one-hot encode (unknown categories for these features encountered at test-time are encoded as all zeros). We one-hot encode any features encountered that only have two unique values.
        one_hot_threshold = 10000  # FIXME research memory constraints
        for feature in self.features:
            feature_data = df[feature]
            num_unique_vals = len(feature_data.unique())
            if feature in continuous_featnames:
                if '__nlp__' in feature:
                    continue
                if np.abs(feature_data.skew()) > self.params['proc.skew_threshold']:
                    types_of_features['skewed'].append(feature)
                else:
                    types_of_features['continuous'].append(feature)
            elif (feature in categorical_featnames) and (num_unique_vals <= one_hot_threshold):
                types_of_features['onehot'].append(feature)
        return types_of_features
