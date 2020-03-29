import logging
import re

import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.utils.tabular.ml.models.abstract.abstract_model import AbstractModel
from autogluon.utils.tabular.ml.models.lr.lr_preprocessing_utils import NlpDataPreprocessor, OheFeaturesGenerator, NumericDataPreprocessor

logger = logging.getLogger(__name__)


class AbstractLinearModel(AbstractModel):

    def __init__(self, path: str, name: str, problem_type: str, objective_func, num_classes=None, hyperparameters=None, features=None,
                 feature_types_metadata=None, debug=0, regression_option='ridge', **kwargs):
        super().__init__(path=path, name=name, problem_type=problem_type, objective_func=objective_func, hyperparameters=hyperparameters, features=features,
                         feature_types_metadata=feature_types_metadata, debug=debug)
        self.types_of_features = None
        self.regression_option = regression_option
        self.pipeline = None

        if self.problem_type == REGRESSION:
            self._get_regression_model(kwargs)
        else:
            self._model_type = LogisticRegression

        self.model_params = None
        self.set_default_params()

    def _get_regression_model(self):
        if self.regression_option == 'ridge':
            self._model_type = Ridge
        elif self.regression_option == 'lasso':
            self._model_type = Lasso
        else:
            logger.warning('Unknown value for regression_option {} - supported types are [ridge, lasso] - falling back to ridge'.format(self.regression_option))
            self.regression_option = 'ridge'
            self._model_type = Ridge

    def set_default_params(self):
        # TODO: get seed from seeds provider
        if self.problem_type == REGRESSION:
            default_params = {'C': None, 'random_state': 0, 'fit_intercept': True}
            if self.regression_option == 'ridge':
                default_params['solver'] = 'auto'
        else:
            default_params = {'C': None, 'random_state': 0, 'solver': self._get_solver(), 'n_jobs': -1, 'fit_intercept': True}
        self.model_params = list(default_params.keys())
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
        raise NotImplementedError()

    def __get_feature_type_if_present(self, feature_type):
        """ Returns crude categorization of feature types """
        return self.feature_types_metadata[feature_type] if feature_type in self.feature_types_metadata else []

    def _get_solver(self):
        if self.problem_type == BINARY:
            solver = 'lbfgs'  # TODO use liblinear for smaller datasets
        elif self.problem_type == MULTICLASS:
            solver = 'saga'  # another option is lbfgs
        else:
            solver = 'lbfgs'
        return solver

    # TODO: handle collinear features - they will impact results quality
    def preprocess(self, X: DataFrame, is_train=False, vect_max_features=1000):
        X = X.copy()
        feature_types = self._get_types_of_features(X)
        if is_train:
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
        default_params = {
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

    # TODO: It could be possible to adaptively set max_iter [1] to approximately respect time_limit based on sample-size, feature-dimensionality, and the solver used.
    #  [1] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#examples-using-sklearn-linear-model-logisticregression
    def fit(self, X_train, Y_train, X_test=None, Y_test=None, time_limit=None, **kwargs):
        hyperparams = self.params.copy()

        if self.problem_type == BINARY:
            Y_train = Y_train.astype(int).values

        X_train = self.preprocess(X_train, is_train=True, vect_max_features=hyperparams['vectorizer_dict_size'])

        params = {k: v for k, v in self.params.items() if k in self.model_params}

        # Ridge/Lasso are using alpha instead of C, which is C^-1
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
        if self.problem_type == REGRESSION:
            # For numerical reasons, using alpha = 0 with the Lasso object is not advised, so we add epsilon
            params['alpha'] = 1 / (params['C'] if params['C'] != 0 else 1e-8)
            params.pop('C', None)

        model = self._model_type(**params)
        self.model = model.fit(X_train, Y_train)

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


class LinearModel(AbstractLinearModel):
    def _select_features(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
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


class LinearModelOnlyTextFeatures(AbstractLinearModel):
    def _select_features(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
        for feature in self.features:
            if feature in language_featnames:
                types_of_features['language'].append(feature)
        return types_of_features


class LinearModelNoTextFeatures(AbstractLinearModel):
    def _select_features(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
        # continuous = numeric features to rescale
        # skewed = features to which we will apply power (ie. log / box-cox) transform before normalization
        # onehot = features to one-hot encode (unknown categories for these features encountered at test-time are encoded as all zeros). We one-hot encode any features encountered that only have two unique values.
        one_hot_threshold = 2 if len(language_featnames) > 0 else 10000
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
