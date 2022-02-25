import logging
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from autogluon.common.features.types import R_BOOL, R_INT, R_FLOAT, R_CATEGORY, R_OBJECT, S_TEXT_AS_CATEGORY
from autogluon.core.constants import BINARY, REGRESSION

from .hyperparameters.parameters import get_param_baseline, INCLUDE, IGNORE, ONLY, _get_solver, preprocess_params_set
from .hyperparameters.searchspaces import get_default_searchspace
from .lr_preprocessing_utils import NlpDataPreprocessor, OheFeaturesGenerator, NumericDataPreprocessor
from autogluon.core.models import AbstractModel

logger = logging.getLogger(__name__)


# TODO: Can Bagged LinearModels be combined during inference to 1 model by averaging their weights?
#  What about just always using refit_full model? Should we even bag at all? Do we care that its slightly overfit?
class LinearModel(AbstractModel):
    """
    Linear model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    Model backend differs depending on problem_type:

        'binary' & 'multiclass': https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        'regression': https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = None

    def _get_model_type(self):
        penalty = self.params.get('penalty', 'L2')
        if self.params_aux.get('use_daal', False):
            # Disabled by default until more testing is done, appears to give 20x training speedup when enabled
            try:
                # TODO: Add more granular switch, currently this affects all future LR models even if they had `use_daal=False`
                from sklearnex import patch_sklearn
                patch_sklearn("ridge")
                patch_sklearn("lasso")
                patch_sklearn("logistic")
                logger.log(15, '\tUsing daal4py LR backend...')
            except:
                pass
        from sklearn.linear_model import LogisticRegression, Ridge, Lasso
        if self.problem_type == REGRESSION:
            if penalty == 'L2':
                model_type = Ridge
            elif penalty == 'L1':
                model_type = Lasso
            else:
                raise AssertionError(f'Unknown value for penalty "{penalty}" - supported types are ["L1", "L2"]')
        else:
            model_type = LogisticRegression
        return model_type

    def _tokenize(self, s):
        return re.split('[ ]+', s)

    def _get_types_of_features(self, df):
        """ Returns dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = ordered list of feature-names falling into each category.
            Each value is a list of feature-names corresponding to columns in original dataframe.
            TODO: ensure features with zero variance have already been removed before this function is called.
        """
        feature_types = self._feature_metadata.get_type_group_map_raw()

        categorical_featnames = feature_types[R_CATEGORY] + feature_types[R_OBJECT] + feature_types[R_BOOL]
        continuous_featnames = feature_types[R_FLOAT] + feature_types[R_INT]  # + self.__get_feature_type_if_present('datetime')
        language_featnames = []  # TODO: Disabled currently, have to pass raw text data features here to function properly
        valid_features = categorical_featnames + continuous_featnames + language_featnames
        if len(categorical_featnames) + len(continuous_featnames) + len(language_featnames) != df.shape[1]:
            unknown_features = [feature for feature in df.columns if feature not in valid_features]
            df = df.drop(columns=unknown_features)
        self._features_internal = list(df.columns)  # FIXME: Don't edit _features_internal

        types_of_features = {'continuous': [], 'skewed': [], 'onehot': [], 'language': []}
        return self._select_features(df, types_of_features, categorical_featnames, language_featnames, continuous_featnames)

    def _select_features(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
        features_selector = {
            INCLUDE: self._select_features_handle_text_include,
            ONLY: self._select_features_handle_text_only,
            IGNORE: self._select_features_handle_text_ignore,
        }.get(self.params.get('handle_text', IGNORE), self._select_features_handle_text_ignore)
        return features_selector(df, types_of_features, categorical_featnames, language_featnames, continuous_featnames)

    # TODO: handle collinear features - they will impact results quality
    def _preprocess(self, X, is_train=False, **kwargs):
        if is_train:
            feature_types = self._get_types_of_features(X)
            X = self._preprocess_train(X, feature_types, self.params['vectorizer_dict_size'])
        else:
            X = self._pipeline.transform(X)
        return X

    def _preprocess_train(self, X, feature_types, vect_max_features):
        transformer_list = []
        if len(feature_types['language']) > 0:
            pipeline = Pipeline(steps=[
                ("preparator", NlpDataPreprocessor(nlp_cols=feature_types['language'])),
                ("vectorizer", TfidfVectorizer(ngram_range=self.params['proc.ngram_range'], sublinear_tf=True, max_features=vect_max_features, tokenizer=self._tokenize)),
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
        self._pipeline = FeatureUnion(transformer_list=transformer_list)
        return self._pipeline.fit_transform(X)

    def _set_default_params(self):
        default_params = {'random_state': 0, 'fit_intercept': True}
        if self.problem_type != REGRESSION:
            default_params.update({'solver': _get_solver(self.problem_type)})
        default_params.update(get_param_baseline())
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return get_default_searchspace(self.problem_type)

    # TODO: It could be possible to adaptively set max_iter [1] to approximately respect time_limit based on sample-size, feature-dimensionality, and the solver used.
    #  [1] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#examples-using-sklearn-linear-model-logisticregression
    def _fit(self,
             X,
             y,
             num_cpus=-1,
             sample_weight=None,
             **kwargs):
        X = self.preprocess(X, is_train=True)
        if self.problem_type == BINARY:
            y = y.astype(int).values

        params = {k: v for k, v in self.params.items() if k not in preprocess_params_set}
        if 'n_jobs' not in params:
            if self.problem_type != REGRESSION:
                params['n_jobs'] = num_cpus

        # Ridge/Lasso are using alpha instead of C, which is C^-1
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
        if self.problem_type == REGRESSION and 'alpha' not in params:
            # For numerical reasons, using alpha = 0 with the Lasso object is not advised, so we add epsilon
            params['alpha'] = 1 / (params['C'] if params['C'] != 0 else 1e-8)
            params.pop('C', None)

        # TODO: copy_X=True currently set during regression problem type, could potentially set to False to avoid unnecessary data copy.
        model_cls = self._get_model_type()
        model = model_cls(**params)

        logger.log(15, f'Training Model with the following hyperparameter settings:')
        logger.log(15, model)

        if sample_weight is None:
            self.model = model.fit(X, y)  # Necessary for rapids models
        else:
            self.model = model.fit(X, y, sample_weight=sample_weight)

    def _select_features_handle_text_include(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
        # continuous = numeric features to rescale
        # skewed = features to which we will apply power (ie. log / box-cox) transform before normalization
        # onehot = features to one-hot encode (unknown categories for these features encountered at test-time are encoded as all zeros). We one-hot encode any features encountered that only have two unique values.
        one_hot_threshold = 10000  # FIXME research memory constraints
        for feature in self._features_internal:
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
        for feature in self._features_internal:
            if feature in language_featnames:
                types_of_features['language'].append(feature)
        return types_of_features

    def _select_features_handle_text_ignore(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
        # continuous = numeric features to rescale
        # skewed = features to which we will apply power (ie. log / box-cox) transform before normalization
        # onehot = features to one-hot encode (unknown categories for these features encountered at test-time are encoded as all zeros). We one-hot encode any features encountered that only have two unique values.
        one_hot_threshold = 10000  # FIXME research memory constraints
        for feature in self._features_internal:
            feature_data = df[feature]
            num_unique_vals = len(feature_data.unique())
            if feature in continuous_featnames:
                if np.abs(feature_data.skew()) > self.params['proc.skew_threshold']:
                    types_of_features['skewed'].append(feature)
                else:
                    types_of_features['continuous'].append(feature)
            elif (feature in categorical_featnames) and (num_unique_vals <= one_hot_threshold):
                types_of_features['onehot'].append(feature)
        return types_of_features

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
            ignored_type_group_special=[S_TEXT_AS_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
