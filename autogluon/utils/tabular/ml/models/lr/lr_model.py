import re
import time

from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS
from autogluon.utils.tabular.ml.models.abstract.abstract_model import AbstractModel
from autogluon.utils.tabular.ml.models.lr.lr_preprocessing_utils import NlpDataPreprocessor, OheFeaturesGenerator, get_one_hot_features, NumericDataPreprocessor


class LRModel(AbstractModel):

    def __init__(self, path: str, name: str, problem_type: str, objective_func, num_classes=None, hyperparameters=None, features=None,
                 feature_types_metadata=None, debug=0):
        super().__init__(path=path, name=name, problem_type=problem_type, objective_func=objective_func, hyperparameters=hyperparameters, features=features,
                         feature_types_metadata=feature_types_metadata, debug=debug)
        self.pipeline = None
        self.cat_one_hot = None

    def tokenize(self, s):
        return re.split('[ ]+', s)

    def preprocess(self, X: DataFrame, is_train=False, vect_max_features=1000):
        X = X.copy()
        nlp_cols = self.feature_types_metadata['nlp']
        cats = self.feature_types_metadata['object']
        cats = [c for c in cats if c in self.cat_one_hot]

        cont = self.feature_types_metadata['int'] + self.feature_types_metadata['float']
        cont = [c for c in cont if '__nlp__' not in c]

        X[cont] = X[cont].fillna(0)

        # TODO: reuse the code from NNs - imputation and quantile transform
        if is_train:
            vectorizer_pipeline = Pipeline(steps=[
                ("preparator", NlpDataPreprocessor(nlp_cols=nlp_cols)),
                ("vectorizer", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, max_features=vect_max_features, tokenizer=self.tokenize))
            ])

            cats_pipeline = Pipeline(steps=[
                ('generator', OheFeaturesGenerator(cats_cols=cats)),
            ])

            numeric_feats_pipeline = Pipeline(steps=[
                ('generator', NumericDataPreprocessor(cont_cols=cont)),
                ('scaler', StandardScaler())
            ])

            self.pipeline = FeatureUnion(transformer_list=[
                ('cats', cats_pipeline),
                ('vect', vectorizer_pipeline),
                ('cont', numeric_feats_pipeline),
            ])
            self.pipeline.fit(X)

        X = self.pipeline.transform(X)
        return X

    def _set_default_params(self):
        default_params = {
            'model_type': 'LR',
            'C': 1,
            'vectorizer_dict_size': 3000,
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
        time_start = time.time()
        hyperparams = self.params.copy()

        self.cat_one_hot = get_one_hot_features(X_train)

        # See solver options here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        if self.problem_type == BINARY:
            Y_train = Y_train.astype(int).values
            solver = 'liblinear'
        elif self.problem_type == MULTICLASS:
            solver = 'saga'  # another option is lbfgs
        else:
            solver = 'liblinear'

        X_train = self.preprocess(X_train, is_train=True, vect_max_features=hyperparams['vectorizer_dict_size'])

        # TODO: get seed from seeds provider
        self.model = LogisticRegression(C=hyperparams['C'], random_state=17, solver=solver)
        self.model.fit(X_train, Y_train)

    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options=None, **kwargs):
        self.fit(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, **kwargs)
        hpo_model_performances = {self.name: self.score(X_test, Y_test)}
        hpo_results = {}
        self.save()
        hpo_models = {self.name: self.path}

        return hpo_models, hpo_model_performances, hpo_results
