from abc import abstractmethod

import numpy as np
import pandas as pd
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.lr.lr_preprocessing_utils import OheFeaturesGenerator
from autogluon.core.features.types import R_INT, R_FLOAT, R_CATEGORY, R_OBJECT
from autogluon.tabular.models.lr.hyperparameters.parameters import get_param_baseline, INCLUDE, IGNORE, ONLY, _get_solver, preprocess_params_set

class IModelsModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    @abstractmethod
    def get_model(self):
        return NotImplemented

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)

        if is_train:
            categorical_featnames = self._get_types_of_features(X)
            self._feature_generator = OheFeaturesGenerator(cats_cols=categorical_featnames['categorical']) # LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator is not None:
            X = X.copy()
            X = X.fillna(0)
            X = self._feature_generator.transform(X=X)
        return X.toarray().astype(np.float32)

    def _get_types_of_features(self, df):
        """ Returns dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = ordered list of feature-names falling into each category.
            Each value is a list of feature-names corresponding to columns in original dataframe.
            TODO: ensure features with zero variance have already been removed before this function is called.
        """
        feature_types = self._feature_metadata.get_type_group_map_raw()

        categorical_featnames = feature_types[R_CATEGORY] + feature_types[R_OBJECT] + feature_types['bool']
        continuous_featnames = feature_types[R_FLOAT] + feature_types[R_INT]  # + self.__get_feature_type_if_present('datetime')
        language_featnames = []  # TODO: Disabled currently, have to pass raw text data features here to function properly
        valid_features = categorical_featnames + continuous_featnames + language_featnames
        if len(categorical_featnames) + len(continuous_featnames) + len(language_featnames) != df.shape[1]:
            unknown_features = [feature for feature in df.columns if feature not in valid_features]
            df = df.drop(columns=unknown_features)
        self._features_internal = list(df.columns)  # FIXME: Don't edit _features_internal
        return {'categorical': categorical_featnames}
        # types_of_features = {'continuous': [], 'skewed': [], 'onehot': [], 'language': []}
        # return self._select_features(df, types_of_features, categorical_featnames, language_featnames, continuous_featnames)

    def _fit(self,
             X: pd.DataFrame,  # training data
             y: pd.Series,  # training labels
             **kwargs):

        model_cls = self.get_model()
        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        self.model = model_cls(**params)
        self.model.fit(X, y)

    def _set_default_params(self):
        default_params = {
            'random_state': 0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            get_features_kwargs=dict(
                valid_raw_types=['int', 'float', 'category'],
            ),
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params


class RuleFitModel(IModelsModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_model(self):
        from imodels import RuleFitRegressor, RuleFitClassifier

        if self.problem_type in ['regression', 'softclass']:
            return RuleFitRegressor
        else:
            return RuleFitClassifier


class GreedyTreeModel(IModelsModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_model(self):
        from imodels import GreedyTreeClassifier
        from sklearn.tree import DecisionTreeRegressor

        if self.problem_type in ['regression', 'softclass']:
            return DecisionTreeRegressor
        else:
            return GreedyTreeClassifier


class GlobalSparseTreeModel(IModelsModel):
    '''todo: properly set up GOSDT (right now it basically just uses DecisionTrees)
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_model(self):
        from imodels import GlobalSparseTreeClassifier

        if self.problem_type in ['binary']:
            return GlobalSparseTreeClassifier
        else:
            raise Exception('GlobalSparseTreeClassifier only supports binary classification!')


class BayesianRuleSetModel(IModelsModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_model(self):
        from imodels import BayesianRuleSetClassifier

        if self.problem_type in ['binary']:
            return BayesianRuleSetClassifier
        else:
            raise Exception('Bayesian Rule Set only supports binary classification!')


class BoostedRulesModel(IModelsModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_model(self):
        from imodels import BoostedRulesClassifier

        if self.problem_type in ['binary']:
            return BoostedRulesClassifier
        else:
            raise Exception('Boosted Rule Set only supports binary classification!')


class CorelsRuleListModel(IModelsModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_model(self):
        from imodels import CorelsRuleListClassifier

        if self.problem_type in ['binary']:
            return CorelsRuleListClassifier
        else:
            raise Exception('Corels only supports binary classification!')
