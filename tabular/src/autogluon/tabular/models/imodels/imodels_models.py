from abc import abstractmethod

import numpy as np
import pandas as pd
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator


class IModelsModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)

    @abstractmethod
    def get_model(self):
        return NotImplemented

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
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        if self.problem_type in ['regression', 'softclass']:
            return DecisionTreeRegressor
        else:
            return DecisionTreeClassifier


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
