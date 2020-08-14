import logging
import re

import numpy as np
import pandas as pd
from pandas import DataFrame
from gluonts.model.seq2seq import MQCNNEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from gluonts.model.seq2seq import MQCNNEstimator
# from .hyperparameters.parameters import get_param_baseline, get_model_params, get_default_params, INCLUDE, IGNORE, ONLY
# from .hyperparameters.searchspaces import get_default_searchspace
# from .lr_preprocessing_utils import NlpDataPreprocessor, OheFeaturesGenerator, NumericDataPreprocessor
from ...constants import BINARY, REGRESSION
from ....ml.models.abstract.abstract_model import AbstractModel
from .hyperparameters.parameters import get_default_parameters
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName


class MQCNNModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = get_default_parameters()

    def _fit(self, X_train, y_train, X_val=None, y_val=None, time_limit=None, **kwargs):
        X_train = self.preprocess(X_train)
        print(X_train)
        estimator = MQCNNEstimator.from_hyperparameters(**self.params)
        self.model = estimator.train(X_train)

    def preprocess(self, X):
        # TODO: transform dataframe into gluon-ts ListDataset
        target_values = X.drop("index", axis=1).values
        processed_X = ListDataset([
            {
                FieldName.TARGET: target,
                FieldName.START: pd.Timestamp("2020-01-01", freq="1D"),
            }
            for (target, ) in zip(target_values,)
        ], freq="D")
        return processed_X