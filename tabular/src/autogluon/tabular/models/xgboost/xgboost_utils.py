import numpy as np
from collections import OrderedDict
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS

from ..tabular_nn.categorical_encoders import OneHotMergeRaresHandleUnknownEncoder


_ag_to_xgbm_metric_dict = {
    BINARY: dict(
        accuracy='error',
        log_loss='logloss',
        roc_auc='auc'
    ),
    MULTICLASS: dict(
        accuracy='merror',
        log_loss='mlogloss',        
    ),
    REGRESSION: dict(
        mean_absolute_error='mae',
        mean_squared_error='rmse', # TODO: not supported from default eavl metric. Firstly, use `rsme` refenced by catboost model.
        root_mean_squared_error='rmse',

    ),
}


def convert_ag_metric_to_xgbm(ag_metric_name, problem_type):
    return _ag_to_xgbm_metric_dict.get(problem_type, dict()).get(ag_metric_name, None)


def _softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def func_generator(metric, is_higher_better, needs_pred_proba, problem_type):
    if needs_pred_proba:
        if problem_type == MULTICLASS:
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = _softmax(y_hat)
                res = metric(y_true, y_hat)
                return metric.name, -1 * res if is_higher_better else res
        elif problem_type == SOFTCLASS:
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = _softmax(y_hat)
                res = metric(y_true, y_hat)
                return metric.name, -1 * res if is_higher_better else res
        elif problem_type == BINARY:
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = _sigmoid(y_hat)
                res = metric(y_true, y_hat)
                return metric.name, -1 * res if is_higher_better else res
        else:
            def function_template(y_hat, data):
                y_true = data.get_label()
                res = metric(y_true, y_hat)
                return metric.name, -1 * res if is_higher_better else res
    else:
        if problem_type == MULTICLASS:
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = y_hat.argmax(axis=1)
                res = metric(y_true, y_hat)
                return metric.name, -1 * res if is_higher_better else res
        elif problem_type == SOFTCLASS:
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = y_hat.argmax(axis=1)
                res = metric(y_true, y_hat)
                return metric.name, -1 * res if is_higher_better else res
        elif problem_type == BINARY:
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = _sigmoid(y_hat)
                y_hat = np.round(y_hat)
                res = metric(y_true, y_hat)
                return metric.name, -1 * res if is_higher_better else res
        else:
            def function_template(y_hat, data):
                y_true = data.get_label()
                res = metric(y_true, y_hat)
                return metric.name, -1 * res if is_higher_better else res

    return function_template


class OheFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, max_levels=None):
        self._feature_map = OrderedDict()  # key: feature_name, value: feature_type
        self.labels = OrderedDict()
        self.cat_cols = []
        self.other_cols = []
        self.ohe_encs = None
        self.max_levels=max_levels

    def fit(self, X, y=None):
        self.cat_cols = list(X.select_dtypes(include='category').columns)
        self.other_cols = list(X.select_dtypes(exclude='category').columns)
        self.ohe_encs = OneHotMergeRaresHandleUnknownEncoder(max_levels=self.max_levels)

        if self.cat_cols:
            self.ohe_encs.fit(X[self.cat_cols])
            assert len(self.cat_cols) == len(self.ohe_encs.categories_)
            
            for cat_col, categories in zip(self.cat_cols, self.ohe_encs.categories_):
                categories_ = categories.tolist()
                self.labels[cat_col] = categories_
                # Update feature map ({name: type})
                for category in categories_:
                    self._feature_map[f"{cat_col}_{category}"] = 'i'  # one-hot encoding data type is boolean

        if self.other_cols:
            for c in self.other_cols:
                self._feature_map[c] = 'int' if X[c].dtypes == int else 'float'
        return self

    def transform(self, X, y=None):
        X_list = []
        if self.cat_cols:
            X_list.append(self.ohe_encs.transform(X[self.cat_cols]))
        if self.other_cols:
            X_list.append(csr_matrix(X[self.other_cols]))
        return hstack(X_list, format="csr")

    def get_feature_names(self):
        return list(self._feature_map.keys())

    def get_feature_types(self):
        return list(self._feature_map.values())

    def get_original_feature_names(self):
        return self.cat_cols + self.other_cols
