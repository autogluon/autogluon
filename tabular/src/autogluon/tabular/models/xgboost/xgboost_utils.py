import numpy as np
from collections import OrderedDict
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS

from ..tabular_nn.utils.categorical_encoders import OneHotMergeRaresHandleUnknownEncoder


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
        mean_squared_error='rmse',
        root_mean_squared_error='rmse',
    ),
}


def convert_ag_metric_to_xgbm(ag_metric_name, problem_type):
    return _ag_to_xgbm_metric_dict.get(problem_type, dict()).get(ag_metric_name, None)


def func_generator(metric, problem_type: str):
    """Create a custom metric compatible with XGBoost, based on the XGBoost 1.6+ API"""
    sign = -1 if metric.greater_is_better else 1
    needs_pred_proba = not metric.needs_pred
    if needs_pred_proba:
        def custom_metric(y_true, y_hat):
            return sign * metric(y_true, y_hat)
    else:
        if problem_type in [MULTICLASS, SOFTCLASS]:
            def custom_metric(y_true, y_hat):
                y_hat = y_hat.argmax(axis=1)
                return sign * metric(y_true, y_hat)
        elif problem_type == BINARY:
            def custom_metric(y_true, y_hat):
                y_hat = np.round(y_hat)
                return sign * metric(y_true, y_hat)
        else:
            def custom_metric(y_true, y_hat):
                return sign * metric(y_true, y_hat)

    return custom_metric


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
