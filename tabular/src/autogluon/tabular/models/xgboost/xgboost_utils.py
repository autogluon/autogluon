from collections import OrderedDict

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS
from autogluon.core.metrics import Scorer

from ..tabular_nn.utils.categorical_encoders import OneHotMergeRaresHandleUnknownEncoder

_ag_to_xgbm_metric_dict = {
    BINARY: dict(accuracy="error", log_loss="logloss", roc_auc="auc"),
    MULTICLASS: dict(
        accuracy="merror",
        log_loss="mlogloss",
    ),
    REGRESSION: dict(
        mean_absolute_error="mae",
        mean_squared_error="rmse",
        root_mean_squared_error="rmse",
    ),
}


def convert_ag_metric_to_xgbm(ag_metric_name, problem_type):
    return _ag_to_xgbm_metric_dict.get(problem_type, dict()).get(ag_metric_name, None)


def func_generator(metric: Scorer, problem_type: str):
    """
    Create a custom metric compatible with XGBoost, based on the XGBoost 1.6+ API.
    Note that XGBoost needs lower is better metrics.

    Params:
    -------
    metric : Scorer
        The autogluon Scorer object to be converted into an XGBoost custom metric.
    problem_type: str
        The current problem type.

    Returns:
    --------
    Callable[y_true, y_hat]
        XGBoost custom metric wrapper function.
    """
    needs_pred_proba = not metric.needs_pred
    sign = -1 if metric.greater_is_better else 1

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

    # Note: Must include "_" prefix due to xgboost internal metric naming conflicts (e.g. precision)
    custom_metric.__name__ = f"_{metric.name}"

    return custom_metric


def learning_curve_func_generator(metric: Scorer, problem_type: str, use_error: bool = False):
    """
    Create a custom metric compatible with XGBoost inputs (but in greater is better format).
    NOTE: Do not use these custom metrics with XGBoost internally.

    Documentation of XGBoost support for Custom Metrics:
    Trying to use Multiple Custom Metrics:
        https://stackoverflow.com/questions/44527485/how-to-pass-multiple-custom-metrics-eval-metric-in-python-xgboost
    Multiple Custom Not possible: https://github.com/dmlc/xgboost/issues/2408
    Possible Workaround: https://github.com/dmlc/xgboost/issues/1125 -> Didn't work
    Resolution: Instead, use custom metrics by passing in list of AutoGluon Scorers into custom metric callback

    Params:
    -------
    metric : Scorer
        The autogluon Scorer object to be converted into an XGBoost custom metric.
    problem_type: str
        The current problem type.
    use_error: bool
        Whether the custom metric should be computed in error or score format.

    Returns:
    --------
    Callable[y_true, y_hat]
        XGBoost custom metric wrapper function.
    """
    sign = -1 if metric.greater_is_better else 1
    func = func_generator(metric=metric, problem_type=problem_type)

    def custom_metric(y_true, y_hat):
        result = sign * func(y_true, y_hat)
        if use_error:
            return metric.convert_score_to_error(result)
        return result

    # Set custom metric name to scorer metric name
    # Note: no need for _ prefix because these metrics aren't
    # to be used by xgboost internally
    custom_metric.__name__ = metric.name

    return custom_metric


class OheFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, max_levels=None):
        self._feature_map = OrderedDict()  # key: feature_name, value: feature_type
        self.labels = OrderedDict()
        self.cat_cols = []
        self.other_cols = []
        self.ohe_encs = None
        self.max_levels = max_levels

    def fit(self, X, y=None):
        self.cat_cols = list(X.select_dtypes(include="category").columns)
        self.other_cols = list(X.select_dtypes(exclude="category").columns)
        self.ohe_encs = OneHotMergeRaresHandleUnknownEncoder(max_levels=self.max_levels)

        if self.cat_cols:
            self.ohe_encs.fit(X[self.cat_cols])
            assert len(self.cat_cols) == len(self.ohe_encs.categories_)

            for cat_col, categories in zip(self.cat_cols, self.ohe_encs.categories_):
                categories_ = categories.tolist()
                self.labels[cat_col] = categories_
                # Update feature map ({name: type})
                for category in categories_:
                    self._feature_map[f"{cat_col}_{category}"] = "i"  # one-hot encoding data type is boolean

        if self.other_cols:
            for c in self.other_cols:
                self._feature_map[c] = "int" if X[c].dtypes == int else "float"
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
