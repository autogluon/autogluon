import logging

from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS

logger = logging.getLogger(__name__)


CATBOOST_QUANTILE_PREFIX = "Quantile:"
# Mapping from non-optimizable eval_metric to optimizable loss_function.
# See https://catboost.ai/docs/en/concepts/loss-functions-regression#usage-information
CATBOOST_EVAL_METRIC_TO_LOSS_FUNCTION = {
    "MedianAbsoluteError": "MAE",
    "SMAPE": "MAPE",
    "R2": "RMSE",
}


# TODO: Add weight support?
# TODO: Can these be optimized? What computational cost do they have compared to the default catboost versions?
class CustomMetric:
    def __init__(self, metric, is_higher_better, needs_pred_proba):
        self.metric = metric
        self.is_higher_better = is_higher_better
        self.needs_pred_proba = needs_pred_proba

    @staticmethod
    def get_final_error(error, weight):
        return error

    def is_max_optimal(self):
        return self.is_higher_better

    def evaluate(self, approxes, target, weight):
        raise NotImplementedError


def get_catboost_metric_from_ag_metric(metric, problem_type, quantile_levels=None):
    if problem_type == SOFTCLASS:
        from .catboost_softclass_utils import SoftclassCustomMetric

        if metric.name != "soft_log_loss":
            logger.warning("Setting metric=soft_log_loss, the only metric supported for softclass problem_type")
        return SoftclassCustomMetric(metric=None, is_higher_better=True, needs_pred_proba=True)
    elif problem_type == BINARY:
        metric_map = dict(
            log_loss="Logloss",
            accuracy="Accuracy",
            roc_auc="AUC",
            f1="Logloss",  # f1 uses Logloss because f1 in CatBoost is not reliable (causes errors between versions)
            f1_macro="Logloss",
            f1_micro="Logloss",
            f1_weighted="Logloss",
            balanced_accuracy="BalancedAccuracy",
            recall="Recall",
            recall_macro="Recall",
            recall_micro="Recall",
            recall_weighted="Recall",
            precision="Precision",
            precision_macro="Precision",
            precision_micro="Precision",
            precision_weighted="Precision",
        )
        metric_class = metric_map.get(metric.name, "Logloss")
    elif problem_type == MULTICLASS:
        metric_map = dict(
            log_loss="MultiClass",
            accuracy="Accuracy",
        )
        metric_class = metric_map.get(metric.name, "MultiClass")
    elif problem_type == REGRESSION:
        metric_map = dict(
            mean_squared_error="RMSE",
            root_mean_squared_error="RMSE",
            mean_absolute_error="MAE",
            mean_absolute_percentage_error="MAPE",
            # Non-optimizable metrics, see CATBOOST_EVAL_METRIC_TO_LOSS_FUNCTION
            median_absolute_error="MedianAbsoluteError",
            symmetric_mean_absolute_percentage_error="SMAPE",
            r2="R2",
        )
        metric_class = metric_map.get(metric.name, "RMSE")
    elif problem_type == QUANTILE:
        if quantile_levels is None:
            raise AssertionError(f"quantile_levels must be provided for problem_type = {problem_type}")
        if not all(0 < q < 1 for q in quantile_levels):
            raise AssertionError(
                f"quantile_levels must fulfill 0 < q < 1, provided quantile_levels: {quantile_levels}"
            )
        # Loss function MultiQuantile: can only be used if len(quantile_levels) >= 2, otherwise we must use Quantile:
        if len(quantile_levels) == 1:
            metric_class = f"{CATBOOST_QUANTILE_PREFIX}alpha={quantile_levels[0]}"
        else:
            quantile_string = ",".join(str(q) for q in quantile_levels)
            metric_class = f"Multi{CATBOOST_QUANTILE_PREFIX}alpha={quantile_string}"
    else:
        raise AssertionError(f"CatBoost does not support {problem_type} problem type.")

    return metric_class
