"""Definitions of standard measures for fairness and performance"""
import numpy as np

from autogluon.core.metrics import roc_auc
from .group_metric_classes import GroupMetric, AddGroupMetrics, BaseGroupMetric, Utility  # pylint: disable=unused-import # noqa

# N.B. BaseGroupMetric and Utility are needed for type declarations

# Basic parity measures for fairness
count = GroupMetric(lambda TP, FP, FN, TN: TP + FP + FN + TN, 'Number of Datapoints')
pos_data_count = GroupMetric(lambda TP, FP, FN, TN: TP + FN, 'Positive Count')
neg_data_count = GroupMetric(lambda TP, FP, FN, TN: FP + TN, 'Negative Count')
pos_data_rate = GroupMetric(lambda TP, FP, FN, TN: (TP + FN) / (TP + FP + FN + TN), 'Positive Label Rate')
neg_data_rate = GroupMetric(lambda TP, FP, FN, TN: (TN + FP) / (TP + FP + FN + TN), 'Negative Label Rate')
pos_pred_rate = GroupMetric(lambda TP, FP, FN, TN: (TP + FP) / (TP + FP + FN + TN), 'Positive Prediction Rate')
neg_pred_rate = GroupMetric(lambda TP, FP, FN, TN: (TN + FN) / (TP + FP + FN + TN), 'Negative Prediction Rate')

# Standard metrics see sidebar of https://en.wikipedia.org/wiki/Precision_and_recall
true_pos_rate = GroupMetric(lambda TP, FP, FN, TN: (TP) / (1e-6 + TP + FN), 'True Positive Rate')
true_neg_rate = GroupMetric(lambda TP, FP, FN, TN: (TN) / (1e-6 + FP + TN), 'True Negative Rate')
false_pos_rate = GroupMetric(lambda TP, FP, FN, TN: (FP) / (1e-6 + FP + TN), 'False Positive Rate')
false_neg_rate = GroupMetric(lambda TP, FP, FN, TN: (FN) / (1e-6 + TP + FN), 'False Negative Rate')
pos_pred_val = GroupMetric(lambda TP, FP, FN, TN: (TP) / (1e-6 + TP + FP), 'Positive Predicted Value')
neg_pred_val = GroupMetric(lambda TP, FP, FN, TN: (TN) / (1e-6 + TN + FN), 'Negative Predicted Value')

# Existing binary metrics for autogluon
accuracy = GroupMetric(lambda TP, FP, FN, TN: (TP + TN) / (TP + FP + FN + TN), 'Accuracy')
balanced_accuracy = GroupMetric(lambda TP, FP, FN, TN: (TP / (1e-6 + TP + FN) + TN / (1e-6 + TN + FP)) / 2,
                                'Balanced Accuracy')
min_accuracy = GroupMetric(lambda TP, FP, FN, TN: np.minimum(TP / (1e-6 + TP + FN), TN / (1e-6 + TN + FP)),
                           'Minimum-Label-Accuracy')  # common in min-max fairness literature
f1 = GroupMetric(lambda TP, FP, FN, TN: (2 * TP) / (1e-6 + 2 * TP + FP + FN), 'F1 score')
precision = pos_pred_val.rename('Precision')
recall = true_pos_rate.rename('Recall')
mcc = GroupMetric(lambda TP, FP, FN, TN:
                  (TP * TN - FP * FN) / (1e-6 + np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))),
                  'MCC')

default_accuracy_metrics = {'accuracy': accuracy,
                            'balanced_accuracy': balanced_accuracy,
                            'f1': f1,
                            'mcc': mcc}
additional_ag_metrics = {
    'precision': precision,
    'recall': recall,
    'roc_auc': roc_auc}
standard_metrics = {
    'true_pos_rate': true_pos_rate,
    'true_neg_rate': true_neg_rate,
    'false_pos_rate': false_pos_rate,
    'false_neg_rate': false_neg_rate,
    'pos_pred_val': pos_pred_val,
    'neg_pred_val': neg_pred_val
}

ag_metrics = {**default_accuracy_metrics, **additional_ag_metrics}

count_metrics = {'count': count,
                 'pos_data_count': pos_data_count,
                 'neg_data_count': neg_data_count,
                 'pos_data_rate': pos_data_rate,
                 'pos_pred_rate': pos_pred_rate}
default_group_metrics = {**ag_metrics, **count_metrics}

extended_group_metrics = {**default_accuracy_metrics, **standard_metrics, **count_metrics}

# Postprocessing Clarify metrics
# https://mkai.org/learn-how-amazon-sagemaker-clarify-helps-detect-bias
class_imbalance = pos_data_rate.diff.rename('Class Imbalance')
demographic_parity = pos_pred_rate.diff.rename('Demographic Parity')
disparate_impact = pos_pred_rate.ratio.rename('Disparate Impact')
acceptance_rate = precision.rename('Acceptance Rate')
cond_accept = GroupMetric(lambda TP, FP, FN, TN: (TP + FN) / (1e-6 + TP + FP), 'Conditional Acceptance Rate')
cond_reject = GroupMetric(lambda TP, FP, FN, TN: (TN + FP) / (1e-6 + TN + FN), 'Conditional Rejectance Rate')
specificity = true_neg_rate.rename('Specificity')
rejection_rate = neg_pred_val.rename('Rejection Rate')
error_ratio = GroupMetric(lambda TP, FP, FN, TN: FP / (1e-6 + FN), 'Error Ratio')
treatment_equality = error_ratio.diff.rename('Treatment Equality')

gen_entropy = GroupMetric(lambda TP, FP, FN, TN: ((TP + FP + TN + FN) * (TP + FP * 4 + TN) /
                          (TP + FP * 2 + TN + 10**-6)**2 - 1) / 2, 'Generalized Entropy', False)
clarify_metrics = {
    'class_imbalance': class_imbalance,
    'demographic_parity': demographic_parity,
    'disparate_impact': disparate_impact,
    'accuracy.diff': accuracy.diff,
    'recall.diff': recall.diff,
    'cond_accept.diff': cond_accept.diff,
    'acceptance_rate.diff': acceptance_rate.diff,
    'specificity.diff': specificity.diff,
    'cond_reject.diff': cond_reject.diff,
    'rejection_rate.diff': rejection_rate.diff,
    'treatment_equality': treatment_equality,
    'gen_entropy': gen_entropy}


# Existing fairness definitions.
# Binary definitions from: https://fairware.cs.umass.edu/papers/Verma.pdf
# As all definitions just say 'these should be equal' we report the max difference in values as a measure of inequality.

statistical_parity = demographic_parity.rename('Statistical Parity')
predictive_parity = precision.diff.rename('Predictive Parity')
predictive_equality = false_neg_rate.diff.rename('Predictive Equality')
equal_opportunity = recall.diff.rename('Equal Opportunity')
equalized_odds = AddGroupMetrics(true_pos_rate.diff, true_neg_rate.diff, 'Equalized Odds')
cond_use_accuracy = AddGroupMetrics(pos_pred_val.diff, neg_pred_val.diff, 'Conditional Use Accuracy')
accuracy_parity = accuracy.diff.rename('Accuracy Parity')

verma_metrics = {
    'statistical_parity': statistical_parity,
    'predictive_parity': predictive_parity,
    'false_pos_rate.diff': false_pos_rate.diff,
    'false_neg_rate.diff': false_neg_rate.diff,
    'equalized_odds': equalized_odds,
    'cond_use_accuracy': cond_use_accuracy,
    'predictive_equality': predictive_equality,
    'accuracy.diff': accuracy.diff,
    'treatment_equality': treatment_equality
}

rate_metrics = {'pos_pred_rate': pos_pred_rate.diff, **{k: v.diff for k, v in standard_metrics.items()}}
