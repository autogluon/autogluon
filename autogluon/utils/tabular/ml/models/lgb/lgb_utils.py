import numpy as np

from ...constants import MULTICLASS


def func_generator(metric, is_higher_better, needs_pred_proba, problem_type):
    if needs_pred_proba:
        if problem_type == MULTICLASS:
            def function_template(y_hat, data):
                y_true = data.get_label()

                y_hat = y_hat.reshape(len(np.unique(y_true)), -1).T

                return metric.name, metric(y_true, y_hat), is_higher_better
        else:
            def function_template(y_hat, data):
                y_true = data.get_label()
                return metric.name, metric(y_true, y_hat), is_higher_better
    else:
        if problem_type == MULTICLASS:
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = y_hat.reshape(len(np.unique(y_true)), -1)
                y_hat = y_hat.argmax(axis=0)
                return metric.name, metric(y_true, y_hat), is_higher_better
        else:
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = np.round(y_hat)
                return metric.name, metric(y_true, y_hat), is_higher_better
    return function_template
