import os

import numpy as np
from pandas import DataFrame, Series

from autogluon import try_import_lightgbm
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


def construct_dataset(x: DataFrame, y: Series, location=None, reference=None, params=None, save=False, weight=None):
    try_import_lightgbm()
    import lightgbm as lgb

    dataset = lgb.Dataset(data=x, label=y, reference=reference, free_raw_data=True, params=params, weight=weight)

    if save:
        assert location is not None
        saving_path = f'{location}.bin'
        if os.path.exists(saving_path):
            os.remove(saving_path)

        os.makedirs(os.path.dirname(saving_path), exist_ok=True)
        dataset.save_binary(saving_path)
        # dataset_binary = lgb.Dataset(location + '.bin', reference=reference, free_raw_data=False)# .construct()

    return dataset
