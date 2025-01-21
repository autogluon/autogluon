import inspect

import numpy as np
import pandas
import pandas as pd

from autogluon.core.pseudolabeling.pseudolabeling import filter_pseudo


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_default_pseudo_test_args():
    default_args = get_default_args(filter_pseudo)
    sample_percent = default_args["percent_sample"]
    min_percent = default_args["min_proportion_prob"]
    max_percent = default_args["max_proportion_prob"]
    threshold = default_args["threshold"]

    return sample_percent, min_percent, max_percent, threshold


def test_regression_pseudofilter():
    y_reg_fake = np.random.rand(200)
    y_reg_fake = pandas.Series(data=y_reg_fake)

    sample_percent, _, _, _ = get_default_pseudo_test_args()
    pseudo_idxes = filter_pseudo(y_reg_fake, problem_type="regression")
    assert len(pseudo_idxes) == int(sample_percent * len(y_reg_fake))


def test_classification_pseudofilter():
    _, min_percent, max_percent, threshold = get_default_pseudo_test_args()
    middle_percent = (max_percent + min_percent) / 2
    num_rows = 100
    num_above_threshold = int(middle_percent * num_rows)
    num_below_threshold = num_rows - num_above_threshold
    y_reg_fake = np.ones((num_above_threshold))

    # Test if percent preds is below min threshold
    y_reg_fake_below_min = np.zeros((num_rows, 2))
    y_reg_fake_below_min = pd.DataFrame(data=y_reg_fake_below_min)
    pseudo_indices_ans = filter_pseudo(y_reg_fake_below_min, "binary")
    assert num_rows == len(pseudo_indices_ans)

    # Test if percent preds is above max threshold
    y_reg_fake_above_max = pandas.DataFrame(data=y_reg_fake)
    pseudo_indices_ans = filter_pseudo(y_reg_fake_above_max, "binary")
    num_rows_threshold = max(np.ceil(max_percent * len(y_reg_fake_above_max)), 1)
    curr_threshold = y_reg_fake_above_max.loc[num_rows_threshold - 1]
    answer = len(y_reg_fake_above_max >= curr_threshold)
    assert answer == len(pseudo_indices_ans)

    # Test if normal functionality beginning
    y_reg_fake = np.column_stack((y_reg_fake, np.zeros((num_above_threshold))))
    y_reg_fake = np.row_stack((y_reg_fake, np.zeros((num_below_threshold, 2))))

    y_reg_fake = pandas.DataFrame(data=y_reg_fake)
    pseudo_flag = y_reg_fake.max(axis=1) > threshold
    pseudo_indices_ans = pseudo_flag[pseudo_flag == True]

    pseudo_idxes = filter_pseudo(y_reg_fake, "binary")
    assert len(pseudo_idxes) == len(pseudo_indices_ans)
