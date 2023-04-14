import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import autogluon.eda.analysis as eda
import autogluon.eda.visualization as viz

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "resources"))

SAMPLE_SIZE = 200


def load_adult_data():
    train_data = os.path.join(RESOURCE_PATH, "adult", "train_data.csv")
    test_data = os.path.join(RESOURCE_PATH, "adult", "test_data.csv")
    train = pd.read_csv(train_data).sample(SAMPLE_SIZE, random_state=0)
    test = pd.read_csv(test_data).sample(SAMPLE_SIZE, random_state=0)
    data = (train, test)
    return data


def sim_cov_shift(train, test, p_nonmarr=0.75, val=False):
    """Simulate covariate shift by biasing training set toward married"""
    data = pd.concat((train, test))
    data.loc[:, "race"] = data["race"].str.strip()
    data.loc[:, "sex"] = data["sex"].str.strip()
    data.loc[:, "marital-status"] = data["marital-status"].str.strip()
    data.index = pd.Index(range(data.shape[0]))
    data_married = data["marital-status"] == "Married-civ-spouse"
    p_married = (0.5 + data_married.mean() - p_nonmarr) / data_married.mean()
    train_p = data_married * p_married + (1 - data_married) * p_nonmarr
    train_ind = np.random.binomial(1, train_p) == 1
    train_cs = data[train_ind]
    test_cs = data[~train_ind]
    if val:
        train_cs, val_cs = train_test_split(train_cs)
        return train_cs, val_cs, test_cs
    else:
        return train_cs, test_cs


class TestShift(unittest.TestCase):
    def test_shift(self):
        train, test = load_adult_data()
        train, test = sim_cov_shift(train, test)
        with tempfile.TemporaryDirectory() as path:
            shft_ana = eda.shift.XShiftDetector(
                train_data=train,
                test_data=test,
                label="class",
                classifier_kwargs={"path": os.path.join(path, "AutogluonModels")},
                classifier_fit_kwargs={"hyperparameters": {"RF": {}}},
                pvalue_thresh=0.1,
            )
            shft_ana.fit()
            shft_viz = viz.shift.XShiftSummary(headers=True)
            shft_viz.render(shft_ana.state)
            result = shft_ana.state.xshift_results
            assert result.pop("feature_importance", None).shape == (14, 6)
            assert result.pop("pvalue", -100) > 0
            assert result.pop("test_statistic", -100) > 0
            assert len(result.pop("shift_features")) > 0
            assert result == {
                "detection_status": True,
                "eval_metric": "roc_auc",
                "pvalue_threshold": 0.1,
            }
