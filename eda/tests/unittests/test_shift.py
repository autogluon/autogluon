import pandas as pd
import numpy as np
import autogluon.eda.analysis as eda
import autogluon.eda.visualization as viz
from sklearn.model_selection import train_test_split
import unittest

S3_URL = 'https://autogluon.s3.us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification/'
SAMPLE_SIZE = 200


def load_adult_data():
    train_data = S3_URL + 'train_data.csv'
    test_data = S3_URL + 'test_data.csv'
    train = pd.read_csv(train_data).sample(SAMPLE_SIZE, random_state=0)
    test = pd.read_csv(test_data).sample(SAMPLE_SIZE, random_state=0)
    data = (train, test)
    return data


def sim_cov_shift(train, test, p_nonmarr=.75, val=False):
    """Simulate covariate shift by biasing training set toward married
    """
    data = pd.concat((train, test))
    data.loc[:, 'race'] = data['race'].str.strip()
    data.loc[:, 'sex'] = data['sex'].str.strip()
    data.loc[:, 'marital-status'] = data['marital-status'].str.strip()
    data.index = pd.Index(range(data.shape[0]))
    data_married = data['marital-status'] == "Married-civ-spouse"
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
        analysis_args = dict(
            train_data=train,
            test_data=test,
            label='class',
            classifier_kwargs={'path': 'AutogluonModels'},
            classifier_fit_kwargs={'hyperparameters': {'XGB': {}}}
        )
        shft_ana = eda.shift.XShiftDetector(**analysis_args)
        shft_ana.fit()
        viz_args = dict(headers=True)
        shft_viz = viz.shift.XShiftSummary(**viz_args)
        shft_viz.render(shft_ana.state)
        res = shft_ana.state.xshift_results
        assert all(res[k] is not None for k in ['detection_status', 'pvalue', 'pvalue_threshold', 'eval_metric',
                                                'feature_importance'])
