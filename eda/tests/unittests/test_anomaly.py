import pandas as pd
from autogluon.eda.analysis import AnomalyDetector
from autogluon.eda.visualization import AnomalyVisualization
import autogluon.eda.auto as auto
import unittest


s3_url = 'https://autogluon.s3.us-west-2.amazonaws.com/datasets/ano_test_hep.csv'
#s3_url = '../data/ano_test_hep.csv'

def load_data():
    A = pd.read_csv(s3_url, index_col=0)
    A.loc[30:50, 'cat'] = 'dog'
    train_data, test_data = A[:40], A[40:]
    return train_data, test_data

class TestAnomaly(unittest.TestCase):
    def test_ano_analysis(self):
        train, test = load_data()
        analysis_args = dict(
            train_data=train,
            test_data=test,
            label='18',
            num_anomalies=10,
        )
        ano_ana = AnomalyDetector(**analysis_args)
        ano_ana.fit()
        assert len(ano_ana.state.top_test_anomalies) == 10

    def test_ano_analysis_train_only(self):
        train, test = load_data()
        analysis_args = dict(
            train_data=train,
            label='18',
            num_anomalies=10,
            fit_train=True,
        )
        ano_ana = AnomalyDetector(**analysis_args)
        ano_ana.fit()

    def test_ano_auto(self):
        train, test = load_data()
        analysis_args = dict(
            train_data=train,
            test_data=test,
            label='class',
        )
        ano_analysis_args = dict(
            num_anomalies=6,
        )
        ano_viz_args = dict(headers=True)
        auto.analyze(**analysis_args, anlz_facets=[
            AnomalyDetector(**ano_analysis_args)
        ],
        viz_facets=[
            AnomalyVisualization(**ano_viz_args)
        ]
        )
        pass
