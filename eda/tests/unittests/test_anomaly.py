import pandas as pd
import numpy as np
from autogluon.eda.analysis import AnomalyDetector


s3_url = 'https://autogluon.s3.us-west-2.amazonaws.com/datasets/ano_test_hep.csv'

def load_data():
    A = pd.read_csv(s3_url)
    train_data, test_data = A[:40], A[40:]
    return train_data, test_data

def test_ano_analysis():
    train, test = load_data()
    analysis_args = dict(
        train_data=train,
        test_data=test,
        label='18',
    )
    ano_ana = AnomalyDetector(**analysis_args)
    ano_ana.fit()
    assert ano_ana.state.test_ano_scores.shape[0] == 40
    assert ano_ana.state.test_ano_pred.shape[0] == 40
    pass

def test_ano_auto():
    train, test = load_data()
    analysis_args = dict(
        train_data=train,
        test_data=test,
        label='class',
    )
    viz_args = dict(headers=True)
    auto.analyze(**analysis_args, anlz_facets=[

    ],
    viz_facets=[

    ]
    )
    pass
