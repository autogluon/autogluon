import unittest

import pandas as pd
from autogluon.eda.analysis.interaction import NonparametricAssociation
from autogluon.eda.visualization import NonparametricSignificanceVisualization


s3_url = 'https://autogluon.s3.us-west-2.amazonaws.com/datasets/ano_test_hep.csv'


def load_data():
    A = pd.read_csv(s3_url)
    return A


class TestNPA(unittest.TestCase):
    def test_tests(self):
        test = load_data()
        n_cat_thresh = 20
        int_cols = [k for k, col in test.iteritems()
                    if col.dtype == float
                    and len(col.unique()) <= n_cat_thresh]
        test[int_cols] = test[int_cols].astype('object')
        test_parms_kruskal = NonparametricAssociation.nonparametric_test(test, '0', '1')
        test_parms_chi = NonparametricAssociation.nonparametric_test(test, '2', '1')
        test_parms_spear = NonparametricAssociation.nonparametric_test(test, '0', '17')
        assert test_parms_kruskal['test'] == 'kruskal'
        assert test_parms_chi['test'] == 'chi2'
        assert test_parms_spear['test'] == 'spearman'

    def test_analysis_vis(self):
        test = load_data()
        n_cat_thresh = 20
        int_cols = [k for k, col in test.iteritems()
                    if col.dtype == float
                    and len(col.unique()) <= n_cat_thresh]
        test[int_cols] = test[int_cols].astype('object')
        analysis_args = dict(
            data=test,
            association_cols=['0', '1', '2', '17'],
        )
        npa = NonparametricAssociation(**analysis_args)
        npa.fit()
        viz_args = dict(headers=True)
        viz = NonparametricSignificanceVisualization(**viz_args)
        viz.render(npa.state)
