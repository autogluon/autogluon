import numpy as np
import pandas as pd

from autogluon.eda import AnalysisState
from autogluon.eda.analysis import Sampler, Namespace
from autogluon.eda.analysis.base import BaseAnalysis


class SomeAnalysis(BaseAnalysis):

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.args = args.copy()


def test_Sampler():
    df_train = pd.DataFrame(np.random.randint(0, 100, size=(10, 4)), columns=list('ABCD'))
    df_test = pd.DataFrame(np.random.randint(0, 100, size=(20, 4)), columns=list('EFGH'))
    assert df_train.shape == (10, 4)
    assert df_test.shape == (20, 4)

    analysis = BaseAnalysis(train_data=df_train, test_data=df_test, children=[
        Namespace(namespace='ns_sampler', children=[
            Sampler(sample=5, children=[
                SomeAnalysis()
            ])
        ]),
        Namespace(namespace='ns_sampler_none', children=[
            Sampler(sample=None, children=[
                SomeAnalysis()
            ])
        ]),
        Namespace(namespace='ns_no_sampler', children=[
            SomeAnalysis()
        ])
    ])

    state = analysis.fit()
    assert state.ns_sampler.args.train_data.shape == (5, 4)
    assert state.ns_sampler.args.test_data.shape == (5, 4)
    assert state.ns_sampler.sample_size == 5

    assert state.ns_sampler_none.args.train_data.shape == (10, 4)
    assert state.ns_sampler_none.args.test_data.shape == (20, 4)
    assert state.ns_sampler_none.sample_size is None

    assert state.ns_no_sampler.args.train_data.shape == (10, 4)
    assert state.ns_no_sampler.args.test_data.shape == (20, 4)
    assert state.ns_no_sampler.sample_size is None


def test_Sampler_frac():
    df_train = pd.DataFrame(np.random.randint(0, 100, size=(10, 4)), columns=list('ABCD'))
    df_test = pd.DataFrame(np.random.randint(0, 100, size=(20, 4)), columns=list('EFGH'))
    assert df_train.shape == (10, 4)
    assert df_test.shape == (20, 4)

    analysis = BaseAnalysis(train_data=df_train, test_data=df_test, children=[
        Sampler(sample=0.5, children=[
            SomeAnalysis()
        ])
    ])

    state = analysis.fit()
    assert state.sample_size == 0.5
    assert state.args.train_data.shape == (5, 4)
    assert state.args.test_data.shape == (10, 4)
