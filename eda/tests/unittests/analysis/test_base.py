from unittest.mock import MagicMock

import pandas as pd
import pytest

from autogluon.eda import AnalysisState
from autogluon.eda.analysis import Namespace
from autogluon.eda.analysis.base import BaseAnalysis


def test_abstractanalysis_parameter_shadowing():
    a: BaseAnalysis = BaseAnalysis(
        x='x', y='y',
        children=[
            BaseAnalysis(x='q'),
            BaseAnalysis(x='w', children=[
                BaseAnalysis(x='z', q='q'),
            ]),
        ]
    )

    assert a._gather_args() == {'x': 'x', 'y': 'y'}
    assert a.children[0]._gather_args() == {'x': 'q', 'y': 'y'}
    assert a.children[1]._gather_args() == {'x': 'w', 'y': 'y'}
    assert a.children[1].children[0]._gather_args() == {'x': 'z', 'y': 'y', 'q': 'q'}


def test_abstractanalysis_available_datasets():
    state = AnalysisState(
        train_data=pd.DataFrame(),
        test_data=pd.DataFrame(),
        tuning_data=pd.DataFrame(),
        val_data=pd.DataFrame(),
    )

    keys = []
    for ds, df in BaseAnalysis().available_datasets(state):
        assert state[ds] is df
        assert ds in state
        keys.append(ds)
    assert list(keys) == list(state.keys())


def test_abstractanalysis_available_datasets_some_present():
    state = AnalysisState(
        train_data=pd.DataFrame(),
    )

    keys = []
    for ds, df in BaseAnalysis().available_datasets(state):
        assert state[ds] is df
        assert ds in state
        keys.append(ds)
    assert list(keys) == list(state.keys())


def test_abstractanalysis_fit_is_not_called_if_cannot_handle():
    a = BaseAnalysis()
    a.can_handle = MagicMock(return_value=False)
    a._fit = MagicMock()
    a.fit()
    a._fit.assert_not_called()


def test_abstractanalysis_fit_is_called_if_can_handle():
    a = BaseAnalysis()
    a.can_handle = MagicMock(return_value=True)
    a._fit = MagicMock()
    a.fit()
    a._fit.assert_called()


def test_abstractanalysis_fit_on_inner_before_outer_raises_exception():
    inner_analysis = BaseAnalysis(a=1, b=2)
    outer_analysis = BaseAnalysis(b=3, c=4, children=[inner_analysis])
    with pytest.raises(AssertionError):
        inner_analysis.fit()
    state = outer_analysis.fit()
    assert state is not None


def test_abstractanalysis_fit_gathers_args():
    inner_analysis = BaseAnalysis(a=1, b=2)
    outer_analysis = BaseAnalysis(b=3, c=4, children=[inner_analysis])
    inner_analysis._fit = MagicMock()
    outer_analysis._fit = MagicMock()

    state = outer_analysis.fit(arg_a=10, arg_b=11)
    assert inner_analysis.state is state
    assert outer_analysis.state is state

    outer_analysis._fit.assert_called_once_with({}, {'b': 3, 'c': 4}, arg_a=10, arg_b=11)
    inner_analysis._fit.assert_called_once_with({}, {'a': 1, 'b': 2, 'c': 4}, arg_a=10, arg_b=11)


def test_namespaces():
    def side_effect(state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.upd = fit_kwargs

    inner_analysis: BaseAnalysis = BaseAnalysis(arg=1)
    inner_analysis._fit = MagicMock()
    inner_analysis._fit.side_effect = side_effect

    # write outputs into ns1
    state = BaseAnalysis(children=[
        Namespace(namespace='ns1', children=[inner_analysis])
    ]).fit(op='fit1')

    # write outputs into ns2
    state = BaseAnalysis(state=state, children=[
        Namespace(namespace='ns2', children=[inner_analysis])
    ]).fit(op='fit2')

    # update outputs in ns2
    state = BaseAnalysis(state=state, children=[
        Namespace(namespace='ns2', children=[inner_analysis])
    ]).fit(op='fit3')

    assert state == {'ns1': {'upd': {'op': 'fit1'}}, 'ns2': {'upd': {'op': 'fit3'}}}
