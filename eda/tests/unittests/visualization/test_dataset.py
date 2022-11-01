import numpy as np
import pandas as pd
import pytest

from autogluon.eda import AnalysisState
from autogluon.eda.visualization import DatasetStatistics


@pytest.mark.parametrize(
    "state_present,expected",
    [('dataset_stats', True),
     ('missing_statistics', True),
     ('raw_type', True),
     ('special_types', True),
     ('unknown_type', False),
     ]
)
def test_DatasetStatistics(state_present, expected):
    assert DatasetStatistics().can_handle(AnalysisState({state_present: ''})) is expected


@pytest.mark.parametrize("field", ['dataset_stats', 'raw_type', 'variable_type', 'special_types'])
def test__merge_analysis_facets__single_values(field):
    expected_result = {'some_stat': 'value'}
    state = AnalysisState({field: {'ds': expected_result}})
    if field == 'dataset_stats':
        assert DatasetStatistics._merge_analysis_facets('ds', state) == expected_result
    else:
        assert DatasetStatistics._merge_analysis_facets('ds', state) == {field: expected_result}


def test__merge_analysis_facets__single_values__missing_statistics():
    state = AnalysisState({'missing_statistics': {'ds': {'count': [1, 2], 'ratio': [0.1, 0.2], 'some_field': ['a', 'b']}}})
    assert DatasetStatistics._merge_analysis_facets('ds', state) == {'missing_count': [1, 2], 'missing_ratio': [0.1, 0.2]}


def test__merge_analysis_facets__multiple_values():
    state = AnalysisState({
        'dataset_stats': {'ds': {'dataset_stats': 'value'}},
        'missing_statistics': {'ds': {'count': [1, 2], 'ratio': [0.1, 0.2], 'some_field': ['a', 'b']}},
        'raw_type': {'ds': {'raw_type': 'value'}},
        'variable_type': {'ds': {'variable_type': 'value'}},
        'special_types': {'ds': {'special_types': 'value'}},
    })
    assert DatasetStatistics._merge_analysis_facets('ds', state) == {
        'dataset_stats': 'value',
        'missing_count': [1, 2],
        'missing_ratio': [0.1, 0.2],
        'raw_type': {'raw_type': 'value'},
        'special_types': {'special_types': 'value'},
        'variable_type': {'variable_type': 'value'},
    }


def test__fix_counts():
    df = pd.DataFrame({
        'a': [1.0, np.NaN],
        'b': [1.0, np.NaN],
        'c': [1, np.NaN],
        'd': [1, 2],
    })
    expected_out = {
        'a': {0: 1, 1: ''},
        'b': {0: 1.0, 1: '--NA--'},
        'c': {0: 1, 1: ''},
        'd': {0: 1, 1: 2}
    }
    assert DatasetStatistics._fix_counts(df, cols=['a', 'c', 'd']).fillna('--NA--').to_dict() == expected_out
