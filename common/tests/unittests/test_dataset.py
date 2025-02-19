import pandas as pd
import pytest

from autogluon.common import TabularDataset


def test_tabular_dataset_from_dataframe():
    # Test that TabularDataset works with pandas DataFrame input
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    ds = TabularDataset(df)

    assert isinstance(ds, pd.DataFrame)
    assert ds.equals(df)
    assert type(ds) == type(df)


def test_tabular_dataset_from_list():
    # Test that TabularDataset works with list input
    data = [[1, 'a'], [2, 'b'], [3, 'c']]
    ds = TabularDataset(data, columns=['A', 'B'])

    expected = pd.DataFrame(data, columns=['A', 'B'])
    assert ds.equals(expected)


def test_tabular_dataset_from_dict():
    # Test that TabularDataset works with dict input
    data = {'A': [1, 2, 3], 'B': ['a', 'b', 'c']}
    ds = TabularDataset(data)

    expected = pd.DataFrame(data)
    assert ds.equals(expected)


def test_tabular_dataset_empty():
    # Test that TabularDataset works with empty DataFrame
    df = pd.DataFrame()
    ds = TabularDataset(df)

    assert isinstance(ds, pd.DataFrame)
    assert ds.empty
    assert ds.equals(df)


def test_tabular_dataset_invalid_input():
    # Test that TabularDataset raises error for invalid input
    invalid_obj = object()  # An object() instance cannot be converted to DataFrame
    with pytest.raises(ValueError, match="DataFrame constructor not properly called!"):
        TabularDataset(invalid_obj)
