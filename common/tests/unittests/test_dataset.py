import pandas as pd

from autogluon.common import TabularDataset


def test_tabular_dataset():
    data = {"col1": [1, 2, 3, 4], "col2": ["a", "b", "b", "c"]}

    df_1 = pd.DataFrame(data)
    df_2 = TabularDataset(data)

    assert isinstance(df_1, pd.DataFrame)
    assert df_1.equals(df_2)
    assert type(df_1) == pd.DataFrame
    assert type(df_1) == type(df_2)
