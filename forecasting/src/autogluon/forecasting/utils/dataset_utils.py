from gluonts.dataset.repository.datasets import dataset_recipes
from ..task.forecasting.dataset import TimeSeriesDataset
import pandas as pd

__all__ = ['gluonts_builtin_datasets', 'rebuild_tabular', 'time_series_dataset', 'train_test_split']


def gluonts_builtin_datasets():
    return list(dataset_recipes.keys())


def rebuild_tabular(X, time_column, target_column, index_column=None):
    """
    X: dataframe to rebuild, should have the form of:
    >>> X
      index_column time_column  target_column
    0            A  2020-01-22              1
    1            A  2020-01-23              2
    2            A  2020-01-24              3
    3            B  2020-01-22              1
    4            B  2020-01-23              2
    5            B  2020-01-24              3
    6            C  2020-01-22              1
    7            C  2020-01-23              2
    8            C  2020-01-24              3

    index_column: time series index, in the above example, there are three ts: A, B, C,
                  if index_column is None, we will assume that the dataset contains only one time series

    time_column: time of a data, in the form "YYYY-MM-DD HH:MM:SS", we are assuming that each time series contains the same time sequence,
                 and the freq in each time series does not change.

    target_column: values used for prediction, integers.

    output:
    a new dataframe in the form that each line contains a time series
    transformed example would be:
    >>> X
          index_column  2020-01-22  2020-01-23  2020-01-24
    0            A           1           2           3
    1            C           1           2           3
    2            B           1           2           3

    """
    if index_column is None:
        X = X[[time_column, target_column]]
        X["index_column"] = ["time_series" for i in range(X.shape[0])]
        index_column = "index_column"
    time_list = sorted(list(set(X[time_column])))
    freq = pd.infer_freq(time_list)
    if freq is None:
        raise ValueError("Freq cannot be inferred. Check your dataset.")

    def reshape_dataframe(df):
        """
        for each time occurs in the dataset, we select the target value corresponding to
        each time series, and use dataframe.pivot() to convert it to one column, where the column name is the
        time, each row is the corresponding target value for each time series.
        """
        df = df.sort_values(by=index_column)
        data_dic = {index_column: sorted(list(set(df[index_column])))}

        for time in time_list:
            tmp = df[df[time_column] == time][[index_column, time_column, target_column]]
            tmp = tmp.pivot(index=index_column, columns=time_column, values=target_column)
            tmp_values = tmp[time].values
            data_dic[time] = tmp_values
        return pd.DataFrame(data_dic)

    X = reshape_dataframe(X)
    return X


def train_test_split(df, prediction_length):
    test_ds = df.copy()
    train_ds = df.iloc[:, :-prediction_length]
    return train_ds, test_ds


def time_series_dataset(data, index_column=None, target_column="target", time_column="date"):
    rebuilt_data = rebuild_tabular(data,
                                   index_column=index_column,
                                   target_column=target_column,
                                   time_column=time_column)
    if index_column is None:
        index_column = "index_column"
    return TimeSeriesDataset(rebuilt_data, index_column=index_column)