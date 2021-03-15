import pandas as pd
from copy import deepcopy
from gluonts.dataset.repository.datasets import dataset_recipes
from gluonts.dataset.common import Dataset, ListDataset, FileDataset
from gluonts.dataset.field_names import FieldName


__all__ = ['TimeSeriesDataset', 'gluonts_builtin_datasets', 'rebuild_tabular', 'time_series_dataset', 'train_test_split_dataframe',
           'train_test_split_gluonts']


class TimeSeriesDataset(ListDataset):

    def __init__(self, df, index_column=None):
        """
        transform a dataframe in the following form to a gluon-ts dataset
        >>> X
              index_column  2020-01-22  2020-01-23  2020-01-24
        0            A           1           2           3
        1            C           1           2           3
        2            B           1           2           3

        """

        if index_column is not None:
            target = df.drop(index_column, axis=1)
            self.index = df[index_column].values
        else:
            target = df.copy()
            self.index = ["index_column"]
        target_values = target.values
        date_list = target.columns
        self.last_date = date_list[-1]
        self.freq = pd.infer_freq(date_list)
        if self.freq is None:
            raise ValueError("Freq cannot be inferred. Check your dataset.")
        data = [
            {
                FieldName.TARGET: target,
                FieldName.START: pd.Timestamp(date_list[0], freq=self.freq),
                FieldName.ITEM_ID: item_id
            }
            for (target, item_id) in zip(target_values, self.index)
        ]
        super(TimeSeriesDataset, self).__init__(data, self.freq)

    def get_freq(self):
        return self.freq

    def get_index(self):
        return self.index

    def get_last_date(self):
        return self.last_date


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


def train_test_split_dataframe(data, prediction_length):
    test_ds = data.copy()
    train_ds = data.iloc[:, :-prediction_length]
    return train_ds, test_ds


def train_test_split_gluonts(data, prediction_length, freq=None):
    train_data_lst = []
    test_data_lst = []

    if freq is None:
        if isinstance(data, ListDataset):
            freq = data.list_data[0]["start"].freq
        else:
            raise ValueError("Cannot infer freq if the data is not ListDataset.")

    for entry in data:
        test_data_lst.append(entry.copy())
        tmp = {}
        for k, v in entry.items():
            if "dynamic" in k.lower():
                tmp[k] = v[..., : -prediction_length]
            elif "target" in k.lower():
                tmp[k] = v[..., : -prediction_length]
            else:
                tmp[k] = deepcopy(v)
        train_data_lst.append(tmp)
    train_ds = ListDataset(train_data_lst, freq=freq)
    test_ds = ListDataset(test_data_lst, freq=freq)
    return train_ds, test_ds


def time_series_dataset(data, index_column=None, target_column="target", time_column="date", chosen_ts=None):
    rebuilt_data = rebuild_tabular(data,
                                   index_column=index_column,
                                   target_column=target_column,
                                   time_column=time_column)
    if index_column is None:
        index_column = "index_column"
    if chosen_ts is not None:
        rebuilt_data = rebuilt_data.loc[rebuilt_data[index_column].isin(chosen_ts)]
    return TimeSeriesDataset(rebuilt_data, index_column=index_column)