from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
import pandas as pd

__all__ = ["TimeSeriesDataset", "transform_tabular_to_gluonts_dataset", "gluonts_builtin_datasets"]


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
        data_dic = {index_column: list(set(df[index_column]))}

        for time in time_list:
            tmp = df[df[time_column] == time][[index_column, time_column, target_column]]
            tmp = tmp.pivot(index=index_column, columns=time_column, values=target_column)
            tmp_values = tmp[time].values
            data_dic[time] = tmp_values
        return pd.DataFrame(data_dic)

    X = reshape_dataframe(X)
    return X, freq


def transform_tabular_to_gluonts_dataset(X, freq, index_column=None):
    """
    transform a dataframe in the following form to a gluon-ts dataset
    >>> X
          index_column  2020-01-22  2020-01-23  2020-01-24
    0            A           1           2           3
    1            C           1           2           3
    2            B           1           2           3

    """
    if index_column is not None:
        target = X.drop(index_column, axis=1)
    else:
        target = X.copy()
    target_values = target.values
    date_list = target.columns
    processed_X = ListDataset([
        {
            FieldName.TARGET: target,
            FieldName.START: pd.Timestamp(date_list[0], freq=freq),
        }
        for (target,) in zip(target_values, )
    ], freq=freq)
    return processed_X, freq


class TimeSeriesDataset:

    def __init__(self, dataset_name=None, train_path=None, test_path=None, is_train=True, prediction_length=None,
                 index_column="index", time_column="date", target_column="target"):
        """
        dataset used for Forecasting Task

        dataset_name: you can choose a built-in dataset in gluon-ts, default is the "electricity" dataset in gluon-ts

        train_path: path to the train data

        test_path: path to the test data, if not specified while train_path is specified, automatic split will be performed

        is_train: whether the dataset is used for training, splitting will only happen when is_train == True

        prediction_length: prediction_length for the forecasting task, need to be specified for automatic splitting

        index_column: time series index, in the above example, there are three ts: A, B, C

        time_column: date of a data

        target_column: the prediction target
        """
        self.is_train = is_train
        if dataset_name is not None and dataset_name not in gluonts_builtin_datasets():
            raise ValueError("Dataset {} is not available in gluonts.\n"
                             " Current available ones:\n"
                             "{}".format(dataset_name, gluonts_builtin_datasets()))

        if dataset_name is not None:
            if dataset_name not in gluonts_builtin_datasets():
                raise ValueError("Dataset {} is not available in gluonts.\n"
                                 " Current available ones:\n"
                                 "{}".format(dataset_name, gluonts_builtin_datasets()))
            try:
                dataset = get_dataset(dataset_name)
            except:
                dataset = get_dataset(dataset_name, regenerate=True)
            self.is_train = True
            self.train_data = dataset.train
            self.test_data = dataset.test
            self.freq = dataset.metadata.freq
            self.prediction_length = dataset.metadata.prediction_length
        else:
            if train_path is None:
                raise ValueError("You must specify one of train_path and dataset_name to create TimeSeriesDataset.")
            train_csv, freq = rebuild_tabular(X=pd.read_csv(train_path),
                                              index_column=index_column,
                                              time_column=time_column,
                                              target_column=target_column)
            if is_train:
                if test_path is None:
                    if prediction_length is None:
                        raise ValueError("prediction length has to be specified for auto train-test split.")
                    else:
                        train_csv, test_csv = self.train_test_split(train_csv, prediction_length)
                else:
                    test_csv, _ = rebuild_tabular(X=pd.read_csv(test_path),
                                                  index_column=index_column,
                                                  time_column=time_column,
                                                  target_column=target_column)

                self.train_data, freq = transform_tabular_to_gluonts_dataset(X=train_csv,
                                                                             freq=freq,
                                                                             index_column=index_column)

                self.test_data, _ = transform_tabular_to_gluonts_dataset(X=test_csv,
                                                                         freq=freq,
                                                                         index_column=index_column)

                self.freq = freq
                self.prediction_length = prediction_length if prediction_length is not None \
                    else len(list(self.test_data)[0]["target"]) - len(list(self.train_data)[0]["target"])
            else:
                self.train_data, freq = transform_tabular_to_gluonts_dataset(X=train_csv,
                                                                             freq=freq,
                                                                             index_column=index_column)
                self.freq = freq
                self.prediction_length = prediction_length
                self.test_data = None

    def train_test_split(self, ds, prediction_length):

        test_ds = ds.copy()
        train_ds = ds.iloc[:, :-prediction_length]
        return train_ds, test_ds
