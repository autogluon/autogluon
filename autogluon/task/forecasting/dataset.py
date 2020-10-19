from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
import pandas as pd

__all__ = ["ForecastingDataset", "transform_tabular_to_gluonts_dataset", "gluonts_builtin_datasets"]


def gluonts_builtin_datasets():
    return list(dataset_recipes.keys())


def rebuild_tabular(X, index_column, date_column, target_column):
    """
    X: dataframe to rebuild, should have the form of:
    >>> X
      index_column date_column  target_column
    0            A  2020-01-22              1
    1            A  2020-01-23              2
    2            A  2020-01-24              3
    3            B  2020-01-22              1
    4            B  2020-01-23              2
    5            B  2020-01-24              3
    6            C  2020-01-22              1
    7            C  2020-01-23              2
    8            C  2020-01-24              3
    index_column: time series index, in the above example, there are three ts: A, B, C
    date_column: date of a data
    target_column: the predict target

    output:
    a new time series in the form that each line contains a time series
    transformed example would be:
    >>> X
          index_column  2020-01-22  2020-01-23  2020-01-24
    0            A           1           2           3
    1            C           1           2           3
    2            B           1           2           3

    """
    date_list = sorted(list(set(X[date_column])))
    freq = pd.infer_freq(date_list)

    def reshape_dataframe(df):
        data_dic = {index_column: list(set(df[index_column]))}

        for date in date_list:
            tmp = df[df[date_column] == date][[index_column, date_column, target_column]]
            tmp = tmp.pivot(index=index_column, columns=date_column, values=target_column)
            tmp_values = tmp[date].values
            data_dic[date] = tmp_values
        return pd.DataFrame(data_dic)

    X = reshape_dataframe(X)
    return X, freq


def transform_tabular_to_gluonts_dataset(X, freq, index_column):
    """
    transform a dataframe in the following form to a gluon-ts dataset
    >>> X
          index_column  2020-01-22  2020-01-23  2020-01-24
    0            A           1           2           3
    1            C           1           2           3
    2            B           1           2           3

    """
    date_list = X.columns[1:]
    target_values = X.drop(index_column, axis=1).values
    processed_X = ListDataset([
        {
            FieldName.TARGET: target,
            FieldName.START: pd.Timestamp(date_list[0], freq=freq),
        }
        for (target,) in zip(target_values, )
    ], freq=freq)
    return processed_X, freq


class ForecastingDataset:

    def __init__(self, dataset_name="electricity", is_gluonts=True, train_path=None, test_path=None, is_train=True, prediction_length=None,
                 index_column="index", date_column="date", target_column="target"):
        """
        dataset used for Forecasting Task

        dataset_name: you can choose a built-in dataset in gluon-ts, default is the "electricity" dataset in gluon-ts

        is_gluonts: boolean to determine whether the dataset is from gluon-ts or not, if train_path is
                    specified, is_gluonts will be automatically turned to False

        train_path: path to the train data

        test_path: path to the test data, if not specified while train_path is specified, automatic split will be performed

        is_train: whether the dataset is used for training, splitting will only happen when is_train == True

        prediction_length: prediction_length for the forecasting task, need to be specified for automatic splitting

        index_column: time series index, in the above example, there are three ts: A, B, C

        date_column: date of a data

        target_column: the predict target
        """
        self.is_train = is_train
        if dataset_name not in gluonts_builtin_datasets():
            raise ValueError("Dataset {} is not available in gluonts.\n"
                             " Current available ones:\n"
                             "{}".format(dataset_name, gluonts_builtin_datasets()))
        if train_path is not None and is_gluonts:
            is_gluonts = False
            print("Warning: is_gluonts is set to False since train_path and test_path are specified.")
        if is_gluonts:
            try:
                dataset = get_dataset(dataset_name)
            except:
                dataset = get_dataset(dataset_name, regenerate=True)
            self.is_train = True
            self.train_ds = dataset.train
            self.test_ds = dataset.test
            self.freq = dataset.metadata.freq
            self.prediction_length = dataset.metadata.prediction_length
        else:
            train_csv, freq = rebuild_tabular(pd.read_csv(train_path), index_column, date_column, target_column)
            if is_train:
                if test_path is None:
                    if prediction_length is None:
                        raise ValueError("prediction length has to be specified for auto train-test split.")
                    else:
                        train_csv, test_csv = self.train_test_split(train_csv, prediction_length)
                else:
                    test_csv, _ = rebuild_tabular(pd.read_csv(test_path), index_column, date_column, target_column)

                self.train_ds, freq = transform_tabular_to_gluonts_dataset(X=train_csv,
                                                                           freq=freq,
                                                                           index_column=index_column)

                self.test_ds, _ = transform_tabular_to_gluonts_dataset(X=test_csv,
                                                                       freq=freq,
                                                                       index_column=index_column)

                self.freq = freq
                self.prediction_length = prediction_length if prediction_length is not None \
                    else len(list(self.test_ds)[0]["target"]) - len(list(self.train_ds)[0]["target"])
            else:
                self.train_ds, freq = transform_tabular_to_gluonts_dataset(X=train_csv,
                                                                           freq=freq,
                                                                           index_column=index_column)
                self.freq = freq
                self.prediction_length = prediction_length
                self.test_ds = None

    def train_test_split(self, ds, prediction_length):

        test_ds = ds.copy()
        train_ds = ds.iloc[:, :-prediction_length]
        return train_ds, test_ds


if __name__ == '__main__':
    dataset = ForecastingDataset(train_path="/Users/yixiaxia/Desktop/亚马逊工作/autogluon/examples/tabular/COV19/processed_train.csv",
                                 test_path="/Users/yixiaxia/Desktop/亚马逊工作/autogluon/examples/tabular/COV19/processed_test.csv",
                                 # prediction_length=19,
                                 index_column="name",
                                 target_column="ConfirmedCases",
                                 date_column="Date")
    print(dataset.freq, dataset.prediction_length)