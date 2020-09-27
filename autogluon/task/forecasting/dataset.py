from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
import pandas as pd


__all__ = ["ForecastingDataset", "transform_tabular_to_gluonts_dataset", "gluonts_builtin_datasets"]


def gluonts_builtin_datasets():
    return list(dataset_recipes.keys())


def transform_tabular_to_gluonts_dataset(X, index_column, date_column, target_column):
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

    def __init__(self, dataset_name="electricity", is_gluonts=True, train_path=None, test_path=None,
                 index_column="index", date_column="date", target_column="target"):
        if dataset_name not in gluonts_builtin_datasets():
            raise ValueError("Dataset {} is not available in gluonts.\n"
                             " Current available ones:\n"
                             "{}".format(dataset_name, gluonts_builtin_datasets()))
        if train_path is not None and test_path is not None and is_gluonts:
            is_gluonts = False
            print("Warning: is_gluonts is set to False since train_path and test_path are specified.")
        if is_gluonts:
            try:
                dataset = get_dataset(dataset_name)
            except:
                dataset = get_dataset(dataset_name, regenerate=True)
            self.train_ds = dataset.train
            self.test_ds = dataset.test
            self.freq = dataset.metadata.freq
            self.prediction_length = dataset.metadata.prediction_length
        else:
            self.train_ds, freq = transform_tabular_to_gluonts_dataset(pd.read_csv(train_path),
                                                                 index_column=index_column,
                                                                 date_column=date_column,
                                                                 target_column=target_column)
            self.test_ds, _ = transform_tabular_to_gluonts_dataset(pd.read_csv(test_path),
                                                                index_column=index_column,
                                                                date_column=date_column,
                                                                target_column=target_column)
            self.freq = freq
            self.prediction_length = len(list(self.test_ds)[0]["target"]) - len(list(self.train_ds)[0]["target"])


if __name__ == '__main__':
    dataset = ForecastingDataset(train_path="/Users/yixiaxia/Desktop/亚马逊工作/autogluon/examples/tabular/COV19/processed_train.csv",
                                 test_path="/Users/yixiaxia/Desktop/亚马逊工作/autogluon/examples/tabular/COV19/processed_test.csv",
                                 index_column="name",
                                 target_column="ConfirmedCases",
                                 date_column="Date")
    print(dataset.freq, dataset.prediction_length)