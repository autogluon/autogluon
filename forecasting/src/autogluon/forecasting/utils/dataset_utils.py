import pandas as pd
from copy import deepcopy
from gluonts.dataset.repository.datasets import dataset_recipes
from gluonts.dataset.common import Dataset, ListDataset, FileDataset
from gluonts.dataset.field_names import FieldName
from autogluon.features.generators import IdentityFeatureGenerator, CategoryFeatureGenerator, FillNaFeatureGenerator
from autogluon.core.features.types import R_INT, R_FLOAT, R_CATEGORY
import logging

__all__ = ['TimeSeriesDataset', 'gluonts_builtin_datasets', 'rebuild_tabular', 'time_series_dataset', 'train_test_split_dataframe',
           'train_test_split_gluonts']

logger = logging.getLogger()


class TimeSeriesDataset(ListDataset):

    def __init__(self, df, index_column=None, static_features=None, prev_inferred=None):
        """
        transform a dataframe in the following form to a gluon-ts dataset
        >>> X
              index_column  2020-01-22  2020-01-23  2020-01-24
        0            A           1           2           3
        1            C           1           2           3
        2            B           1           2           3

        """
        if static_features is None:
            self.static_cat_features = None
            self.static_real_features = None
            self.cardinality = None
        else:
            self.static_cat_features, self.static_real_features, self.cardinality = extract_static_feature(index_column, static_features, prev_inferred=prev_inferred)

        if index_column is not None:
            target = df.drop(index_column, axis=1)
            self.index = df[index_column].values
        else:
            target = df.copy()
            self.index = ["index_column"]
        # check whether index in static features corresponds to index in data
        if static_features is not None:
            if sorted(static_features[index_column]) != sorted(self.index):
                raise ValueError(f"Index column does not match between static features and the data given.")
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
                FieldName.ITEM_ID: item_id,
                FieldName.FEAT_STATIC_CAT: self.static_cat_features.loc[self.static_cat_features[index_column].isin([item_id])].drop(index_column, axis=1).values[0] if self.static_cat_features is not None else [],
                FieldName.FEAT_STATIC_REAL: self.static_real_features.loc[self.static_real_features[index_column].isin([item_id])].drop(index_column, axis=1).values[0] if self.static_real_features is not None else [],
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

    def use_feat_static_cat(self):
        if self.static_cat_features is None:
            return False
        else:
            return len(self.static_cat_features) != 0

    def use_feat_static_real(self):
        if self.static_real_features is None:
            return False
        else:
            return len(self.static_real_features) != 0

    def static_cat_columns(self):
        return self.static_cat_features.columns if self.static_cat_features is not None else None

    def static_real_columns(self):
        return self.static_real_features.columns if self.static_real_features is not None else None

    def get_static_cat_cardinality(self):
        return self.cardinality


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
    # check for uniform date
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
        # check for dataset with multiple targets for a certain time series/time
        if any(df[[index_column, time_column]].value_counts() > 1):
            for combination in df[[index_column, time_column]].value_counts().index:
                raise ValueError(f"Containing multiple targets for time series {combination[0]} and time {combination[1]}. "
                                 "Please check your dataset.")
        # check whether need auto-padding
        need_padding = False
        if any(df[time_column].value_counts() != len(df[index_column].unique())):
            logger.log(30, "Warning: autogluon requires each index to be observed over the same set of time values. \n"
                           "As this is not the case in your data, we are automatically padding the dataset with all missing (index, time) combinations which may take some time. \n"
                           "To do this yourself, simply add rows to the Dataframe with target value = NA for each missing (index,time) combination.")
            need_padding = True

        for time in time_list:
            tmp = df[df[time_column] == time][[index_column, time_column, target_column]]
            tmp = tmp.pivot(index=index_column, columns=time_column, values=target_column)
            # automatically padding with NAN if for some time series missing a date target if needed
            if need_padding:
                for index in data_dic[index_column]:
                    if index not in tmp.index:
                        tmp.loc[index, time] = None
            tmp_values = tmp[time].values
            data_dic[time] = tmp_values
        return pd.DataFrame(data_dic)

    X = reshape_dataframe(X)
    return X

# TODO: Improve the way to do the split
def train_test_split_dataframe(data, prediction_length):
    test_ds = data.copy()
    train_ds = data.iloc[:, :-prediction_length]
    if all(data.iloc[:, prediction_length:]):
        logger.log(30, "Warning: All targets used for validation is NAN.")
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


def time_series_dataset(data,
                        index_column=None,
                        target_column="target",
                        time_column="date",
                        chosen_ts=None,
                        static_features=None,
                        prev_inferred=None):
    rebuilt_data = rebuild_tabular(data,
                                   index_column=index_column,
                                   target_column=target_column,
                                   time_column=time_column)
    if index_column is None:
        index_column = "index_column"
    if chosen_ts is not None:
        rebuilt_data = rebuilt_data.loc[rebuilt_data[index_column].isin(chosen_ts)]
        if static_features is not None:
            static_features = static_features.loc[static_features[index_column].isin(chosen_ts)]
    return TimeSeriesDataset(rebuilt_data, index_column=index_column, static_features=static_features, prev_inferred=prev_inferred)


def extract_static_feature(index_column, features, prev_inferred=None):
    if prev_inferred is not None:
        logger.log(30, "Using previous inferred feature columns...")
        logger.log(30, f"Static Cat Features Dataframe including {list([prev_inferred['static_cat_columns']])}")
        logger.log(30, f"Static Real Features Dataframe including {list(prev_inferred['static_real_columns'])}")
        static_cat_features = features[prev_inferred["static_cat_columns"]]
        static_real_features = features[prev_inferred["static_real_columns"]]
        cardinality = prev_inferred["cardinality"]
    else:
        if index_column is None:
            raise ValueError("Index column is not given for static features.")
        indices = features[index_column]
        features = features.drop(index_column, axis=1)
        cardinality = []
        for column in features.columns:
            try:
                features[column] = features[column].astype(R_FLOAT)
                if len(features[column].unique()) <= 10:
                    logger.log(30, f"static feature column {column} has 10 or less unique values, assuming it is categorical.")
                    features[column] = features[column].astype(R_CATEGORY)
                    cardinality.append(len(features[column].unique()))
            except ValueError:
                logger.log(30, f"Cannot convert column {column} to float, assuming it is categorical.")
                if len(features[column].unqiue()) == len(features[column]):
                    logger.log(30, f"Categorical feature {column} has different values for all rows, discarding it.")
                else:
                    cardinality.append(len(features[column].unique()))
        # Extracting static real features and fillna with mean
        static_real_features = IdentityFeatureGenerator(
            infer_features_in_args={"valid_raw_types": [R_INT, R_FLOAT]}
        ).fit_transform(features)
        for column in static_real_features:
            static_real_features[column].fillna(static_real_features[column].mean(), inplace=True)
        # Extracting static cat features, na is deal with as an additional category.
        static_cat_features = IdentityFeatureGenerator(
            infer_features_in_args={"invalid_raw_types": [R_INT, R_FLOAT]}
        ).fit_transform(features)
        static_cat_features = CategoryFeatureGenerator().fit_transform(static_cat_features)

        static_cat_features[index_column] = indices
        static_real_features[index_column] = indices
    return static_cat_features, static_real_features, cardinality
