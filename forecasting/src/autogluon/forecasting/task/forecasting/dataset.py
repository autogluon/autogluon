from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
import pandas as pd


__all__ = ["TimeSeriesDataset"]


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
