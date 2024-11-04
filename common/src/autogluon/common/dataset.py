import pandas as pd

from .loaders import load_pd

__all__ = ["TabularDataset"]


# FIXME: Add unit tests
class TabularDataset(pd.DataFrame):
    """
    A dataset in tabular format (with rows = samples, columns = features/variables).
    This object returns a pandas DataFrame when initialized and all existing pandas methods can be applied to it.
    For full list of methods/attributes, see pandas Dataframe documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

    Parameters
    ----------
    data : :class:`pd.DataFrame` or str
        If str, path to data file (CSV or Parquet format).
        If you already have your data in a :class:`pd.DataFrame`, you can specify it here.

    Note: In addition to these attributes, `TabularDataset` also shares all the same attributes and methods of a pandas Dataframe.
    For a detailed list, see:  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

    Examples
    --------
    >>> import pandas as pd
    >>> from autogluon.common import TabularDataset
    >>> train_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
    >>> train_data_pd = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
    >>> assert isinstance(train_data, pd.DataFrame)  # True
    >>> assert train_data.equals(train_data_pd)  # True
    >>> assert type(train_data) == type(train_data_pd)  # True
    """

    def __new__(cls, data, **kwargs):
        if isinstance(data, str):
            data = load_pd.load(data)
        return pd.DataFrame(data, **kwargs)
