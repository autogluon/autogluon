import pandas as pd

from .loaders import load_pd
from .savers import save_pd

__all__ = ["TabularDataset"]


class TabularDataset:
    """
    A dataset in tabular format (with rows = samples, columns = features/variables).
    This class returns a :class:`pd.DataFrame` when initialized and all existing pandas methods can be applied to it.
    For full list of methods/attributes, see pandas Dataframe documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

    The purpose of this class is to provide an easy-to-use shorthand for loading a pandas DataFrame to use in AutoGluon.

    Parameters
    ----------
    data : str, :class:`pd.DataFrame`, :class:`np.ndarray`, Iterable, or dict
        If str, path to data file (CSV or Parquet format).
        If you already have your data in a :class:`pd.DataFrame`, you can specify it here. In this case, the same DataFrame will be returned with no changes.

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

    def __new__(cls, data, **kwargs) -> pd.DataFrame:
        if isinstance(data, str):
            data = cls.load(path=data)
        return pd.DataFrame(data, **kwargs)

    @classmethod
    def load(cls, path: str, **kwargs) -> pd.DataFrame:
        return load_pd.load(path, **kwargs)

    @classmethod
    def save(cls, path: str, df: pd.DataFrame, **kwargs):
        save_pd.save(path=path, df=df, **kwargs)
