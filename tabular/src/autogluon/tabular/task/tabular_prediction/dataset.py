
import pandas as pd

from autogluon.core.utils import warning_filter

with warning_filter():
    from autogluon.core.utils.loaders import load_pd

__all__ = ['TabularDataset']


class TabularDataset(pd.DataFrame):
    """
    A dataset in tabular format (with rows = samples, columns = features/variables). 
    This object is essentially a pandas DataFrame (with some extra attributes) and all existing pandas methods can be applied to it. 
    For full list of methods/attributes, see pandas Dataframe documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

    Parameters
    ----------
    data : :class:`pd.DataFrame` or str
        Path to data file (csv or parquet) or loaded :class:`pd.DataFrame`.
        If you already have your data in a :class:`pd.DataFrame`, you can directly provide it by specifying `data`.

    Attributes
    ----------
    file_path: (str)
        Path to data file from which this `TabularDataset` was created.
        None if `data` was a :class:`pd.DataFrame`.
    
    Note: In addition to these attributes, `TabularDataset` also shares all the same attributes and methods of a pandas Dataframe. 
    For a detailed list, see:  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

    Examples
    --------
    >>> from autogluon.tabular import TabularDataset
    >>> train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    >>> test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    >>> train_data.head(30)
    >>> train_data.columns
    """

    _metadata = ['file_path']  # preserved properties that will be copied to a new instance of TabularDataset

    @property
    def _constructor(self):
        return TabularDataset

    @property
    def _constructor_sliced(self):
        return pd.Series

    def __init__(self, data, **kwargs):
        if isinstance(data, str):
            file_path = data
            data = load_pd.load(file_path)
        else:
            file_path = None
        super().__init__(data, **kwargs)
        self.file_path = file_path
