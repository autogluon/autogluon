import warnings
import pandas as pd

from ...utils import warning_filter
with warning_filter():
    from ...utils.tabular.utils.loaders import load_pd

__all__ = ['TabularDataset']


class TabularDataset(pd.DataFrame):
    """
    A dataset in tabular format (with rows = samples, columns = features/variables). 
    This object is essentially a pandas DataFrame (with some extra attributes) and all existing pandas methods can be applied to it. 
    For full list of methods/attributes, see pandas Dataframe documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

    Parameters
    ----------
    file_path : (str, optional)
        Path to the data file (may be on local filesystem or URL to cloud s3 bucket). 
        At least one of `file_path` and `df` arguments must be specified when constructing a new TabularDataset.
    df : (pandas.DataFrame, optional)
        If you already have your data in a pandas Dataframe, you can directly provide it by specifying `df`. 
        At least one of `file_path` and `df` arguments must be specified when constructing new TabularDataset.
    feature_types : (dict, optional)
        Mapping from column_names to string describing data type of each column. 
        If not specified, AutoGluon's fit() will automatically infer what type of data each feature contains.
    subsample : (int, optional)
        If specified = k, we only keep first k rows of the provided dataset.
    name : (str, optional)
         Optional name to assign to dataset (has no effect beyond being accessible via TabularDataset.name).

    Attributes
    ----------
    name: (str)
        An optional name assigned to this TabularDataset.
    file_path: (str)
        Path to data file from which this TabularDataset was created.
    feature_types: (dict) 
        Maps column-names to string describing the data type of each column in this TabularDataset.
    subsample: (int) 
        Describes size of subsample retained in this TabularDataset (None if this is original dataset).
    
    Note: In addition to these attributes, TabularDataset also shares all attributes and methods of a pandas Dataframe. 
    For detailed list, see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

    Examples
    --------
    >>> from autogluon import TabularPrediction as task  # Note: TabularPrediction.Dataset == TabularDataset.
    >>> train_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/train.csv')
    >>> test_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/test.csv')
    """
    
    _metadata = ['name', 'file_path', 'feature_types', 'subsample'] # preserved properties that will be copied to a new instance of TabularDataset
    
    @property
    def _constructor(self):
        return TabularDataset
    
    @property
    def _constructor_sliced(self):
        return pd.Series
    
    def __init__(self, *args, **kwargs):
        file_path = kwargs.get('file_path', None)
        name = kwargs.get('name', None)
        feature_types = kwargs.get('feature_types', None)
        df = kwargs.get('df', None)
        subsample = kwargs.get('subsample', None)
        construct_from_df = False # whether or not we are constructing new dataset object from scratch based on provided DataFrame.
        # if df is None and file_path is None: # Cannot be used currently!
        #     raise ValueError("Must specify either named argument 'file_path' or 'df' in order to construct tabular Dataset")
        if df is not None: # Create Dataset from existing Python DataFrame:
            construct_from_df = True
            if type(df) != pd.DataFrame:
                raise ValueError("'df' must be existing pandas DataFrame. To read dataset from file instead, use 'file_path' string argument.")
            if file_path is not None:
                warnings.warn("Both 'df' and 'file_path' supplied. Creating dataset based on DataFrame 'df' rather than reading from file_path.")
            df = df.copy(deep=True)
        elif file_path is not None: # Read from file to create dataset
            construct_from_df = True
            df = load_pd.load(file_path)
        if construct_from_df: # Construct new Dataset object based off of DataFrame
            if subsample is not None:
                if type(subsample) != int or subsample <= 1:
                    raise ValueError("'subsample' must be of type int and > 1")
                df = df.head(subsample)
            super().__init__(df)
            self.file_path = file_path
            self.name = name
            self.feature_types = feature_types
            self.subsample = subsample
        else:
            super().__init__(*args, **kwargs)
