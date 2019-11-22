import math, warnings
import numpy as np
import pandas as pd 

from autogluon.tabular.utils.loaders import load_pd # TODO: move this
from ...core import * # TODO: needed?

__all__ = ['TabularDataset']

class TabularDataset(pd.DataFrame):
    """
    A dataset in tabular format (with rows = samples, columns = features/variables).
    This object is essentially a pandas DataFrame (with extra slots) and all the Pandas methods can be applied to it. 
    """
    
    _metadata = ['name', 'file_path', 'feature_types'] # preserved properties that will be copied to a new instance of TabularDataset
    
    @property
    def _constructor(self):
        return TabularDataset
    
    @property
    def _constructor_sliced(self):
        return pd.Series
    
    def __init__(self, *args, **kwargs):
        """ 
        Creates a new TabularDataset object.
    Args:
        file_path : (str)
            Path to the data file.
        name : (str)
             Name to assign to dataset (has no effect beyond being accessible via Dataset.name).
        df : (pandas DataFrame)
            If you already have your data in a pandas Dataframe, you can directly provide it by specifying df.
            At least one of file_path and df arguments must be specified when constructing new TabularDataset.
        feature_types : (dict)
            Mapping from column_name to str describing type of each column.
            If not specified, AutoGluon's fit() will automatically infer what type of data each feature contains.
        subsample : (int, default = None)
            If specified, we only keep first K rows of the provided dataset.
    """
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


""" OLD Code.  # TODO: remove

    def __init__(self, name=None, file_path=None, df=None, feature_types=None, subsample=None):
        if df is not None:
            if type(df) != pd.DataFrame:
                raise ValueError("'df' must be existing pandas DataFrame. To read dataset from file instead, use 'file_path' string argument.")
            if file_path is not None:
                warnings.warn("Both 'df' and 'file_path' supplied. Creating dataset based on DataFrame 'df' rather than reading from file_path.")
            df = df.copy()
        elif file_path is not None:
            if type(file_path) != str:
                raise ValueError("'file_path' must be a string specifying location of data file")
            df = load_pd.load(file_path)
        # Dataset df will be empty if neither 'file_path' nor 'df' arguments are specified
        # else:
        #    raise ValueError("One of 'file_path' and 'df' arguments must be specified")
        if subsample is not None:
            if type(subsample) != int:
                raise ValueError("'subsample' must be of type int")
            df = df.head(subsample)
        super().__init__(df)
        # Set attributes:
        self.name = name
        self.file_path = file_path
        self.feature_types = feature_types
"""    
