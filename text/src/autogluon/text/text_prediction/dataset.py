import collections
import collections.abc
import numpy as np
import pandas as pd
import json
from autogluon.core.utils.loaders import load_pd
from . import constants as _C
from .column_property import CategoricalColumnProperty, TextColumnProperty, NumericalColumnProperty,\
    get_column_properties_from_metadata
from autogluon_contrib_nlp.base import INT_TYPES, FLOAT_TYPES, BOOL_TYPES
from typing import List, Optional, Union, Dict, Tuple



# TODO(sxjscience) Later when we have autogluon.core.features, try to revise the code here.

def infer_column_types(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        label_columns: Union[str, List[str]],
        problem_type: Optional[str] = None,
        provided_column_types: Optional[Dict] = None) -> collections.OrderedDict:
    """Inference the column types of the data frame

    Parameters
    ----------
    train_df
        The training Pandas DataFrame
    valid_df
        The validation Pandas DataFrame
    label_columns
        The chosen column names of the table
    problem_type
        The type of the problem
    provided_column_types
        Additional dictionary that you can use to
        {'col_name': {'type': type_string}}

    Returns
    -------
    column_types
        Dictionary of column types

    """
    if label_columns is None:
        label_columns_set = set()
    elif isinstance(label_columns, str):
        label_columns_set = set([label_columns])
    else:
        label_columns_set = set(label_columns)
    column_types = collections.OrderedDict()
    # Process all feature columns
    column_properties_from_metadata = get_column_properties_from_metadata(metadata)
    for col_name in train_df.columns:
        if provided_column_properties is not None and col_name in provided_column_properties:
            column_properties[col_name] = provided_column_properties[col_name].clone()
            column_properties[col_name].parse(df[col_name])
            continue
        if col_name in column_properties_from_metadata:
            column_properties[col_name] = column_properties_from_metadata[col_name].clone()
            column_properties[col_name].parse(df[col_name])
            continue
        idx = df[col_name].first_valid_index()
        if idx is None:
            # No valid index, it should have been handled previously
            raise ValueError('Column Name="{}" has no valid data and is ignored.'.format(col_name))
        ele = df[col_name][idx]
        # Try to inference the categorical column
        if isinstance(ele, collections.abc.Hashable) and not isinstance(ele, FLOAT_TYPES):
            # Try to tell if the column is a categorical column
            is_categorical, allow_missing = is_categorical_column(
                df[col_name],
                is_label_columns=col_name in label_columns_set)
            if is_categorical:
                column_properties[col_name] = CategoricalColumnProperty(allow_missing=allow_missing)
                column_properties[col_name].parse(df[col_name])
                continue
        if isinstance(ele, str):
            column_properties[col_name] = TextColumnProperty()
            column_properties[col_name].parse(df[col_name])
            continue
        # Raise error if we find an entity column
        if isinstance(ele, list):
            if isinstance(ele[0], (tuple, dict)):
                raise ValueError('An Entity column "{}" is found but no metadata is given.'
                                 .format(col_name))
        elif isinstance(ele, dict):
            raise ValueError('An Entity column "{}" is found but no metadata is given.'
                             .format(col_name))
        column_properties[col_name] = NumericalColumnProperty()
        column_properties[col_name].parse(df[col_name])
    return column_properties


def normalize_df(df, convert_text_to_numerical=True, remove_none=True):
    """Try to convert the text columns in the input data-frame to numerical columns

    Parameters
    ----------
    df
        The DataFrame
    convert_text_to_numerical
        Whether to convert text columns to numerical columns
    remove_none
        Whether to try to remove None values in the sample.

    Returns
    -------
    new_df
        The normalized dataframe
    """
    conversion_cols = dict()
    for col_name in df.columns:
        col = df[col_name]
        idx = col.first_valid_index()
        if idx is not None:
            val = col[idx]
            if isinstance(val, str):
                num_missing = col.isnull().sum().sum().item()
                if num_missing > 0 and remove_none:
                    col = col.fillna('')
                    conversion_cols[col_name] = col
                if convert_text_to_numerical:
                    try:
                        new_col = pd.to_numeric(col)
                        conversion_cols[col_name] = new_col
                    except Exception:
                        pass
                    finally:
                        pass
    if len(conversion_cols) == 0:
        return df
    else:
        new_df = df.copy()
        for col_name in conversion_cols:
            new_df[col_name] = conversion_cols[col_name]
        return new_df


def infer_problem_type(column_types, label_column):
    """

    Parameters
    ----------
    column_types
        Types of the columns
    label_column
        Name of the label column

    Returns
    -------
    problem_type
        Type of the problem
    label_shape
        Shape of the label
    """
    if column_properties[label_col_name].type == _C.CATEGORICAL:
        return _C.CLASSIFICATION, column_properties[label_col_name].num_class
    elif column_properties[label_col_name].type == _C.NUMERICAL:
        return _C.REGRESSION, column_properties[label_col_name].shape
    else:
        raise NotImplementedError('Cannot infer the problem type')


class TabularDataset:
    def __init__(self, path_or_df: Union[str, pd.DataFrame],
                 *,
                 columns=None,
                 label_columns=None,
                 column_metadata: Optional[Union[str, Dict]] = None,
                 column_properties: Optional[collections.OrderedDict] = None,
                 categorical_default_handle_missing_value=True):
        """

        Parameters
        ----------
        path_or_df
            The path or dataframe of the tabular dataset for NLP.
        columns
            The chosen columns to load the data
        label_columns
            The name of the label columns. This helps to infer the column properties.
        column_metadata
            The metadata object that describes the property of the columns in the dataset
        column_properties
            The given column properties
        categorical_default_handle_missing_value
            Whether to handle missing value in categorical columns by default
        """
        super().__init__()
        if isinstance(path_or_df, pd.DataFrame):
            df = path_or_df
        else:
            df = load_pd.load(path_or_df)
        if columns is not None:
            if not isinstance(columns, list):
                columns = [columns]
            df = df[columns]
        df = normalize_df(df)
        if column_metadata is None:
            column_metadata = dict()
        elif isinstance(column_metadata, str):
            with open(column_metadata, 'r') as f:
                column_metadata = json.load(f)
        # Inference the column properties
        column_properties = get_column_properties(
            df,
            metadata=column_metadata,
            label_columns=label_columns,
            provided_column_properties=column_properties,
            categorical_default_handle_missing_value=categorical_default_handle_missing_value)
        for col_name, prop in column_properties.items():
            if prop.type == _C.TEXT:
                df[col_name] = df[col_name].fillna('').apply(str)
            elif prop.type == _C.NUMERICAL:
                df[col_name] = df[col_name].fillna(-1).apply(np.array)
        self._table = df
        self._column_properties = column_properties

    @property
    def columns(self):
        return list(self._table.columns)

    @property
    def table(self):
        return self._table

    @property
    def column_properties(self):
        return self._column_properties

    def __str__(self):
        ret = 'Columns:\n\n'
        for col_name in self.column_properties.keys():
            ret += '- ' + str(self.column_properties[col_name])
        ret += '\n'
        return ret
