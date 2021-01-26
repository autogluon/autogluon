import collections
import pandas as pd
import warnings
from typing import Union, Optional, List, Dict
from autogluon_contrib_nlp.base import INT_TYPES, FLOAT_TYPES, BOOL_TYPES
from . import constants as _C


def is_categorical_column(data: pd.Series,
                          valid_data: Optional[pd.Series] = None,
                          threshold: int = 100,
                          ratio: float = 0.1,
                          is_label: bool = False) -> bool:
    """Check whether the column is a categorical column.

    If the number of unique elements in the column is smaller than

        min(#Total Sample * ratio, threshold),

    it will be treated as a categorical column.

    Parameters
    ----------
    data
        The column data
    valid_data
        Additional validation data
    threshold
        The threshold for detecting categorical column
    ratio
        The ratio for detecting categorical column
    is_label
        Whether the column is a label column.

    Returns
    -------
    is_categorical
        Whether the column is a categorical column
    """
    if data.dtype.name == 'category':
        return True
    else:
        threshold = min(int(len(data) * ratio), threshold)
        unique_values = data.unique()
        if len(unique_values) < threshold:
            if is_label:
                # If it is a label column, we will usually treat it as a categorical column.
                # We will use the validation data as additional verification of whether the
                # inference result is correct or not.
                if valid_data is not None:
                    unique_values_set = set(unique_values)
                    valid_unique_values = valid_data.unique()
                    for ele in valid_unique_values:
                        if ele not in unique_values_set:
                            return False
            return True
        return False


def is_numerical_column(data: pd.Series,
                        valid_data: Optional[pd.Series] = None):
    """Try to identify if a column is a numerical column.

    We adopted a very simple rule to verify if the column is a numerical column.

    Parameters
    ----------
    data
        The training data series
    valid_data
        The validation data series

    Returns
    -------
    is_numerical
        Whether the column is a numerical column
    """
    try:
        numerical_data = pd.to_numeric(data)
        if valid_data is not None:
            numerical_valid_data = pd.to_numeric(valid_data)
        return True
    except:
        return False


def infer_problem_type(label_column_property):
    """Infer the type of the problem based on the column property

    Parameters
    ----------
    label_column_property

    Returns
    -------
    problem_type
        classification or regression
    problem_label_shape
        For classification problem it will be the number of classes.
        For regression problem, it will be the label shape.
    """
    if label_column_property.type == _C.CATEGORICAL:
        return _C.CLASSIFICATION, label_column_property.num_class
    elif label_column_property.type == _C.NUMERICAL:
        return _C.REGRESSION, label_column_property.shape
    else:
        raise NotImplementedError


def infer_column_types(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        label_columns: Union[str, List[str]],
        problem_type: Optional[str] = None,
        provided_column_types: Optional[Dict] = None) -> collections.OrderedDict:
    """Infer the column types of the data frame

    Parameters
    ----------
    train_df
        The training Pandas DataFrame
    valid_df
        The validation Pandas DataFrame
    label_columns
        The chosen label column names
    problem_type
        The type of the problem
    provided_column_types
        Additional dictionary that you can use to specify the columns types that you know.
        {'col_name': TYPE}

    Returns
    -------
    column_types
        Dictionary of column types
        If the column does not contain any useful information, we will filter the column with
        type = NULL
    """
    if label_columns is None:
        label_columns_set = set()
    elif isinstance(label_columns, str):
        label_columns_set = set([label_columns])
    elif isinstance(label_columns, (list, tuple)):
        label_columns_set = set(label_columns)
    else:
        raise NotImplementedError
    column_types = collections.OrderedDict()
    # Process all feature columns

    for col_name in train_df.columns:
        if provided_column_types is not None and col_name in provided_column_types:
            column_types[col_name] = provided_column_types[col_name]
            continue
        if col_name in label_columns_set:
            num_train_missing = train_df[col_name].is_null().sum()
            num_valid_missing = valid_df[col_name].is_null().sum()
            if num_train_missing > 0:
                raise ValueError(f'Label column "{col_name}" contains missing values in the '
                                 f'training data frame. You may want to filter your data because '
                                 f'missing label is currently not supported.')
            if num_valid_missing > 0:
                raise ValueError(f'Label column "{col_name}" contains missing values in the '
                                 f'validation data frame. You may want to filter your data because '
                                 f'missing label is currently not supported.')
            if problem_type == _C.CLASSIFICATION:
                column_types[col_name] = _C.CATEGORICAL
                continue
            elif problem_type == _C.REGRESSION:
                column_types[col_name] = _C.NUMERICAL
                continue
        # Identify columns that provide no information
        idx = train_df[col_name].first_valid_index()
        if idx is None or len(train_df[col_name].unique()) == 1:
            # No valid index, thus, we will just ignore the column
            if col_name not in label_columns_set:
                column_types[col_name] = _C.NULL
            else:
                warnings.warn(f'Label column "{col_name}" contains only one label. You may want'
                              f' to check your dataset again.')
        ele = train_df[col_name][idx]
        # Try to inference the categorical column

    return column_properties
