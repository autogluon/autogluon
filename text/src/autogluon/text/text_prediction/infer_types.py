import collections
import pandas as pd
import warnings
from typing import Union, Optional, List, Dict
from autogluon_contrib_nlp.base import INT_TYPES, FLOAT_TYPES, BOOL_TYPES
from autogluon.core.constants import MULTICLASS, BINARY, REGRESSION
from .constants import NULL, CATEGORICAL, NUMERICAL, TEXT


def is_categorical_column(data: pd.Series,
                          valid_data: pd.Series,
                          threshold: int = 20,
                          ratio: float = 0.1,
                          oov_ratio_threshold: float = 0.1,
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
        The ratio detecting categorical column
    oov_ratio_threshold
        The out-of-vocabulary ratio between training and validation.
        This is used to determine if the column is a categorical column.
        Usually, a categorical column can tolerate a small OOV ratio
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
        data_value_counts = data.value_counts(dropna=False)
        key_set = set(data_value_counts.keys())
        if len(data_value_counts) < threshold:
            valid_value_counts = valid_data.value_counts(dropna=False)
            total_valid_num = len(valid_data)
            oov_num = 0
            for k, v in zip(valid_value_counts.keys(), valid_value_counts.values()):
                if k not in key_set:
                    oov_num += v
            if oov_num / total_valid_num > oov_ratio_threshold:
                return False
            if is_label and oov_num != 0:
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


def infer_column_problem_types(
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
        label_set = set()
    elif isinstance(label_columns, str):
        label_set = set([label_columns])
    elif isinstance(label_columns, (list, tuple)):
        label_set = set(label_columns)
    else:
        raise NotImplementedError
    column_types = collections.OrderedDict()
    # Process all feature columns

    for col_name in train_df.columns:
        is_label = col_name in label_set
        if provided_column_types is not None and col_name in provided_column_types:
            column_types[col_name] = provided_column_types[col_name]
            continue
        if is_label:
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
            if problem_type == MULTICLASS or problem_type == BINARY:
                column_types[col_name] = CATEGORICAL
                continue
            elif problem_type == REGRESSION:
                column_types[col_name] = NUMERICAL
                continue
        # Identify columns that provide no information
        idx = train_df[col_name].first_valid_index()
        if idx is None or len(train_df[col_name].unique()) == 1:
            # No valid index, thus, we will just ignore the column
            if not is_label:
                column_types[col_name] = NULL
            else:
                warnings.warn(f'Label column "{col_name}" contains only one label. You may want'
                              f' to check your dataset again.')
        ele = train_df[col_name][idx]
        # Try to inference the categorical column
        if is_categorical_column(train_df[col_name], valid_df[col_name],
                                 is_label=is_label):
            column_types[col_name] = CATEGORICAL
        elif is_numerical_column(train_df[col_name], valid_df[col_name]):
            column_types[col_name] = NUMERICAL
        else:
            column_types[col_name] = TEXT
    return column_types


def infer_problem_type(column_types, label_column, train_df):
    """Inference the type of the problem based on type of the column and
    the training and validation data.

    Parameters
    ----------
    column_types
        Type of the columns
    label_column
        The label column
    train_df
        The training dataframe

    Returns
    -------
    problem_type
        Type of the problem
    """
    if column_types[label_column] == CATEGORICAL:
        if len(train_df[label_column].value_counts()) == 2:
            return BINARY
        else:
            return MULTICLASS
    elif column_types[label_column] == NUMERICAL:
        return REGRESSION
    else:
        raise ValueError(f'The label column "{label_column}" has type'
                         f' "{column_types[label_column]}" and is supported yet.')
