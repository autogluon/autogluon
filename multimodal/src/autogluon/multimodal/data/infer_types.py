import collections
import json
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import PIL

from ..constants import (
    AUTOMM,
    BINARY,
    CATEGORICAL,
    CLASSIFICATION,
    IMAGE,
    MULTICLASS,
    NULL,
    NUMERICAL,
    REGRESSION,
    ROIS,
    TEXT,
)

logger = logging.getLogger(AUTOMM)


def is_categorical_column(
    data: pd.Series,
    valid_data: pd.Series,
    threshold: int = None,
    ratio: Optional[float] = None,
    oov_ratio_threshold: Optional[float] = None,
    is_label: bool = False,
) -> bool:
    """
    Identify whether a column is one categorical column.
    If the number of unique elements in the column is smaller than

        min(#Total Sample * ratio, threshold),

    it will be treated as a categorical column.

    Parameters
    ----------
    data
        One column of a multimodal pd.DataFrame for training.
    valid_data
        One column of a multimodal pd.DataFrame for validation.
    threshold
        The threshold for detecting categorical column.
    ratio
        The ratio detecting categorical column.
    oov_ratio_threshold
        The out-of-vocabulary ratio between training and validation.
        This is used to determine if the column is a categorical column.
        Usually, a categorical column can tolerate a small OOV ratio.
    is_label
        Whether the column is a label column.

    Returns
    -------
    Whether the column is a categorical column.
    """
    if data.dtype.name == "category":
        return True
    else:
        if threshold is None:
            if is_label:
                # TODO(?) The following logic will be problematic if the task is few-shot learning.
                threshold = 100
                oov_ratio_threshold = 0
                ratio = 0.1
            else:
                threshold = 20
                oov_ratio_threshold = 0
                ratio = 0.1
        threshold = min(int(len(data) * ratio), threshold)
        data_value_counts = data.value_counts(dropna=False)
        key_set = set(data_value_counts.keys())
        if len(data_value_counts) < threshold:
            valid_value_counts = valid_data.value_counts(dropna=False)
            total_valid_num = len(valid_data)
            oov_num = 0
            for k, v in zip(valid_value_counts.keys(), valid_value_counts.values):
                if k not in key_set:
                    oov_num += v
            if is_label and oov_num != 0:
                return False
            if oov_num / total_valid_num > oov_ratio_threshold:
                return False
            return True
        return False


def is_rois_column(data: pd.Series) -> bool:
    """
    Identify if a column is one rois column.

    Parameters
    ----------
    X
        One column of a multimodal pd.DataFrame for training.

    Returns
    -------
    Whether the column is a rois column.
    """
    idx = data.first_valid_index()
    if isinstance(data[idx], list):
        return (
            len(data[idx])
            and isinstance(data[idx][0], dict)
            and set(["xmin", "ymin", "xmax", "ymax", "class"]).issubset(data[idx][0].keys())
        )
    else:
        isinstance(data[idx], str)
        rois = {}
        try:
            rois = json.loads(data[idx][0])
        except:
            pass
        return (
            rois
            and isinstance(rois, dict)
            and set(["xmin", "ymin", "xmax", "ymax", "class"]).issubset(data[idx][0].keys())
        )


def is_numerical_column(
    data: pd.Series,
    valid_data: Optional[pd.Series] = None,
) -> bool:
    """
    Identify if a column is a numerical column.
    Here it uses a very simple rule to verify if this is a numerical column.

    Parameters
    ----------
    data
        One column of a multimodal pd.DataFrame for training.
    valid_data
        One column of a multimodal pd.DataFrame for validation.

    Returns
    -------
    Whether the column is a numerical column.
    """
    try:
        numerical_data = pd.to_numeric(data)
        if valid_data is not None:
            numerical_valid_data = pd.to_numeric(valid_data)
        return True
    except:
        return False


def is_imagepath_column(
    data: pd.Series,
    col_name: str,
) -> bool:
    """
    Identify if a column is one image-path column.
    Here it counts the failures when trying PIL.Image.open() on a sampled subset.
    If over 90% attempts fail, this column isn't an image-path column.

    Parameters
    ----------
    data
        One column of a multimodal pd.DataFrame for training.
    col_name
        Name of column.

    Returns
    -------
    Whether the column is an image-path column.
    """
    sample_num = min(len(data), 500)
    data = data.sample(n=sample_num, random_state=0)
    data = data.apply(lambda ele: str(ele).split(";")).tolist()
    failure_count = 0
    for image_paths in data:
        success = False
        for img_path in image_paths:
            try:
                img = PIL.Image.open(img_path)
                success = True
                break
            except:
                pass
        if not success:
            failure_count += 1
    failure_ratio = failure_count / sample_num

    # Tolerate high failure rate in case that many image files may be corrupted.
    if failure_ratio <= 0.9:
        if failure_ratio > 0:
            logger.warning(
                f"Among {sample_num} sampled images in column '{col_name}', "
                f"{failure_ratio:.0%} images can't be open. "
                "You may need to thoroughly check your data to see the percentage of missing images, "
                "and estimate the potential influence. By default, we skip the samples with missing images. "
                "You can also set hyperparameter 'data.image.missing_value_strategy' to be 'zero', "
                "which uses a zero image to replace any missing image."
            )
        return True
    else:
        return False


def is_text_column(X: pd.Series) -> bool:
    """
    Identify if a column is one text column.

    Parameters
    ----------
    X
        One column of a multimodal pd.DataFrame for training.

    Returns
    -------
    Whether the column is a text column.
    """
    if len(X) > 5000:
        # Sample to speed-up type inference
        X = X.sample(n=5000, random_state=0)
    X_unique = X.unique()
    num_unique = len(X_unique)
    num_rows = len(X)
    unique_ratio = num_unique / num_rows
    if unique_ratio <= 0.01:
        return False
    try:
        avg_words = pd.Series(X_unique).str.split().str.len().mean()
    except AttributeError:
        return False
    if avg_words < 3:
        return False

    return True


def infer_column_types(
    data: pd.DataFrame,
    valid_data: Optional[pd.DataFrame] = None,
    label_columns: Union[str, List[str]] = None,
    provided_column_types: Optional[Dict] = None,
    allowable_column_types: Optional[List[str]] = None,
    fallback_column_type: Optional[str] = None,
) -> Dict:
    """
    Infer the column types of a multimodal pd.DataFrame.

    Parameters
    ----------
    data
        The multimodal pd.DataFrame for training.
    valid_data
        The multimodal pd.DataFrame for validation.
    label_columns
        The label column names.
    provided_column_types
        Additional dictionary that you can use to specify the columns types that you know.
        {'col_name': TYPE, ...}
    allowable_column_types
        What column types are allowed. This is the prior knowledge inferred from the model type.
    fallback_column_type
        What's the fallback column type if the detected type if out of the allowable_column_types.

    Returns
    -------
    column_types
        A dictionary containing the mappings from column names to their modality types.
        If the column does not contain any useful information, we will set its column type NULL.
    """
    if label_columns is None:
        label_columns = []
    if isinstance(label_columns, str):
        label_columns = [label_columns]

    column_types = collections.OrderedDict()

    if valid_data is None:
        valid_data = data
        is_training = False
    else:
        is_training = True

    for col_name in data.columns:
        if provided_column_types is not None and col_name in provided_column_types:
            column_types[col_name] = provided_column_types[col_name]
            continue
        # Identify columns that provide no information
        idx = data[col_name].first_valid_index()
        if idx is None:
            # No valid index, thus, we will just ignore the column
            column_types[col_name] = NULL
            continue
        if not isinstance(data[col_name][idx], list) and len(data[col_name].unique()) == 1 and is_training:
            column_types[col_name] = NULL
            continue

        if is_rois_column(data[col_name]):
            column_types[col_name] = ROIS
        elif is_categorical_column(
            data[col_name], valid_data[col_name], is_label=col_name in label_columns
        ):  # Infer categorical column
            column_types[col_name] = CATEGORICAL
        elif is_numerical_column(data[col_name], valid_data[col_name]):  # Infer numerical column
            column_types[col_name] = NUMERICAL
        elif is_imagepath_column(data[col_name], col_name):  # Infer image-path column
            column_types[col_name] = IMAGE
        elif is_text_column(data[col_name]):  # Infer text column
            column_types[col_name] = TEXT
        else:  # All the other columns are treated as categorical
            column_types[col_name] = CATEGORICAL

    if allowable_column_types and fallback_column_type:
        column_types = set_fallback_column_type(
            column_types=column_types,
            allowable_column_types=allowable_column_types,
            fallback_column_type=fallback_column_type,
        )

    return column_types


def check_missing_values(
    data: pd.DataFrame,
    column_name: str,
    split: Optional[str] = "",
):
    num_missing_values = data[column_name].isnull().sum()
    if num_missing_values > 0:
        raise ValueError(
            f"Label column '{column_name}' contains missing values in the "
            f"{split} dataframe. You may want to filter your data because "
            "missing label is currently not supported."
        )


def infer_label_column_type_by_problem_type(
    column_types: Dict,
    label_columns: Union[str, List[str]],
    problem_type: str,
    data: Optional[pd.DataFrame] = None,
    valid_data: Optional[pd.DataFrame] = None,
    allowable_label_types: Optional[List[str]] = (CATEGORICAL, NUMERICAL),
    fallback_label_type: Optional[str] = CATEGORICAL,
):
    """
    Infer the label column types based on problem type.

    Parameters
    ----------
    column_types
        Types of columns in a pd.DataFrame.
    label_columns
        The label columns in a pd.DataFrame.
    problem_type
        Type of problem.
    data
        A pd.DataFrame.
    valid_data
        A validation pd.DataFrame.
    allowable_label_types
        Which label types are allowed.
    fallback_label_type
        If a label type is not within the allowable_label_types, replace it with this fallback_label_type.

    Returns
    -------
    Column types with the label columns' types inferred from the problem type.
    """
    if isinstance(label_columns, str):
        label_columns = [label_columns]

    for col_name in label_columns:
        # Make sure the provided label columns are in the dataframe.
        assert (
            col_name in column_types
        ), f"Column {col_name} is not in {column_types.keys()}. Make sure calling `infer_column_types()` first."
        if data is not None:
            check_missing_values(data=data, column_name=col_name, split="training")
        if valid_data is not None:
            check_missing_values(data=valid_data, column_name=col_name, split="validation")
        if column_types[col_name] == NULL:
            raise ValueError(
                f"Label column '{col_name}' contains only one label class. Make sure it has at least two label classes."
            )
        if problem_type in [MULTICLASS, BINARY, CLASSIFICATION]:
            column_types[col_name] = CATEGORICAL
        elif problem_type == REGRESSION:
            column_types[col_name] = NUMERICAL

        if column_types[col_name] not in allowable_label_types:
            column_types[col_name] = fallback_label_type

    return column_types


def infer_problem_type_output_shape(
    label_column: str,
    column_types: Optional[Dict] = None,
    data: Optional[pd.DataFrame] = None,
    provided_problem_type: Optional[str] = None,
) -> Tuple[str, int]:
    """
    Infer the problem type and output shape based on the label column type and training data.
    Binary classification should have class number 2, while multi-class classification's class
    number should be larger than 2. For regression, the output is restricted to 1 scalar.

    Parameters
    ----------
    column_types
        Types of columns in a multimodal pd.DataFrame.
    label_column
        The label column in a multimodal pd.DataFrame.
    data
        The multimodal pd.DataFrame for training.
    provided_problem_type
        The provided problem type.

    Returns
    -------
    problem_type
        Type of problem.
    output_shape
        Shape of output.
    """
    if provided_problem_type is not None:
        if provided_problem_type == MULTICLASS or provided_problem_type == BINARY:
            class_num = len(data[label_column].unique())
            err_msg = (
                f"Provided problem type is '{provided_problem_type}' while the number of "
                f"unique values in the label column is {class_num}."
            )
            if provided_problem_type == BINARY and class_num != 2:
                raise AssertionError(err_msg)
            elif provided_problem_type == MULTICLASS and class_num <= 2:
                raise AssertionError(err_msg)
            return provided_problem_type, class_num
        if provided_problem_type == BINARY:
            return BINARY, 2
        elif provided_problem_type == MULTICLASS:
            class_num = len(data[label_column].value_counts())
            return MULTICLASS, class_num
        elif provided_problem_type == CLASSIFICATION:
            class_num = len(data[label_column].value_counts())
            if class_num == 2:
                return BINARY, 2
            else:
                return MULTICLASS, class_num
        elif provided_problem_type == REGRESSION:
            return provided_problem_type, 1
        else:
            raise ValueError(
                f"Problem type '{provided_problem_type}' doesn't have a valid output shape "
                f"for training. The supported problem types are"
                f" '{BINARY}', '{MULTICLASS}', '{REGRESSION}', '{CLASSIFICATION}'"
            )

    else:
        if column_types[label_column] == CATEGORICAL:
            class_num = len(data[label_column].unique())
            if class_num == 2:
                return BINARY, 2
            else:
                return MULTICLASS, class_num
        elif column_types[label_column] == NUMERICAL:
            return REGRESSION, 1
        else:
            raise ValueError(
                f"The label column '{label_column}' has type"
                f" '{column_types[label_column]}', which is not supported yet."
            )


def set_fallback_column_type(column_types: Dict, allowable_column_types: List[str], fallback_column_type: str) -> Dict:
    """
    Filter the auto-detected column types to make sure that all column types are allowable.
    Use the fallback type to replace those out of the allowable_column_types.

    Parameters
    ----------
    column_types
        The inferred column types.
    allowable_column_types
        The column types which are allowed by the model type.
    fallback_column_type
        Fallback to this type if some invalid column type is found.

    Returns
    -------
    The filtered column types.
    """
    for col_name, col_type in column_types.items():
        if col_type not in allowable_column_types:
            column_types[col_name] = fallback_column_type

    return column_types
