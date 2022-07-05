import logging
import collections
import pandas as pd
import warnings
import PIL
from typing import Union, Optional, List, Dict, Tuple
from ..constants import NULL, CATEGORICAL, NUMERICAL, TEXT, IMAGE_PATH, MULTICLASS, BINARY, REGRESSION, AUTOMM

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


def check_if_nlp_feature(X: pd.Series) -> bool:
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


def infer_column_problem_types(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    label_columns: Union[str, List[str]],
    problem_type: Optional[str] = None,
    provided_column_types: Optional[Dict] = None,
) -> Tuple[collections.OrderedDict, str, int]:
    """
    Infer the column types of a multimodal pd.DataFrame and the problem type.

    Parameters
    ----------
    train_df
        The multimodal pd.DataFrame for training.
    valid_df
        The multimodal pd.DataFrame for validation.
    label_columns
        The label column names.
    problem_type
        Type of problem.
    provided_column_types
        Additional dictionary that you can use to specify the columns types that you know.
        {'col_name': TYPE, ...}

    Returns
    -------
    column_types
        A dictionary containing the mappings from column names to their modality types.
        If the column does not contain any useful information, we will set its column type NULL.
    problem_type
        The inferred problem type.
    output_shape
        Shape of output.
    """
    if isinstance(label_columns, str):
        label_columns = [label_columns]
    elif isinstance(label_columns, (list, tuple)):
        pass
    else:
        raise NotImplementedError(f"label_columns is not supported. label_columns={label_columns}.")
    label_set = set(label_columns)
    assert len(label_set) == 1, "Currently, only a single label column is supported."
    column_types = collections.OrderedDict()
    # Process all feature columns

    for col_name in train_df.columns:
        is_label = col_name in label_set
        if provided_column_types is not None and col_name in provided_column_types:
            column_types[col_name] = provided_column_types[col_name]
            continue
        if is_label:
            num_train_missing = train_df[col_name].isnull().sum()
            num_valid_missing = valid_df[col_name].isnull().sum()
            if num_train_missing > 0:
                raise ValueError(
                    f"Label column '{col_name}' contains missing values in the "
                    "training data frame. You may want to filter your data because "
                    "missing label is currently not supported."
                )
            if num_valid_missing > 0:
                raise ValueError(
                    f"Label column '{col_name}' contains missing values in the "
                    "validation data frame. You may want to filter your data because "
                    "missing label is currently not supported."
                )
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
                warnings.warn(
                    f"Label column '{col_name}' contains only one label. You may need to check your dataset again."
                )
        # Use the following way for type inference
        # 1) Infer categorical column
        # 2) Infer numerical column
        # 3) Infer image-path column
        # 4) Infer text column
        # 4) All the other columns are treated as categorical
        if is_categorical_column(train_df[col_name], valid_df[col_name], is_label=is_label):
            column_types[col_name] = CATEGORICAL
        elif is_numerical_column(train_df[col_name], valid_df[col_name]):
            column_types[col_name] = NUMERICAL
        elif is_imagepath_column(train_df[col_name], col_name):
            column_types[col_name] = IMAGE_PATH
        elif check_if_nlp_feature(train_df[col_name]):
            column_types[col_name] = TEXT
        else:
            column_types[col_name] = CATEGORICAL
    problem_type, output_shape = infer_problem_type_output_shape(
        column_types=column_types,
        label_column=label_columns[0],
        data_df=train_df,
        provided_problem_type=problem_type,
    )
    return column_types, problem_type, output_shape


def infer_problem_type_output_shape(
    column_types: dict,
    label_column: str,
    data_df: pd.DataFrame,
    provided_problem_type=None,
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
    data_df
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
            class_num = len(data_df[label_column].unique())
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
            class_num = len(data_df[label_column].value_counts())
            return MULTICLASS, class_num
        else:
            return provided_problem_type, 1

    else:
        if column_types[label_column] == CATEGORICAL:
            class_num = len(data_df[label_column].unique())
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
