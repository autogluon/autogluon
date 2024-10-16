import base64
import collections
import json
import logging
import warnings
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import PIL
import pytesseract

from autogluon.core.utils import infer_problem_type as infer_basic_problem_type

from ..constants import (
    BINARY,
    CATEGORICAL,
    CLASSIFICATION,
    DOCUMENT_IMAGE,
    DOCUMENT_PDF,
    IDENTIFIER,
    IMAGE,
    IMAGE_BASE64_STR,
    IMAGE_BYTEARRAY,
    IMAGE_PATH,
    MULTICLASS,
    NER,
    NER_ANNOTATION,
    NULL,
    NUMERICAL,
    OBJECT_DETECTION,
    REGRESSION,
    ROIS,
    SEMANTIC_SEGMENTATION,
    SEMANTIC_SEGMENTATION_GT,
    SEMANTIC_SEGMENTATION_IMG,
    TEXT,
    TEXT_NER,
)
from .utils import is_rois_input

logger = logging.getLogger(__name__)


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
        try:
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
            else:
                return False
        except:
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
    if isinstance(data[idx], str):
        try:
            rois = json.loads(data[idx][0])
            # TODO: better infer logic / input format to not confuse with other modality
            return rois and isinstance(rois, list) and len(data[idx][0]) == 5
        except:
            return False
    else:
        return is_rois_input(data[idx])


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


def is_image_column(
    data: pd.Series,
    col_name: str,
    image_type: str,
    sample_n: Optional[int] = 500,
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
    image_type
        The image type to check.
    sample_n
        Number of sample images to open for sanity check.

    Returns
    -------
    Whether the column is an image-path column.
    """
    sample_num = min(len(data), sample_n)
    data = data.sample(n=sample_num, random_state=0)
    if image_type == IMAGE_PATH:
        data = data.apply(lambda ele: str(ele).split(";")).tolist()
    elif image_type in [IMAGE_BYTEARRAY, IMAGE_BASE64_STR]:
        data = data.tolist()
    else:
        raise ValueError(f"Unsupported image type: {image_type}")

    failure_count = 0
    for images in data:
        success = False
        if not isinstance(images, list):
            images = [images]
        for per_image in images:
            try:
                if image_type == IMAGE_PATH:
                    with PIL.Image.open(per_image) as img:
                        pass
                elif image_type == IMAGE_BYTEARRAY:
                    with PIL.Image.open(BytesIO(per_image)) as img:
                        pass
                elif image_type == IMAGE_BASE64_STR:
                    with PIL.Image.open(BytesIO(base64.b64decode(per_image))) as img:
                        pass
                else:
                    raise ValueError(f"Unsupported image type: {image_type}")
            except:
                success = False
                break

            success = True
        if not success:
            failure_count += 1
    failure_ratio = failure_count / sample_num
    # Tolerate high failure rate in case that many image files may be corrupted.
    if failure_ratio <= 0.9:
        if failure_ratio > 0:
            warnings.warn(
                f"Among {sample_num} sampled images in column '{col_name}', "
                f"{failure_ratio:.0%} images can't be open. "
                "You may need to thoroughly check your data to see the percentage of missing images, "
                "and estimate the potential influence. By default, we use an image with zero pixels. "
                "You can also set hyperparameter 'data.image.missing_value_strategy' to be 'skip', "
                "which skips samples that contain a missing image.",
                UserWarning,
            )
        return True
    else:
        return False


def is_document_pdf_column(
    data: pd.Series,
    col_name: str,
    sample_m: Optional[int] = 500,
) -> bool:
    """
    Identify if a column is a pdf document column.

    Parameters
    ----------
    data
        One column of a multimodal pd.DataFrame for training.
    col_name
        Name of column.
    sample_m
        Number of sample images used to check if images are documents images.

    Returns
    -------
    Whether the column is a pdf document column.
    """

    if len(data) > sample_m:
        # Sample to speed-up type inference
        data = data.sample(n=sample_m, random_state=0)

    for docs in data:
        try:
            if not isinstance(docs, list):
                docs = [docs]
            for per_doc in docs:
                # If there is non-pdf document, return False
                if not per_doc.endswith(".pdf"):
                    return False
        except:
            return False

    return True


def is_document_image_column(
    data: pd.Series,
    col_name: str,
    image_type: Optional[str] = IMAGE_PATH,
    sample_m: Optional[int] = 10,
    text_len_threshold: Optional[int] = 100,
) -> bool:
    """
    Identify if a column is a document image column.

    Parameters
    ----------
    data
        One column of a multimodal pd.DataFrame for training.
    col_name
        Name of column.
    image_type
        The image type to check. Set to IMAGE_PATH by default.
    sample_m
        Number of sample images used to check if images are documents images.
    text_len_threshold
        If the average text length is longer than text_len_threshold, the images will be considered as document images.
    Returns
    -------
    Whether the column is a document image column.
    """

    # TODO: Add support for other types (e.g., pdf) of document.

    words_len = []
    if len(data) > sample_m:
        # Sample to speed-up type inference
        data = data.sample(n=sample_m, random_state=0)
    failure_count = 0
    for images in data:
        success = False
        if not isinstance(images, list):
            images = [images]
        for per_image in images:
            try:
                # convert images to string
                with PIL.Image.open(per_image) as doc_image:
                    words = pytesseract.image_to_string(doc_image)
                    words_len.append(len(words))
            except Exception as e:
                words_len.append(0)
                success = False
                break
            success = True
        if not success:
            failure_count += 1

    if (1 - failure_count / sample_m) >= 0.8:
        logger.debug(f"Average length of words of this dataset is {sum(words_len) / len(words_len)}.")
        if sum(words_len) / len(words_len) > text_len_threshold:
            return True
        else:
            return False
    else:
        False


def is_text_column(data: pd.Series) -> bool:
    """
    Identify if a column is one text column.

    Parameters
    ----------
    data
        One column of a multimodal pd.DataFrame for training.

    Returns
    -------
    Whether the column is a text column.
    """
    if len(data) > 5000:
        # Sample to speed-up type inference
        data = data.sample(n=5000, random_state=0)
    try:
        data_unique = data.unique()
        num_unique = len(data_unique)
        num_rows = len(data)
        unique_ratio = num_unique / num_rows
        if unique_ratio <= 0.01:
            return False
        try:
            avg_words = pd.Series(data_unique).str.split().str.len().mean()
        except AttributeError:
            return False
        if avg_words < 3:
            return False
    except:
        return False
    return True


def is_identifier_column(data: pd.Series, col_name: str, id_mappings: Dict[str, Dict]) -> bool:
    """
    Check if a column is one identifier column.

    Parameters
    ----------
    data
        One column of multimodal pd.DataFrame.
    col_name
        Name of the column.
    id_mappings
        Id-to-content mappings. The contents can be text, image, etc.
        This is used when the dataframe contains the query/response indexes instead of their contents.

    Returns
    -------
    Whether the column is an identifier column.
    """
    if not id_mappings or col_name not in id_mappings:
        return False

    sample_num = min(len(data), 500)
    data = data.sample(n=sample_num, random_state=0).tolist()
    failure_count = 0
    for index in data:
        try:
            per_value = id_mappings[col_name][index]
        except:
            failure_count += 1

    if failure_count == 0:
        return True
    elif 0 < failure_count < sample_num:
        warnings.warn(
            f"Among {sample_num} sampled indexes in column {col_name}, "
            f"we can't index all their values from the id_mappings ({failure_count} failures). "
            f"You may need to assure that all the indexes of column {col_name} exist in your id_mappings.",
            UserWarning,
        )
    else:
        return False


def infer_id_mappings_types(id_mappings: Union[Dict[str, Dict], Dict[str, pd.Series]]) -> Dict:
    """
    Infer the data types in id_mappings.

    Parameters
    ----------
    id_mappings
        Id-to-content mappings. The contents can be text, image, etc.

    Returns
    -------
    A dictionary containing the data types in id_mappings.
    """
    id_mappings_types = collections.OrderedDict()
    if id_mappings is None:
        return id_mappings_types

    for per_name, per_id_mappings in id_mappings.items():
        if isinstance(per_id_mappings, dict):
            per_id_mappings = pd.Series(per_id_mappings.values())
        elif isinstance(per_id_mappings, pd.Series):
            pass
        else:
            raise ValueError(
                f"Invalid per_id_mappings type: {type(per_id_mappings)}. Make sure the id_mappings is a dict of dicts or a dict of pd.Series."
            )
        if is_image_column(per_id_mappings, col_name=per_name, image_type=IMAGE_PATH):
            id_mappings_types[per_name] = IMAGE_PATH
        elif is_text_column(per_id_mappings):
            id_mappings_types[per_name] = TEXT
        elif is_image_column(per_id_mappings, col_name=per_name, image_type=IMAGE_BYTEARRAY):
            id_mappings_types[per_name] = IMAGE_BYTEARRAY
        elif is_image_column(per_id_mappings, col_name=per_name, image_type=IMAGE_BASE64_STR):
            id_mappings_types[per_name] = IMAGE_BASE64_STR
        else:
            raise ValueError(
                f"{per_name} in the id_mappings has an invalid type. Currently, we only support image and text types."
            )
    return id_mappings_types


def infer_column_types(
    data: pd.DataFrame,
    valid_data: Optional[pd.DataFrame] = None,
    label_columns: Union[str, List[str]] = None,
    provided_column_types: Optional[Dict] = None,
    allowable_column_types: Optional[List[str]] = None,
    fallback_column_type: Optional[str] = None,
    id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
    problem_type: Optional[str] = None,
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
    id_mappings
        Id-to-content mappings. The contents can be text, image, etc.
        This is used when the dataframe contains the query/response indexes instead of their contents.
    problem_type
        The problem type used to update some column types.

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

    id_mappings_types = infer_id_mappings_types(id_mappings)

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
        if (
            (not isinstance(data[col_name][idx], (list, dict, set, bytearray)))
            and len(data[col_name].unique()) == 1
            and is_training
        ):
            column_types[col_name] = NULL
            continue
        # TODO: valid check for collections

        if is_rois_column(data[col_name]):
            column_types[col_name] = ROIS
        # keep the elif here because ROIS need to skip the categorical check
        # where error occurs due to List type input
        elif is_identifier_column(data[col_name], col_name=col_name, id_mappings=id_mappings):
            column_types[col_name] = f"{id_mappings_types[col_name]}_{IDENTIFIER}"
        elif is_categorical_column(
            data[col_name], valid_data[col_name], is_label=col_name in label_columns
        ):  # Infer categorical column
            column_types[col_name] = CATEGORICAL
        elif is_numerical_column(data[col_name], valid_data[col_name]):  # Infer numerical column
            column_types[col_name] = NUMERICAL
        elif is_image_column(data[col_name], col_name=col_name, image_type=IMAGE_PATH):  # Infer image-path column
            # Check if it is document image or not.
            if is_document_image_column(data[col_name], col_name=col_name):
                column_types[col_name] = DOCUMENT_IMAGE
            elif problem_type == SEMANTIC_SEGMENTATION:
                column_types[col_name] = SEMANTIC_SEGMENTATION_IMG
            else:
                column_types[col_name] = IMAGE_PATH
        elif is_document_pdf_column(data[col_name], col_name=col_name):
            column_types[col_name] = DOCUMENT_PDF
        elif is_text_column(data[col_name]):  # Infer text column
            column_types[col_name] = TEXT
        elif is_image_column(
            data[col_name], col_name=col_name, image_type=IMAGE_BYTEARRAY
        ):  # Infer image-bytearray column
            column_types[col_name] = IMAGE_BYTEARRAY
        elif is_image_column(
            data[col_name], col_name=col_name, image_type=IMAGE_BASE64_STR
        ):  # Infer image-base64str column
            column_types[col_name] = IMAGE_BASE64_STR
        else:  # All the other columns are treated as categorical
            column_types[col_name] = CATEGORICAL

    if allowable_column_types and fallback_column_type:
        column_types = set_fallback_column_type(
            column_types=column_types,
            allowable_column_types=allowable_column_types,
            fallback_column_type=fallback_column_type,
        )
    if problem_type == NER:
        column_types = infer_ner_column_type(column_types=column_types)

    if problem_type and label_columns:
        column_types = infer_label_column_type_by_problem_type(
            column_types=column_types,
            label_columns=label_columns,
            problem_type=problem_type,
            data=data,
            valid_data=valid_data if is_training else None,
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
    problem_type: Optional[str],
    data: Optional[pd.DataFrame] = None,
    valid_data: Optional[pd.DataFrame] = None,
    allowable_label_types: Optional[List[str]] = (
        CATEGORICAL,
        NUMERICAL,
        NER_ANNOTATION,
        ROIS,
        SEMANTIC_SEGMENTATION_GT,
    ),
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
    if label_columns is None:
        return column_types

    if isinstance(label_columns, str):
        label_columns = [label_columns]

    for col_name in label_columns:
        # For zero-shot inference, label column is unnecessary
        if col_name not in column_types:
            continue
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
        elif problem_type == NER:
            column_types[col_name] = NER_ANNOTATION
        elif problem_type == OBJECT_DETECTION:
            column_types[col_name] = ROIS
        elif problem_type == SEMANTIC_SEGMENTATION:
            column_types[col_name] = SEMANTIC_SEGMENTATION_GT

        if column_types[col_name] not in allowable_label_types:
            column_types[col_name] = fallback_label_type

    return column_types


def infer_rois_column_type(
    column_types: Dict,
    data: Optional[pd.DataFrame] = None,
):
    for col_name in column_types.keys():
        if is_rois_column(data[col_name]):
            column_types[col_name] = ROIS
    return column_types


def infer_problem_type(
    y_train_data: Optional[pd.DataFrame] = None,
    provided_problem_type: Optional[str] = None,
) -> str:
    """
    Infer the basic (binary, multiclass, and regression) problem types if problem type is None or classification.
    Only training data are used. Assume users provide valid validation data.

    Parameters
    ----------
    y_train_data
        The label column of training data.
    provided_problem_type
        Provided problem type

    Returns
    -------
    Inferred problem type
    """
    if provided_problem_type in [None, CLASSIFICATION]:
        problem_type = infer_basic_problem_type(y=y_train_data)
        # trust users' prior knowledge
        if provided_problem_type == CLASSIFICATION and problem_type == REGRESSION:
            problem_type = MULTICLASS
    else:
        problem_type = provided_problem_type

    return problem_type


def infer_output_shape(
    label_column: str,
    problem_type: str,
    data: pd.DataFrame,
) -> int:
    """
    Infer the output shape based on the label column type, training data, and problem type.
    Binary classification should have class number 2, while multi-class classification's class
    number should be larger than 2. For regression, the output is restricted to 1 scalar.

    Parameters
    ----------
    label_column
        The label column in a multimodal pd.DataFrame.
    data
        The multimodal pd.DataFrame for training.
    problem_type
        Problem type which can be inferred or provided.

    Returns
    -------
    problem_type
        Type of problem.
    output_shape
        Shape of output.
    """
    if label_column is None:
        return None

    if problem_type in [BINARY, MULTICLASS, REGRESSION, CLASSIFICATION]:
        class_num = len(data[label_column].unique())
        err_msg = (
            f"Problem type is '{problem_type}' while the number of "
            f"unique values in the label column is {class_num}."
        )
        if problem_type == BINARY:
            if class_num != 2:
                raise AssertionError(err_msg)
            return 2
        elif problem_type == MULTICLASS:
            if class_num <= 2:
                raise AssertionError(err_msg)
            return class_num
        elif problem_type == REGRESSION:
            if class_num < 1:
                raise AssertionError(err_msg)
            return 1
        else:
            return class_num
    elif problem_type in [NER, OBJECT_DETECTION, SEMANTIC_SEGMENTATION]:
        return None
    else:
        raise ValueError(
            f"Problem type '{problem_type}' doesn't have a valid output shape "
            f"for training. The supported problem types are"
            f" '{BINARY}', '{MULTICLASS}', '{REGRESSION}',"
            f" '{CLASSIFICATION}', '{NER}',"
            f" '{OBJECT_DETECTION}', "
            f" '{SEMANTIC_SEGMENTATION}"
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
        if not col_type.startswith(tuple(allowable_column_types)):
            column_types[col_name] = fallback_column_type

    return column_types


def infer_ner_column_type(column_types: Dict):
    """
    Replace the first text with text_ner for the ner problem type
    because no text_ner is detected in infer_column_types.

    Parameters
    ----------
    column_types
        The column types of a dataframe.

    Returns
    -------
    The columns types with one text_ner inside it.
    """
    # if there is already one column has type text_ner, no action is taken.
    if any([col_type.startswith(TEXT_NER) for col_type in column_types.values()]):
        return column_types

    for column in (
        column_types.keys()
    ):  # column_types is an ordered dict, so column_types.keys() returns the keys in the order of insertions.
        if column_types[column].startswith(TEXT):
            column_types[column] = column_types[column].replace(TEXT, TEXT_NER)
            break

    return column_types
