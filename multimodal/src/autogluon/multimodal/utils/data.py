import logging
import os
import random
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import PIL
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn

from autogluon.core.utils.loaders import load_pd
from autogluon.core.utils.utils import default_holdout_frac

from ..constants import (
    AUTOMM,
    BINARY,
    CATEGORICAL,
    DEFAULT_SHOT,
    DOCUMENT,
    FEW_SHOT,
    IMAGE,
    IMAGE_PATH,
    LABEL,
    MMLAB_MODELS,
    NER,
    NER_ANNOTATION,
    NER_TEXT,
    NUMERICAL,
    ROIS,
    TEXT,
    TEXT_NER,
)
from ..data import (
    CategoricalProcessor,
    DocumentProcessor,
    ImageProcessor,
    LabelProcessor,
    MixupModule,
    MMDetProcessor,
    MMOcrProcessor,
    MultiModalFeaturePreprocessor,
    NerLabelEncoder,
    NerProcessor,
    NumericalProcessor,
    TextProcessor,
)
from ..data.infer_types import is_image_column

logger = logging.getLogger(__name__)


def init_df_preprocessor(
    config: DictConfig,
    column_types: Dict,
    label_column: Optional[str] = None,
    train_df_x: Optional[pd.DataFrame] = None,
    train_df_y: Optional[pd.Series] = None,
):
    """
    Initialize the dataframe preprocessor by calling .fit().

    Parameters
    ----------
    config
        A DictConfig containing only the data config.
    column_types
        A dictionary that maps column names to their data types.
        For example: `column_types = {"item_name": "text", "image": "image_path",
        "product_description": "text", "height": "numerical"}`
        may be used for a table with columns: "item_name", "brand", "product_description", and "height".
    label_column
        Name of the column that contains the target variable to predict.
    train_df_x
        A pd.DataFrame containing only the feature columns.
    train_df_y
        A pd.Series object containing only the label column.

    Returns
    -------
    Initialized dataframe preprocessor.
    """
    if label_column is not None and column_types[label_column] == NER_ANNOTATION:
        label_generator = NerLabelEncoder(config)
    else:
        label_generator = None

    df_preprocessor = MultiModalFeaturePreprocessor(
        config=config.data,
        column_types=column_types,
        label_column=label_column,
        label_generator=label_generator,
    )
    df_preprocessor.fit(
        X=train_df_x,
        y=train_df_y,
    )

    return df_preprocessor


def create_data_processor(
    data_type: str,
    config: DictConfig,
    model: nn.Module,
):
    """
    Create one data processor based on the data type and model.

    Parameters
    ----------
    data_type
        Data type.
    config
        The config may contain information required by creating a data processor.
        In future, we may move the required config information into the model.config
        to make the data processor conditioned only on the model itself.
    model
        The model.

    Returns
    -------
    One data processor.
    """
    model_config = getattr(config.model, model.prefix)
    if data_type == IMAGE:
        data_processor = ImageProcessor(
            model=model,
            train_transform_types=model_config.train_transform_types,
            val_transform_types=model_config.val_transform_types,
            norm_type=model_config.image_norm,
            size=model_config.image_size,
            max_img_num_per_col=model_config.max_img_num_per_col,
            missing_value_strategy=config.data.image.missing_value_strategy,
        )
    elif data_type == TEXT:
        data_processor = TextProcessor(
            model=model,
            tokenizer_name=model_config.tokenizer_name,
            max_len=model_config.max_text_len,
            insert_sep=model_config.insert_sep,
            text_segment_num=model_config.text_segment_num,
            stochastic_chunk=model_config.stochastic_chunk,
            text_detection_length=OmegaConf.select(model_config, "text_aug_detect_length"),
            text_trivial_aug_maxscale=OmegaConf.select(model_config, "text_trivial_aug_maxscale"),
            train_augment_types=OmegaConf.select(model_config, "text_train_augment_types"),
            template_config=getattr(config.data, "templates", OmegaConf.create({"turn_on": False})),
            normalize_text=getattr(config.data.text, "normalize_text", False),
        )
    elif data_type == CATEGORICAL:
        data_processor = CategoricalProcessor(
            model=model,
        )
    elif data_type == NUMERICAL:
        data_processor = NumericalProcessor(
            model=model,
            merge=model_config.merge,
        )
    elif data_type == LABEL:
        data_processor = LabelProcessor(model=model)
    elif data_type == TEXT_NER:
        data_processor = NerProcessor(
            model=model,
            max_len=model_config.max_text_len,
            config=config,
        )
    elif data_type == ROIS:
        data_processor = MMDetProcessor(
            model=model,
            max_img_num_per_col=model_config.max_img_num_per_col,
            missing_value_strategy=config.data.image.missing_value_strategy,
        )
    elif data_type == DOCUMENT:
        data_processor = DocumentProcessor(
            model=model,
            train_transform_types=model_config.train_transform_types,
            val_transform_types=model_config.val_transform_types,
            norm_type=model_config.image_norm,
            size=model_config.image_size,
            text_max_len=model_config.max_text_len,
            missing_value_strategy=config.data.document.missing_value_strategy,
        )
    else:
        raise ValueError(f"unknown data type: {data_type}")

    return data_processor


def create_fusion_data_processors(
    config: DictConfig,
    model: nn.Module,
    requires_label: Optional[bool] = True,
    requires_data: Optional[bool] = True,
):
    """
    Create the data processors for late-fusion models. This function creates one processor for
    each modality of each model. For example, if one model config contains BERT, ViT, and CLIP, then
    BERT would have its own text processor, ViT would have its own image processor, and CLIP would have
    its own text and image processors. This is to support training arbitrary combinations of single-modal
    and multimodal models since two models may share the same modality but have different processing. Text
    sequence length is a good example. BERT's sequence length is generally 512, while CLIP uses sequences of
    length 77.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model".
    model
        The model object.

    Returns
    -------
    A dictionary with modalities as the keys. Each modality has a list of processors.
    Note that "label" is also treated as a modality for convenience.
    """
    data_processors = {
        IMAGE: [],
        TEXT: [],
        CATEGORICAL: [],
        NUMERICAL: [],
        LABEL: [],
        ROIS: [],
        TEXT_NER: [],
        DOCUMENT: [],
    }

    model_dict = {model.prefix: model}

    if model.prefix.lower().startswith("fusion"):
        for per_model in model.model:
            model_dict[per_model.prefix] = per_model

    assert sorted(list(model_dict.keys())) == sorted(config.model.names)

    for per_name, per_model in model_dict.items():
        model_config = getattr(config.model, per_model.prefix)
        if model_config.data_types is not None:
            data_types = model_config.data_types.copy()
        else:
            data_types = None

        if per_name == NER_TEXT:
            # create a multimodal processor for NER.
            data_processors[TEXT_NER].append(
                create_data_processor(
                    data_type=TEXT_NER,
                    config=config,
                    model=per_model,
                )
            )
            requires_label = False
            if data_types is not None and TEXT_NER in data_types:
                data_types.remove(TEXT_NER)
        elif per_name.lower().startswith(MMLAB_MODELS):
            # create a multimodal processor for NER.
            data_processors[ROIS].append(
                create_data_processor(
                    data_type=ROIS,
                    config=config,
                    model=per_model,
                )
            )
            if data_types is not None and IMAGE in data_types:
                data_types.remove(IMAGE)

        if requires_label:
            # each model has its own label processor
            label_processor = create_data_processor(
                data_type=LABEL,
                config=config,
                model=per_model,
            )
            data_processors[LABEL].append(label_processor)

        if requires_data and data_types:
            for data_type in data_types:
                per_data_processor = create_data_processor(
                    data_type=data_type,
                    model=per_model,
                    config=config,
                )
                data_processors[data_type].append(per_data_processor)

    # Only keep the modalities with non-empty processors.
    data_processors = {k: v for k, v in data_processors.items() if len(v) > 0}

    if TEXT_NER in data_processors and LABEL in data_processors:
        # LabelProcessor is not needed for NER tasks as annotations are handled in NerProcessor.
        data_processors.pop(LABEL)
    return data_processors


def assign_feature_column_names(
    data_processors: Dict,
    df_preprocessor: MultiModalFeaturePreprocessor,
):
    """
    Assign feature column names to data processors.
    This is to patch the data processors saved by AutoGluon 0.4.0.

    Parameters
    ----------
    data_processors
        The data processors.
    df_preprocessor
        The dataframe preprocessor.

    Returns
    -------
    The data processors with feature column names added.
    """
    for per_modality in data_processors:
        if per_modality == LABEL or per_modality == TEXT_NER:
            continue
        for per_model_processor in data_processors[per_modality]:
            # requires_column_info=True is used for feature column distillation.
            per_model_processor.requires_column_info = False
            if per_modality == IMAGE:
                per_model_processor.image_column_names = df_preprocessor.image_path_names
            elif per_modality == TEXT:
                per_model_processor.text_column_names = df_preprocessor.text_feature_names
            elif per_modality == NUMERICAL:
                per_model_processor.numerical_column_names = df_preprocessor.numerical_feature_names
            elif per_modality == CATEGORICAL:
                per_model_processor.categorical_column_names = df_preprocessor.categorical_feature_names
            else:
                raise ValueError(f"Unknown modality: {per_modality}")

    return data_processors


def turn_on_off_feature_column_info(
    data_processors: Dict,
    flag: bool,
):
    """
    Turn on or off returning feature column information in data processors.
    Since feature column information is not always required in training models,
    we optionally turn this flag on or off.

    Parameters
    ----------
    data_processors
        The data processors.
    flag
        True/False
    """
    for per_modality_processors in data_processors.values():
        for per_model_processor in per_modality_processors:
            # label processor doesn't have requires_column_info.
            if hasattr(per_model_processor, "requires_column_info"):
                per_model_processor.requires_column_info = flag


def try_to_infer_pos_label(
    data_config: DictConfig,
    label_encoder: LabelEncoder,
    problem_type: str,
):
    """
    Try to infer positive label for binary classification, which is used in computing some metrics, e.g., roc_auc.
    If positive class is not provided, then use pos_label=1 by default.
    If the problem type is not binary classification, then return None.

    Parameters
    ----------
    data_config
        A DictConfig object containing only the data configurations.
    label_encoder
        The label encoder of classification tasks.
    problem_type
        Type of problem.

    Returns
    -------

    """
    if problem_type != BINARY:
        return None

    pos_label = OmegaConf.select(data_config, "pos_label", default=None)
    if pos_label is not None:
        logger.debug(f"pos_label: {pos_label}\n")
        pos_label = label_encoder.transform([pos_label]).item()
    else:
        pos_label = 1

    logger.debug(f"pos_label: {pos_label}")
    return pos_label


def get_mixup(
    model_config: DictConfig,
    mixup_config: DictConfig,
    num_classes: int,
):
    """
    Get the mixup state for loss function choice.
    Now the mixup can only support image data.
    And the problem type can not support Regression.
    Parameters
    ----------
    model_config
        The model configs to find image model for the necessity of mixup.
    mixup_config
        The mixup configs for mixup and cutmix.
    num_classes
        The number of classes in the task. Class <= 1 will cause faults.

    Returns
    -------
    The mixup is on or off.
    """
    model_active = False
    names = model_config.names
    if isinstance(names, str):
        names = [names]
    for model_name in names:
        permodel_config = getattr(model_config, model_name)
        if hasattr(permodel_config.data_types, IMAGE):
            model_active = True
            break

    mixup_active = False
    if mixup_config is not None and mixup_config.turn_on:
        mixup_active = (
            mixup_config.mixup_alpha > 0 or mixup_config.cutmix_alpha > 0.0 or mixup_config.cutmix_minmax is not None
        )

    mixup_state = model_active & mixup_active & ((num_classes is not None) and (num_classes > 1))
    mixup_fn = None
    if mixup_state:
        mixup_args = dict(
            mixup_alpha=mixup_config.mixup_alpha,
            cutmix_alpha=mixup_config.cutmix_alpha,
            cutmix_minmax=mixup_config.cutmix_minmax,
            prob=mixup_config.prob,
            switch_prob=mixup_config.switch_prob,
            mode=mixup_config.mode,
            label_smoothing=mixup_config.label_smoothing,
            num_classes=num_classes,
        )
        mixup_fn = MixupModule(**mixup_args)
    return mixup_state, mixup_fn


def data_to_df(
    data: Union[pd.DataFrame, Dict, List],
    required_columns: Optional[List] = None,
    all_columns: Optional[List] = None,
    header: Optional[str] = None,
):
    """
    Convert the input data to a dataframe.

    Parameters
    ----------
    data
        Input data provided by users during prediction/evaluation.
    required_columns
        Required columns.
    all_columns
        All the possible columns got from training data. The column order is preserved.
    header
        Provided header to create a dataframe.

    Returns
    -------
    A dataframe with required columns.
    """
    has_header = True
    if isinstance(data, pd.DataFrame):
        pass
    elif isinstance(data, dict):
        data = pd.DataFrame(data)
    elif isinstance(data, list):
        assert len(data) > 0, f"Expected data to have length > 0, but got {data} of len {len(data)}"
        if header is None:
            has_header = False
            data = pd.DataFrame(data)
        else:
            data = pd.DataFrame({header: data})
    elif isinstance(data, str):
        df = pd.DataFrame([data])
        col_name = list(df.columns)[0]
        if is_image_column(df[col_name], col_name=col_name, image_type=IMAGE_PATH):
            has_header = False
            data = df
        else:
            data = load_pd.load(data)
    else:
        raise NotImplementedError(
            f"The format of data is not understood. "
            f'We have type(data)="{type(data)}", but a pd.DataFrame was required.'
        )

    if required_columns and all_columns:
        detected_columns = data.columns.values.tolist()
        missing_columns = []
        for per_col in required_columns:
            if per_col not in detected_columns:
                missing_columns.append(per_col)

        if len(missing_columns) > 0:
            # assume no column names are provided and users organize data in the same column order of training data.
            if len(detected_columns) == len(all_columns):
                if has_header:
                    warnings.warn(
                        f"Replacing detected dataframe columns `{detected_columns}` with columns "
                        f"`{all_columns}` from training data."
                        "Double check the correspondences between them to avoid unexpected behaviors.",
                        UserWarning,
                    )
                data.rename(dict(zip(detected_columns, required_columns)), axis=1, inplace=True)
            else:
                raise ValueError(
                    f"Dataframe columns `{detected_columns}` are detected, but columns `{missing_columns}` are missing. "
                    f"Please double check your input data to provide all the "
                    f"required columns `{required_columns}`."
                )

    return data


def infer_scarcity_mode_by_data_size(df_train: pd.DataFrame, scarcity_threshold: int = 50):
    """
    Infer based on the number of training sample the data scarsity. Select mode accordingly from [DEFAULT_SHOT, FEW_SHOT, ZERO_SHOT].

    Parameters
    ---------------
    df_train
        Training dataframe
    scarcity_threshold
        Threshold number of samples when to select FEW_SHOT mode

    Returns
    --------
    Mode in  [DEFAULT_SHOT, FEW_SHOT, ZERO_SHOT]
    """
    row_num = len(df_train)
    if row_num < scarcity_threshold:
        return FEW_SHOT
    else:
        return DEFAULT_SHOT


def infer_dtypes_by_model_names(model_config: DictConfig):
    """
    Get data types according to model types.

    Parameters
    ----------
    model_config
        Model config from `config.model`.

    Returns
    -------
    The data types allowed by models and the default fallback data type.
    """
    allowable_dtypes = []
    fallback_dtype = None
    for per_model in model_config.names:
        per_model_dtypes = OmegaConf.select(model_config, f"{per_model}.data_types")
        if per_model_dtypes:
            allowable_dtypes.extend(per_model_dtypes)

    allowable_dtypes = set(allowable_dtypes)
    if allowable_dtypes == {IMAGE, TEXT}:
        fallback_dtype = TEXT
    elif len(allowable_dtypes) == 1:
        fallback_dtype = list(allowable_dtypes)[0]

    return allowable_dtypes, fallback_dtype


def split_train_tuning_data(train_data, tuning_data, holdout_frac, is_classification, label_column, seed):
    """
    Split training and tuning data.

    Parameters
    ----------
    train_data
        Provided training data.
    tuning_data
        Provided tuning data.
    holdout_frac
        Fraction of train_data to holdout as tuning_data.
    is_classification
        Whether is a classification task.
    label_column
        Header of label column.
    seed
        Random seed.

    Returns
    -------
    The splitted training and tuning data.
    """
    if tuning_data is None:
        if is_classification:
            stratify = train_data[label_column]
        else:
            stratify = None
        if holdout_frac is None:
            val_frac = default_holdout_frac(len(train_data), hyperparameter_tune=False)
        else:
            val_frac = holdout_frac
        train_data, tuning_data = train_test_split(
            train_data,
            test_size=val_frac,
            stratify=stratify,
            random_state=np.random.RandomState(seed),
        )

    return train_data, tuning_data


def get_detected_data_types(column_types: Dict):
    """
    Extract data types from column types.

    Parameters
    ----------
    column_types
        A dataframe's column types.

    Returns
    -------
    A list of detected data types.
    """
    data_types = []
    for col_type in column_types.values():
        if col_type.startswith(IMAGE) and IMAGE not in data_types:
            data_types.append(IMAGE)
        elif col_type.startswith(TEXT_NER) and TEXT_NER not in data_types:
            data_types.append(TEXT_NER)
        elif col_type.startswith(TEXT) and TEXT not in data_types:
            data_types.append(TEXT)
        elif col_type.startswith(DOCUMENT) and DOCUMENT not in data_types:
            data_types.append(DOCUMENT)
        elif col_type.startswith(NUMERICAL) and NUMERICAL not in data_types:
            data_types.append(NUMERICAL)
        elif col_type.startswith(CATEGORICAL) and CATEGORICAL not in data_types:
            data_types.append(CATEGORICAL)
        elif col_type.startswith(ROIS) and ROIS not in data_types:
            data_types.append(ROIS)

    return data_types
