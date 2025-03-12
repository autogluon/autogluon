import logging
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from torch import nn

from autogluon.core.utils import default_holdout_frac, generate_train_test_split_combined
from autogluon.core.utils.loaders import load_pd

from ..constants import (
    BINARY,
    CATEGORICAL,
    DEFAULT_SHOT,
    DOCUMENT,
    FEW_SHOT,
    IDENTIFIER,
    IMAGE,
    IMAGE_PATH,
    LABEL,
    MMDET_IMAGE,
    MMLAB_MODELS,
    MULTICLASS,
    NER_ANNOTATION,
    NER_TEXT,
    NUMERICAL,
    REGRESSION,
    ROIS,
    SAM,
    SEMANTIC_SEGMENTATION_IMG,
    TEXT,
    TEXT_NER,
)
from .collator import DictCollator
from .infer_types import is_image_column
from .label_encoder import NerLabelEncoder
from .mixup import MixupModule
from .preprocess_dataframe import MultiModalFeaturePreprocessor
from .process_categorical import CategoricalProcessor
from .process_document import DocumentProcessor
from .process_image import ImageProcessor
from .process_label import LabelProcessor
from .process_mmlab import MMDetProcessor
from .process_ner import NerProcessor
from .process_numerical import NumericalProcessor
from .process_semantic_seg_img import SemanticSegImageProcessor
from .process_text import TextProcessor

logger = logging.getLogger(__name__)


def get_collate_fn(
    df_preprocessor: Union[MultiModalFeaturePreprocessor, List[MultiModalFeaturePreprocessor]],
    data_processors: Union[Dict, List[Dict]],
    per_gpu_batch_size: Optional[int] = None,
):
    """
    Collect collator functions for each modality input of every model.
    These collator functions are wrapped by the "Dict" collator function,
    which can then be used by the Pytorch DataLoader.

    Parameters
    ----------
    df_preprocessor
        One or a list of dataframe preprocessors.
    data_processors
        One or a list of data processor dicts.
    per_gpu_batch_size
        Mini-batch size for each GPU.

    Returns
    -------
    A "Dict" collator wrapping other collators.
    """
    if isinstance(df_preprocessor, MultiModalFeaturePreprocessor):
        df_preprocessor = [df_preprocessor]
    if isinstance(data_processors, dict):
        data_processors = [data_processors]

    collate_fn = {}
    for per_preprocessor, per_data_processors_group in zip(df_preprocessor, data_processors):
        for per_modality in per_data_processors_group:
            per_modality_column_names = per_preprocessor.get_column_names(modality=per_modality)
            if per_modality_column_names:
                for per_model_processor in per_data_processors_group[per_modality]:
                    if per_model_processor.prefix.lower().startswith(MMLAB_MODELS):
                        collate_fn.update(
                            per_model_processor.collate_fn(
                                per_modality_column_names, per_gpu_batch_size=per_gpu_batch_size
                            )
                        )
                    else:
                        collate_fn.update(per_model_processor.collate_fn(per_modality_column_names))
    return DictCollator(collate_fn)


def apply_df_preprocessor(
    data: pd.DataFrame,
    df_preprocessor: MultiModalFeaturePreprocessor,
    modalities: Iterable,
):
    """
    Preprocess one dataframe with one df_preprocessor.

    Parameters
    ----------
    data
        A pandas dataframe.
    df_preprocessor
        One dataframe preprocessor object.
    modalities
        A list of data modalities to preprocess.

    Returns
    -------
    modality_features
        Preprocessed features of given modalities.
    modality_types
        Minor modality types of each major modality.
    sample_num
        Number of samples.
    """
    lengths = []
    modality_features = {}
    modality_types = {}
    for per_modality in modalities:
        per_modality_features, per_modality_types = getattr(df_preprocessor, f"transform_{per_modality}")(data)
        modality_features[per_modality] = per_modality_features
        modality_types[per_modality] = per_modality_types
        if per_modality_features:
            lengths.append(len(per_modality_features[next(iter(per_modality_features))]))
    assert len(set(lengths)) == 1  # make sure each modality has the same sample num
    sample_num = lengths[0]

    return modality_features, modality_types, sample_num


def apply_data_processor(
    per_sample_features: Dict,
    data_processors: Dict,
    data_types: Dict,
    is_training: bool,
    load_only=False,
):
    """
    Process one sample's features.

    Parameters
    ----------
    per_sample_features
        Features of one sample.
    data_processors
        A dict of data processors.
    data_types
        Data types of all columns.
    is_training
        Whether is training.
    load_only
        Whether to only load the data. Other processing steps may happen in dataset.__getitem__.

    Returns
    -------
    The processed features of one sample.
    """
    sample_features = {}
    for per_modality, per_modality_processors in data_processors.items():
        for per_model_processor in per_modality_processors:
            if per_modality in per_sample_features and per_sample_features[per_modality]:
                sample_features.update(
                    per_model_processor(
                        per_sample_features[per_modality],
                        data_types[per_modality],
                        is_training=is_training,
                        load_only=load_only,
                    )
                    if per_model_processor.prefix.lower().startswith(MMDET_IMAGE)
                    else per_model_processor(
                        per_sample_features[per_modality],
                        data_types[per_modality],
                        is_training=is_training,
                    )
                )

    return sample_features


def get_per_sample_features(
    modality_features: Dict, modality_types: Dict, idx: int, id_mappings: Optional[Dict] = None
):
    """
    Extract the modality features of one sample.

    Parameters
    ----------
    modality_features
        Modality features of all samples.
    modality_types
        Data types of all columns.
    idx
        The sample index.
    id_mappings
        Id-to-content mappings. The contents can be text, image, etc.
        This is used when the dataframe contains the query/response indexes instead of their contents.

    Returns
    -------
    One sample's modality features.
    """
    ret = dict()
    for per_modality, per_modality_features in modality_features.items():
        if per_modality_features:
            per_modality_ret = dict()
            for per_col_name, per_col_features in per_modality_features.items():
                per_sample_features = per_col_features[idx]
                if (
                    modality_types
                    and modality_types[per_modality]
                    and modality_types[per_modality][per_col_name].endswith(IDENTIFIER)
                ):
                    per_sample_features = id_mappings[per_col_name][per_sample_features]

                per_modality_ret[per_col_name] = per_sample_features
            ret[per_modality] = per_modality_ret

    return ret


def default_holdout_frac(num_train_rows, hyperparameter_tune=False):
    """Returns default holdout_frac used in fit().
    Between row count 5,000 and 25,000 keep 0.1 holdout_frac, as we want to grow validation set to a stable 2500 examples.
    """
    if num_train_rows < 5000:
        holdout_frac = max(0.1, min(0.2, 500.0 / num_train_rows))
    else:
        holdout_frac = max(0.01, min(0.1, 2500.0 / num_train_rows))

    if hyperparameter_tune:
        holdout_frac = min(
            0.2, holdout_frac * 2
        )  # We want to allocate more validation data for HPO to avoid overfitting

    return holdout_frac


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
    if label_column in column_types and column_types[label_column] == NER_ANNOTATION:
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


def get_image_transforms(model_config: DictConfig, model_name: str, advanced_hyperparameters: Dict):
    """
    Get the image transforms of one image-related model.
    Use the transforms in advanced_hyperparameters with higher priority.

    Parameters
    ----------
    model_config
        Config of one model.
    model_name
        Name of one model.
    advanced_hyperparameters
        The advanced hyperparameters whose values are complex objects.

    Returns
    -------
    The image transforms used in training and validation.
    """
    train_transform_key = f"model.{model_name}.train_transforms"
    val_transform_key = f"model.{model_name}.val_transforms"
    if advanced_hyperparameters and train_transform_key in advanced_hyperparameters:
        train_transforms = advanced_hyperparameters[train_transform_key]
    else:
        train_transforms = model_config.train_transforms
        train_transforms = list(train_transforms)

    if advanced_hyperparameters and val_transform_key in advanced_hyperparameters:
        val_transforms = advanced_hyperparameters[val_transform_key]
    else:
        val_transforms = model_config.val_transforms
        val_transforms = list(val_transforms)

    return train_transforms, val_transforms


def create_data_processor(
    data_type: str,
    config: DictConfig,
    model: nn.Module,
    advanced_hyperparameters: Optional[Dict] = None,
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
        train_transforms, val_transforms = get_image_transforms(
            model_config=model_config,
            model_name=model.prefix,
            advanced_hyperparameters=advanced_hyperparameters,
        )
        data_processor = ImageProcessor(
            model=model,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            max_image_num_per_column=model_config.max_image_num_per_column,
            missing_value_strategy=config.data.image.missing_value_strategy,
            dropout=config.data.modality_dropout,
        )
    elif data_type == TEXT:
        data_processor = TextProcessor(
            model=model,
            insert_sep=model_config.insert_sep,
            stochastic_chunk=model_config.stochastic_chunk,
            text_detection_length=model_config.text_aug_detect_length,
            text_trivial_aug_maxscale=model_config.text_trivial_aug_maxscale,
            train_augment_types=model_config.text_train_augment_types,
            normalize_text=config.data.text.normalize_text,
            template_config=config.data.templates,
            dropout=config.data.modality_dropout,
        )
    elif data_type == CATEGORICAL:
        data_processor = CategoricalProcessor(
            model=model,
            dropout=config.data.modality_dropout,
        )
    elif data_type == NUMERICAL:
        data_processor = NumericalProcessor(
            model=model,
            merge=model_config.merge,
            dropout=config.data.modality_dropout,
        )
    elif data_type == LABEL:
        data_processor = LabelProcessor(model=model)
    elif data_type == TEXT_NER:
        data_processor = NerProcessor(
            model=model,
            max_len=model_config.max_text_len,
            entity_map=config.entity_map,
        )
    elif data_type == ROIS:
        data_processor = MMDetProcessor(
            model=model,
            max_img_num_per_col=model_config.max_img_num_per_col,
            missing_value_strategy=config.data.image.missing_value_strategy,
        )
    elif data_type == DOCUMENT:
        train_transforms, val_transforms = get_image_transforms(
            model_config=model_config,
            model_name=model.prefix,
            advanced_hyperparameters=advanced_hyperparameters,
        )
        data_processor = DocumentProcessor(
            model=model,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            size=model_config.image_size,
            text_max_len=model_config.max_text_len,
            missing_value_strategy=config.data.document.missing_value_strategy,
        )
    elif data_type == SEMANTIC_SEGMENTATION_IMG:
        data_processor = SemanticSegImageProcessor(
            model=model,
            img_transforms=model_config.img_transforms,
            gt_transforms=model_config.gt_transforms,
            train_transforms=model_config.train_transforms,
            val_transforms=model_config.val_transforms,
            ignore_label=model_config.ignore_label,
        )
    else:
        raise ValueError(f"unknown data type: {data_type}")

    return data_processor


def create_fusion_data_processors(
    config: DictConfig,
    model: nn.Module,
    requires_label: Optional[bool] = True,
    requires_data: Optional[bool] = True,
    advanced_hyperparameters: Optional[Dict] = None,
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
        SEMANTIC_SEGMENTATION_IMG: [],
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
        elif per_name == SAM:
            data_processors[SEMANTIC_SEGMENTATION_IMG].append(
                create_data_processor(
                    data_type=SEMANTIC_SEGMENTATION_IMG,
                    config=config,
                    model=per_model,
                )
            )
            if data_types is not None and SEMANTIC_SEGMENTATION_IMG in data_types:
                data_types.remove(SEMANTIC_SEGMENTATION_IMG)
            requires_label = False

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
                    advanced_hyperparameters=advanced_hyperparameters,
                )
                data_processors[data_type].append(per_data_processor)

    # Only keep the modalities with non-empty processors.
    data_processors = {k: v for k, v in data_processors.items() if len(v) > 0}

    if TEXT_NER in data_processors and LABEL in data_processors:
        # LabelProcessor is not needed for NER tasks as annotations are handled in NerProcessor.
        data_processors.pop(LABEL)
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


def split_train_tuning_data(
    data: pd.DataFrame,
    holdout_frac: float = None,
    problem_type: str = None,
    label_column: str = None,
    random_state: int = 0,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits `data` into `train_data` and `tuning_data`.
    If the problem_type is one of ['binary', 'multiclass']:
        The split will be done with stratification on the label column.
        Will guarantee at least 1 sample of every class in `data` will be present in `train_data`.
            If only 1 sample of a class exists, it will always be put in `train_data` and not `tuning_data`.

    Parameters
    ----------
    data : pd.DataFrame
        The data to be split
    holdout_frac : float, default = None
        The ratio of data to use as validation.
        If 0.2, 20% of the data will be used for validation, and 80% for training.
        If None, the ratio is automatically determined,
        ranging from 0.2 for small row count to 0.01 for large row count.
    random_state : int, default = 0
        The random state to use when splitting the data, to make the splitting process deterministic.
        If None, a random value is used.

    Returns
    -------
    Tuple of (train_data, tuning_data) of the split `data`
    """
    if holdout_frac is None:
        holdout_frac = default_holdout_frac(num_train_rows=len(data), hyperparameter_tune=False)

    # TODO: Hack since the recognized problem types are only binary, multiclass, and regression
    #  Problem types used for purpose of stratification, so regression = no stratification
    if problem_type in [BINARY, MULTICLASS]:
        problem_type_for_split = problem_type
    else:
        problem_type_for_split = REGRESSION

    train_data, tuning_data = generate_train_test_split_combined(
        data=data,
        label=label_column,
        test_size=holdout_frac,
        problem_type=problem_type_for_split,
        random_state=random_state,
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
