import pytz
import datetime
import os
import functools
import logging
import pandas as pd
import pickle
import collections
import copy
import torch
from torch import nn
import warnings
from contextlib import contextmanager
from typing import Optional, List, Any, Dict, Tuple, Union
from nptyping import NDArray
from omegaconf import OmegaConf, DictConfig
from autogluon.core.metrics import get_metric

from .models import (
    HFAutoModelForTextPrediction,
    TimmAutoModelForImagePrediction,
    CLIPForImageText,
    CategoricalMLP,
    NumericalMLP,
    MultimodalFusionMLP,
    NumericalTransformer,
    CategoricalTransformer,
    MultimodalFusionTransformer,
)
from .data import (
    ImageProcessor,
    TextProcessor,
    CategoricalProcessor,
    NumericalProcessor,
    LabelProcessor,
    MultiModalFeaturePreprocessor,
)
from .constants import (
    ACCURACY, RMSE, R2, PEARSONR, SPEARMANR, ALL_MODALITIES,
    IMAGE, TEXT, CATEGORICAL, NUMERICAL,
    LABEL, MULTICLASS, BINARY, REGRESSION,
    Y_PRED_PROB, Y_PRED, Y_TRUE, AUTOMM,
    CLIP, TIMM_IMAGE, HF_TEXT, NUMERICAL_MLP,
    CATEGORICAL_MLP, FUSION_MLP, NUMERICAL_TRANSFORMER,
    CATEGORICAL_TRANSFORMER, FUSION_TRANSFORMER,
    ROC_AUC, AVERAGE_PRECISION, LOG_LOSS,
)
from .presets import (
    list_model_presets,
    get_preset,
)

logger = logging.getLogger(AUTOMM)


def infer_metrics(
        problem_type: Optional[str] = None,
        eval_metric_name: Optional[str] = None,
):
    """
    Infer the validation metric and the evaluation metric if not provided.
    Validation metric is for early-stopping and selecting the best model checkpoints.
    Evaluation metric is to report performance to users.

    If the evaluation metric is provided, then we use it as the validation metric.
    But there are some exceptions that validation metric is different from evaluation metric.
    For example, if the provided evaluation metric is `r2`, we set the validation metric as `rmse`
    since `torchmetrics.R2Score` may encounter errors for per gpu batch size 1. Another example is
    that `torchmetrics.AUROC` requires that both positive and negative examples are available in a mini-batch.
    When training a large model, the per gpu batch size is probably small, leading to an incorrect
    roc_auc score.


    Parameters
    ----------
    problem_type
        Type of problem.
    eval_metric_name
        Name of evaluation metric provided by users.

    Returns
    -------
    validation_metric_name
        Name of validation metric.
    eval_metric_name
        Name of evaluation metric.
    """
    if eval_metric_name is not None:
        validation_metric_name = eval_metric_name
        return validation_metric_name, eval_metric_name

    if problem_type in [MULTICLASS, BINARY]:
        eval_metric_name = ACCURACY
    elif problem_type == REGRESSION:
        eval_metric_name = RMSE
    else:
        raise NotImplementedError(
            f"Problem type: {problem_type} is not supported yet!"
        )

    validation_metric_name = eval_metric_name

    return validation_metric_name, eval_metric_name


def get_config(
        config: Union[dict, DictConfig],
        overrides: Optional[Union[str, List[str], Dict]] = None,
):
    """
    Construct configurations for model, data, optimization, and environment.
    It supports to overrides some default configurations.

    Parameters
    ----------
    config
        A dictionary including four keys: "model", "data", "optimization", and "environment".
        If any key is not not given, we will fill in with the default value.

        The value of each key can be a string, yaml path, or DictConfig object. For example:
        config = {
                        "model": "fusion_mlp_image_text_tabular",
                        "data": "default",
                        "optimization": "adamw",
                        "environment": "default",
                    }
            or
            config = {
                        "model": "/path/to/model/config.yaml",
                        "data": "/path/to/data/config.yaml",
                        "optimization": "/path/to/optimization/config.yaml",
                        "environment": "/path/to/environment/config.yaml",
                    }
            or
            config = {
                        "model": OmegaConf.load("/path/to/model/config.yaml"),
                        "data": OmegaConf.load("/path/to/data/config.yaml"),
                        "optimization": OmegaConf.load("/path/to/optimization/config.yaml"),
                        "environment": OmegaConf.load("/path/to/environment/config.yaml"),
                    }
    overrides
        This is to override some default configurations.
            For example, changing the text and image backbones can be done by formatting:

            a string
            overrides = "model.hf_text.checkpoint_name=google/electra-small-discriminator
            model.timm_image.checkpoint_name=swin_small_patch4_window7_224"

            or a list of strings
            overrides = ["model.hf_text.checkpoint_name=google/electra-small-discriminator",
            "model.timm_image.checkpoint_name=swin_small_patch4_window7_224"]

            or a dictionary
            overrides = {
                            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
                            "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
                        }

    Returns
    -------
    Configurations as a DictConfig object
    """
    if config is None:
        config = get_preset(list_model_presets()[0])
    if not isinstance(config, DictConfig):
        all_configs = []
        for k, default_value in [('model', 'fusion_mlp_image_text_tabular'),
                                 ('data', 'default'),
                                 ('optimization', 'adamw'),
                                 ('environment', 'default')]:
            if k not in config:
                config[k] = default_value
        for k, v in config.items():
            if isinstance(v, dict):
                per_config = OmegaConf.create(v)
            elif isinstance(v, DictConfig):
                per_config = v
            elif isinstance(v, str):
                if v.lower().endswith((".yaml", ".yml")):
                    per_config = OmegaConf.load(os.path.expanduser(v))
                else:
                    cur_path = os.path.dirname(os.path.abspath(__file__))
                    config_path = os.path.join(cur_path, "configs", k, f"{v}.yaml")
                    per_config = OmegaConf.load(config_path)
            else:
                raise ValueError(f"Unknown configuration type: {type(v)}")

            all_configs.append(per_config)

        config = OmegaConf.merge(*all_configs)
    verify_model_names(config.model)
    logger.debug(f"overrides: {overrides}")
    if overrides is not None:
        # avoid manipulating user-provided overrides
        overrides = copy.deepcopy(overrides)
        # apply customized model names
        overrides = parse_dotlist_conf(overrides)  # convert to a dict
        config.model = customize_model_names(
            config=config.model,
            customized_names=overrides.get("model.names", None),
        )
        # remove `model.names` from overrides since it's already applied.
        overrides.pop("model.names", None)
        # apply all the overrides
        config = apply_omegaconf_overrides(config, overrides=overrides, check_key_exist=True)
    verify_model_names(config.model)
    return config


def verify_model_names(config: DictConfig):
    """
    Verify whether provided model names are valid.

    Parameters
    ----------
    config
        Config should have a attribute `names`, which contains a list of
        attribute names, e.g., ["timm_image", "hf_text"]. And each string in
        `config.names` should also be a attribute of `config`, e.g, `config.timm_image`.
    """
    # must have attribute `names`
    assert hasattr(config, "names")
    # assure no duplicate names
    assert len(config.names) == len(set(config.names))
    # verify that strings in `config.names` match the keys of `config`.
    keys = list(config.keys())
    keys.remove("names")
    assert set(config.names).issubset(set(keys)), \
        f"`{config.names}` do not match config keys {keys}"

    # verify that no name starts with another one
    names = sorted(config.names, key=lambda ele: len(ele), reverse=True)
    for i in range(len(names)):
        if names[i].startswith(tuple(names[i+1:])):
            raise ValueError(
                f"name {names[i]} starts with one of another name: {names[i+1:]}"
            )


def get_name_prefix(
        name: str,
        prefixes: List[str],
):
    """
    Get a name's prefix from some available candidates.

    Parameters
    ----------
    name
        A name string
    prefixes
        Available prefixes.

    Returns
    -------
        Prefix of the name.
    """
    search_results = [pre for pre in prefixes if name.lower().startswith(pre)]
    if len(search_results) == 0:
        return None
    elif len(search_results) >= 2:
        raise ValueError(
            f"Model name `{name}` is mapped to multiple models, "
            f"which means some names in `{prefixes}` have duplicate prefixes."
        )
    else:
        return search_results[0]


def customize_model_names(
        config: DictConfig,
        customized_names: Union[str, List[str]],
):
    """
    Customize attribute names of `config` with the provided names.
    A valid customized name string should start with one available name
    string in `config`.

    Parameters
    ----------
    config
        Config should have a attribute `names`, which contains a list of
        attribute names, e.g., ["timm_image", "hf_text"]. And each string in
        `config.names` should also be a attribute of `config`, e.g, `config.timm_image`.
    customized_names
        The provided names to replace the existing ones in `config.names` as well as
        the corresponding attribute names. For example, if `customized_names` is
        ["timm_image_123", "hf_text_abc"], then `config.timm_image` and `config.hf_text`
        are changed to `config.timm_image_123` and `config.hf_text_abc`.

    Returns
    -------
        A new config with its first-level attributes customized by the provided names.
    """
    if not customized_names:
        return config

    if isinstance(customized_names, str):
        customized_names = OmegaConf.from_dotlist([f'names={customized_names}']).names

    new_config = OmegaConf.create()
    new_config.names = []
    available_prefixes = list(config.keys())
    available_prefixes.remove("names")
    for per_name in customized_names:
        per_prefix = get_name_prefix(
            name=per_name,
            prefixes=available_prefixes,
        )
        if per_prefix:
            per_config = getattr(config, per_prefix)
            setattr(new_config, per_name, copy.deepcopy(per_config))
            new_config.names.append(per_name)
        else:
            logger.debug(
                f"Removing {per_name}, which doesn't start with any of these prefixes: {available_prefixes}."
            )

    if len(new_config.names) == 0:
        raise ValueError(
            f"No customized name in `{customized_names}` starts with name prefixes in `{available_prefixes}`."
        )

    return new_config


def select_model(
        config: DictConfig,
        df_preprocessor: MultiModalFeaturePreprocessor,
):
    """
    Filter model config through the detected modalities in the training data.
    If MultiModalFeaturePreprocessor can't detect some modality,
    this function will remove the models that use this modality. This function is to
    maximize the user flexibility in defining the config.
    For example, if one uses the "fusion_mlp_image_text_tabular" as the model config template
    but the training data don't have images, this function will filter out all the models
    using images, such as Swin Transformer and CLIP.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model"
    df_preprocessor
        A MultiModalFeaturePreprocessor object, which has called .fit() on the training data.
        Column names of the same modality are grouped into one list. If a modality's list is empty,
        it means the training data don't have this modality.

    Returns
    -------
    Config with some unused models removed.
    """
    data_status = {}
    for per_modality in ALL_MODALITIES:
        data_status[per_modality] = False
    if len(df_preprocessor.image_path_names) > 0:
        data_status[IMAGE] = True
    if len(df_preprocessor.text_feature_names) > 0:
        data_status[TEXT] = True
    if len(df_preprocessor.categorical_feature_names) > 0:
        data_status[CATEGORICAL] = True
    if len(df_preprocessor.numerical_feature_names) > 0:
        data_status[NUMERICAL] = True

    names = config.model.names
    if isinstance(names, str):
        names = [names]
    selected_model_names = []
    fusion_model_name = []
    for model_name in names:
        model_config = getattr(config.model, model_name)
        if model_config.data_types is None:
            fusion_model_name.append(model_name)
            continue
        model_data_status = [data_status[d_type] for d_type in model_config.data_types]
        if all(model_data_status):
            selected_model_names.append(model_name)
        else:
            delattr(config.model, model_name)

    if len(selected_model_names) == 0:
        raise ValueError("No model is available for this dataset.")
    # only allow no more than 1 fusion model
    assert len(fusion_model_name) <= 1
    if len(selected_model_names) > 1:
        assert len(fusion_model_name) == 1
        selected_model_names.extend(fusion_model_name)
    else:  # remove the fusion model's config make `config.model.names` and the keys of `config.model` consistent.
        if len(fusion_model_name) == 1 and hasattr(config.model, fusion_model_name[0]):
            delattr(config.model, fusion_model_name[0])

    config.model.names = selected_model_names
    logger.debug(f"selected models: {selected_model_names}")

    return config


def init_df_preprocessor(
        config: DictConfig,
        column_types: collections.OrderedDict,
        label_column: str,
        train_df_x: pd.DataFrame,
        train_df_y: pd.Series,
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
    df_preprocessor = MultiModalFeaturePreprocessor(
        config=config,
        column_types=column_types,
        label_column=label_column,
    )
    df_preprocessor.fit(
        X=train_df_x,
        y=train_df_y,
    )

    return df_preprocessor


def init_data_processors(
        config: DictConfig,
        df_preprocessor: MultiModalFeaturePreprocessor,
):
    """
    Create the data processors according to the model config. This function creates one processor for
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
    df_preprocessor
        The dataframe preprocessor.

    Returns
    -------
    A dictionary with modalities as the keys. Each modality has a list of processors.
    Note that "label" is also treated as a modality for convenience.
    """
    names = config.model.names
    if isinstance(names, str):
        names = [names]

    data_processors = {
        IMAGE: [],
        TEXT: [],
        CATEGORICAL: [],
        NUMERICAL: [],
        LABEL: [],
    }
    for model_name in names:
        model_config = getattr(config.model, model_name)
        # each model has its own label processor
        data_processors[LABEL].append(
            LabelProcessor(prefix=model_name)
        )
        if model_config.data_types is None:
            continue
        for d_type in model_config.data_types:
            if d_type == IMAGE:
                data_processors[IMAGE].append(
                    ImageProcessor(
                        prefix=model_name,
                        checkpoint_name=model_config.checkpoint_name,
                        train_transform_types=model_config.train_transform_types,
                        val_transform_types=model_config.val_transform_types,
                        image_column_names=df_preprocessor.image_path_names,
                        norm_type=model_config.image_norm,
                        size=model_config.image_size,
                        max_img_num_per_col=model_config.max_img_num_per_col,
                        missing_value_strategy=config.data.image.missing_value_strategy,
                    )
                )
            elif d_type == TEXT:
                data_processors[TEXT].append(
                    TextProcessor(
                        prefix=model_name,
                        tokenizer_name=model_config.tokenizer_name,
                        checkpoint_name=model_config.checkpoint_name,
                        text_column_names=df_preprocessor.text_feature_names,
                        max_len=model_config.max_text_len,
                        insert_sep=model_config.insert_sep,
                        text_segment_num=model_config.text_segment_num,
                        stochastic_chunk=model_config.stochastic_chunk,
                    )
                )
            elif d_type == CATEGORICAL:
                data_processors[CATEGORICAL].append(
                    CategoricalProcessor(
                        prefix=model_name,
                        categorical_column_names=df_preprocessor.categorical_feature_names,
                    )
                )
            elif d_type == NUMERICAL:
                data_processors[NUMERICAL].append(
                    NumericalProcessor(
                        prefix=model_name,
                        numerical_column_names=df_preprocessor.numerical_feature_names,
                        merge=model_config.merge,
                    )
                )
            else:
                raise ValueError(f"unknown data type: {d_type}")

    assert len(data_processors[LABEL]) > 0

    # Only keep the modalities with non-empty processors.
    data_processors = {k: v for k, v in data_processors.items() if len(v) > 0}
    return data_processors


def create_model(
        config: DictConfig,
        num_classes: int,
        num_numerical_columns: Optional[int] = None,
        num_categories: Optional[List[int]] = None,
        pretrained: Optional[bool] = True,
):
    """
    Create models. It supports the auto models of huggingface text and timm image.
    Multimodal models, e.g., CLIP, should be added case-by-case since their configs and usages
    may be different. It uses MLP for the numerical features, categorical features, and late-fusion.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model".
    num_classes
        The class number for a classification task. It should be 1 for a regression task.
    num_numerical_columns
        The number of numerical columns in the training dataframe.
    num_categories
        The category number for each categorical column in the training dataframe.
    pretrained
        Whether using the pretrained timm models. If pretrained=True, download the pretrained model.

    Returns
    -------
    A Pytorch model.
    """
    names = config.model.names
    if isinstance(names, str):
        names = [names]
    # make sure no duplicate model names
    assert len(names) == len(set(names))
    logger.debug(f"output_shape: {num_classes}")
    all_models = []
    for model_name in names:
        model_config = getattr(config.model, model_name)
        if model_name.lower().startswith(CLIP):
            model = CLIPForImageText(
                prefix=model_name,
                checkpoint_name=model_config.checkpoint_name,
                num_classes=num_classes,
            )
        elif model_name.lower().startswith(TIMM_IMAGE):
            model = TimmAutoModelForImagePrediction(
                prefix=model_name,
                checkpoint_name=model_config.checkpoint_name,
                num_classes=num_classes,
                mix_choice=model_config.mix_choice,
                pretrained=pretrained,
            )
        elif model_name.lower().startswith(HF_TEXT):
            model = HFAutoModelForTextPrediction(
                prefix=model_name,
                checkpoint_name=model_config.checkpoint_name,
                num_classes=num_classes,
            )
        elif model_name.lower().startswith(NUMERICAL_MLP):
            model = NumericalMLP(
                prefix=model_name,
                in_features=num_numerical_columns,
                hidden_features=model_config.hidden_size,
                out_features=model_config.hidden_size,
                num_layers=model_config.num_layers,
                activation=model_config.activation,
                dropout_prob=model_config.drop_rate,
                normalization=model_config.normalization,
                num_classes=num_classes,
            )
        elif model_name.lower().startswith(NUMERICAL_TRANSFORMER):
            model = NumericalTransformer(
                prefix=model_name,
                in_features=num_numerical_columns,
                out_features=model_config.out_features,
                d_token=model_config.d_token,
                n_blocks=model_config.num_trans_blocks,
                attention_n_heads=model_config.num_attn_heads,
                attention_dropout=model_config.attention_dropout,
                residual_dropout=model_config.residual_dropout,
                ffn_dropout=model_config.ffn_dropout,
                attention_normalization=model_config.normalization,
                ffn_normalization=model_config.normalization,
                head_normalization=model_config.normalization,
                ffn_activation=model_config.ffn_activation,
                head_activation=model_config.head_activation,
                embedding_arch=model_config.embedding_arch,
                num_classes=num_classes,
                cls_token=True if len(names) == 1 else False,
            )
        elif model_name.lower().startswith(CATEGORICAL_MLP):
            model = CategoricalMLP(
                prefix=model_name,
                num_categories=num_categories,
                out_features=model_config.hidden_size,
                num_layers=model_config.num_layers,
                activation=model_config.activation,
                dropout_prob=model_config.drop_rate,
                normalization=model_config.normalization,
                num_classes=num_classes,
            )
        elif model_name.lower().startswith(CATEGORICAL_TRANSFORMER):
            model = CategoricalTransformer(
                prefix=model_name,
                num_categories=num_categories,
                out_features=model_config.out_features,
                d_token=model_config.d_token,
                n_blocks=model_config.num_trans_blocks,
                attention_n_heads=model_config.num_attn_heads,
                attention_dropout=model_config.attention_dropout,
                residual_dropout=model_config.residual_dropout,
                ffn_dropout=model_config.ffn_dropout,
                attention_normalization=model_config.normalization,
                ffn_normalization=model_config.normalization,
                head_normalization=model_config.normalization,
                ffn_activation=model_config.ffn_activation,
                head_activation=model_config.head_activation,
                num_classes=num_classes,
                cls_token=True if len(names) == 1 else False,
            )
        elif model_name.lower().startswith(FUSION_MLP):
            fusion_model = functools.partial(
                MultimodalFusionMLP,
                prefix=model_name,
                hidden_features=model_config.hidden_sizes,
                num_classes=num_classes,
                adapt_in_features=model_config.adapt_in_features,
                activation=model_config.activation,
                dropout_prob=model_config.drop_rate,
                normalization=model_config.normalization,
                loss_weight=model_config.weight if hasattr(model_config, "weight") else None,
            )
            continue
        elif model_name.lower().startswith(FUSION_TRANSFORMER):
            fusion_model = functools.partial(
                MultimodalFusionTransformer,
                prefix=model_name,
                hidden_features=model_config.hidden_size,
                num_classes=num_classes,
                n_blocks=model_config.n_blocks,
                attention_n_heads=model_config.attention_n_heads,
                ffn_d_hidden=model_config.ffn_d_hidden,
                attention_dropout=model_config.attention_dropout,
                residual_dropout=model_config.residual_dropout,
                ffn_dropout=model_config.ffn_dropout,
                attention_normalization=model_config.normalization,
                ffn_normalization=model_config.normalization,
                head_normalization=model_config.normalization,
                ffn_activation=model_config.ffn_activation,
                head_activation=model_config.head_activation,
                adapt_in_features=model_config.adapt_in_features,
                loss_weight=model_config.weight if hasattr(model_config, "weight") else None,
            )
            continue
        else:
            raise ValueError(f"unknown model name: {model_name}")

        all_models.append(model)

    if len(all_models) > 1:
        # must have one fusion model if there are multiple independent models
        return fusion_model(models=all_models)
    elif len(all_models) == 1:
        return all_models[0]
    else:
        raise ValueError(f"No available models for {names}")


def save_pretrained_models(
        model: nn.Module,
        config: DictConfig,
        path: str,
) -> DictConfig:
    """
    Save the pretrained models and configs to local to make future loading not dependent on Internet access.
    By loading local checkpoints, Huggingface doesn't need to download pretrained checkpoints from Internet.
    It is called by setting "standalone=True" in "AutoMMPredictor.load()".

    Parameters
    ----------
    model
        One model.
    config
        A DictConfig object. The model config should be accessible by "config.model".
    path
        The path to save pretrained checkpoints.
    """
    requires_saving = any([
        model_name.lower().startswith((CLIP, HF_TEXT)) for model_name in config.model.names
    ])
    if not requires_saving:
        return config

    if len(config.model.names) == 1:
        model = nn.ModuleList([model])
    else:  # assumes the fusion model has a model attribute, a nn.ModuleList
        model = model.model
    for per_model in model:
        if per_model.prefix.lower().startswith((CLIP, HF_TEXT)):
            per_model.model.save_pretrained(os.path.join(path, per_model.prefix))
            model_config = getattr(config.model, per_model.prefix)
            model_config.checkpoint_name = os.path.join('local://', per_model.prefix)

    return config


def convert_checkpoint_name(
        config: DictConfig,
        path: str
) -> DictConfig:  
    """
    Convert the checkpoint name from relative path to absolute path for
    loading the pretrained weights in offline deployment.
    It is called by setting "standalone=True" in "AutoMMPredictor.load()". 

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model".
    path
        The saving path to the pretrained Huggingface models.
    """
    for model_name in config.model.names:
        if model_name.lower().startswith((CLIP, HF_TEXT)):
            model_config = getattr(config.model, model_name)
            if model_config.checkpoint_name.startswith('local://'):
                model_config.checkpoint_name = os.path.join(path, model_config.checkpoint_name[len('local://'):])
                assert os.path.exists(os.path.join(model_config.checkpoint_name, 'config.json')) # guarantee the existence of local configs
                assert os.path.exists(os.path.join(model_config.checkpoint_name, 'pytorch_model.bin'))
                
    return config


def save_text_tokenizers(
        text_processors: List[TextProcessor],
        path: str,
) -> List[TextProcessor]:
    """
    Save all the text tokenizers and record their relative paths, which are
    the corresponding model names, e.g, hf_text.

    Parameters
    ----------
    text_processors
        A list of text processors with tokenizers.
    path
        The root path.

    Returns
    -------
    A list of text processors with tokenizers replaced by their local relative paths.
    """
    for per_text_processor in text_processors:
        per_path = os.path.join(path, per_text_processor.prefix)
        per_text_processor.tokenizer.save_pretrained(per_path)
        per_text_processor.tokenizer = per_text_processor.prefix

    return text_processors


def load_text_tokenizers(
        text_processors: List[TextProcessor],
        path: str,
) -> List[TextProcessor]:
    """
    Load saved text tokenizers. If text processors already have tokenizers,
    then do nothing.

    Parameters
    ----------
    text_processors
        A list of text processors with tokenizers or their relative paths.
    path
        The root path.

    Returns
    -------
    A list of text processors with tokenizers loaded.
    """
    for per_text_processor in text_processors:
        if isinstance(per_text_processor.tokenizer, str):
            per_path = os.path.join(path, per_text_processor.tokenizer)
            per_text_processor.tokenizer = per_text_processor.get_pretrained_tokenizer(
                tokenizer_name=per_text_processor.tokenizer_name,
                checkpoint_name=per_path,
            )

    return text_processors


def make_exp_dir(
        root_path: str,
        job_name: str,
        create: Optional[bool] = True,
):
    """
    Creates the exp dir of format e.g.,: root_path/2022_01_01/job_name_12_00_00/
    This function is to better organize the training runs. It is recommended to call this
    function and pass the returned "exp_dir" to "AutoMMPredictor.fit(save_path=exp_dir)".

    Parameters
    ----------
    root_path
        The basic path where to create saving directories for training runs.
    job_name
        The job names to name training runs.
    create
        Whether to make the directory.

    Returns
    -------
    The formatted directory path.
    """
    tz = pytz.timezone('US/Pacific')
    ct = datetime.datetime.now(tz=tz)
    date_stamp = ct.strftime("%Y_%m_%d")
    time_stamp = ct.strftime("%H_%M_%S")

    # Group logs by day first
    exp_dir = os.path.join(root_path, date_stamp)

    # Then, group by run_name and hour + min + sec to avoid duplicates
    exp_dir = os.path.join(exp_dir, "_".join([job_name, time_stamp]))

    if create:
        os.makedirs(exp_dir, mode=0o777, exist_ok=False)

    return exp_dir


def average_checkpoints(
        checkpoint_paths: List[str],
):
    """
    Average a list of checkpoints' state_dicts.

    Parameters
    ----------
    checkpoint_paths
        A list of model checkpoint paths.

    Returns
    -------
    The averaged state_dict.
    """
    if len(checkpoint_paths) > 1:
        avg_state_dict = {}
        for per_path in checkpoint_paths:
            state_dict = torch.load(per_path, map_location=torch.device("cpu"))["state_dict"]
            for key in state_dict:
                if key in avg_state_dict:
                    avg_state_dict[key] += state_dict[key]
                else:
                    avg_state_dict[key] = state_dict[key]
            del state_dict

        num = torch.tensor(len(checkpoint_paths))
        for key in avg_state_dict:
            avg_state_dict[key] = avg_state_dict[key] / num.to(avg_state_dict[key])
    else:
        avg_state_dict = torch.load(checkpoint_paths[0], map_location=torch.device("cpu"))["state_dict"]

    return avg_state_dict


def compute_score(
        metric_data: dict,
        metric_name: str,
) -> float:
    """
    Use sklearn to compute the score of one metric.

    Parameters
    ----------
    metric_data
        A dictionary with the groundtruth (Y_TRUE) and predicted values (Y_PRED, Y_PRED_PROB).
        The predicted class probabilities are required to compute the roc_auc score.
    metric_name
        The name of metric to compute.

    Returns
    -------
    Computed score.
    """
    metric = get_metric(metric_name)
    if metric.name in [ROC_AUC, AVERAGE_PRECISION]:
        return metric._sign * metric(metric_data[Y_TRUE], metric_data[Y_PRED_PROB][:, 1])
    else:
        return metric._sign * metric(metric_data[Y_TRUE], metric_data[Y_PRED])


def parse_dotlist_conf(conf):
    """Parse the config files that is potentially in the dotlist format to a dictionary

    Parameters
    ----------
    conf
        Apply the conf stored as dotlist, e.g.,
         'aaa=a, bbb=b' or ['aaa=a, ', 'bbb=b'] to {'aaa': 'a', 'bbb': b}

    Returns
    -------
    new_conf
    """
    if isinstance(conf, str):
        conf = conf.split()
        need_parse = True
    elif isinstance(conf, (list, tuple)):
        need_parse = True
    elif isinstance(conf, dict):
        need_parse = False
    else:
        raise ValueError(f'Unsupported format of conf={conf}')
    if need_parse:
        new_conf = dict()
        curr_key = None
        curr_value = ''
        for ele in conf:
            if '=' in ele:
                key, v = ele.split('=')
                if curr_key is not None:
                    new_conf[curr_key] = curr_value
                curr_key = key
                curr_value = v
            else:
                if curr_key is None:
                    raise ValueError(f'Cannot parse the conf={conf}')
                curr_value = curr_value + ' ' + ele
        if curr_key is not None:
            new_conf[curr_key] = curr_value
        return new_conf
    else:
        return conf


def apply_omegaconf_overrides(
        conf: DictConfig,
        overrides: Union[List, Tuple, str, Dict, DictConfig],
        check_key_exist=True,
):
    """
    Apply omegaconf overrides.

    Parameters
    ----------
    conf
        The base configuration.
    overrides
        The overrides can be a string or a list.
    check_key_exist
        Whether to check if all keys in the overrides must exist in the conf.

    Returns
    -------
    new_conf
        The updated configuration.
    """
    overrides = parse_dotlist_conf(overrides)

    def _check_exist_dotlist(C, key_in_dotlist):
        if not isinstance(key_in_dotlist, list):
            key_in_dotlist = key_in_dotlist.split('.')
        if key_in_dotlist[0] in C:
            if len(key_in_dotlist) > 1:
                return _check_exist_dotlist(C[key_in_dotlist[0]], key_in_dotlist[1:])
            else:
                return True
        else:
            return False

    if check_key_exist:
        for ele in overrides.items():
            if not _check_exist_dotlist(conf, ele[0]):
                raise KeyError(f'"{ele[0]}" is not found in the config. You may need to check the overrides. '
                               f'overrides={overrides}')
    override_conf = OmegaConf.from_dotlist([f'{ele[0]}={ele[1]}' for ele in overrides.items()])
    conf = OmegaConf.merge(conf, override_conf)
    return conf


class LogFilter(logging.Filter):
    """
    Filter log messages with patterns.
    """

    def __init__(self, blacklist: Union[str, List[str]]):
        """
        Parameters
        ----------
        blacklist
            Patterns to be suppressed in logging.
        """
        super().__init__()
        if isinstance(blacklist, str):
            blacklist = [blacklist]
        self._blacklist = blacklist

    def filter(self, record):
        """
        Check whether to suppress a logging message.

        Parameters
        ----------
        record
            A logging message.

        Returns
        -------
        If True, no pattern exists in the message, hence printed out.
        If False, some pattern is in the message, hence filtered out.
        """
        matches = [pattern not in record.msg for pattern in self._blacklist]
        return all(matches)


def add_log_filter(target_logger, log_filter):
    """
    Add one log filter to the target logger.

    Parameters
    ----------
    target_logger
        Target logger
    log_filter
        Log filter
    """
    for handler in target_logger.handlers:
        handler.addFilter(log_filter)


def remove_log_filter(target_logger, log_filter):
    """
    Remove one log filter to the target logger.

    Parameters
    ----------
    target_logger
        Target logger
    log_filter
        Log filter
    """
    for handler in target_logger.handlers:
        handler.removeFilter(log_filter)


@contextmanager
def apply_log_filter(log_filter):
    """
    User contextmanager to control the scope of applying one log filter.
    Currently, it is to filter some pytorch lightning's log messages.
    But we can easily extend it to cover more loggers.

    Parameters
    ----------
    log_filter
        Log filter.
    """
    try:
        add_log_filter(logging.getLogger(), log_filter)
        add_log_filter(logging.getLogger("pytorch_lightning"), log_filter)
        yield

    finally:
        remove_log_filter(logging.getLogger(), log_filter)
        remove_log_filter(logging.getLogger("pytorch_lightning"), log_filter)


def modify_duplicate_model_names(
        predictor,
        postfix: str,
        blacklist: List[str],
):
    """
    Modify a predictor's model names if they exist in a blacklist.

    Parameters
    ----------
    predictor
        An AutoMMPredictor object.
    postfix
        The postfix used to change the duplicate names.
    blacklist
        A list of names. The provided predictor can't use model names in the list.

    Returns
    -------
    The predictor guaranteed has no duplicate model names with the backlist names.
    """
    model_names = []
    for n in predictor._config.model.names:
        if n in blacklist:
            new_name = f"{n}_{postfix}"
            assert new_name not in blacklist
            assert new_name not in predictor._config.model.names
            # modify model prefix
            if n == predictor._model.prefix:
                predictor._model.prefix = new_name
            else:
                assert isinstance(predictor._model.model, nn.ModuleList)
                for per_model in predictor._model.model:
                    if n == per_model.prefix:
                        per_model.prefix = new_name
                        break
            # modify data processor prefix
            for per_modality_processors in predictor._data_processors.values():
                for per_processor in per_modality_processors:
                    if n == per_processor.prefix:
                        per_processor.prefix = new_name
            # modify model config keys
            setattr(predictor._config.model, new_name, getattr(predictor._config.model, n))
            delattr(predictor._config.model, n)

            model_names.append(new_name)
        else:
            model_names.append(n)

    predictor._config.model.names = model_names

    return predictor


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
        if per_modality == LABEL:
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

    Returns
    -------
    The data processors with the flag on or off.
    """
    for per_modality_processors in data_processors.values():
        for per_model_processor in per_modality_processors:
            # label processor doesn't have requires_column_info.
            if hasattr(per_model_processor, "requires_column_info"):
                per_model_processor.requires_column_info = flag

    return data_processors
