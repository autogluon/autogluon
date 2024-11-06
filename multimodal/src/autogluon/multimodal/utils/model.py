import functools
import json
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import timm
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from ..constants import (
    ALL_MODALITIES,
    AUTOMM,
    CATEGORICAL,
    CATEGORICAL_MLP,
    CLIP,
    DOCUMENT,
    DOCUMENT_TRANSFORMER,
    FT_TRANSFORMER,
    FUSION_MLP,
    FUSION_NER,
    FUSION_TRANSFORMER,
    HF_TEXT,
    IMAGE,
    MMDET_IMAGE,
    MMOCR_TEXT_DET,
    MMOCR_TEXT_RECOG,
    NER,
    NER_TEXT,
    NUMERICAL,
    NUMERICAL_MLP,
    PEFT_ADDITIVE_STRATEGIES,
    SAM,
    SEMANTIC_SEGMENTATION_IMG,
    T_FEW,
    TEXT,
    TEXT_NER,
    TIMM_IMAGE,
    XYXY,
)
from ..data import MultiModalFeaturePreprocessor
from ..models import (
    CategoricalMLP,
    CLIPForImageText,
    DocumentTransformer,
    FT_Transformer,
    HFAutoModelForNER,
    HFAutoModelForTextPrediction,
    MMDetAutoModelForObjectDetection,
    MMOCRAutoModelForTextDetection,
    MMOCRAutoModelForTextRecognition,
    MultimodalFusionMLP,
    MultimodalFusionNER,
    MultimodalFusionTransformer,
    NumericalMLP,
    SAMForSemanticSegmentation,
    TFewModel,
    TimmAutoModelForImagePrediction,
)
from ..models.utils import inject_adaptation_to_linear_layer

logger = logging.getLogger(__name__)


def select_model(
    config: DictConfig,
    df_preprocessor: MultiModalFeaturePreprocessor,
    strict: Optional[bool] = True,
):
    """
    Filter model config through the detected modalities in the training data.
    If MultiModalFeaturePreprocessor can't detect some modality,
    this function will remove the models that use this modality. This function is to
    maximize the user flexibility in defining the config.
    For example, if one uses the default, including hf_text and timm_image, as the model config template
    but the training data don't have images, this function will filter out timm_image.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model"
    df_preprocessor
        A MultiModalFeaturePreprocessor object, which has called .fit() on the training data.
        Column names of the same modality are grouped into one list. If a modality's list is empty,
        it means the training data don't have this modality.
    strict
        If False, allow retaining one model when partial modalities are available for that model.

    Returns
    -------
    Config with some unused models removed.
    """
    data_status = {}
    for per_modality in ALL_MODALITIES:
        data_status[per_modality] = False
    if len(df_preprocessor.image_feature_names) > 0:
        data_status[IMAGE] = True
    if len(df_preprocessor.text_feature_names) > 0:
        data_status[TEXT] = True
    if len(df_preprocessor.categorical_feature_names) > 0:
        data_status[CATEGORICAL] = True
    if len(df_preprocessor.numerical_feature_names) > 0:
        data_status[NUMERICAL] = True
    if len(df_preprocessor.ner_feature_names) > 0:
        data_status[TEXT_NER] = True
    if len(df_preprocessor.document_feature_names) > 0:
        data_status[DOCUMENT] = True
    if len(df_preprocessor.semantic_segmentation_feature_names) > 0:
        data_status[SEMANTIC_SEGMENTATION_IMG] = True

    names = config.model.names
    if isinstance(names, str):
        names = [names]
    selected_model_names = []
    fusion_model_name = []
    for model_name in names:
        model_config = getattr(config.model, model_name)
        strict = getattr(model_config, "requires_all_dtypes", strict)
        if not model_config.data_types:
            fusion_model_name.append(model_name)
            continue
        model_data_status = [data_status[d_type] for d_type in model_config.data_types]
        if all(model_data_status):
            selected_model_names.append(model_name)
        else:
            if any(model_data_status) and not strict:
                selected_model_names.append(model_name)
            else:
                delattr(config.model, model_name)

    if len(selected_model_names) == 0:
        raise ValueError("No model is available for this dataset.")
    # only allow no more than 1 fusion model
    if len(fusion_model_name) > 1:
        raise ValueError(f"More than one fusion models `{fusion_model_name}` are detected, but only one is allowed.")

    if len(selected_model_names) > 1:
        assert len(fusion_model_name) == 1
        selected_model_names.extend(fusion_model_name)
    elif len(fusion_model_name) == 1 and hasattr(config.model, fusion_model_name[0]):
        delattr(config.model, fusion_model_name[0])

    config.model.names = selected_model_names
    logger.debug(f"selected models: {selected_model_names}")
    for model_name in selected_model_names:
        logger.debug(f"model dtypes: {getattr(config.model, model_name).data_types}")

    # clean up unused model configs
    model_keys = list(config.model.keys())
    for model_name in model_keys:
        if model_name not in selected_model_names + ["names"]:
            delattr(config.model, model_name)

    return config


def create_model(
    model_name: str,
    model_config: DictConfig,
    num_classes: Optional[int] = 0,
    classes: Optional[list] = None,
    num_numerical_columns: Optional[int] = None,
    num_categories: Optional[List[int]] = None,
    pretrained: Optional[bool] = True,
):
    """
    Create a single model.

    Parameters
    ----------
    model_name
        Name of the model.
    model_config
        Config of the model.
    num_classes
        The class number for a classification task. It should be 1 for a regression task.
    classes
        All classes in this dataset.
    num_numerical_columns
        The number of numerical columns in the training dataframe.
    num_categories
        The category number for each categorical column in the training dataframe.
    pretrained
        Whether using the pretrained timm models. If pretrained=True, download the pretrained model.

    Returns
    -------
    A model.
    """
    if model_name.lower().startswith(CLIP):
        model = CLIPForImageText(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
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
            pooling_mode=OmegaConf.select(model_config, "pooling_mode", default="cls"),
            gradient_checkpointing=OmegaConf.select(model_config, "gradient_checkpointing"),
            low_cpu_mem_usage=OmegaConf.select(model_config, "low_cpu_mem_usage", default=False),
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            use_fast=OmegaConf.select(model_config, "use_fast", default=True),
        )
    elif model_name.lower().startswith(T_FEW):
        model = TFewModel(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            length_norm=model_config.length_norm,  # Normalizes length to adjust for length bias in target template
            unlikely_loss=model_config.unlikely_loss,  # Adds loss term that lowers probability of incorrect outputs
            mc_loss=model_config.mc_loss,  # Adds multiple choice cross entropy loss
            num_classes=num_classes,
            gradient_checkpointing=OmegaConf.select(model_config, "gradient_checkpointing"),
            low_cpu_mem_usage=OmegaConf.select(model_config, "low_cpu_mem_usage", default=False),
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
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
            d_token=OmegaConf.select(model_config, "d_token"),
            embedding_arch=OmegaConf.select(model_config, "embedding_arch"),
            num_classes=num_classes,
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
    elif model_name.lower().startswith(DOCUMENT_TRANSFORMER):
        model = DocumentTransformer(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pooling_mode=OmegaConf.select(model_config, "pooling_mode", default="cls"),
            gradient_checkpointing=OmegaConf.select(model_config, "gradient_checkpointing"),
            low_cpu_mem_usage=OmegaConf.select(model_config, "low_cpu_mem_usage", default=False),
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
        )
    elif model_name.lower().startswith(MMDET_IMAGE):
        model = MMDetAutoModelForObjectDetection(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            config_file=OmegaConf.select(model_config, "config_file", default=None),
            classes=classes,
            pretrained=pretrained,
            output_bbox_format=OmegaConf.select(model_config, "output_bbox_format", default=XYXY),
            frozen_layers=OmegaConf.select(model_config, "frozen_layers", default=None),
        )
    elif model_name.lower().startswith(MMOCR_TEXT_DET):
        model = MMOCRAutoModelForTextDetection(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
        )
    elif model_name.lower().startswith(MMOCR_TEXT_RECOG):
        model = MMOCRAutoModelForTextRecognition(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
        )
    elif model_name.lower().startswith(NER_TEXT):
        model = HFAutoModelForNER(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            gradient_checkpointing=OmegaConf.select(model_config, "gradient_checkpointing"),
            low_cpu_mem_usage=OmegaConf.select(model_config, "low_cpu_mem_usage", default=False),
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
        )
    elif model_name.lower().startswith(FUSION_MLP):
        model = functools.partial(
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
    elif model_name.lower().startswith(FUSION_NER):
        model = functools.partial(
            MultimodalFusionNER,
            prefix=model_name,
            hidden_features=model_config.hidden_sizes,
            num_classes=num_classes,
            adapt_in_features=model_config.adapt_in_features,
            activation=model_config.activation,
            dropout_prob=model_config.drop_rate,
            normalization=model_config.normalization,
            loss_weight=model_config.weight if hasattr(model_config, "weight") else None,
        )
    elif model_name.lower().startswith(FUSION_TRANSFORMER):
        model = functools.partial(
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
            additive_attention=OmegaConf.select(model_config, "additive_attention", default=False),
            share_qv_weights=OmegaConf.select(model_config, "share_qv_weights", default=False),
        )
    elif model_name.lower().startswith(FT_TRANSFORMER):
        model = FT_Transformer(
            prefix=model_name,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            embedding_arch=model_config.embedding_arch,
            token_dim=model_config.token_dim,
            hidden_size=model_config.hidden_size,
            hidden_features=model_config.hidden_size,
            num_classes=num_classes,
            num_blocks=model_config.num_blocks,
            attention_n_heads=model_config.attention_n_heads,
            attention_dropout=model_config.attention_dropout,
            attention_normalization=model_config.normalization,
            ffn_hidden_size=model_config.ffn_hidden_size,
            ffn_dropout=model_config.ffn_dropout,
            ffn_normalization=model_config.normalization,
            ffn_activation=model_config.ffn_activation,
            residual_dropout=model_config.residual_dropout,
            head_normalization=model_config.normalization,
            head_activation=model_config.head_activation,
            additive_attention=OmegaConf.select(model_config, "additive_attention", default=False),
            share_qv_weights=OmegaConf.select(model_config, "share_qv_weights", default=False),
            pooling_mode=OmegaConf.select(model_config, "pooling_mode", default="cls"),
            checkpoint_name=model_config.checkpoint_name,
            pretrained=pretrained,
        )
    elif model_name.lower().startswith(SAM):
        model = SAMForSemanticSegmentation(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pretrained=pretrained,
            frozen_layers=OmegaConf.select(model_config, "frozen_layers", default=None),
            num_mask_tokens=OmegaConf.select(model_config, "num_mask_tokens", default=1),
        )
    else:
        raise ValueError(f"unknown model name: {model_name}")

    return model


def create_fusion_model(
    config: DictConfig,
    num_classes: Optional[int] = None,
    classes: Optional[list] = None,
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
    classes
        All classes in this dataset.
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
    names = sorted(names)
    config.model.names = names
    single_models = []
    fusion_model = None

    for model_name in names:
        model_config = getattr(config.model, model_name)
        model = create_model(
            model_name=model_name,
            model_config=model_config,
            num_classes=num_classes,
            classes=classes,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            pretrained=pretrained,
        )

        if isinstance(model, functools.partial):  # fusion model
            if fusion_model is None:
                fusion_model = model
            else:
                raise ValueError(
                    f"More than one fusion models are detected in {names}. Only one fusion model is allowed."
                )
        else:  # single model
            if (
                OmegaConf.select(config, "optimization.efficient_finetune") is not None
                and OmegaConf.select(config, "optimization.efficient_finetune") != "None"
            ):
                model = apply_model_adaptation(model, config)
            single_models.append(model)

    if len(single_models) > 1:
        # must have one fusion model if there are multiple independent models
        return fusion_model(models=single_models)
    elif len(single_models) == 1:
        return single_models[0]
    else:
        raise ValueError(f"No available models for {names}")


def apply_model_adaptation(model: nn.Module, config: DictConfig) -> nn.Module:
    """
    Apply an adaptation to the model for efficient fine-tuning.

    Parameters
    ----------
    model
        A PyTorch model.
    config:
        A DictConfig object. The optimization config should be accessible by "config.optimization".
    """
    if OmegaConf.select(config, "optimization.efficient_finetune") in PEFT_ADDITIVE_STRATEGIES:
        model = inject_adaptation_to_linear_layer(
            model=model,
            efficient_finetune=OmegaConf.select(config, "optimization.efficient_finetune"),
            lora_r=config.optimization.lora.r,
            lora_alpha=config.optimization.lora.alpha,
            module_filter=config.optimization.lora.module_filter,
            filter=config.optimization.lora.filter,
            extra_trainable_params=OmegaConf.select(config, "optimization.extra_trainable_params"),
            conv_lora_expert_num=config.optimization.lora.conv_lora_expert_num,
        )
        model.name_to_id = model.get_layer_ids()  # Need to update name to id dictionary.

    return model


def modify_duplicate_model_names(
    learner,
    postfix: str,
    blacklist: List[str],
):
    """
    Modify a learner's model names if they exist in a blacklist.

    Parameters
    ----------
    learner
        A BaseLearner object.
    postfix
        The postfix used to change the duplicate names.
    blacklist
        A list of names. The provided learner can't use model names in the list.

    Returns
    -------
    The learner guaranteed has no duplicate model names with the blacklist names.
    """
    model_names = []
    for n in learner._config.model.names:
        if n in blacklist:
            new_name = f"{n}_{postfix}"
            assert new_name not in blacklist
            assert new_name not in learner._config.model.names
            # modify model prefix
            if n == learner._model.prefix:
                learner._model.prefix = new_name
            else:
                assert isinstance(learner._model.model, nn.ModuleList)
                for per_model in learner._model.model:
                    if n == per_model.prefix:
                        per_model.prefix = new_name
                        break
            # modify data processor prefix
            for per_modality_processors in learner._data_processors.values():
                for per_processor in per_modality_processors:
                    if n == per_processor.prefix:
                        per_processor.prefix = new_name
            # modify model config keys
            setattr(learner._config.model, new_name, getattr(learner._config.model, n))
            delattr(learner._config.model, n)

            model_names.append(new_name)
        else:
            model_names.append(n)

    learner._config.model.names = model_names

    return learner


def list_timm_models(pretrained=True):
    return timm.list_models(pretrained=pretrained)


def is_lazy_weight_tensor(p: Tensor) -> bool:
    from torch.nn.parameter import UninitializedParameter

    if isinstance(p, UninitializedParameter):
        warnings.warn(
            "A layer with UninitializedParameter was found. "
            "Thus, the total number of parameters detected may be inaccurate."
        )
        return True
    return False
