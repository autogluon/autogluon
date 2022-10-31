import functools
import logging
from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf
from torch import nn

from ..constants import (
    ALL_MODALITIES,
    AUTOMM,
    CATEGORICAL,
    CATEGORICAL_MLP,
    CATEGORICAL_TRANSFORMER,
    CLIP,
    FUSION_MLP,
    FUSION_TRANSFORMER,
    HF_TEXT,
    IMAGE,
    MMDET_IMAGE,
    MMOCR_TEXT_DET,
    MMOCR_TEXT_RECOG,
    NER,
    NUMERICAL,
    NUMERICAL_MLP,
    NUMERICAL_TRANSFORMER,
    T_FEW,
    TEXT,
    TIMM_IMAGE,
)
from ..data import MultiModalFeaturePreprocessor
from ..models import (
    CategoricalMLP,
    CategoricalTransformer,
    CLIPForImageText,
    HFAutoModelForNER,
    HFAutoModelForTextPrediction,
    MMDetAutoModelForObjectDetection,
    MMOCRAutoModelForTextDetection,
    MMOCRAutoModelForTextRecognition,
    MultimodalFusionMLP,
    MultimodalFusionTransformer,
    NumericalMLP,
    NumericalTransformer,
    TFewModel,
    TimmAutoModelForImagePrediction,
)
from ..models.utils import inject_ia3_to_linear_layer, inject_lora_to_linear_layer

logger = logging.getLogger(AUTOMM)


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
        if not model_config.data_types:
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
    if len(fusion_model_name) > 1:
        raise ValueError(f"More than one fusion models `{fusion_model_name}` are detected, but only one is allowed.")

    if len(selected_model_names) > 1:
        assert len(fusion_model_name) == 1
        selected_model_names.extend(fusion_model_name)
    elif len(fusion_model_name) == 1 and hasattr(config.model, fusion_model_name[0]):
        # TODO: Support using categorical_transformer or numerical_transformer alone without a fusion model.
        if selected_model_names[0].lower().startswith((CATEGORICAL_TRANSFORMER, NUMERICAL_TRANSFORMER)):
            selected_model_names.extend(fusion_model_name)
        else:
            # remove the fusion model's config make `config.model.names` and the keys of `config.model` consistent.
            delattr(config.model, fusion_model_name[0])

    config.model.names = selected_model_names
    logger.debug(f"selected models: {selected_model_names}")

    return config


def create_model(
    model_name: str,
    model_config: DictConfig,
    num_classes: Optional[int] = 0,
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
            pretrained=pretrained,
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
            pretrained=pretrained,
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
            cls_token=False,
            embedding_arch=model_config.embedding_arch,
            num_classes=num_classes,
            ffn_d_hidden=OmegaConf.select(model_config, "ffn_d_hidden", default=192),
            additive_attention=OmegaConf.select(model_config, "additive_attention", default=False),
            share_qv_weights=OmegaConf.select(model_config, "share_qv_weights", default=False),
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
            ffn_d_hidden=OmegaConf.select(model_config, "ffn_d_hidden", default=192),
            num_classes=num_classes,
            cls_token=False,
            additive_attention=OmegaConf.select(model_config, "additive_attention", default=False),
            share_qv_weights=OmegaConf.select(model_config, "share_qv_weights", default=False),
        )
    elif model_name.lower().startswith(MMDET_IMAGE):
        model = MMDetAutoModelForObjectDetection(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
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
    elif model_name.lower().startswith(NER):
        model = HFAutoModelForNER(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            gradient_checkpointing=OmegaConf.select(model_config, "gradient_checkpointing"),
            pretrained=pretrained,
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
    else:
        raise ValueError(f"unknown model name: {model_name}")

    return model


def create_fusion_model(
    config: DictConfig,
    num_classes: Optional[int] = None,
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
            if OmegaConf.select(config, "optimization.efficient_finetune"):
                model = apply_model_adaptation(model, config)
            single_models.append(model)

    if len(single_models) > 1:
        # must have one fusion model if there are multiple independent models
        return fusion_model(models=single_models)
    elif len(single_models) == 1:
        if isinstance(single_models[0], NumericalTransformer) or isinstance(single_models[0], CategoricalTransformer):
            # TODO: Support using categorical_transformer or numerical_transformer alone without a fusion model.
            return fusion_model(models=single_models)
        else:
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
    if "lora" in OmegaConf.select(config, "optimization.efficient_finetune"):
        model = inject_lora_to_linear_layer(
            model=model,
            lora_r=config.optimization.lora.r,
            lora_alpha=config.optimization.lora.alpha,
            module_filter=config.optimization.lora.module_filter,
            filter=config.optimization.lora.filter,
        )
    elif "ia3" in OmegaConf.select(config, "optimization.efficient_finetune"):
        model = inject_ia3_to_linear_layer(
            model=model,
            module_filter=config.optimization.lora.module_filter,
            filter=config.optimization.lora.filter,
        )

    model.name_to_id = model.get_layer_ids()  # Need to update name to id dictionary.

    return model


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
    The predictor guaranteed has no duplicate model names with the blacklist names.
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
