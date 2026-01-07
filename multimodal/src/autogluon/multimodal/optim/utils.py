import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn, optim
from transformers import Adafactor
from transformers.trainer_pt_utils import get_parameter_names

from ..constants import (
    BIT_FIT,
    COLUMN_FEATURES,
    CONV_LORA,
    FEATURES,
    IA3,
    IA3_BIAS,
    IA3_LORA,
    IA3_LORA_BIAS,
    IA3_LORA_NORM,
    IA3_NORM,
    LORA,
    LORA_BIAS,
    LORA_NORM,
    NORM_FIT,
    PEFT_STRATEGIES,
)

logger = logging.getLogger(__name__)


def get_optimizer(
    optim_type: str,
    optimizer_grouped_parameters,
    lr: float,
    weight_decay: float,
    eps: Optional[float] = 1e-6,
    betas: Optional[Tuple[float, float]] = (0.9, 0.999),
    momentum: Optional[float] = 0.9,
):
    """
    Choose a Pytorch optimizer based on its name.

    Parameters
    ----------
    optim_type
        Name of optimizer.
    optimizer_grouped_parameters
        The model parameters to be optimized.
    lr
        Learning rate.
    weight_decay
        Optimizer weight decay.
    eps
        Optimizer eps.
    betas
        Optimizer betas.
    momentum
        Momentum used in the SGD optimizer.

    Returns
    -------
    A Pytorch optimizer.
    """
    if optim_type == "adamw":
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            betas=betas,
        )
    elif optim_type == "adam":
        optimizer = optim.Adam(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optim_type == "sgd":
        optimizer = optim.SGD(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optim_type == "adafactor":
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=weight_decay,
            scale_parameter=True,  # Generally recommended to enable scaling
            relative_step=False,
            warmup_init=False,
        )
    else:
        raise ValueError(f"unknown optimizer: {optim_type}")

    return optimizer


def get_weight_decay_param_names(model: nn.Module):
    """
    Set the layer normalization parameters and other layers' bias parameters not to use weight decay.

    Parameters
    ----------
    model
        A Pytorch model.

    Returns
    -------
    A list of parameter names not using weight decay.
    """
    # By default, we should not apply weight decay for all the norm layers
    decay_param_names = get_parameter_names(
        model,
        [nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm],
    )
    decay_param_names = [
        name
        for name in decay_param_names
        if (
            "bias" not in name
            and "cls_token" not in name
            and "categorical_feature_tokenizer" not in name
            and "numerical_feature_tokenizer" not in name
        )
    ]
    return decay_param_names


def get_norm_layer_param_names(model: nn.Module):
    """
    Get parameters associated with the normalization layers

    Parameters
    ----------
    model
        A Pytorch model

    Returns
    -------
    norm_param_names
        A list of normalization parameter names
    """
    all_param_names = [name for name, _ in model.named_parameters()]
    all_param_names_except_norm_names = get_parameter_names(
        model,
        [nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm],
    )
    norm_param_names = [name for name in all_param_names if name not in all_param_names_except_norm_names]
    return norm_param_names


def get_peft_param_names(norm_param_names: List[str], peft: Optional[str] = None, extra_params: Optional[List] = None):
    """
     Get the list of trainable parameters according to the provided efficient finetuning method.

    Parameters
    ----------
    norm_param_names
        The parameters associated with the normalization layers
    peft
        Efficient finetuning strategy. Trainable parameters will be adjusted according to the method.
    extra_params
        Extra parameters to train.

    Returns
    -------
    Get list of trainable parameter names according to the provided efficient finetuning method.
    """
    peft_param_names = []

    if peft == BIT_FIT:
        peft_param_names.append(".*bias*.")
    elif peft == NORM_FIT:
        peft_param_names.append(".*bias*.")
        peft_param_names += norm_param_names
    elif peft in [LORA, IA3, IA3_LORA, CONV_LORA]:
        peft_param_names.append(".*lora_*.")
    elif peft in [LORA_BIAS, IA3_BIAS, IA3_LORA_BIAS]:
        peft_param_names.append(".*lora_*.")
        peft_param_names.append(".*bias*.")
    elif peft in [LORA_NORM, IA3_NORM, IA3_LORA_NORM]:
        peft_param_names.append(".*lora_*.")
        peft_param_names.append(".*bias*.")
        peft_param_names += norm_param_names
    elif peft is not None:
        raise NotImplementedError(
            f"The efficient finetuning strategy '{peft}'"
            f" is not supported. We only support"
            f" {', '.join(PEFT_STRATEGIES)}."
        )

    if extra_params:
        peft_param_names.extend(extra_params)

    return peft_param_names


def remove_parameters_without_grad(
    grouped_parameters: List[Dict],
):
    """
    Remove layers

    Parameters
    ----------
    grouped_parameters
        The grouped parameters or their names output from lr_choice.

    Returns
    -------
    The updated grouped parameters or their names.
    """
    for group_idx, group_param in enumerate(grouped_parameters):
        updated_params = []
        for p in group_param["params"]:
            if p.requires_grad:
                updated_params.append(p)
        grouped_parameters[group_idx]["params"] = updated_params

    return grouped_parameters


def gather_column_features(
    output: Dict[str, Dict],
    column_names: Union[str, List[str]],
):
    """
    Gather column features from models' outputs.
    For each feature name in one model's output, we enumerate the provided column names to see
    whether (partial) the provided columns share one cls feature or they have independent features.

    TODO: return features' masks and use them to filter the losses.

    Parameters
    ----------
    output
        The models' outputs.
    column_names
        The columns whose features we want to get.

    Returns
    -------
    The gathered feature vectors. Each sample should only have one feature vector.
    """
    if isinstance(column_names, str):
        column_names = [column_names]

    gathered_features = []
    # logger.debug(f"gather features for columns: {column_names}")
    for per_model_name, per_model_output in output.items():
        # logger.debug(f"gather column features from model: {per_model_name}")
        for feature_name in per_model_output[COLUMN_FEATURES][FEATURES]:
            # logger.debug(f"processing feature: {feature_name}")
            columns_share_one_feature = []
            for col_name in column_names:
                if col_name in feature_name:
                    # this column feature is part of the cls feature
                    if not (feature_name.startswith(col_name) and feature_name.endswith(col_name)):
                        columns_share_one_feature.append(col_name)
                        # logger.debug(f"column {col_name} is included in feature {feature_name}")
                    else:  # this column's feature is independent of other columns'
                        gathered_features.append(per_model_output[COLUMN_FEATURES][FEATURES][col_name])
                        # logger.debug(f"col_name {col_name} has an independent feature in model: {per_model_name}")

            # two or more columns share one cls feature, and no other columns share it.
            if len(columns_share_one_feature) > 0:
                assert len("_".join(columns_share_one_feature)) == len(feature_name), (
                    f"model `{per_model_name}`'s cls feature name `{feature_name}` doesn't match `{columns_share_one_feature}`"
                )
                gathered_features.append(per_model_output[COLUMN_FEATURES][FEATURES][feature_name])

    if len(gathered_features) > 1:
        # currently only support features of the same shape
        assert all(per_features.shape == gathered_features[0].shape for per_features in gathered_features), (
            "Currently we only support gathering features of the same dimension."
        )

    if len(gathered_features) == 0:
        raise ValueError(f"No features are found for columns names {column_names}.")

    gathered_features = torch.stack(gathered_features, dim=0).mean(dim=0)  # (b, d)

    return gathered_features
