import torch
from typing import Optional, Union, Tuple, List, Dict
from torch import nn
from torch import optim
import json
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_pt_utils import get_parameter_names
import torchmetrics
from ..constants import BINARY, MULTICLASS, REGRESSION, MAX, MIN


def get_loss_func(problem_type: str):
    if problem_type in [BINARY, MULTICLASS]:
        loss_func = nn.CrossEntropyLoss()
    elif problem_type == REGRESSION:
        loss_func = nn.MSELoss()
    else:
        raise NotImplementedError

    return loss_func


def get_metric(
        metric_name: str,
        num_classes: Optional[int] = None,
):
    metric_name = metric_name.lower()
    if metric_name in ["acc", "accuracy"]:
        return torchmetrics.Accuracy(), MAX
    elif metric_name == "rmse":
        return torchmetrics.MeanSquaredError(squared=False), MIN
    elif metric_name == "r2":
        return torchmetrics.R2Score(), MAX
    elif metric_name == "quadratic_kappa":
        return torchmetrics.CohenKappa(num_classes=num_classes,
                                       weights="quadratic"), MAX
    elif metric_name == "roc_auc":
        return torchmetrics.AUROC(), MAX
    else:
        raise ValueError(f"unknown metric_name: {metric_name}")


def get_optimizer(
        optim_type: str,
        optimizer_grouped_parameters,
        lr: float,
        weight_decay: float,
        eps: Optional[float] = 1e-6,
        betas: Optional[Tuple[float, float]] = (0.9, 0.999),
        momentum: Optional[float] = 0.9,
):
    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            betas=betas,
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    else:
        raise ValueError(f"unknown optimizer: {optim_type}")

    return optimizer


def get_lr_scheduler(
        optimizer: optim.Optimizer,
        num_max_steps: int,
        num_warmup_steps: int,
        lr_schedule: str,
        end_lr: Union[float, int],
):

    if lr_schedule == "cosine_decay":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_max_steps,
        )
    elif lr_schedule == "polynomial_decay":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_max_steps,
            lr_end=end_lr,
            power=1,
        )
    elif lr_schedule == "linear_decay":
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_max_steps
        )
    else:
        raise ValueError(f"unknown lr schedule: {lr_schedule}")

    return scheduler


def get_weight_decay_param_names(model: nn.Module):
    decay_param_names = get_parameter_names(model, [nn.LayerNorm])
    decay_param_names = [name for name in decay_param_names if "bias" not in name]
    return decay_param_names


def apply_single_lr(
        model: nn.Module,
        lr: float,
        weight_decay: float,
        return_params: Optional[bool] = True,
):
    decay_param_names = get_weight_decay_param_names(model)
    optimizer_grouped_parameters = [
        {
            "params": [p if return_params else n for n, p in model.named_parameters() if n in decay_param_names],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [p if return_params else n for n, p in model.named_parameters() if n not in decay_param_names],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    return optimizer_grouped_parameters


def apply_two_stages_lr(
        model: nn.Module,
        lr: float,
        lr_mult: Union[float, int],
        weight_decay: float,
        return_params: Optional[bool] = True,
):

    decay_param_names = get_weight_decay_param_names(model)

    optimizer_grouped_parameters = [
        {
            "params": [
                p if return_params else n
                for n, p in model.named_parameters()
                if n in decay_param_names
                   and not any(bb in n for bb in model.head_layer_names)
            ],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [
                p if return_params else n
                for n, p in model.named_parameters()
                if n not in decay_param_names
                   and not any(bb in n for bb in model.head_layer_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p if return_params else n
                for n, p in model.named_parameters()
                if n in decay_param_names
                   and any(bb in n for bb in model.head_layer_names)
            ],
            "weight_decay": weight_decay,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p if return_params else n
                for n, p in model.named_parameters()
                if n not in decay_param_names
                   and any(bb in n for bb in model.head_layer_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    return optimizer_grouped_parameters


def apply_layerwise_lr_decay(
        model: nn.Module,
        lr: float,
        lr_decay: float,
        weight_decay: float,
):
    parameter_group_names = {}
    parameter_group_vars = {}
    decay_param_names = get_weight_decay_param_names(model)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if name in decay_param_names:
            group_name = "decay"
            this_weight_decay = weight_decay
        else:
            group_name = "no_decay"
            this_weight_decay = 0.

        layer_id = model.name_to_id[name]
        group_name = "layer_%d_%s" % (layer_id, group_name)

        if group_name not in parameter_group_names:
            scale = lr_decay ** layer_id

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": scale * lr
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": scale * lr
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())
