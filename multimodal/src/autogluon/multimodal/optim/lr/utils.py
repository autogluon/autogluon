import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn, optim

from ..utils import get_weight_decay_param_names
from .lr_schedulers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    num_max_steps: int,
    num_warmup_steps: int,
    lr_schedule: str,
    end_lr: Union[float, int],
):
    """
    Get the learning rate scheduler from its name. Here we use our defined learning rate
    scheduler instead of those imported from "transformers" because we want to support
    Pytorch lightning's "ddp_spawn" training strategy.

    Parameters
    ----------
    optimizer
        A Pytorch optimizer.
    num_max_steps
        Number of maximum training steps.
    num_warmup_steps
        Number of steps to do learning rate warmup.
    lr_schedule
        Name of the learning rate scheduler.
    end_lr
        The final learning rate after decay.

    Returns
    -------
    A learning rate scheduler.
    """
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
            num_training_steps=num_max_steps,
        )
    elif lr_schedule == "multi_step":
        # TODO: add milestones, gamma into hyperparameters
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 55], gamma=0.1)
    else:
        raise ValueError(f"unknown lr schedule: {lr_schedule}")

    return scheduler


def apply_single_lr(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    return_params: Optional[bool] = True,
    peft: Optional[str] = None,
    trainable_param_names: Optional[List] = None,
    exclude_keys: Optional[List] = None,
):
    """
    Set to use a single learning rate for all parameters. Layer normalization parameters and other
    layers' bias parameters don't use weight decay.

    Parameters
    ----------
    model
        A Pytorch model.
    lr
        Learning rate.
    weight_decay
        Weight decay.
    return_params
        Whether to return parameters or their names. If you want to double-check
        whether the learning rate setup is as expected, you can set "return_params=False",
        and print the layer names along with their learning rates through
        "print("Param groups = %s" % json.dumps(optimizer_grouped_parameters, indent=2))".
    peft
        Efficient finetuning strategy. It will only finetune part of the parameters
    trainable_param_names
        A list of trainable parameters. (Optional)
    exclude_keys
        A list of keys to be excluded.

    Returns
    -------
    The grouped parameters or their names.
    """
    decay_param_names = get_weight_decay_param_names(model)
    decay_grad_param_names = []
    no_decay_grad_param_names = []

    for name, param in model.named_parameters():
        if exclude_keys and any([exc in name for exc in exclude_keys]):
            continue

        if (
            peft is not None
            and trainable_param_names
            and not any([re.match(trainable_param_name, name) for trainable_param_name in trainable_param_names])
        ):
            param.requires_grad = False

        if not param.requires_grad:
            continue  # frozen weights

        if name in decay_param_names:
            if return_params:
                decay_grad_param_names.append(param)
            else:
                decay_grad_param_names.append(name)

        else:
            if return_params:
                no_decay_grad_param_names.append(param)
            else:
                no_decay_grad_param_names.append(name)

    optimizer_grouped_parameters = [
        {
            "params": decay_grad_param_names,
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": no_decay_grad_param_names,
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
    exclude_keys: Optional[List] = None,
):
    """
    Set up the pretrained backbone to use a smaller learning rate (lr * lr_mult).
    The newly added head layers use the normal learning rate (lr).
    Layer normalization parameters and other layers' bias parameters don't use weight decay.

    Parameters
    ----------
    model
        A Pytorch model.
    lr
        The learning rate.
    lr_mult
        The multiplier (0, 1) to scale down the learning rate.
    weight_decay
        Weight decay.
    return_params
        return_params
        Whether to return parameters or their names. If you want to double-check
        whether the learning rate setup is as expected, you can set "return_params=False",
        and print the layer names along with their learning rates through
        "print("Param groups = %s" % json.dumps(optimizer_grouped_parameters, indent=2))".
    exclude_keys
        A list of keys to be excluded.

    Returns
    -------
    The grouped parameters or their names.
    """
    decay_param_names = get_weight_decay_param_names(model)

    optimizer_grouped_parameters = [
        {
            "params": [
                p if return_params else n
                for n, p in model.named_parameters()
                if n in decay_param_names
                and not any(bb in n for bb in model.head_layer_names)
                and (not exclude_keys or not any([exc in n for exc in exclude_keys]))
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
                and (not exclude_keys or not any([exc in n for exc in exclude_keys]))
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
                and (not exclude_keys or not any([exc in n for exc in exclude_keys]))
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
                and (not exclude_keys or not any([exc in n for exc in exclude_keys]))
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
    peft: Optional[str] = None,
    trainable_param_names: Optional[List] = None,
    exclude_keys: Optional[List] = None,
):
    """
    Assign monotonically decreasing learning rates for layers from the output end to the input end.
    The intuition behind is that later layers are more task-related compared to the early layers.
    Layer normalization parameters and other layers' bias parameters don't use weight decay.
    If you want to double-check whether the learning rate setup is as expected,
    you can print the layer names along with their learning rates through
    "print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))".

    Parameters
    ----------
    model
        A Pytorch model.
    lr
        The learning rate.
    lr_decay
        The learning rate decay factor (0, 1).
    weight_decay
        Weight decay.
    peft
        Efficient finetuning strategy. It will only finetune part of the parameters
    trainable_param_names
        A list of trainable parameters. (Optional)
    exclude_keys
        A list of keys to be excluded.

    Returns
    -------
    The grouped parameters based on their layer ids and whether using weight decay.
    """
    parameter_group_names = {}
    parameter_group_vars = {}
    decay_param_names = get_weight_decay_param_names(model)

    for name, param in model.named_parameters():
        if name.startswith("_orig_mod."):
            name = "".join(name.split("_orig_mod."))
        if exclude_keys and any([exc in name for exc in exclude_keys]):
            continue
        layer_id = model.name_to_id[name]
        if layer_id == 0:  # Set top layer (e.g. head, fusion_mlp, adapter) as being trainable.
            param.requires_grad = True
        elif (
            peft is not None
            and trainable_param_names
            and not any([re.match(trainable_param_name, name) for trainable_param_name in trainable_param_names])
        ):
            param.requires_grad = False

        if not param.requires_grad:
            continue  # frozen weights

        if name in decay_param_names:
            group_name = "decay"
            this_weight_decay = weight_decay
        else:
            group_name = "no_decay"
            this_weight_decay = 0.0

        layer_id = model.name_to_id[name]
        group_name = "layer_%d_%s" % (layer_id, group_name)

        if group_name not in parameter_group_names:
            scale = lr_decay**layer_id
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": scale * lr,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": scale * lr,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())
