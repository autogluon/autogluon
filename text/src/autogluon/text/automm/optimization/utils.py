from typing import Optional, Union, Tuple, List, Dict
import functools
from torch import nn
from torch import optim
from torch.nn import functional as F
from transformers.trainer_pt_utils import get_parameter_names
import torchmetrics
from .lr_scheduler import (
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from ..constants import BINARY, MULTICLASS, REGRESSION, MAX, MIN, NORM_FIT, BIT_FIT


def get_loss_func(problem_type: str):
    """
    Choose a suitable Pytorch loss module based on the provided problem type.

    Parameters
    ----------
    problem_type
        Type of problem.

    Returns
    -------
    A Pytorch loss module.
    """
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
    """
    Obtain a torchmerics.Metric from its name.
    Define a customized metric function in case that torchmetrics doesn't support some metric.

    Parameters
    ----------
    metric_name
        Name of metric
    num_classes
        Number of classes, used in the quadratic_kappa metric for binary classification.

    Returns
    -------
    torchmetrics.Metric
        A torchmetrics.Metric object.
    mode
        The min/max mode used in selecting model checkpoints.
        - min
             Its means that smaller metric is better.
        - max
            It means that larger metric is better.
    custom_metric_func
        A customized metric function.
    """
    metric_name = metric_name.lower()
    if metric_name in ["acc", "accuracy"]:
        return torchmetrics.Accuracy(), MAX, None
    elif metric_name in ["rmse", "root_mean_squared_error"]:
        return torchmetrics.MeanSquaredError(squared=False), MIN, None
    elif metric_name == "r2":
        return torchmetrics.R2Score(), MAX, None
    elif metric_name == "quadratic_kappa":
        return torchmetrics.CohenKappa(num_classes=num_classes,
                                       weights="quadratic"), MAX, None
    elif metric_name == "roc_auc":
        return torchmetrics.AUROC(), MAX, None
    elif metric_name in ["log_loss", "cross_entropy"]:
        return torchmetrics.MeanMetric(), MIN, \
               functools.partial(F.cross_entropy, reduction="none")
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
            num_training_steps=num_max_steps
        )
    else:
        raise ValueError(f"unknown lr schedule: {lr_schedule}")

    return scheduler


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
    decay_param_names = get_parameter_names(model,
                                            [nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                             nn.GroupNorm])
    decay_param_names = [name for name in decay_param_names if "bias" not in name]
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
        model, [nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm])
    norm_param_names = [name for name in all_param_names if name not in all_param_names_except_norm_names]
    return norm_param_names


def apply_single_lr(
        model: nn.Module,
        lr: float,
        weight_decay: float,
        return_params: Optional[bool] = True,
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

    Returns
    -------
    The grouped parameters or their names.
    """
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
        efficient_finetune: Optional[str] = None,
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
    efficient_finetune
        Efficient finetuning strategy. Can be "bit_fit", "norm_fit". It will only finetune part of the parameters

    Returns
    -------
    The grouped parameters based on their layer ids and whether using weight decay.
    """
    parameter_group_names = {}
    parameter_group_vars = {}
    decay_param_names = get_weight_decay_param_names(model)
    norm_param_names = get_norm_layer_param_names(model)
    for name, param in model.named_parameters():
        if efficient_finetune == BIT_FIT:
            # For bit_fit, we disable tuning everything except the bias terms
            if 'bias' not in name:
                param.requires_grad = False
        elif efficient_finetune == NORM_FIT:
            # For norm-fit, we finetune all the normalization layers and bias layers
            if name not in norm_param_names and 'bias' not in name:
                param.requires_grad = False

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
