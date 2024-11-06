import functools
import logging
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchmetrics
from omegaconf import DictConfig, OmegaConf
from packaging import version
from pytorch_metric_learning import distances, losses, miners
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import Adafactor
from transformers.models.mask2former.modeling_mask2former import Mask2FormerConfig, Mask2FormerLoss
from transformers.trainer_pt_utils import get_parameter_names

from ..constants import (
    ACC,
    ACCURACY,
    AVERAGE_PRECISION,
    BER,
    BINARY,
    BIT_FIT,
    COLUMN_FEATURES,
    CONTRASTIVE_LOSS,
    CONV_LORA,
    COSINE_EMBEDDING_LOSS,
    COSINE_SIMILARITY,
    CROSS_ENTROPY,
    DETECTION_METRICS,
    DIRECT_LOSS,
    EM,
    F1,
    F1_MACRO,
    F1_MICRO,
    F1_WEIGHTED,
    FEATURES,
    FEW_SHOT_CLASSIFICATION,
    FM,
    HIT_RATE,
    IA3,
    IA3_BIAS,
    IA3_LORA,
    IA3_LORA_BIAS,
    IA3_LORA_NORM,
    IA3_NORM,
    IOU,
    LOG_LOSS,
    LORA,
    LORA_BIAS,
    LORA_NORM,
    MAE,
    MULTI_NEGATIVES_SOFTMAX_LOSS,
    MULTICLASS,
    NER,
    NER_TOKEN_F1,
    NORM_FIT,
    OBJECT_DETECTION,
    OVERALL_ACCURACY,
    PAIR_MARGIN_MINER,
    PEARSONR,
    PEFT_STRATEGIES,
    QUADRATIC_KAPPA,
    R2,
    RECALL,
    REGRESSION,
    RMSE,
    ROC_AUC,
    ROOT_MEAN_SQUARED_ERROR,
    SEMANTIC_SEGMENTATION,
    SM,
    SPEARMANR,
)
from .losses import BBCEWithLogitLoss, FocalLoss, MultiNegativesSoftmaxLoss, SoftTargetCrossEntropy, StructureLoss
from .lr_scheduler import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from .semantic_seg_metrics import COD_METRICS_NAMES, Balanced_Error_Rate, Binary_IoU, Multiclass_IoU

logger = logging.getLogger(__name__)


def get_loss_func(
    problem_type: str,
    mixup_active: Optional[bool] = None,
    loss_func_name: Optional[str] = None,
    config: Optional[DictConfig] = None,
    **kwargs,
):
    """
    Choose a suitable Pytorch loss module based on the provided problem type.

    Parameters
    ----------
    problem_type
        Type of problem.
    mixup_active
        The activation determining whether to use mixup.
    loss_func_name
        The name of the function the user wants to use.
    config
        The optimization configs containing values such as i.e. optimization.loss_function
        An example purpose of this config here is to pass through the parameters for focal loss, i.e.:
            alpha = optimization.focal_loss.alpha
    Returns
    -------
    A Pytorch loss module.
    """
    if problem_type in [BINARY, MULTICLASS]:
        if mixup_active:
            loss_func = SoftTargetCrossEntropy()
        else:
            if loss_func_name is not None and loss_func_name.lower() == "focal_loss":
                loss_func = FocalLoss(
                    alpha=OmegaConf.select(config, "focal_loss.alpha"),
                    gamma=OmegaConf.select(config, "focal_loss.gamma", default=2.0),
                    reduction=OmegaConf.select(config, "focal_loss.reduction", default="mean"),
                )
            else:
                loss_func = nn.CrossEntropyLoss()
    elif problem_type == REGRESSION:
        if loss_func_name is not None:
            if "bcewithlogitsloss" in loss_func_name.lower():
                loss_func = nn.BCEWithLogitsLoss()
            else:
                loss_func = nn.MSELoss()
        else:
            loss_func = nn.MSELoss()
    elif problem_type == NER:
        loss_func = nn.CrossEntropyLoss(ignore_index=0)
    elif problem_type in [OBJECT_DETECTION, FEW_SHOT_CLASSIFICATION]:
        return None
    elif problem_type == SEMANTIC_SEGMENTATION:
        if "structure_loss" in loss_func_name.lower():
            loss_func = StructureLoss()
        elif "balanced_bce" in loss_func_name.lower():
            loss_func = BBCEWithLogitLoss()
        elif "mask2former_loss" in loss_func_name.lower():
            weight_dict = {
                "loss_cross_entropy": config.mask2former_loss.loss_cross_entropy_weight,
                "loss_mask": config.mask2former_loss.loss_mask_weight,
                "loss_dice": config.mask2former_loss.loss_dice_weight,
            }
            loss_func = Mask2FormerLoss(
                config=Mask2FormerConfig(num_labels=kwargs["num_classes"]), weight_dict=weight_dict
            )
        else:
            loss_func = nn.BCEWithLogitsLoss()
    elif problem_type is None:
        return None
    else:
        raise NotImplementedError

    return loss_func


class CustomHitRate(torchmetrics.Metric):
    """
    Compute the hit rate when doing semantic search between two group of embeddings.
    We assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.add_state("query_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("response_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("logit_scale", default=[], dist_reduce_fx=None)

    def update(
        self,
        batch_query_embeds: torch.Tensor,
        batch_response_embeds: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
    ):
        self.query_embeddings.append(batch_query_embeds)
        self.response_embeddings.append(batch_response_embeds)
        if logit_scale is not None:
            self.logit_scale.append(logit_scale)

    def compute(self):
        query_embeddings = torch.cat(self.query_embeddings)
        response_embeddings = torch.cat(self.response_embeddings)
        if self.logit_scale:
            logit_scale = torch.mean(torch.stack(self.logit_scale))
        else:
            logit_scale = 1

        return compute_hit_rate(query_embeddings, response_embeddings, logit_scale)


def compute_hit_rate(features_a, features_b, logit_scale, top_ks=[1, 5, 10]):
    """
    Compute symmetric hit rates between two groups of features.

    Parameters
    ----------
    features_a
        One group of features.
    features_b
        The other group of features.
    logit_scale
        The scale of logit (Used in CLIP).
    top_ks
        Consider only the top k elements for each query.

    Returns
    -------
    The accumulated hit rate.
    """
    assert len(features_a) == len(features_b)
    hit_rate = 0
    logits_per_a = (logit_scale * features_a @ features_b.t()).detach().cpu()
    logits_per_b = logits_per_a.t().detach().cpu()

    logits = {"logits_per_a": logits_per_a, "logits_per_b": logits_per_b}
    ground_truth = torch.arange(len(features_b)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]

        for k in top_ks:
            hit_rate += (preds < k).float().mean()

    hit_rate /= len(top_ks) * len(logits)
    return hit_rate


def get_metric(
    metric_name: str,
    num_classes: Optional[int] = None,
    is_matching: Optional[bool] = False,
    problem_type: Optional[str] = None,
):
    """
    Obtain a torchmerics.Metric from its name.
    Define a customized metric function in case that torchmetrics doesn't support some metric.

    Parameters
    ----------
    metric_name
        Name of metric.
    num_classes
        Number of classes.
    is_matching
        Whether is matching.
    problem_type
        Type of problem, e.g., binary and multiclass.

    Returns
    -------
    torchmetrics.Metric
        A torchmetrics.Metric object.
    custom_metric_func
        A customized metric function.
    """
    metric_name = metric_name.lower()
    if metric_name in [ACC, ACCURACY, OVERALL_ACCURACY]:
        # use MULTICLASS since the head output dim is 2 for the binary problem type.
        return torchmetrics.Accuracy(task=MULTICLASS, num_classes=num_classes), None
    elif metric_name == NER_TOKEN_F1:
        return torchmetrics.F1Score(task=MULTICLASS, num_classes=num_classes, ignore_index=1), None
    elif metric_name in [RMSE, ROOT_MEAN_SQUARED_ERROR]:
        return torchmetrics.MeanSquaredError(squared=False), None
    elif metric_name == R2:
        return torchmetrics.R2Score(), None
    elif metric_name == QUADRATIC_KAPPA:
        return (
            torchmetrics.CohenKappa(task=problem_type, num_classes=num_classes, weights="quadratic"),
            None,
        )
    elif metric_name == ROC_AUC:
        return torchmetrics.AUROC(task=problem_type, num_classes=num_classes), None
    elif metric_name == AVERAGE_PRECISION:
        return torchmetrics.AveragePrecision(task=problem_type, num_classes=num_classes)
    elif metric_name in [LOG_LOSS, CROSS_ENTROPY]:
        return torchmetrics.MeanMetric(), functools.partial(F.cross_entropy, reduction="none")
    elif metric_name == COSINE_EMBEDDING_LOSS:
        return torchmetrics.MeanMetric(), functools.partial(F.cosine_embedding_loss, reduction="none")
    elif metric_name == PEARSONR:
        return torchmetrics.PearsonCorrCoef(), None
    elif metric_name == SPEARMANR:
        if is_matching:  # TODO: add support for matching.
            raise ValueError("spearman relation is not supported for matching yet.")
        else:
            return torchmetrics.SpearmanCorrCoef(), None
    elif metric_name == F1:
        return torchmetrics.F1Score(task=problem_type, num_classes=num_classes), None
    elif metric_name in [F1_MACRO, F1_MICRO, F1_WEIGHTED]:
        average = metric_name.split("_")[1]
        return torchmetrics.F1Score(task=problem_type, num_classes=num_classes, average=average), None
    elif metric_name in DETECTION_METRICS:
        return (
            MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False),
            None,
        )  # TODO: remove parameter hardcodings here, and add class_metrics
    elif metric_name == DIRECT_LOSS:
        return (
            torchmetrics.MeanMetric(nan_strategy="warn"),
            None,
        )  # This only works for detection where custom_metric is not required for BaseAggregator
    elif metric_name in [RECALL, HIT_RATE]:
        if is_matching:
            return CustomHitRate(), None
        else:  # TODO: support recall for general classification tasks.
            raise ValueError("Recall is not supported yet.")
    elif metric_name == BER:
        return Balanced_Error_Rate(), None
    elif metric_name in [SM, EM, FM, MAE]:
        return COD_METRICS_NAMES[metric_name], None
    elif metric_name == IOU:
        if num_classes == 1:
            return Binary_IoU(), None
        else:
            return Multiclass_IoU(num_classes=num_classes), None
    else:
        raise ValueError(f"Unknown metric {metric_name}")


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


def apply_single_lr(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    return_params: Optional[bool] = True,
    efficient_finetune: Optional[str] = None,
    trainable_param_names: Optional[List] = None,
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
    decay_grad_param_names = []
    no_decay_grad_param_names = []

    for name, param in model.named_parameters():
        if (
            efficient_finetune is not None
            and efficient_finetune != "None"
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
                if n in decay_param_names and not any(bb in n for bb in model.head_layer_names)
            ],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [
                p if return_params else n
                for n, p in model.named_parameters()
                if n not in decay_param_names and not any(bb in n for bb in model.head_layer_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p if return_params else n
                for n, p in model.named_parameters()
                if n in decay_param_names and any(bb in n for bb in model.head_layer_names)
            ],
            "weight_decay": weight_decay,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p if return_params else n
                for n, p in model.named_parameters()
                if n not in decay_param_names and any(bb in n for bb in model.head_layer_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    return optimizer_grouped_parameters


def get_trainable_params_efficient_finetune(
    norm_param_names: List[str], efficient_finetune: Optional[str] = None, extra_params: Optional[List] = None
):
    """
     Get the list of trainable parameters according to the provided efficient finetuning method.

    Parameters
    ----------
    norm_param_names
        The parameters associated with the normalization layers
    efficient_finetune
        Efficient finetuning strategy. Trainable parameters will be adjusted according to the method.

    Returns
    -------
    Get list of trainable parameter names according to the provided efficient finetuning method.
    """
    trainable_param_names = []

    if efficient_finetune == BIT_FIT:
        trainable_param_names.append(".*bias*.")
    elif efficient_finetune == NORM_FIT:
        trainable_param_names.append(".*bias*.")
        trainable_param_names += norm_param_names
    elif efficient_finetune in [LORA, IA3, IA3_LORA, CONV_LORA]:
        trainable_param_names.append(".*lora_*.")
    elif efficient_finetune in [LORA_BIAS, IA3_BIAS, IA3_LORA_BIAS]:
        trainable_param_names.append(".*lora_*.")
        trainable_param_names.append(".*bias*.")
    elif efficient_finetune in [LORA_NORM, IA3_NORM, IA3_LORA_NORM]:
        trainable_param_names.append(".*lora_*.")
        trainable_param_names.append(".*bias*.")
        trainable_param_names += norm_param_names
    elif efficient_finetune is not None and efficient_finetune != "None":
        raise NotImplementedError(
            f"The efficient finetuning strategy '{efficient_finetune}'"
            f" is not supported. We only support"
            f" {', '.join(PEFT_STRATEGIES)}."
        )

    if extra_params:
        trainable_param_names.extend(extra_params)

    return trainable_param_names


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


def apply_layerwise_lr_decay(
    model: nn.Module,
    lr: float,
    lr_decay: float,
    weight_decay: float,
    efficient_finetune: Optional[str] = None,
    trainable_param_names: Optional[List] = None,
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
        Efficient finetuning strategy. It will only finetune part of the parameters

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
        layer_id = model.name_to_id[name]
        if layer_id == 0:  # Set top layer (e.g. head, fusion_mlp, adapter) as being trainable.
            param.requires_grad = True
        elif (
            efficient_finetune is not None
            and efficient_finetune != "None"
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
                assert (
                    len("_".join(columns_share_one_feature)) == len(feature_name)
                ), f"model `{per_model_name}`'s cls feature name `{feature_name}` doesn't match `{columns_share_one_feature}`"
                gathered_features.append(per_model_output[COLUMN_FEATURES][FEATURES][feature_name])

    if len(gathered_features) > 1:
        # currently only support features of the same shape
        assert all(
            per_features.shape == gathered_features[0].shape for per_features in gathered_features
        ), "Currently we only support gathering features of the same dimension."

    if len(gathered_features) == 0:
        raise ValueError(f"No features are found for columns names {column_names}.")

    gathered_features = torch.stack(gathered_features, dim=0).mean(dim=0)  # (b, d)

    return gathered_features


def get_metric_learning_distance_func(
    name: str,
):
    """
    Return one pytorch metric learning's distance function based on its name.

    Parameters
    ----------
    name
        distance function name

    Returns
    -------
    A distance function from the pytorch metric learning package.
    """
    if name.lower() == COSINE_SIMILARITY:
        return distances.CosineSimilarity()
    else:
        raise ValueError(f"Unknown distance measure: {name}")


def infer_matcher_loss(data_format: str, problem_type: str):
    """
    Infer the loss type to train the matcher.

    Parameters
    ----------
    data_format
        The training data format, e.g., pair or triplet.
    problem_type
        Type of problem.

    Returns
    -------
    The loss name.
    """
    if data_format == "pair":
        if problem_type is None:
            return [MULTI_NEGATIVES_SOFTMAX_LOSS]
        elif problem_type == BINARY:
            return [CONTRASTIVE_LOSS]
        elif problem_type == REGRESSION:
            return ["cosine_similarity_loss"]
        else:
            raise ValueError(f"Unsupported data format {data_format} with problem type {problem_type}")
    elif data_format == "triplet":
        if problem_type is None:
            return [MULTI_NEGATIVES_SOFTMAX_LOSS]
        else:
            raise ValueError(f"Unsupported data format {data_format} with problem type {problem_type}")
    else:
        raise ValueError(f"Unsupported data format: {data_format}")


def get_matcher_loss_func(
    data_format: str,
    problem_type: str,
    loss_type: Optional[str] = None,
    pos_margin: Optional[float] = None,
    neg_margin: Optional[float] = None,
    distance_type: Optional[str] = None,
):
    """
    Return a list of pytorch metric learning's loss functions based on their names.

    Parameters
    ----------
    data_format
        The training data format, e.g., pair or triplet.
    problem_type
        Type of problem.
    loss_type
        The provided loss type.
    pos_margin
        The positive margin in computing the metric learning loss.
    neg_margin
        The negative margin in computing the metric learning loss.
    distance_type
        The distance function type.

    Returns
    -------
    A loss function of metric learning.
    """

    allowable_loss_types = infer_matcher_loss(data_format=data_format, problem_type=problem_type)
    if loss_type is not None:
        assert loss_type in allowable_loss_types, f"data format {data_format} can't use loss {loss_type}."
    else:
        loss_type = allowable_loss_types[0]

    if loss_type.lower() == CONTRASTIVE_LOSS:
        return losses.ContrastiveLoss(
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            distance=get_metric_learning_distance_func(distance_type),
        )
    elif loss_type.lower() == MULTI_NEGATIVES_SOFTMAX_LOSS:
        return MultiNegativesSoftmaxLoss(
            local_loss=True,
            gather_with_grad=True,
            cache_labels=False,
        )
    else:
        raise ValueError(f"Unknown metric learning loss: {loss_type}")


def get_matcher_miner_func(
    miner_type: str,
    pos_margin: float,
    neg_margin: float,
    distance_type: str,
):
    """
    Return a pytorch metric learning's miner functions based on their names.
    The miners are used to mine the positive and negative examples.

    Parameters
    ----------
    miner_type
        The miner function type.
    pos_margin
        The positive margin used by the miner function.
    neg_margin
        The negative margin used by the miner function.
    distance_type
        The distance function type.

    Returns
    -------
    A miner function to mine positive and negative samples.
    """
    if miner_type.lower() == PAIR_MARGIN_MINER:
        return miners.PairMarginMiner(
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            distance=get_metric_learning_distance_func(distance_type),
        )
    else:
        raise ValueError(f"Unknown metric learning miner: {miner_type}")


def generate_metric_learning_labels(
    num_samples: int,
    match_label: int,
    labels: torch.Tensor,
):
    """
    Generate labels to compute the metric learning loss of one mini-batch.
    For n samples, it generates 2*n labels since each match has two sides, each of which
    has one label. If we know the matching label, then it determines the two sides' labels
    according to whether their label is the matching label. If the matching label is None,
    it assigns a unique label for each side.

    Parameters
    ----------
    num_samples
        number of samples.
    match_label
        The matching label, which can be None.
    labels
        The sample labels used in the supervised setting. It's required only when match_label is not None.

    Returns
    -------
    The labels used in computing the metric learning loss.
    """
    device = labels.device
    labels_1 = torch.arange(num_samples, device=device)

    if match_label is not None:
        labels_2 = torch.arange(num_samples, num_samples * 2, device=device)
        # users need to specify the match_label based on the raw label's semantic meaning.
        mask = labels == match_label
        labels_2[mask] = labels_1[mask]
    else:
        labels_2 = torch.arange(num_samples, device=device)

    metric_learning_labels = torch.cat([labels_1, labels_2], dim=0)

    return metric_learning_labels
