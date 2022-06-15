import logging
from typing import Optional, Union, Tuple, List, Dict
import functools
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from transformers.trainer_pt_utils import get_parameter_names
import torchmetrics
from omegaconf import OmegaConf, DictConfig
from pytorch_metric_learning import losses, miners, distances
from .lr_scheduler import (
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from ..constants import (
    BINARY,
    MULTICLASS,
    REGRESSION,
    MAX,
    MIN,
    NORM_FIT,
    BIT_FIT,
    ACC,
    ACCURACY,
    RMSE,
    ROOT_MEAN_SQUARED_ERROR,
    R2,
    QUADRATIC_KAPPA,
    ROC_AUC,
    AVERAGE_PRECISION,
    LOG_LOSS,
    CROSS_ENTROPY,
    PEARSONR,
    SPEARMANR,
    CONTRASTIVE_LOSS,
    COSINE_SIMILARITY,
    PAIR_MARGIN_MINER,
    COLUMN_FEATURES,
    FEATURES,
    MASKS,
    AUTOMM,
    COSINE_EMBEDDING_LOSS,
    LORA,
    LORA_BIAS,
    LORA_NORM,
)
import warnings
from .soft_target_crossentropy import SoftTargetCrossEntropy

logger = logging.getLogger(AUTOMM)


def get_loss_func(
    problem_type: str,
    mixup_active: bool,
    loss_func_name: Optional[str] = None,
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

    Returns
    -------
    A Pytorch loss module.
    """
    if problem_type in [BINARY, MULTICLASS]:
        if mixup_active:
            loss_func = SoftTargetCrossEntropy()
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
    else:
        raise NotImplementedError

    return loss_func


def get_metric(
    metric_name: str,
    problem_type: str,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
):
    """
    Obtain a torchmerics.Metric from its name.
    Define a customized metric function in case that torchmetrics doesn't support some metric.

    Parameters
    ----------
    metric_name
        Name of metric
    problem_type
        The type of the problem.
    num_classes
        Number of classes, used in the quadratic_kappa metric for binary classification.
    pos_label
        The label (0 or 1) of binary classification's positive class, which is used in some metrics, e.g., AUROC.

    Returns
    -------
    torchmetrics.Metric
        A torchmetrics.Metric object.
    custom_metric_func
        A customized metric function.
    """
    metric_name = metric_name.lower()
    if metric_name in [ACC, ACCURACY]:
        return torchmetrics.Accuracy(), None
    elif metric_name in [RMSE, ROOT_MEAN_SQUARED_ERROR]:
        return torchmetrics.MeanSquaredError(squared=False), None
    elif metric_name == R2:
        return torchmetrics.R2Score(), None
    elif metric_name == QUADRATIC_KAPPA:
        return (
            torchmetrics.CohenKappa(num_classes=num_classes, weights="quadratic"),
            None,
        )
    elif metric_name == ROC_AUC:
        return torchmetrics.AUROC(pos_label=pos_label), None
    elif metric_name == AVERAGE_PRECISION:
        return torchmetrics.AveragePrecision(pos_label=pos_label), None
    elif metric_name in [LOG_LOSS, CROSS_ENTROPY]:
        return torchmetrics.MeanMetric(), functools.partial(F.cross_entropy, reduction="none")
    elif metric_name == COSINE_EMBEDDING_LOSS:
        return torchmetrics.MeanMetric(), functools.partial(F.cosine_embedding_loss, reduction="none")
    elif metric_name == PEARSONR:
        return torchmetrics.PearsonCorrCoef(), None
    elif metric_name == SPEARMANR:
        return torchmetrics.SpearmanCorrCoef(), None
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
            if "bias" not in name:
                param.requires_grad = False
        elif efficient_finetune == NORM_FIT:
            # For norm-fit, we finetune all the normalization layers and bias layers
            if name not in norm_param_names and "bias" not in name:
                param.requires_grad = False
        elif efficient_finetune == LORA:
            # For LoRA adaptation we only fine-tune LoRA weights
            if "lora_" not in name:
                param.requires_grad = False
        elif efficient_finetune == LORA_BIAS:
            # For LoRA adapation we fine-tune LoRA and all bias weights
            if "lora_" not in name and "bias" not in name:
                param.requires_grad = False
        elif efficient_finetune == LORA_NORM:
            # For LoRA adapation we fine-tune LoRA and normalization and bias layers
            if "lora_" not in name and name not in norm_param_names and "bias" not in name:
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


def update_config_by_rules(
    problem_type: str,
    config: DictConfig,
):
    """
    Modify configs based on the need of loss func.
    Now it support changing the preprocessing of numerical label into Minmaxscaler while using BCEloss.

    Parameters
    ----------
    problem_type
        The type of the problem of the project.
    config
        The config of the project. It is a Dictconfig object.

    Returns
    -------
    The modified config.
    """
    loss_func = OmegaConf.select(config, "optimization.loss_function")
    if loss_func is not None:
        if problem_type == REGRESSION and "bce" in loss_func.lower():
            # We are using BCELoss for regression problems. Need to first scale the labels.
            config.data.label.numerical_label_preprocessing = "minmaxscaler"
        elif loss_func != "auto":
            warnings.warn(
                f"Received loss function={loss_func} for problem={problem_type}. "
                "Currently, we only support using BCE loss for regression problems and choose "
                "the loss_function automatically otherwise.",
                UserWarning,
            )
    return config


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
                assert len("_".join(columns_share_one_feature)) == len(
                    feature_name
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


def get_metric_learning_loss_funcs(
    matches: List[DictConfig],
):
    """
    Return a list of pytorch metric learning's loss functions based on their names.

    Parameters
    ----------
    matches
        A list of matches from the matcher config.

    Returns
    -------
    A list of loss functions from the pytorch metric learning package.
    """
    metric_learning_loss_funcs = []
    for per_match in matches:
        if per_match.loss.type.lower() == CONTRASTIVE_LOSS:
            metric_learning_loss_funcs.append(
                losses.ContrastiveLoss(
                    pos_margin=per_match.loss.pos_margin,
                    neg_margin=per_match.loss.neg_margin,
                    distance=get_metric_learning_distance_func(per_match.distance.type),
                )
            )
        else:
            raise ValueError(f"Unknown metric learning loss: {per_match.loss.type}")

    return metric_learning_loss_funcs


def get_metric_learning_miner_funcs(
    matches: List[DictConfig],
):
    """
    Return a list of pytorch metric learning's miner functions based on their names.
    The miners are used to mine the positive and negative examples.

    Parameters
    ----------
    matches
        A list of matches from the matcher config.

    Returns
    -------
    A list of miner functions from the pytorch metric learning package.
    """
    metric_learning_miner_funcs = []
    for per_match in matches:
        if per_match.miner.type.lower() == PAIR_MARGIN_MINER:
            metric_learning_miner_funcs.append(
                miners.PairMarginMiner(
                    pos_margin=per_match.miner.pos_margin,
                    neg_margin=per_match.miner.neg_margin,
                    distance=get_metric_learning_distance_func(per_match.distance.type),
                )
            )
        else:
            raise ValueError(f"Unknown metric learning miner: {per_match.miner.type}")

    return metric_learning_miner_funcs


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
    labels_1 = torch.arange(num_samples)

    if match_label is not None:
        labels_2 = torch.arange(num_samples, num_samples * 2)
        # users need to specify the match_label based on the raw label's semantic meaning.
        mask = labels == match_label
        labels_2[mask] = labels_1[mask]
    else:
        labels_2 = torch.arange(num_samples)

    metric_learning_labels = torch.cat([labels_1, labels_2], dim=0)

    return metric_learning_labels


def compute_probability(
    logits: Optional[torch.Tensor] = None,
    embeddings1: Optional[torch.Tensor] = None,
    embeddings2: Optional[torch.Tensor] = None,
    reverse_prob: Optional[bool] = False,
):
    """
    Compute probabilities from logits or embedding pairs.

    Parameters
    ----------
    logits
        The output of a model's head layer.
    embeddings1
        Feature embeddings of one side in matching.
    embeddings2
        Feature embeddings 2 of the other side in matching.
    reverse_prob
        Whether to reverse the probability.

    Returns
    -------
    Probabilities.
    """
    if logits is not None:
        prob = F.softmax(logits.float(), dim=1)[:, 1]
    else:
        cosine_similarity = F.cosine_similarity(embeddings1, embeddings2)
        prob = 0.5 * (cosine_similarity + 1)

    if reverse_prob:
        prob = 1 - prob

    return prob
