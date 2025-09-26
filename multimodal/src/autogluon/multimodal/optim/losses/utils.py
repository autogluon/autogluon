import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_metric_learning import distances, losses, miners
from torch import nn
from transformers.models.mask2former.modeling_mask2former import Mask2FormerConfig, Mask2FormerLoss

from ...constants import (
    BINARY,
    CONTRASTIVE_LOSS,
    COSINE_SIMILARITY,
    FEW_SHOT_CLASSIFICATION,
    MULTI_NEGATIVES_SOFTMAX_LOSS,
    MULTICLASS,
    MULTILABEL,
    NER,
    OBJECT_DETECTION,
    PAIR_MARGIN_MINER,
    REGRESSION,
    SEMANTIC_SEGMENTATION,
)
from .bce_loss import BBCEWithLogitLoss
from .focal_loss import FocalLoss
from .lemda_loss import LemdaLoss
from .softmax_losses import MultiNegativesSoftmaxLoss, SoftTargetCrossEntropy
from .structure_loss import StructureLoss

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
        The optimization configs containing values such as i.e. optim.loss_func
        An example purpose of this config here is to pass through the parameters for focal loss, i.e.:
            alpha = optim.focal_loss.alpha
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
                    alpha=config.focal_loss.alpha,
                    gamma=config.focal_loss.gamma,
                    reduction=config.focal_loss.reduction,
                )
            else:
                loss_func = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
                logger.debug(f"loss_func.label_smoothing: {loss_func.label_smoothing}")
    elif problem_type == MULTILABEL:
        # Use BCEWithLogitsLoss for multilabel classification
        loss_func = nn.BCEWithLogitsLoss()
        logger.debug("Using BCEWithLogitsLoss for multilabel classification")
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
    elif problem_type in [None, OBJECT_DETECTION, FEW_SHOT_CLASSIFICATION]:
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
    else:
        raise NotImplementedError

    return loss_func


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


def get_aug_loss_func(config: Optional[DictConfig] = None, problem_type: Optional[str] = None):
    """
    Return the loss function for lemda augmentation

    Parameters
    ----------
    config
        The optimization configuration.
    problem_type
        Problem type (binary, multimclass, or regression)

    Returns
    -------
    Augmentation loss function.
    """
    loss_func = None
    if config.lemda.turn_on:
        loss_func = LemdaLoss(
            mse_weight=config.lemda.mse_weight,
            kld_weight=config.lemda.kld_weight,
            consist_weight=config.lemda.consist_weight,
            consist_threshold=config.lemda.consist_threshold,
            problem_type=problem_type,
        )

    return loss_func
