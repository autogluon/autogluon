import logging
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from scipy.special import softmax
from torch import nn
from torch.nn.modules.loss import _Loss

from ..constants import AUTOMM, BBOX, COLUMN_FEATURES, FEATURES, LOGITS, MASKS, PROBABILITY, TEXT, SCORE
from .environment import compute_num_gpus, infer_precision, move_to_device

logger = logging.getLogger(AUTOMM)


def logits_to_prob(logits: np.ndarray):
    """
    Convert logits to probabilities.

    Parameters
    ----------
    logits
        The logits output of a classification head.

    Returns
    -------
    Probabilities.
    """
    assert logits.ndim == 2
    prob = softmax(logits, axis=1)
    return prob


def tensor_to_ndarray(tensor: torch.Tensor):
    """
    Convert Pytorch tensor to numpy array.

    Parameters
    ----------
    tensor
        A Pytorch tensor.

    Returns
    -------
    A ndarray.
    """
    return tensor.detach().cpu().float().numpy()


def extract_from_output(outputs: List[Dict], ret_type: str, as_ndarray: Optional[bool] = True):
    """
    Extract desired information, e.g., logits or features, from a list of model outputs.
    Support returning a concatenated tensor/ndarray or a dictionary of tensors/ndarrays.

    Parameters
    ----------
    ret_type
        What kind of information to extract from model outputs.
    outputs
        A list of model outputs.
    as_ndarray
        Whether to convert Pytorch tensor to numpy array. (Default True)

    Returns
    -------
    The desired information from model outputs.
    """
    if ret_type == LOGITS:
        logits = [ele[LOGITS] for ele in outputs]
        ret = torch.cat(logits)
    elif ret_type == PROBABILITY:
        probability = [ele[PROBABILITY] for ele in outputs]
        ret = torch.cat(probability)
    elif ret_type == FEATURES:
        features = [ele[FEATURES] for ele in outputs]
        ret = torch.cat(features)
    elif ret_type == COLUMN_FEATURES:
        ret = {}
        column_features = [ele[COLUMN_FEATURES][FEATURES] for ele in outputs]  # a list of dicts
        for feature_name in column_features[0].keys():
            ret[feature_name] = torch.cat([ele[feature_name] for ele in column_features])
    elif ret_type == MASKS:
        ret = {}
        feature_masks = [ele[COLUMN_FEATURES][MASKS] for ele in outputs]  # a list of dicts
        for feature_name in feature_masks[0].keys():
            ret[feature_name] = torch.cat([ele[feature_name] for ele in feature_masks])
    elif ret_type == BBOX:
        # outputs shape: num_batch, 1(["bbox"]), batch_size, 2(if using mask_rcnn)/na, 80, n, 5
        if len(outputs[0]["bbox"][0]) == 2:  # additional axis for mask_rcnn
            return [bbox[0] for ele in outputs for bbox in ele[BBOX]]
        else:
            return [bbox for ele in outputs for bbox in ele[BBOX]]
    elif ret_type == TEXT:
        return [ele[TEXT] for ele in outputs]  # single image
    elif ret_type == SCORE:
        return [ele[SCORE] for ele in outputs]
    else:
        raise ValueError(f"Unknown return type: {ret_type}")

    if as_ndarray:
        if isinstance(ret, torch.Tensor):
            ret = tensor_to_ndarray(ret)
        elif isinstance(ret, dict):
            ret = {k: tensor_to_ndarray(v) for k, v in ret.items()}
        else:
            raise ValueError(f"Unsupported ret type: {type(ret)}")
    return ret


def infer_batch(
    batch: Dict, model: nn.Module, precision: Union[str, int], num_gpus: int, model_postprocess_fn: Callable
):
    """
    Perform inference for a batch.

    Parameters
    ----------
    batch
        The batch data.
    model
        A Pytorch model.
    precision
        The desired precision used in inference.
    loss_func
        The loss function used in training the model.

    Returns
    -------
    Model output.
    """
    num_gpus = compute_num_gpus(config_num_gpus=num_gpus, strategy="dp")
    precision = infer_precision(num_gpus=num_gpus, precision=precision, as_torch=True)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    batch_size = len(batch[next(iter(batch))])
    if 1 < num_gpus <= batch_size:
        model = nn.DataParallel(model)
    model.to(device).eval()
    batch = move_to_device(batch, device=device)
    with torch.autocast(device_type=device_type, dtype=precision):
        with torch.no_grad():
            output = model(batch)
            if model_postprocess_fn:
                output = model_postprocess_fn(output)

    return output[model.prefix]
