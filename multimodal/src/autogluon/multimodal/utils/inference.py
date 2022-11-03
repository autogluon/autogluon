import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import nn

from ..constants import (
    AUTOMM,
    BBOX,
    COLUMN_FEATURES,
    FEATURES,
    IMAGE,
    LOGITS,
    MASKS,
    NER,
    NER_ANNOTATION,
    NER_RET,
    PROBABILITY,
    SCORE,
    TEXT,
    TOKEN_WORD_MAPPING,
    WORD_OFFSETS,
)
from .environment import get_precision_context, move_to_device
from .misc import tensor_to_ndarray

logger = logging.getLogger(AUTOMM)


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
        ret = torch.cat(logits).nan_to_num(nan=-1e4)
    elif ret_type == PROBABILITY:
        probability = [ele[PROBABILITY] for ele in outputs]
        ret = torch.cat(probability).nan_to_num(nan=0)
    elif ret_type == FEATURES:
        features = [ele[FEATURES] for ele in outputs]
        ret = torch.cat(features).nan_to_num(nan=0)
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
        return [ele[BBOX] for ele in outputs]
    elif ret_type == TEXT:
        return [ele[TEXT] for ele in outputs]  # single image
    elif ret_type == SCORE:
        return [ele[SCORE] for ele in outputs]
    elif ret_type == NER_RET:
        ner_pred = []
        as_ndarray = False
        for ele in outputs:
            logits_label = ele[NER_ANNOTATION].detach().cpu().numpy()
            token_word_mapping = ele[TOKEN_WORD_MAPPING].detach().cpu().numpy()
            word_offsets = ele[WORD_OFFSETS].detach().cpu().numpy()
            for token_preds, mappings, offsets in zip(logits_label, token_word_mapping, word_offsets):
                pred_one_sentence, word_offset = [], []
                counter = 0
                temp = set()
                for token_pred, mapping in zip(token_preds, mappings):
                    if mapping != -1 and mapping not in temp:
                        temp.add(mapping)
                        word_offset.append(list(offsets[counter]))
                        pred_one_sentence.append(token_pred)
                        counter += 1
                ner_pred.append((pred_one_sentence, word_offset))
        return ner_pred
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
    num_gpus
        Number of GPUs.
    model_postprocess_fn
        The post-processing function for the model output.

    Returns
    -------
    Model output.
    """
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    batch_size = len(batch[next(iter(batch))])
    if 1 < num_gpus <= batch_size:
        model = nn.DataParallel(model)
    model.to(device).eval()
    batch = move_to_device(batch, device=device)
    precision_context = get_precision_context(precision=precision, device_type=device_type)
    with precision_context, torch.no_grad():
        output = model(batch)
        if model_postprocess_fn:
            output = model_postprocess_fn(output)

    if isinstance(model, nn.DataParallel):
        model = model.module
    else:
        model = model
    return output[model.prefix]


def use_realtime(data: pd.DataFrame, data_processors: Dict, batch_size: int):
    """
    Determine whether to use the realtime inference based on the sample number
    and the data modalities. Loading image data requires more time than text.
    Thus, we set a small threshold for image data. We may also consider the
    model size in future, but we need to ensure this function is efficient since
    using this function also costs additional inference time.

    Parameters
    ----------
    data
        A dataframe.
    data_processors
        A dict of data processors.
    batch_size
        The batch size from config.

    Returns
    -------
    Whether to use the realtime inference.
    """
    realtime = False
    sample_num = len(data)
    if IMAGE in data_processors and len(data_processors[IMAGE]) > 0:  # has image
        if sample_num <= min(10, batch_size):
            realtime = True
    elif TEXT in data_processors and len(data_processors[TEXT]) > 0:  # has text but no image
        if sample_num <= min(100, batch_size):
            realtime = True
    else:  # only has tabular data
        if sample_num <= min(200, batch_size):
            realtime = True

    return realtime
