import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from scipy.special import softmax
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
    OBJECT_DETECTION,
    PROBABILITY,
    QUERY,
    RESPONSE,
    SCORE,
    TEXT,
    TOKEN_WORD_MAPPING,
    WORD_OFFSETS,
)
from ..data.preprocess_dataframe import MultiModalFeaturePreprocessor
from ..data.utils import apply_data_processor, apply_df_preprocessor, get_collate_fn, get_per_sample_features
from ..models.utils import run_model
from .environment import (
    compute_inference_batch_size,
    compute_num_gpus,
    get_precision_context,
    infer_precision,
    move_to_device,
)
from .log import LogFilter, apply_log_filter
from .matcher import compute_matching_probability
from .misc import tensor_to_ndarray

logger = logging.getLogger(__name__)


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
            logits = softmax(ele[LOGITS].detach().cpu().numpy(), axis=-1)
            token_word_mapping = ele[TOKEN_WORD_MAPPING].detach().cpu().numpy()
            word_offsets = ele[WORD_OFFSETS].detach().cpu().numpy()
            for token_preds, logit, mappings, offsets in zip(logits_label, logits, token_word_mapping, word_offsets):
                pred_one_sentence, word_offset, pred_proba = [], [], []
                counter = 0
                temp = set()
                for token_pred, mapping, lt in zip(token_preds, mappings, logit):
                    if mapping != -1 and mapping not in temp:
                        temp.add(mapping)
                        word_offset.append(list(offsets[counter]))
                        pred_one_sentence.append(token_pred)
                        pred_proba.append(lt)
                        counter += 1
                ner_pred.append((pred_one_sentence, word_offset, pred_proba))
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
    batch: Dict,
    model: nn.Module,
    precision: Union[str, int],
    num_gpus: int,
    model_postprocess_fn: Callable = None,
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
        output = run_model(model, batch)
        if model_postprocess_fn:
            output = model_postprocess_fn(output)

    if isinstance(model, nn.DataParallel):
        model = model.module
    else:
        model = model
    output = move_to_device(output, device=torch.device("cpu"))
    return output[model.prefix]


def infer_matcher_batch(
    batch: Dict,
    query_model: nn.Module,
    response_model: nn.Module,
    signature: str,
    match_label: int,
    precision: Union[str, int],
    num_gpus: int,
):
    """
    Perform matcher inference for a batch.

    Parameters
    ----------
    batch
        The batch data.
    query_model
        Query model.
    response_model
        Response model.
    signature
        query, response, or None.
    match_label
        0 or 1.
    precision
        The desired precision used in inference.
    num_gpus
        Number of GPUs.

    Returns
    -------
    Model output.
    """
    if signature is None or signature == QUERY:
        output = infer_batch(
            batch=batch,
            model=query_model,
            precision=precision,
            num_gpus=num_gpus,
        )
        query_embeddings = output[FEATURES]

    if signature is None or signature == RESPONSE:
        output = infer_batch(
            batch=batch,
            model=response_model,
            precision=precision,
            num_gpus=num_gpus,
        )
        response_embeddings = output[FEATURES]

    if signature == QUERY:
        return {FEATURES: query_embeddings}
    elif signature == RESPONSE:
        return {FEATURES: response_embeddings}
    else:
        match_prob = compute_matching_probability(
            embeddings1=query_embeddings,
            embeddings2=response_embeddings,
        )
        if match_label == 0:
            probability = torch.stack([match_prob, 1 - match_prob]).t()
        else:
            probability = torch.stack([1 - match_prob, match_prob]).t()

        return {PROBABILITY: probability}


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


def process_batch(
    data: pd.DataFrame,
    df_preprocessor: Union[MultiModalFeaturePreprocessor, List[MultiModalFeaturePreprocessor]],
    data_processors: Union[Dict, List[Dict]],
    id_mappings: Union[Dict[str, Dict], Dict[str, pd.Series]] = None,
):
    """
    process data to get a batch.

    Parameters
    ----------
    data
        A dataframe.
    df_preprocessor
        Dataframe preprocessors.
    data_processors
        Data processors.
    id_mappings
        Id-to-content mappings. The contents can be text, image, etc.
        This is used when the dataframe contains the query/response indexes instead of their contents.

    Returns
    -------
    A dict of tensors.
    """
    if isinstance(df_preprocessor, MultiModalFeaturePreprocessor):
        df_preprocessor = [df_preprocessor]
    if isinstance(data_processors, dict):
        data_processors = [data_processors]

    modality_features = dict()
    modality_types = dict()
    sample_num = dict()

    for i, (per_preprocessor, per_processors_group) in enumerate(zip(df_preprocessor, data_processors)):
        modality_features[i], modality_types[i], sample_num[i] = apply_df_preprocessor(
            data=data,
            df_preprocessor=per_preprocessor,
            modalities=per_processors_group.keys(),
        )
    sample_num = list(sample_num.values())
    assert len(set(sample_num)) == 1
    sample_num = sample_num[0]

    processed_features = []
    for i in range(sample_num):
        per_sample_features = dict()
        for group_id, per_processors_group in enumerate(data_processors):
            per_sample_features_group = get_per_sample_features(
                modality_features=modality_features[group_id],
                modality_types=modality_types[group_id],
                idx=i,
                id_mappings=id_mappings,
            )
            per_sample_features_group = apply_data_processor(
                per_sample_features=per_sample_features_group,
                data_processors=per_processors_group,
                feature_modalities=modality_types[group_id],
                is_training=False,
            )
            per_sample_features.update(per_sample_features_group)

        processed_features.append(per_sample_features)

    collate_fn = get_collate_fn(
        df_preprocessor=df_preprocessor, data_processors=data_processors, per_gpu_batch_size=sample_num
    )
    batch = collate_fn(processed_features)

    return batch


def realtime_predict(
    data: pd.DataFrame,
    df_preprocessor: Union[MultiModalFeaturePreprocessor, List[MultiModalFeaturePreprocessor]],
    data_processors: Union[Dict, List[Dict]],
    num_gpus: int,
    precision: Union[int, str],
    model: Optional[nn.Module] = None,
    query_model: Optional[nn.Module] = None,
    response_model: Optional[nn.Module] = None,
    model_postprocess_fn: Optional[Callable] = None,
    is_matching: Optional[bool] = False,
    id_mappings: Union[Dict[str, Dict], Dict[str, pd.Series]] = None,
    signature: Optional[str] = None,
    match_label: Optional[int] = None,
) -> List[Dict]:
    """
    Perform realtime inference.

    Parameters
    ----------
    data
        A dataframe.
    df_preprocessor
        Dataframe preprocessors.
    data_processors
        Data processors.
    num_gpus
        Number of GPUs.
    precision
        The precision used in inference.
    model
        Predictor's model.
    query_model
        Matcher's query model.
    response_model
        Matcher's response model.
    model_postprocess_fn
        Model postprocessing function.
    is_matching
        Whether is matching.
    id_mappings
        Id-to-content mappings. The contents can be text, image, etc.
        This is used when the dataframe contains the query/response indexes instead of their contents.
    signature
        query or response.
    match_label
        0 or 1.

    Returns
    -------
    A list of output dicts.
    """

    batch = process_batch(
        data=data,
        df_preprocessor=df_preprocessor,
        data_processors=data_processors,
        id_mappings=id_mappings,
    )
    if is_matching:
        output = infer_matcher_batch(
            batch=batch,
            query_model=query_model,
            response_model=response_model,
            signature=signature,
            match_label=match_label,
            precision=precision,
            num_gpus=num_gpus,
        )
    else:
        output = infer_batch(
            batch=batch,
            model=model,
            precision=precision,
            num_gpus=num_gpus,
            model_postprocess_fn=model_postprocess_fn,
        )
    return [output]


def predict(
    predictor,
    data: Union[pd.DataFrame, dict, list],
    requires_label: bool,
    id_mappings: Union[Dict[str, Dict], Dict[str, pd.Series]] = None,
    signature: Optional[str] = None,
    realtime: Optional[bool] = None,
    is_matching: Optional[bool] = False,
) -> List[Dict]:
    """
    Perform inference for predictor or matcher.

    Parameters
    ----------
    predictor
        A predictor or matcher object.
    data
        The data for inference.
    requires_label
        Whether uses label during inference.
    id_mappings
        Id-to-content mappings. The contents can be text, image, etc.
        This is used when the dataframe contains the query/response indexes instead of their contents.
    signature
        query or response.
    realtime
        Whether use realtime infernece.
    is_matching
        Whether is matching.
    seed
        random seed.

    Returns
    -------
    A list of output dicts.
    """
    if is_matching:
        data, df_preprocessor, data_processors, match_label = predictor._on_predict_start(
            data=data,
            id_mappings=id_mappings,
            requires_label=requires_label,
            signature=signature,
        )
    else:
        data, df_preprocessor, data_processors = predictor._on_predict_start(
            data=data,
            requires_label=requires_label,
        )

    strategy = "dp"  # default used in inference.

    num_gpus = compute_num_gpus(config_num_gpus=predictor._config.env.num_gpus, strategy=strategy)

    if predictor._problem_type == OBJECT_DETECTION:
        strategy = "ddp"

    if strategy == "ddp" and predictor._fit_called:
        num_gpus = 1  # While using DDP, we can only use single gpu after fit is called

    if num_gpus <= 1:
        # Force set strategy to be None if it's cpu-only or we have only one GPU.
        strategy = None

    precision = infer_precision(num_gpus=num_gpus, precision=predictor._config.env.precision, cpu_only_warning=False)

    if not realtime:
        batch_size = compute_inference_batch_size(
            per_gpu_batch_size=predictor._config.env.per_gpu_batch_size,
            eval_batch_size_ratio=OmegaConf.select(predictor._config, "env.eval_batch_size_ratio"),
            per_gpu_batch_size_evaluation=predictor._config.env.per_gpu_batch_size_evaluation,  # backward compatibility.
            num_gpus=num_gpus,
            strategy=strategy,
        )

    if realtime is None:
        realtime = use_realtime(data=data, data_processors=data_processors, batch_size=batch_size)

    if predictor._problem_type == OBJECT_DETECTION:
        realtime = False

    if realtime:
        if is_matching:
            outputs = realtime_predict(
                query_model=predictor._query_model,
                response_model=predictor._response_model,
                data=data,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                num_gpus=num_gpus,
                precision=precision,
                is_matching=True,
                id_mappings=id_mappings,
                signature=signature,
                match_label=match_label,
            )
        else:
            outputs = realtime_predict(
                model=predictor._model,
                data=data,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                num_gpus=num_gpus,
                precision=precision,
                model_postprocess_fn=predictor._model_postprocess_fn,
            )
    else:
        if is_matching:
            outputs = predictor._default_predict(
                data=data,
                id_mappings=id_mappings,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                num_gpus=num_gpus,
                precision=precision,
                batch_size=batch_size,
                strategy=strategy,
                match_label=match_label,
                signature=signature,
            )
        else:
            outputs = predictor._default_predict(
                data=data,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                num_gpus=num_gpus,
                precision=precision,
                batch_size=batch_size,
                strategy=strategy,
            )

    return outputs
