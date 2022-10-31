import copy
import functools
import heapq
import logging
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from omegaconf import DictConfig
from torch import nn

from ..constants import AUTOMM, FUSION, QUERY, RESPONSE
from .data import data_to_df
from .model import create_model

logger = logging.getLogger(AUTOMM)


def get_fusion_model_dict(
    model,
    single_models: Optional[Dict] = None,
):
    """
    Take apart a late-fusion model into a dict of single models and a fusion piece.

    Parameters
    ----------
    model
        A late-fusion model.
    single_models
        A dict of single models.

    Returns
    -------
    single_models
        A dict of single models.
    fusion_model
        The fusion part of a late-fusion model.
    """
    if not single_models:
        single_models = {}
    fusion_model = None
    if model.prefix.startswith(FUSION):  # fusion model
        fusion_model = model
        models = model.model
        model.model = None
    else:
        models = [model]

    for per_model in models:
        if per_model.prefix.endswith(QUERY):
            model_name = per_model.prefix[:-6]  # cut off query
        elif per_model.prefix.endswith(RESPONSE):
            model_name = per_model.prefix[:-9]
        else:
            raise ValueError(f"Model prefix {per_model.prefix} doesn't end with {QUERY} or {RESPONSE}.")

        if model_name not in single_models:
            single_models[model_name] = per_model

    return single_models, fusion_model


def create_fusion_model_dict(
    config: DictConfig,
    single_models: Optional[Dict] = None,
):
    """
    Create a dict of single models and fusion piece based on a late-fusion config.

    Parameters
    ----------
    config
        The model config.
    single_models
        A dict of single models used in the late-fusion.

    Returns
    -------
    single_models
        A dict of single models.
    fusion_model
        The fusion part of a late-fusion model.
    """
    if not single_models:
        single_models = {}
    fusion_model = None
    for model_name in config.names:
        model_config = getattr(config, model_name)
        if not model_name.lower().startswith(FUSION):
            if model_name.endswith(QUERY):
                model_name = model_name[:-6]  # cut off query
            elif model_name.endswith(RESPONSE):
                model_name = model_name[:-9]
            else:
                raise ValueError(f"Model name {model_name} doesn't end with {QUERY} or {RESPONSE}.")

            if model_name in single_models:
                continue
        model = create_model(
            model_name=model_name,
            model_config=model_config,
        )
        if model_name.lower().startswith(FUSION):
            fusion_model = model
        else:
            single_models[model_name] = model

    return single_models, fusion_model


def make_siamese(
    query_config: DictConfig,
    response_config: DictConfig,
    single_models: Dict,
    query_fusion_model: Union[nn.Module, functools.partial],
    response_fusion_model: Union[nn.Module, functools.partial],
    share_fusion: bool,
    initialized: Optional[bool] = False,
):
    """
    Build a siamese network, in which the query and response share the same encoders for the same modalities.

    Parameters
    ----------
    query_config
        The query config.
    response_config
        The response config.
    single_models
        A dict of single models used in the late-fusion.
    query_fusion_model
        The fusion piece of the query model.
    response_fusion_model
        The fusion piece of the response model.
    share_fusion
        Whether the query and response share the fusion piece.
    initialized
        Whether the fusion piece is initialized.

    Returns
    -------
    The query and response models satisfying the siamese constraint.
    """
    query_model_names = [n for n in query_config.model.names if not n.lower().startswith(FUSION)]
    query_fusion_model_name = [n for n in query_config.model.names if n.lower().startswith(FUSION)]
    assert len(query_fusion_model_name) <= 1
    if len(query_fusion_model_name) == 1:
        query_fusion_model_name = query_fusion_model_name[0]
    response_model_names = [n for n in response_config.model.names if not n.lower().startswith(FUSION)]
    response_fusion_model_name = [n for n in response_config.model.names if n.lower().startswith(FUSION)]
    assert len(response_fusion_model_name) <= 1
    if len(response_fusion_model_name) == 1:
        response_fusion_model_name = response_fusion_model_name[0]

    logger.debug(f"single model names: {list(single_models.keys())}")
    logger.debug(f"query fusion model name: {query_fusion_model_name}")
    logger.debug(f"response fusion model name: {response_fusion_model_name}")

    # use shallow copy to create query single models
    query_single_models = []
    for model_name in query_model_names:
        model = copy.copy(single_models[model_name[:-6]])  # cut off _query
        model.prefix = model_name
        query_single_models.append(model)

    # use shallow copy to create response single models
    response_single_models = []
    for model_name in response_model_names:
        model = copy.copy(single_models[model_name[:-9]])  # cut off _response
        model.prefix = model_name
        response_single_models.append(model)

    if len(query_single_models) == 1:
        query_model = query_single_models[0]
    else:
        if initialized:
            query_model = query_fusion_model
            query_model.model = nn.ModuleList(query_single_models)
        else:
            query_model = query_fusion_model(models=query_single_models)

        query_model.prefix = query_fusion_model_name

    if len(response_single_models) == 1:
        response_model = response_single_models[0]
    else:
        if share_fusion:
            response_model = copy.copy(query_model)  # copy query_model rather than query_fusion_model
            response_model.model = nn.ModuleList(response_single_models)
        else:
            if initialized:
                response_model = response_fusion_model
                response_model.model = nn.ModuleList(response_single_models)
            else:
                response_model = response_fusion_model(models=response_single_models)

        response_model.prefix = response_fusion_model_name

    return query_model, response_model


def is_share_fusion(
    query_model_names: List[str],
    response_model_names: List[str],
):
    """
    Check whether the query and response models share the same fusion part.

    Parameters
    ----------
    query_model_names
        Names of single models in the query late-fusion model.
    response_model_names
        Names of single models in the response late-fusion model.

    Returns
    -------
    Whether to share the same fusion part.
    """
    query_model_names = [n for n in query_model_names if not n.lower().startswith(FUSION)]
    response_model_names = [n for n in response_model_names if not n.lower().startswith(FUSION)]
    return sorted(query_model_names) == sorted(response_model_names)


def create_siamese_model(
    query_config: DictConfig,
    response_config: DictConfig,
    query_model: Optional[nn.Module] = None,
    response_model: Optional[nn.Module] = None,
):
    """
    Create the query and response models and make them share the same encoders for the same modalities.

    Parameters
    ----------
    query_config
        The query config.
    response_config
        The response config.
    query_model
        The query model if already created.
    response_model
        The response model if already created.

    Returns
    -------
    The query and response models satisfying the siamese constraint.
    """
    if query_model is None:
        single_models, query_fusion_model = create_fusion_model_dict(
            config=query_config.model,
        )
    else:
        single_models, query_fusion_model = get_fusion_model_dict(
            model=query_model,
        )

    if response_model is None:
        single_models, response_fusion_model = create_fusion_model_dict(
            config=response_config.model,
            single_models=single_models,
        )
    else:
        single_models, response_fusion_model = get_fusion_model_dict(
            model=response_model,
            single_models=single_models,
        )

    share_fusion = is_share_fusion(
        query_model_names=query_config.model.names,
        response_model_names=response_config.model.names,
    )
    query_model, response_model = make_siamese(
        query_config=query_config,
        response_config=response_config,
        single_models=single_models,
        query_fusion_model=query_fusion_model,
        response_fusion_model=response_fusion_model,
        share_fusion=share_fusion,
        initialized=False,
    )

    return query_model, response_model


def compute_semantic_similarity(a: torch.Tensor, b: torch.Tensor, similarity_type: Optional[str] = "cosine"):
    """
    Compute the semantic similarity of each vector in `a` with each vector in `b`.

    Parameters
    ----------
    a
        A tensor with shape (n, dim).
    b
        A tensor with shape (m, dim).
    similarity_type
        Use what function (cosine/dot_prod) to score the similarity (default: cosine).

    Returns
    -------
    A similarity matrix with shape (n, m).
    """
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    if similarity_type == "cosine":
        a = torch.nn.functional.normalize(a, p=2, dim=1)
        b = torch.nn.functional.normalize(b, p=2, dim=1)
    elif similarity_type == "dot_prod":
        pass
    else:
        raise ValueError(
            f"Invalid similarity type: {similarity_type}. The supported types are `cosine` and `dot_prod`."
        )

    return torch.mm(a, b.transpose(0, 1))


def semantic_search(
    matcher,
    query_data: Optional[Union[pd.DataFrame, dict, list]] = None,
    response_data: Optional[Union[pd.DataFrame, dict, list]] = None,
    query_embeddings: Optional[torch.Tensor] = None,
    response_embeddings: Optional[torch.Tensor] = None,
    query_chunk_size: int = 100,
    response_chunk_size: int = 500000,
    top_k: int = 10,
    id_mappings: Optional[Dict[str, Dict]] = None,
    similarity_type: Optional[str] = "cosine",
):
    """
    Perform a cosine similarity search between query data and response data.

    Parameters
    ----------
    query_data
        The query data.
    response_data
        The response data.
    query_embeddings
        2-D query embeddings.
    response_embeddings
        2-D response embeddings.
    id_mappings
        Id-to-content mappings. The contents can be text, image, etc.
        This is used when the dataframe contains the query/response indexes instead of their contents.
    query_chunk_size
        Process queries by query_chunk_size each time.
    response_chunk_size
        Process response data by response_chunk_size each time.
    top_k
        Retrieve top k matching entries.
    similarity_type
        Use what function (cosine/dot_prod) to score the similarity (default: cosine).

    Returns
    -------
    Search results.
    """
    assert (
        query_data is None or query_embeddings is None
    ), "Both query_data and query_embeddings are detected, but you can only use one of them."
    assert query_data is not None or query_embeddings is not None, "Both query_data and query_embeddings are None."
    assert (
        response_data is None or response_embeddings is None
    ), "Both response_data and response_embeddings are detected, but you can only use one of them."
    assert (
        response_data is not None or response_embeddings is not None
    ), "Both response_data and response_embeddings are None."

    if query_embeddings is None:
        query_header = matcher._query[0] if matcher._query is not None else QUERY
        query_data = data_to_df(query_data, header=query_header)
    if response_embeddings is None:
        response_header = matcher._response[0] if matcher._response else RESPONSE
        response_data = data_to_df(response_data, header=response_header)

    if query_embeddings is None:
        num_queries = len(query_data)
    else:
        num_queries = len(query_embeddings)

    if response_embeddings is None:
        num_responses = len(response_data)
    else:
        num_responses = len(response_embeddings)

    queries_result_list = [[] for _ in range(num_queries)]

    for query_start_idx in range(0, num_queries, query_chunk_size):
        if query_embeddings is None:
            batch_query_embeddings = matcher.extract_embedding(
                query_data[query_start_idx : query_start_idx + query_chunk_size],
                signature=QUERY,
                id_mappings=id_mappings,
                as_tensor=True,
            )
        else:
            batch_query_embeddings = query_embeddings[query_start_idx : query_start_idx + query_chunk_size]
        # Iterate over chunks of the corpus
        for response_start_idx in range(0, num_responses, response_chunk_size):
            if response_embeddings is None:
                batch_response_embeddings = matcher.extract_embedding(
                    response_data[response_start_idx : response_start_idx + response_chunk_size],
                    signature=RESPONSE,
                    id_mappings=id_mappings,
                    as_tensor=True,
                )
            else:
                batch_response_embeddings = response_embeddings[
                    response_start_idx : response_start_idx + response_chunk_size
                ]
            # Compute cosine similarities
            scores = compute_semantic_similarity(
                a=batch_query_embeddings,
                b=batch_response_embeddings,
                similarity_type=similarity_type,
            )

            # Get top-k scores
            scores_top_k_values, scores_top_k_idx = torch.topk(
                scores,
                k=min(top_k, len(scores[0])),
                dim=1,
                largest=True,
                sorted=False,
            )
            scores_top_k_values = scores_top_k_values.cpu().tolist()
            scores_top_k_idx = scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(scores)):
                for sub_response_id, score in zip(scores_top_k_idx[query_itr], scores_top_k_values[query_itr]):
                    corpus_id = response_start_idx + sub_response_id
                    query_id = query_start_idx + query_itr
                    if len(queries_result_list[query_id]) < top_k:
                        heapq.heappush(
                            queries_result_list[query_id], (score, corpus_id)
                        )  # heaqp tracks the quantity of the first element in the tuple
                    else:
                        heapq.heappushpop(queries_result_list[query_id], (score, corpus_id))

    # change the data format and sort
    for query_id in range(len(queries_result_list)):
        for doc_itr in range(len(queries_result_list[query_id])):
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {"corpus_id": corpus_id, "score": score}
        queries_result_list[query_id] = sorted(queries_result_list[query_id], key=lambda x: x["score"], reverse=True)

    return queries_result_list
