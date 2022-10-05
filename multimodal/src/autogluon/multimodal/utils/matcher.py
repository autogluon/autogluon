import copy
import functools
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig
from torch import nn

from ..constants import QUERY, RESPONSE
from .model import create_model


def get_fusion_model_dict(
    model,
    single_models: Optional[Dict] = None,
):
    if not single_models:
        single_models = {}
    fusion_model = None
    if model.prefix.startswith("fusion"):  # fusion model
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
    if not single_models:
        single_models = {}
    fusion_model = None
    for model_name in config.model.names:
        model_config = getattr(config.model, model_name)
        if not model_name.lower().startswith("fusion"):
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
        if model_name.lower().startswith("fusion"):
            fusion_model = model
        else:
            single_models[model_name] = model

    return single_models, fusion_model


def build_siamese_network(
    query_config: DictConfig,
    response_config: DictConfig,
    single_models: Dict,
    query_fusion_model: Union[nn.Module, functools.partial],
    response_fusion_model: Union[nn.Module, functools.partial],
    share_fusion: bool,
    initialized: Optional[bool] = False,
):
    query_model_names = [n for n in query_config.model.names if not n.lower().startswith("fusion")]
    query_fusion_model_name = [n for n in query_config.model.names if n.lower().startswith("fusion")]
    assert len(query_fusion_model_name) <= 1
    if len(query_fusion_model_name) == 1:
        query_fusion_model_name = query_fusion_model_name[0]
    response_model_names = [n for n in response_config.model.names if not n.lower().startswith("fusion")]
    response_fusion_model_name = [n for n in response_config.model.names if n.lower().startswith("fusion")]
    assert len(response_fusion_model_name) <= 1
    if len(response_fusion_model_name) == 1:
        response_fusion_model_name = response_fusion_model_name[0]

    print(f"single model names: {list(single_models.keys())}")
    print(f"query fusion model name: {query_fusion_model_name}")
    print(f"response fusion model name: {response_fusion_model_name}")

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
    query_model_names = [n for n in query_model_names if not n.lower().startswith("fusion")]
    response_model_names = [n for n in response_model_names if not n.lower().startswith("fusion")]
    return sorted(query_model_names) == sorted(response_model_names)


def create_siamese_model(
    query_config: DictConfig,
    response_config: DictConfig,
    query_model: Optional[nn.Module] = None,
    response_model: Optional[nn.Module] = None,
):
    if query_model is None:
        single_models, query_fusion_model = create_fusion_model_dict(
            config=query_config,
        )
    else:
        single_models, query_fusion_model = get_fusion_model_dict(
            model=query_model,
        )

    if response_model is None:
        single_models, response_fusion_model = create_fusion_model_dict(
            config=response_config,
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
    query_model, response_model = build_siamese_network(
        query_config=query_config,
        response_config=response_config,
        single_models=single_models,
        query_fusion_model=query_fusion_model,
        response_fusion_model=response_fusion_model,
        share_fusion=share_fusion,
        initialized=False,
    )

    return query_model, response_model
