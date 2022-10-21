import logging
from typing import Dict, List, Optional, Tuple, Union
import copy

from ..constants import AUTOMM, QUERY, RESPONSE
from .config import get_config, customize_model_names
from .data import create_fusion_data_processors
from .model import create_fusion_model
from .matcher import create_siamese_model

logger = logging.getLogger(AUTOMM)


def init_pretrained(
    pipeline: Optional[str],
    hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
):
    """
    Zero shot initialization.

    Parameters
    ----------
    hyperparameters
        The customized hyperparameters used to override the default.
        Users need to use it to choose one model, e.g., {"model.names": ["clip"]}.

    Returns
    -------
    config
        A DictConfig object containing the configurations for zero-shot learning.
    model
        The model with pre-trained weights.
    data_processors
        The data processors associated with the pre-trained model.
    """
    config = get_config(presets=pipeline, overrides=hyperparameters)
    assert (
        len(config.model.names) == 1
    ), f"Zero shot mode only supports using one model, but detects multiple models {config.model.names}"
    model = create_fusion_model(config=config, pretrained=True)

    data_processors = create_fusion_data_processors(
        config=config,
        model=model,
    )

    return config, model, data_processors


def init_pretrained_matcher(
    pipeline: Optional[str],
    hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
):
    """
    Zero shot initialization.

    Parameters
    ----------
    hyperparameters
        The customized hyperparameters used to override the default.
        Users need to use it to choose one model, e.g., {"model.names": ["clip"]}.

    Returns
    -------
    config
        A DictConfig object containing the configurations for zero-shot learning.
    query_config
    response_config
    query_model
    response_model
        The model with pre-trained weights.
    query_processors
        The data processors associated with the pre-trained model.
    response_processors
    """
    config = get_config(
        presets=pipeline,
        overrides=hyperparameters,
        extra=["matcher"],
    )
    assert (
        len(config.model.names) == 1
    ), f"Zero shot mode only supports using one model, but detects multiple models {config.model.names}"

    query_config = copy.deepcopy(config)
    # customize config model names to make them consistent with model prefixes.
    query_config.model = customize_model_names(
        config=query_config.model, customized_names=[f"{n}_{QUERY}" for n in query_config.model.names]
    )

    response_config = copy.deepcopy(config)
    # customize config model names to make them consistent with model prefixes.
    response_config.model = customize_model_names(
        config=response_config.model,
        customized_names=[f"{n}_{RESPONSE}" for n in response_config.model.names],
    )

    query_model, response_model = create_siamese_model(
        query_config=query_config,
        response_config=response_config,
    )

    query_processors = create_fusion_data_processors(
        model=query_model,
        config=query_config,
        requires_label=False,
        requires_data=True,
    )

    response_processors = create_fusion_data_processors(
        model=response_model,
        config=response_config,
        requires_label=False,
        requires_data=True,
    )

    return config, query_config, response_config, query_model, response_model, query_processors, response_processors