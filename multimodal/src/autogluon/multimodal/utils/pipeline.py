import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

from ..constants import AUTOMM, QUERY, RESPONSE
from .config import customize_model_names, get_config
from .data import create_fusion_data_processors
from .matcher import create_siamese_model
from .model import create_fusion_model

logger = logging.getLogger(AUTOMM)


def init_pretrained(
    problem_type: Optional[str] = None,
    presets: Optional[str] = None,
    hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
    num_classes: Optional[int] = None,
    classes: Optional[list] = None,
    init_scratch: Optional[bool] = False,
):
    """
    Zero shot initialization.

    Parameters
    ----------
    problem_type
        Problem type.
    presets
        Presets regarding model quality, e.g., best_quality, high_quality, and medium_quality.
    hyperparameters
        The customized hyperparameters used to override the default.

    Returns
    -------
    config
        A DictConfig object containing the configurations for zero-shot learning.
    model
        The model with pre-trained weights.
    data_processors
        The data processors associated with the pre-trained model.
    """
    config = get_config(problem_type=problem_type, presets=presets, overrides=hyperparameters)
    assert (
        len(config.model.names) == 1
    ), f"Zero shot mode only supports using one model, but detects multiple models {config.model.names}"
    model = create_fusion_model(config=config, pretrained=(not init_scratch), num_classes=num_classes, classes=classes)

    data_processors = create_fusion_data_processors(
        config=config,
        model=model,
    )

    return config, model, data_processors


def init_pretrained_matcher(
    pipeline: Optional[str] = None,
    presets: Optional[str] = None,
    hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
):
    """
    Zero shot initialization.

    Parameters
    ----------
    pipeline
        Matching pipeline.
    presets
        Presets regarding model quality, e.g., best_quality, high_quality, and medium_quality.
    hyperparameters
        The customized hyperparameters used to override the default.

    Returns
    -------
    config
        A DictConfig object containing the configurations for the pipeline.
    query_config
        Configurations for the query model and related.
    response_config
        Configurations for the response model and related.
    query_model
        Query model with pre-trained weights.
    response_model
        Response model with pre-trained weights.
    query_processors
        The data processors associated with the query model.
    response_processors
        The data processors associated with the response model.
    """
    config = get_config(
        problem_type=pipeline,
        presets=presets,
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
