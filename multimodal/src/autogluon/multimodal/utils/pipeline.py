import logging
from typing import Dict, List, Optional, Tuple, Union

from ..constants import AUTOMM
from .config import get_config
from .data import create_fusion_data_processors
from .model import create_fusion_model

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
