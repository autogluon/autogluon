import logging
import os
from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf
from torch import nn

from ..constants import AUTOMM, HF_MODELS, LAST_CHECKPOINT
from ..data import TextProcessor

logger = logging.getLogger(AUTOMM)


def save_pretrained_model_configs(
    model: nn.Module,
    config: DictConfig,
    path: str,
) -> DictConfig:
    """
    Save the pretrained model configs to local to make future loading not dependent on Internet access.
    By initializing models with local configs, Huggingface doesn't need to download pretrained weights from Internet.

    Parameters
    ----------
    model
        One model.
    config
        A DictConfig object. The model config should be accessible by "config.model".
    path
        The path to save pretrained model configs.
    """
    # TODO? Fix hardcoded model names.
    requires_saving = any([model_name.lower().startswith(HF_MODELS) for model_name in config.model.names])
    if not requires_saving:
        return config

    if (
        len(config.model.names) == 1
    ):  # TODO: Not sure this is a sufficient check. Hyperparameter "model.names" : ["hf_text", "fusion_mlp"] fails here.
        model = nn.ModuleList([model])
    else:  # assumes the fusion model has a model attribute, a nn.ModuleList
        model = model.model
    for per_model in model:
        if per_model.prefix.lower().startswith(HF_MODELS):
            per_model.config.save_pretrained(os.path.join(path, per_model.prefix))
            model_config = getattr(config.model, per_model.prefix)
            model_config.checkpoint_name = os.path.join("local://", per_model.prefix)

    return config


def save_text_tokenizers(
    text_processors: List[TextProcessor],
    path: str,
) -> List[TextProcessor]:
    """
    Save all the text tokenizers and record their relative paths, which are
    the corresponding model names, e.g, hf_text.

    Parameters
    ----------
    text_processors
        A list of text processors with tokenizers.
    path
        The root path.

    Returns
    -------
    A list of text processors with tokenizers replaced by their local relative paths.
    """
    for per_text_processor in text_processors:
        per_path = os.path.join(path, per_text_processor.prefix)
        per_text_processor.tokenizer.save_pretrained(per_path)
        per_text_processor.tokenizer = per_text_processor.prefix

    return text_processors


def process_save_path(path, resume: Optional[bool] = False, raise_if_exist: Optional[bool] = True):
    """
    Convert the provided path to an absolute path and check whether it is valid.
    If a path exists, either raise error or return None.
    A None path can be identified by the `setup_outputdir` to generate a random path.

    Parameters
    ----------
    path
        A provided path.
    resume
        Whether this is a path to resume training.
    raise_if_exist
        Whether to raise error if the path exists.

    Returns
    -------
    A complete and verified path or None.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if resume:
        assert os.path.isfile(os.path.join(path, LAST_CHECKPOINT)), (
            f"Trying to resume training from '{path}'. "
            f"However, it does not contain the last checkpoint file: '{LAST_CHECKPOINT}'. "
            "Are you using a correct path?"
        )
    elif os.path.isdir(path):
        if raise_if_exist:
            raise ValueError(
                f"Path {path} already exists."
                "Specify a new path to avoid accidentally overwriting a saved predictor."
            )
        else:
            path = None

    return path
